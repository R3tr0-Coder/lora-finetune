"""
src/trainer.py

Production LoRA fine-tuning script — mirrors the notebook but runs from the
command line for automated pipelines, CI, and remote GPU jobs.

Usage:
    python -m src.trainer --config configs/lora_tinyllama.yaml
    python -m src.trainer --config configs/lora_tinyllama.yaml --max_steps 50

The script is deliberately model-agnostic: change BASE_MODEL in the config
and everything adapts automatically.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from datasets import load_dataset
    from peft import (
        LoraConfig, PeftModel, TaskType,
        get_peft_model, prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, TrainingArguments,
    )
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
except Exception:
    pass  # mocked in tests / unavailable in CPU-only CI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: str = "outputs/lora-run"
    hub_model_id: Optional[str] = None

    # Dataset
    dataset_name: str = "timdettmers/openassistant-guanaco"
    dataset_text_col: str = "text"
    response_template: str = "### Assistant:"
    max_seq_len: int = 1024

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # Quantisation
    use_4bit: bool = True
    bnb_compute_dtype: str = "bfloat16"

    # Training
    batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = -1
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.001
    logging_steps: int = 10
    save_steps: int = 100
    report_to: str = "tensorboard"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# ── Builder functions ─────────────────────────────────────────────────────────

def build_bnb_config(cfg: TrainConfig) -> Optional[BitsAndBytesConfig]:
    if not cfg.use_4bit:
        return None
    dtype = getattr(torch, cfg.bnb_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def build_tokenizer(cfg: TrainConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        logger.info("Added [PAD] token — vocab size now %d", len(tokenizer))
    return tokenizer


def build_model(cfg: TrainConfig, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    bnb_config = build_bnb_config(cfg)
    dtype = getattr(torch, cfg.bnb_compute_dtype)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype if not cfg.use_4bit else None,
    )
    logger.info("Base model loaded in %.1fs", time.time() - t0)
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    return model


def apply_lora(model: AutoModelForCausalLM, cfg: TrainConfig) -> AutoModelForCausalLM:
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "LoRA applied — trainable: %d / %d  (%.3f%%)",
        trainable, total, 100 * trainable / total,
    )
    return model


def build_training_args(cfg: TrainConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        report_to=[cfg.report_to],
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        seed=cfg.seed,
        data_seed=cfg.seed,
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )


# ── Main training function ────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> str:
    """
    Run a full fine-tuning job from config.

    Returns
    -------
    str
        Path to the saved adapter directory.
    """
    logger.info("Starting LoRA fine-tuning: %s → %s", cfg.base_model, cfg.output_dir)

    tokenizer = build_tokenizer(cfg)
    model     = build_model(cfg, tokenizer)
    model     = apply_lora(model, cfg)

    dataset  = load_dataset(cfg.dataset_name)
    response_ids = tokenizer.encode(cfg.response_template, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_ids, tokenizer=tokenizer
    )

    def fmt(ex):
        return ex[cfg.dataset_text_col] if isinstance(ex[cfg.dataset_text_col], list) \
               else [ex[cfg.dataset_text_col]]

    training_args = build_training_args(cfg)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        formatting_func=fmt,
        data_collator=collator,
        max_seq_length=cfg.max_seq_len,
        dataset_num_proc=2,
        packing=False,
    )

    t0     = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    logger.info("Training complete: %.1f min | loss: %.4f", elapsed / 60, result.training_loss)

    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    adapter_dir = str(Path(cfg.output_dir) / "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Adapter saved → %s", adapter_dir)

    if cfg.hub_model_id:
        logger.info("Pushing adapter to Hub: %s", cfg.hub_model_id)
        model.push_to_hub(cfg.hub_model_id, private=True)
        tokenizer.push_to_hub(cfg.hub_model_id, private=True)

    return adapter_dir


def merge(cfg: TrainConfig, adapter_dir: str) -> str:
    """
    Merge LoRA adapter into the base model and save the result.

    Returns
    -------
    str
        Path to the merged model directory.
    """
    logger.info("Merging adapter → base model")
    dtype     = getattr(torch, cfg.bnb_compute_dtype)
    tokenizer = build_tokenizer(cfg)

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    base.resize_token_embeddings(len(tokenizer))

    peft_model   = PeftModel.from_pretrained(base, adapter_dir)
    merged_model = peft_model.merge_and_unload()

    merged_dir = str(Path(cfg.output_dir) / "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    logger.info("Merged model saved → %s", merged_dir)
    return merged_dir


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning CLI")
    parser.add_argument("--config",    required=True, help="Path to YAML config file")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--merge",     action="store_true",    help="Merge adapter after training")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
        logger.info("max_steps overridden → %d", cfg.max_steps)

    adapter_dir = train(cfg)

    if args.merge:
        merge(cfg, adapter_dir)


if __name__ == "__main__":
    main()
