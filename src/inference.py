"""
src/inference.py

Inference helpers — load a fine-tuned model (adapter or merged) and generate
responses. Designed to be imported in notebooks or used standalone.

All torch / transformers imports are lazy so this module can be imported
in CPU-only CI environments without a working CUDA stack.

Usage:
    from src.inference import load_model, generate

    model, tokenizer = load_model("outputs/lora-tinyllama/adapter",
                                  base_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(generate(model, tokenizer, "Explain gradient descent simply."))
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Lazy imports ──────────────────────────────────────────────────────────────
# These fail in CPU-only CI environments where CUDA shared libs are absent.
# Tests mock the model/tokenizer objects directly, so the imports are never
# actually needed there.

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except Exception:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:
    pass  # mocked in tests / unavailable in CPU-only CI


def _no_grad():
    """Return torch.no_grad() when torch is available, else a nullcontext."""
    if _TORCH_AVAILABLE:
        return _torch.no_grad()
    return contextlib.nullcontext()


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(
    model_path: str,
    base_model_path: Optional[str] = None,
    load_in_4bit: bool = False,
    torch_dtype=None,   # defaults to torch.bfloat16 when torch is available
):
    """
    Load a model for inference.

    Handles two cases:
      1. model_path is a merged model directory → load directly
      2. model_path is a PEFT adapter + base_model_path provided → load base + adapter

    Parameters
    ----------
    model_path:
        Path to merged model directory OR PEFT adapter directory.
    base_model_path:
        Path/ID of the base model. Required only when loading an adapter.
    load_in_4bit:
        Enable 4-bit inference via bitsandbytes (reduces VRAM by ~75%).
    torch_dtype:
        Computation dtype. Defaults to torch.bfloat16 when torch is available.
    """
    if torch_dtype is None and _TORCH_AVAILABLE:
        torch_dtype = _torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    if base_model_path is not None:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype if not load_in_4bit else None,
        )
        base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, model_path)
        logger.info("Loaded adapter from %s on top of %s", model_path, base_model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch_dtype if not load_in_4bit else None,
        )
        logger.info("Loaded merged model from %s", model_path)

    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    prompt_template: str = "### Human: {prompt}\n### Assistant:",
) -> str:
    """
    Generate a response for the given prompt.

    Parameters
    ----------
    prompt:
        The user's question or instruction.
    system_prompt:
        Optional system-level context prepended to the conversation.
    max_new_tokens:
        Maximum number of tokens to generate.
    temperature:
        Sampling temperature (0 = greedy, >0 = sampling).
    top_p:
        Nucleus sampling threshold.
    top_k:
        Top-k sampling threshold.
    repetition_penalty:
        Penalises repetition (1.0 = no penalty, >1 = discourage repeats).
    prompt_template:
        Format string with a {prompt} placeholder.

    Returns
    -------
    str
        Generated text with the input prompt stripped.
    """
    model.eval()

    formatted = prompt_template.format(prompt=prompt)
    if system_prompt:
        formatted = f"{system_prompt}\n\n{formatted}"

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=2048 - max_new_tokens,
    )
    # Move to device if the encoding supports it
    # (BatchEncoding has .to(); plain dicts from mocks don't)
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with _no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    **generate_kwargs,
) -> list[str]:
    """
    Generate responses for a list of prompts (sequential, deterministic ordering).
    """
    return [generate(model, tokenizer, p, **generate_kwargs) for p in prompts]
