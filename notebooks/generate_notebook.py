"""
Generates lora_finetune.ipynb via nbformat.
Run: python notebooks/generate_notebook.py
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
    "colab": {"provenance": []},
}

cells = []
def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# ──────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
# LoRA Fine-Tuning with PEFT + Hugging Face Transformers

**What this notebook does:**
Supervised fine-tuning (SFT) of an open-source causal LLM using Low-Rank Adaptation (LoRA)
via the `peft` library. We fine-tune `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on a
conversational instruction dataset so the walkthrough runs on a **free-tier Colab T4 GPU
(≈ 16 GB VRAM) in under 30 minutes**.

**What you'll practise:**
- Quantising a model to 4-bit with `bitsandbytes` (QLoRA)
- Configuring a `LoraConfig` and inspecting trainable parameter counts
- Building a `DataCollatorForCompletionOnlyLM` pipeline
- Training with `SFTTrainer` (TRL) and logging to TensorBoard / W&B
- Merging LoRA adapters back into the base model for deployment
- Running a VRAM budget analysis and plotting loss curves

> **Swap the model:** Change `BASE_MODEL` in Section 1 to
> `meta-llama/Llama-3.2-1B`, `mistralai/Mistral-7B-v0.1`, or any causal LM.
"""))

# ── 0. Install ────────────────────────────────────────────────────────────────
cells.append(md("## 0 · Environment Setup"))
cells.append(code("""\
%pip install -q \\
    transformers>=4.40.0 \\
    datasets>=2.18.0 \\
    peft>=0.10.0 \\
    trl>=0.8.6 \\
    bitsandbytes>=0.43.0 \\
    accelerate>=0.29.0 \\
    sentencepiece \\
    einops \\
    wandb \\
    tensorboard

import importlib
for pkg in ["transformers", "datasets", "peft", "trl", "bitsandbytes", "accelerate"]:
    v = importlib.import_module(pkg).__version__
    print(f"{pkg:>20}  {v}")
"""))

# ── 1. Imports ────────────────────────────────────────────────────────────────
cells.append(md("## 1 · Imports & Configuration"))
cells.append(code("""\
import os, math, time, warnings
warnings.filterwarnings("ignore")

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"Device  : {p.name}")
    print(f"VRAM    : {p.total_memory / 1e9:.1f} GB")
else:
    print("Device  : CPU (GPU recommended)")
"""))

# ── 1b. Config ────────────────────────────────────────────────────────────────
cells.append(code("""\
# ═══════════════════════════════════════════════════════════════════════════════
#  PROJECT CONFIGURATION — edit here, nowhere else
# ═══════════════════════════════════════════════════════════════════════════════

BASE_MODEL    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME  = "timdettmers/openassistant-guanaco"   # ~10k instruction pairs
OUTPUT_DIR    = "../outputs/lora-tinyllama"
HUB_MODEL_ID  = None   # e.g. "your-username/tinyllama-lora-demo"

# ── LoRA hyperparameters ──────────────────────────────────────────────────────
LORA_R               = 16      # rank — higher = more capacity, more VRAM
LORA_ALPHA           = 32      # scaling = alpha / r; rule-of-thumb: 2 × r
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = ["q_proj", "v_proj"]
# add "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj" for more capacity

# ── Quantisation ─────────────────────────────────────────────────────────────
USE_4BIT           = True           # QLoRA (NF4) — set False for full bfloat16
BNB_COMPUTE_DTYPE  = torch.bfloat16

# ── Training ──────────────────────────────────────────────────────────────────
MAX_SEQ_LEN    = 1024
BATCH_SIZE     = 4        # per-device
GRAD_ACCUM     = 4        # effective batch = BATCH_SIZE × GRAD_ACCUM = 16
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 1
WARMUP_RATIO   = 0.03
LR_SCHEDULER   = "cosine"
WEIGHT_DECAY   = 0.001
LOGGING_STEPS  = 10
SAVE_STEPS     = 100
MAX_STEPS      = -1       # -1 = full epoch; set e.g. 50 for a quick smoke test

SEED = 42

print("Configuration loaded ✓")
print(f"  Model   : {BASE_MODEL}")
print(f"  Dataset : {DATASET_NAME}")
print(f"  LoRA r  : {LORA_R}   alpha: {LORA_ALPHA}   targets: {LORA_TARGET_MODULES}")
print(f"  QLoRA   : {USE_4BIT}")
print(f"  Batch   : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")
"""))

# ── 2. VRAM estimate ──────────────────────────────────────────────────────────
cells.append(md("""\
## 2 · VRAM Budget Analysis

Before loading anything, estimate GPU memory usage so we can catch
OOM issues before they happen.
"""))
cells.append(code("""\
def vram_estimate(
    params_b: float,
    bits: int = 4,
    batch: int = 4,
    seq_len: int = 1024,
    hidden: int = 2048,
) -> dict:
    \"\"\"
    Rule-of-thumb VRAM breakdown for QLoRA fine-tuning.

    Args:
        params_b: model parameter count in billions
        bits:     quantisation width (4 = QLoRA, 16 = bfloat16)
        batch:    per-device batch size
        seq_len:  maximum sequence length
        hidden:   model hidden dimension (for activation estimate)
    \"\"\"
    weights_gb  = params_b * 1e9 * (bits / 8) / 1e9
    act_gb      = batch * seq_len * hidden * 2 * 4 / 1e9   # ~4 layers in flight
    lora_params = params_b * 0.01                           # ~1% trainable
    grad_opt_gb = lora_params * 1e9 * 2 * 8 / 1e9          # grad + Adam states
    overhead_gb = 1.5
    total_gb    = weights_gb + act_gb + grad_opt_gb + overhead_gb
    return {
        "weights_GB":    round(weights_gb, 2),
        "activations_GB": round(act_gb, 2),
        "grad+opt_GB":   round(grad_opt_gb, 2),
        "overhead_GB":   round(overhead_gb, 2),
        "total_GB":      round(total_gb, 2),
    }


estimate = vram_estimate(
    params_b=1.1,
    bits=4 if USE_4BIT else 16,
    batch=BATCH_SIZE,
    seq_len=MAX_SEQ_LEN,
    hidden=2048,
)

print("VRAM Budget Estimate")
print("=" * 38)
for k, v in estimate.items():
    print(f"  {k:<20} {v:>6.2f} GB")

if torch.cuda.is_available():
    available = torch.cuda.get_device_properties(0).total_memory / 1e9
    headroom  = available - estimate["total_GB"]
    print(f"\\n  Available VRAM    {available:>6.1f} GB")
    status = "✓ safe" if headroom > 1 else "⚠  tight — consider reducing BATCH_SIZE"
    print(f"  Headroom          {headroom:>6.1f} GB  {status}")
"""))

# ── 3. Tokeniser ──────────────────────────────────────────────────────────────
cells.append(md("## 3 · Tokeniser"))
cells.append(code("""\
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="right",   # required for causal LM loss computation
)

# Add pad token if missing (common for instruction-tuned LLMs)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    print("Added [PAD] token")

print(f"Vocab size : {tokenizer.vocab_size:,}")
print(f"Pad token  : {tokenizer.pad_token!r}  (id {tokenizer.pad_token_id})")
print(f"BOS token  : {tokenizer.bos_token!r}")
print(f"EOS token  : {tokenizer.eos_token!r}")

# Sanity check: round-trip a sample string
_enc = tokenizer("LoRA fine-tuning is parameter-efficient.", return_tensors="pt")
_dec = tokenizer.decode(_enc["input_ids"][0])
print(f"\\nRound-trip : {_dec!r}  ({_enc['input_ids'].shape[1]} tokens)")
"""))

# ── 4. Load model ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 4 · Load Base Model (QLoRA)

`BitsAndBytesConfig` with `load_in_4bit=True` and `bnb_4bit_quant_type="nf4"` (NormalFloat4)
gives better quality than plain int4. `bnb_4bit_use_double_quant=True` applies a second
quantisation pass to the quantisation constants, saving ~0.4 GB on a 7B model.

`prepare_model_for_kbit_training` then:
1. Casts all LayerNorm layers to `float32` (prevents NaN gradients)
2. Enables gradient checkpointing (trades recomputation for ~40% VRAM saving)
"""))
cells.append(code("""\
bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
) if USE_4BIT else None

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=BNB_COMPUTE_DTYPE if not USE_4BIT else None,
)
print(f"Model loaded in {time.time() - t0:.1f}s")

# Resize if we added a pad token
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False   # incompatible with gradient checkpointing

if torch.cuda.is_available():
    used     = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM after load — allocated: {used:.2f} GB  reserved: {reserved:.2f} GB")
"""))

# ── 5. LoRA config ────────────────────────────────────────────────────────────
cells.append(md(r"""\
## 5 · LoRA Configuration

LoRA inserts trainable low-rank matrices $A \in \mathbb{R}^{d \times r}$ and
$B \in \mathbb{R}^{r \times k}$ alongside frozen weight matrices $W$. The adapted
forward pass is:

$$W' = W + \frac{\alpha}{r} \cdot BA$$

Only $A$ and $B$ are updated — typically **0.1–1%** of total parameters.
"""))
cells.append(code("""\
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",           # "none" is standard; "all" adds trainable bias terms
    inference_mode=False,
)

model = get_peft_model(model, lora_config)

# ── Parameter report ──────────────────────────────────────────────────────────
total      = sum(p.numel() for p in model.parameters())
trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen     = total - trainable
pct        = 100 * trainable / total

print("Parameter Count")
print("=" * 42)
print(f"  Total       : {total:>15,}")
print(f"  Trainable   : {trainable:>15,}   ({pct:.3f} %)")
print(f"  Frozen      : {frozen:>15,}")
print()
print("Trainable modules:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name:<55}  {list(param.shape)}")
"""))

# ── 6. Dataset ────────────────────────────────────────────────────────────────
cells.append(md("""\
## 6 · Dataset

We use `timdettmers/openassistant-guanaco` — ~10k instruction pairs formatted as:

```
### Human: <question>
### Assistant: <answer>
```

This simple delimiter works perfectly with `DataCollatorForCompletionOnlyLM`,
which masks everything before `### Assistant:` from the loss.
"""))
cells.append(code("""\
raw = load_dataset(DATASET_NAME)
print(raw)
print("\\nSample (first 400 chars):")
print(raw["train"][0]["text"][:400])
"""))

cells.append(code("""\
# ── Token-length distribution ──────────────────────────────────────────────────
import random, statistics

def token_stats(dataset, tokenizer, col="text", n=500) -> dict:
    \"\"\"Sample token lengths to guide MAX_SEQ_LEN selection.\"\"\"
    idx = random.sample(range(len(dataset)), min(n, len(dataset)))
    lengths = [
        len(tokenizer(dataset[i][col], truncation=False)["input_ids"])
        for i in idx
    ]
    s = sorted(lengths)
    return {
        "min":    s[0],
        "p25":    s[len(s)//4],
        "median": int(statistics.median(lengths)),
        "p75":    s[3*len(s)//4],
        "p95":    s[int(.95*len(s))],
        "max":    s[-1],
        "mean":   round(statistics.mean(lengths), 1),
    }

stats = token_stats(raw["train"], tokenizer)
print(f"Token-length distribution (n=500 sample)")
print("=" * 45)
for k, v in stats.items():
    flag = "  ← would truncate" if int(v) > MAX_SEQ_LEN else ""
    print(f"  {k:<8}  {int(v):>5}{flag}")
print(f"\\nMAX_SEQ_LEN = {MAX_SEQ_LEN}")
"""))

# ── 7. Formatting ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 7 · Formatting & Completion-Only Collator

`DataCollatorForCompletionOnlyLM` masks the *prompt* tokens so the model only
learns to predict the **assistant turn**. Without this, the model wastes
capacity learning to mimic the human side and training loss is misleadingly low.
"""))
cells.append(code("""\
RESPONSE_TEMPLATE = "### Assistant:"

def formatting_func(example):
    \"\"\"Called by SFTTrainer on each batch element. Returns a list of strings.\"\"\"
    if isinstance(example["text"], list):
        return example["text"]
    return [example["text"]]

# Encode the template so the collator can locate it in token space
response_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
print(f"Response template tokens : {response_ids}")
print(f"  → decoded              : {tokenizer.decode(response_ids)!r}")

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_ids,
    tokenizer=tokenizer,
)

# Verify on one example
sample_text = formatting_func(raw["train"][0])[0]
print(f"\\nFormatted sample (first 300 chars):\\n{sample_text[:300]}")
"""))

# ── 8. TrainingArguments ──────────────────────────────────────────────────────
cells.append(md("## 8 · TrainingArguments"))
cells.append(code("""\
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # Batching
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS,

    # Optimisation
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=LR_SCHEDULER,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    optim="paged_adamw_8bit",   # 8-bit AdamW reduces optimiser VRAM by ~50%
    bf16=True,
    fp16=False,

    # Logging
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],   # swap to ["wandb"] after wandb.login()
    run_name="lora-tinyllama-demo",

    # Checkpointing
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    load_best_model_at_end=False,

    # Reproducibility & efficiency
    seed=SEED,
    data_seed=SEED,
    group_by_length=True,          # packs similar-length examples → less padding
    ddp_find_unused_parameters=False,
)

n_samples   = len(raw["train"])
eff_batch   = BATCH_SIZE * GRAD_ACCUM
steps_epoch = math.ceil(n_samples / eff_batch)
total_steps = steps_epoch * NUM_EPOCHS if MAX_STEPS == -1 else MAX_STEPS

print(f"Train samples        : {n_samples:,}")
print(f"Effective batch size : {eff_batch}")
print(f"Steps / epoch        : {steps_epoch:,}")
print(f"Total steps          : {total_steps:,}")
"""))

# ── 8b. W&B optional ──────────────────────────────────────────────────────────
cells.append(md("### Optional: Weights & Biases"))
cells.append(code("""\
# Uncomment to enable W&B experiment tracking (free account at wandb.ai)
# import wandb
# wandb.login()
# training_args.report_to = ["wandb"]
# os.environ["WANDB_PROJECT"] = "lora-finetune-demo"

# TensorBoard is active by default. Launch with:
# %load_ext tensorboard
# %tensorboard --logdir {OUTPUT_DIR}/runs
print("Logging: TensorBoard  →", OUTPUT_DIR + "/runs")
"""))

# ── 9. SFTTrainer ─────────────────────────────────────────────────────────────
cells.append(md("""\
## 9 · SFTTrainer

`SFTTrainer` (TRL) wraps `Trainer` with:
- Automatic dataset formatting via `formatting_func`
- Built-in support for the completion-only collator
- Optional example packing (`packing=True`) for higher GPU utilisation
"""))
cells.append(code("""\
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=raw["train"],
    tokenizer=tokenizer,
    formatting_func=formatting_func,
    data_collator=collator,
    max_seq_length=MAX_SEQ_LEN,
    dataset_num_proc=2,
    packing=False,   # True packs multiple short examples per MAX_SEQ_LEN window
)

print(f"Training samples : {len(raw['train']):,}")
print(f"Trainable params : {trainable:,}  ({pct:.3f}% of total)")
trainer.model.print_trainable_parameters()
"""))

# ── 10. Train ─────────────────────────────────────────────────────────────────
cells.append(md("## 10 · Train"))
cells.append(code("""\
print("Starting training…  (grab a coffee ☕)")
t_start = time.time()

train_result = trainer.train()

elapsed = time.time() - t_start
print(f"\\nTraining complete in {elapsed / 60:.1f} min")
print(f"Final train loss : {train_result.training_loss:.4f}")

trainer.save_state()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
"""))

# ── 11. Loss curve ────────────────────────────────────────────────────────────
cells.append(md("## 11 · Loss Curve"))
cells.append(code("""\
import json, pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_loss(state_path: str, output_dir: str = OUTPUT_DIR):
    \"\"\"Parse trainer_state.json and plot raw + EMA loss.\"\"\"
    p = pathlib.Path(state_path)
    if not p.exists():
        print(f"State file not found: {p}"); return

    with open(p) as f:
        history = [e for e in json.load(f).get("log_history", []) if "loss" in e]

    if not history:
        print("No loss entries found."); return

    steps  = [e["step"] for e in history]
    losses = [e["loss"]  for e in history]

    alpha, ema_val, ema_vals = 0.1, losses[0], []
    for l in losses:
        ema_val = alpha * l + (1 - alpha) * ema_val
        ema_vals.append(ema_val)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses,   color="#b0c4de", lw=1,   alpha=0.7, label="raw loss")
    ax.plot(steps, ema_vals, color="#1f77b4", lw=2.5, label="EMA (α=0.1)")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title(f"LoRA Fine-Tuning — {BASE_MODEL.split('/')[-1]}", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    out = f"{output_dir}/loss_curve.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Final loss: {losses[-1]:.4f}  |  EMA: {ema_vals[-1]:.4f}")
    print(f"Saved: {out}")


plot_loss(f"{OUTPUT_DIR}/trainer_state.json")
"""))

# ── 12. Save adapter ──────────────────────────────────────────────────────────
cells.append(md("""\
## 12 · Save & Inspect the LoRA Adapter

The adapter is tiny — for a 1.1B model with `r=16` on `q_proj`+`v_proj`
it weighs **≈ 8–12 MB** vs 2.2 GB for the full bfloat16 model.
This makes it practical to version-control adapters in Git LFS or swap
them at inference time for different tasks.
"""))
cells.append(code("""\
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

adapter_files = list(pathlib.Path(ADAPTER_DIR).rglob("*"))
total_mb = sum(f.stat().st_size for f in adapter_files if f.is_file()) / 1e6
print(f"Adapter saved → {ADAPTER_DIR}")
for f in sorted(adapter_files):
    if f.is_file():
        print(f"  {f.name:<40}  {f.stat().st_size / 1e6:.2f} MB")
print(f"\\nTotal adapter size: {total_mb:.2f} MB")
"""))

# ── 13. Inference pre-merge ───────────────────────────────────────────────────
cells.append(md("## 13 · Inference with the Adapter (Pre-Merge)"))
cells.append(code("""\
def generate(model, tokenizer, prompt: str,
             max_new_tokens: int = 256,
             temperature: float = 0.7,
             top_p: float = 0.9) -> str:
    \"\"\"Generate a response, returning only the newly generated text.\"\"\"
    model.eval()
    formatted = f"### Human: {prompt}\\n### Assistant:"
    inputs = tokenizer(
        formatted, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LEN - max_new_tokens,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a Python function to compute Fibonacci using memoisation.",
    "What are three benefits of LoRA over full fine-tuning?",
]

print("=" * 62)
for prompt in PROMPTS:
    print(f"Human : {prompt}")
    print(f"Model : {generate(model, tokenizer, prompt)}")
    print("-" * 62)
"""))

# ── 14. Merge ─────────────────────────────────────────────────────────────────
cells.append(md("""\
## 14 · Merge Adapter → Base Model

`merge_and_unload()` folds the LoRA deltas ($\\frac{\\alpha}{r} \\cdot BA$)
into the original weight matrices in-place, producing a standard HuggingFace
model with no PEFT dependency. Use this checkpoint for production deployment
or quantisation (GPTQ, AWQ).
"""))
cells.append(code("""\
MERGED_DIR = f"{OUTPUT_DIR}/merged"
os.makedirs(MERGED_DIR, exist_ok=True)

print("Loading base model for clean merge (bfloat16)…")
t0 = time.time()

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base.resize_token_embeddings(len(tokenizer))

peft_model   = PeftModel.from_pretrained(base, ADAPTER_DIR)
merged_model = peft_model.merge_and_unload()
print(f"Merged in {time.time() - t0:.1f}s")

assert not hasattr(merged_model, "peft_config"), "Merge incomplete"
print("✓ No PEFT wrapper in merged model")

merged_model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model saved → {MERGED_DIR}")
"""))

# ── 15. Inference post-merge ──────────────────────────────────────────────────
cells.append(md("## 15 · Inference with the Merged Model"))
cells.append(code("""\
print("Responses from merged model (no PEFT overhead):")
print("=" * 62)
for prompt in PROMPTS:
    print(f"Human : {prompt}")
    print(f"Model : {generate(merged_model, tokenizer, prompt)}")
    print("-" * 62)
"""))

# ── 16. Hub push ──────────────────────────────────────────────────────────────
cells.append(md("## 16 · Push to Hugging Face Hub (Optional)"))
cells.append(code("""\
if HUB_MODEL_ID:
    from huggingface_hub import login
    # login()  # authenticate interactively if needed

    merged_model.push_to_hub(HUB_MODEL_ID, private=True)
    tokenizer.push_to_hub(HUB_MODEL_ID, private=True)
    print(f"✓ Merged model → {HUB_MODEL_ID}")

    # Push adapter separately (much smaller; useful for sharing)
    model.push_to_hub(f"{HUB_MODEL_ID}-adapter", private=True)
    print(f"✓ Adapter only → {HUB_MODEL_ID}-adapter")
else:
    print("HUB_MODEL_ID not set — skipping.")
    print("Set it in Section 1 and rerun this cell.")
"""))

# ── 17. Experiment matrix ─────────────────────────────────────────────────────
cells.append(md("""\
## 17 · Experiment Matrix

Sweep LoRA rank and target modules to understand the capacity–VRAM trade-off
before committing to a longer training run.
"""))
cells.append(code("""\
import pandas as pd

rows = []
for r in [4, 8, 16, 32, 64]:
    for targets in [["q_proj","v_proj"],
                    ["q_proj","k_proj","v_proj","o_proj"],
                    ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]]:
        for bits in [4, 16]:
            # TinyLlama: 22 layers, hidden=2048
            params_per_module = 2048 * r + r * 2048
            trainable = params_per_module * len(targets) * 22
            pct   = 100 * trainable / 1.1e9
            vram  = vram_estimate(1.1, bits=bits, batch=4, seq_len=1024, hidden=2048)
            rows.append({
                "r":           r,
                "targets":     len(targets),
                "quant":       f"{bits}-bit",
                "trainable_M": f"{trainable/1e6:.1f}",
                "% total":     f"{pct:.3f}",
                "VRAM_GB":     vram["total_GB"],
            })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print("\\nRule of thumb: r=16, q+v targets is a good starting point for most tasks.")
"""))

# ── 18. Next steps ────────────────────────────────────────────────────────────
cells.append(md("""\
## 18 · Next Steps & Extensions

| What to try | How |
|---|---|
| **Larger model** | `BASE_MODEL = "meta-llama/Llama-3.2-3B"` |
| **More adapters** | Add `"k_proj","o_proj","gate_proj","up_proj","down_proj"` |
| **DPO alignment** | Replace `SFTTrainer` with TRL's `DPOTrainer` |
| **RLHF / PPO** | Use TRL's `PPOTrainer` + a reward model |
| **Quantise for inference** | Run GPTQ or AWQ on the merged model |
| **Serve with vLLM** | `from vllm import LLM; llm = LLM(MERGED_DIR)` |
| **Eval with lm-eval** | `lm_eval --model hf --model_args pretrained={MERGED_DIR} --tasks hellaswag,arc_easy` |
| **MLflow tracking** | `training_args.report_to = ["mlflow"]` |

---

### Key things to say in an interview

> *"I used QLoRA — 4-bit NF4 quantisation plus LoRA — so the entire fine-tune
> fit on a single T4. Only the low-rank adapter matrices (about 0.3% of
> parameters) are trained; everything else is frozen."*

> *"I targeted `q_proj` and `v_proj` because those are the attention projections
> most responsible for in-context behaviour. Adding all seven MLP and attention
> matrices gives more capacity but costs more VRAM."*

> *"I used `DataCollatorForCompletionOnlyLM` to mask the prompt from the loss.
> Without that, the model wastes capacity imitating the human turn and the
> loss number is artificially lower."*

> *"After training I merge the adapter back with `merge_and_unload()`, which
> folds BA into W so inference has zero PEFT overhead and the checkpoint is a
> standard HuggingFace model."*
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb.cells = cells
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_finetune.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written → {out_path}")
