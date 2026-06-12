# 🧠 LoRA Fine-Tuning with PEFT + Hugging Face Transformers

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.40+-FF6B35?style=flat)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-0.10+-1C3C3C?style=flat)](https://huggingface.co/docs/peft)
[![TRL](https://img.shields.io/badge/TRL-0.8+-412991?style=flat)](https://huggingface.co/docs/trl)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-blue?style=flat&logo=pytest)](tests/)

Supervised fine-tuning (SFT) of an open-source causal LLM using **Low-Rank Adaptation (LoRA)**
via the `peft` library. The full pipeline — quantisation, training, evaluation, and adapter
merging — runs on a **free-tier Colab T4 GPU (16 GB) in under 30 minutes**.

---

## What this demonstrates

| Skill | Where |
|---|---|
| QLoRA 4-bit quantisation (`bitsandbytes` NF4) | `src/trainer.py`, notebook §4 |
| `LoraConfig` setup, rank sweep, module targeting | notebook §5, §17 |
| `DataCollatorForCompletionOnlyLM` (prompt masking) | notebook §7 |
| `SFTTrainer` training loop with `paged_adamw_8bit` | notebook §9–10 |
| VRAM budget analysis (pre-flight) | `src/utils.py`, notebook §2 |
| Loss curve plotting (EMA smoothing) | notebook §11 |
| Adapter save / reload / merge with `merge_and_unload` | notebook §12–14 |
| Hugging Face Hub push (adapter + merged) | notebook §16 |
| TensorBoard / W&B logging | notebook §8 |
| CLI training script with YAML config | `src/trainer.py` |
| Pytest suite — no GPU needed | `tests/` |

---

## Architecture

```
User Prompt
     │
     ▼
┌────────────────────────────────────────────────────┐
│                   QLoRA Model                      │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │         Frozen Base Weights (4-bit NF4)      │  │
│  │  W ∈ ℝ^{d×k}  — quantised, not updated      │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │ + LoRA delta                  │
│  ┌──────────────────▼───────────────────────────┐  │
│  │         Trainable LoRA Adapters (~0.3%)       │  │
│  │  A ∈ ℝ^{d×r}  ·  B ∈ ℝ^{r×k}               │  │
│  │  delta W = (α/r) · B·A                       │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
     │
     ▼ merge_and_unload()
┌────────────────────────────────────────────────────┐
│   Merged Model  W' = W + (α/r)·BA  (no PEFT dep)  │
└────────────────────────────────────────────────────┘
```

### Key design decisions

- **4-bit NF4 + double quantisation** — reduces base weight VRAM from ~2.2 GB (bfloat16)
  to ~0.6 GB for TinyLlama, enabling fine-tuning on commodity GPUs.
- **`q_proj` + `v_proj` targeting** — these attention projections are most responsible
  for in-context behaviour. The experiment matrix (notebook §17) shows the capacity–VRAM
  trade-off for adding more modules.
- **Completion-only loss mask** — `DataCollatorForCompletionOnlyLM` masks the human-turn
  tokens so the model only learns to predict the assistant response.
- **`paged_adamw_8bit`** — reduces optimiser VRAM by ~50% vs full AdamW with negligible
  quality loss.

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/R3tr0-Coder/lora-finetune.git
cd lora-finetune
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the notebook (recommended)

Open in Colab (T4 GPU):

```
notebooks/lora_finetune.ipynb
```

Or regenerate it locally:

```bash
python notebooks/generate_notebook.py
jupyter lab notebooks/lora_finetune.ipynb
```

### 3. Run the CLI trainer

```bash
# Full epoch (~25 min on T4)
python -m src.trainer --config configs/lora_tinyllama.yaml

# Quick smoke test (50 steps, ~2 min)
python -m src.trainer --config configs/lora_tinyllama.yaml --max_steps 50

# Train + merge in one shot
python -m src.trainer --config configs/lora_tinyllama.yaml --merge
```

### 4. Run tests (no GPU needed)

```bash
pytest tests/ -v
```

---

## Project Structure

```
lora-finetune/
├── notebooks/
│   ├── lora_finetune.ipynb        # Main walkthrough (42 cells)
│   └── generate_notebook.py       # Regenerate notebook from source
├── src/
│   ├── trainer.py                 # CLI training script + TrainConfig
│   ├── inference.py               # load_model(), generate(), batch_generate()
│   └── utils.py                   # vram_estimate, count_parameters, plot_loss
├── configs/
│   ├── lora_tinyllama.yaml        # TinyLlama-1.1B (free T4)
│   └── lora_llama32_1b.yaml       # Llama-3.2-1B (gated — needs HF login)
├── tests/
│   ├── conftest.py
│   ├── test_utils.py              # VRAM, param count, token stats, TrainConfig
│   └── test_inference.py          # generate(), batch_generate() — all mocked
├── outputs/                       # Generated at runtime (gitignored)
│   └── lora-tinyllama/
│       ├── adapter/               # LoRA adapter files (~10 MB)
│       ├── merged/                # Merged base+adapter model
│       ├── trainer_state.json
│       └── loss_curve.png
├── .github/workflows/ci.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Swap the model

Change `base_model` in `configs/lora_tinyllama.yaml` (or `BASE_MODEL` in the notebook).
The pipeline adapts automatically:

| Model | VRAM (4-bit) | Notes |
|---|---|---|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~4 GB | Free Colab T4 — default |
| `meta-llama/Llama-3.2-1B-Instruct` | ~5 GB | Gated — accept Meta ToS |
| `meta-llama/Llama-3.2-3B-Instruct` | ~8 GB | Gated — better quality |
| `mistralai/Mistral-7B-v0.1` | ~10 GB | Best quality on T4 |
| `meta-llama/Llama-3.1-8B-Instruct` | ~12 GB | Needs A10 or better |

---

## Extending

| What | How |
|---|---|
| **More adapter modules** | Add `"k_proj","o_proj","gate_proj","up_proj","down_proj"` to `lora_target_modules` |
| **DPO alignment** | Replace `SFTTrainer` with TRL's `DPOTrainer` after SFT |
| **RLHF / PPO** | Use TRL's `PPOTrainer` + a separate reward model |
| **GPTQ / AWQ quantisation** | Run on the merged model for 2–4× inference speedup |
| **vLLM serving** | `from vllm import LLM; llm = LLM("outputs/lora-tinyllama/merged")` |
| **lm-evaluation-harness** | `lm_eval --model hf --model_args pretrained=outputs/lora-tinyllama/merged --tasks hellaswag,arc_easy` |
| **MLflow tracking** | `report_to: mlflow` in YAML config |
| **Multi-GPU** | `accelerate launch --multi_gpu -m src.trainer --config ...` |

---

## Interview talking points

> *"I used QLoRA — 4-bit NF4 quantisation plus LoRA — so the entire fine-tune fit on a
> single T4. The base model weights are frozen; only the low-rank adapter matrices
> (about 0.3% of total parameters) are trained."*

> *"I targeted `q_proj` and `v_proj` because those are the attention projections most
> responsible for in-context behaviour. The experiment matrix in the notebook shows the
> capacity–VRAM trade-off for adding more modules."*

> *"I used `DataCollatorForCompletionOnlyLM` to mask the prompt tokens from the loss.
> Without that, the model wastes capacity imitating the human turn and the training
> loss is artificially low."*

> *"After training I merge the adapter with `merge_and_unload()`, which folds BA into W
> so inference has zero PEFT overhead and the checkpoint is a standard HuggingFace model
> you can load anywhere."*

> *"The whole training loop is in a YAML-configurable CLI script so it can be dropped
> into any CI/CD pipeline or cloud GPU job — no notebook required."*

---

## License

MIT — use freely, attribution appreciated.
