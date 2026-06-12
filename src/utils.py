"""
src/utils.py

Shared utility functions used by both the notebook and the trainer script.

- vram_estimate()      : pre-flight VRAM budget analysis
- count_parameters()   : trainable vs frozen parameter report
- plot_loss_curve()    : parse trainer_state.json and render loss chart
- token_length_stats() : dataset token distribution analysis
"""

from __future__ import annotations

import json
import logging
import pathlib
import random
import statistics
from typing import Optional

# Lazy torch import — avoids import-time OSError in CPU-only CI environments
# where CUDA shared libraries are absent. VRAM queries only run when a GPU
# is actually present.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except Exception:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── VRAM Budget ───────────────────────────────────────────────────────────────

def vram_estimate(
    params_b: float,
    bits: int = 4,
    batch: int = 4,
    seq_len: int = 1024,
    hidden: int = 2048,
) -> dict:
    """
    Rule-of-thumb VRAM breakdown for QLoRA fine-tuning.

    Args:
        params_b: model parameter count in billions (e.g. 1.1 for TinyLlama)
        bits:     quantisation width — 4 for QLoRA, 16 for full bfloat16
        batch:    per-device batch size
        seq_len:  maximum sequence length
        hidden:   model hidden dimension

    Returns:
        dict with keys: weights_GB, activations_GB, grad+opt_GB,
                        overhead_GB, total_GB
    """
    weights_gb  = params_b * 1e9 * (bits / 8) / 1e9
    # Rough activation estimate: batch × seq × hidden × dtype_bytes × layers_in_flight
    act_gb      = batch * seq_len * hidden * 2 * 4 / 1e9
    lora_params = params_b * 0.01           # ~1% of params are trainable
    grad_opt_gb = lora_params * 1e9 * 2 * 8 / 1e9   # grad + Adam m + v states
    overhead_gb = 1.5
    total_gb    = weights_gb + act_gb + grad_opt_gb + overhead_gb
    return {
        "weights_GB":     round(weights_gb, 2),
        "activations_GB": round(act_gb, 2),
        "grad+opt_GB":    round(grad_opt_gb, 2),
        "overhead_GB":    round(overhead_gb, 2),
        "total_GB":       round(total_gb, 2),
    }


def vram_report(cfg_params_b: float, bits: int, batch: int,
                seq_len: int, hidden: int) -> None:
    """Print a formatted VRAM report and compare against available GPU memory."""
    est = vram_estimate(cfg_params_b, bits, batch, seq_len, hidden)
    print("VRAM Budget Estimate")
    print("=" * 38)
    for k, v in est.items():
        print(f"  {k:<22} {v:>6.2f} GB")

    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        available = _torch.cuda.get_device_properties(0).total_memory / 1e9
        headroom  = available - est["total_GB"]
        print(f"\n  Available VRAM      {available:>6.1f} GB")
        status = "✓ safe" if headroom > 1 else "⚠  tight — reduce batch or seq_len"
        print(f"  Headroom            {headroom:>6.1f} GB  {status}")


# ── Parameter Counting ────────────────────────────────────────────────────────

def count_parameters(model) -> dict:
    """
    Count total, trainable, and frozen parameters.

    Returns
    -------
    dict with keys: total, trainable, frozen, trainable_pct
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return {
        "total":         total,
        "trainable":     trainable,
        "frozen":        frozen,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0.0,
    }


def print_parameter_report(model) -> None:
    """Print a formatted parameter count report."""
    stats = count_parameters(model)
    print("Parameter Count")
    print("=" * 42)
    print(f"  Total       : {stats['total']:>15,}")
    print(f"  Trainable   : {stats['trainable']:>15,}   ({stats['trainable_pct']:.3f} %)")
    print(f"  Frozen      : {stats['frozen']:>15,}")
    print()
    print("Trainable modules:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:<55}  {list(param.shape)}")


# ── Loss Curve ────────────────────────────────────────────────────────────────

def plot_loss_curve(
    trainer_state_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    ema_alpha: float = 0.1,
) -> None:
    """
    Parse trainer_state.json and render a train-loss plot with EMA smoothing.

    Args:
        trainer_state_path: path to Trainer's trainer_state.json
        output_path:        where to save the PNG (None = don't save)
        title:              chart title (auto-generated if None)
        ema_alpha:          exponential moving average smoothing factor
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.warning("matplotlib not installed — skipping loss curve plot")
        return

    p = pathlib.Path(trainer_state_path)
    if not p.exists():
        logger.warning("trainer_state.json not found at %s", p)
        return

    with open(p) as f:
        history = [e for e in json.load(f).get("log_history", []) if "loss" in e]

    if not history:
        logger.warning("No loss entries in %s", p)
        return

    steps  = [e["step"] for e in history]
    losses = [e["loss"]  for e in history]

    ema_val, ema_vals = losses[0], []
    for l in losses:
        ema_val = ema_alpha * l + (1 - ema_alpha) * ema_val
        ema_vals.append(ema_val)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses,   color="#b0c4de", lw=1,   alpha=0.7, label="raw loss")
    ax.plot(steps, ema_vals, color="#1f77b4", lw=2.5,
            label=f"EMA (α={ema_alpha})")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title(title or "LoRA Fine-Tuning Loss", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        logger.info("Loss curve saved → %s", output_path)

    plt.show()
    print(f"Final loss: {losses[-1]:.4f}  |  EMA: {ema_vals[-1]:.4f}")


# ── Token Length Stats ────────────────────────────────────────────────────────

def token_length_stats(
    dataset,
    tokenizer,
    col: str = "text",
    n: int = 500,
    max_seq_len: Optional[int] = None,
) -> dict:
    """
    Sample token-length distribution to guide MAX_SEQ_LEN selection.

    Args:
        dataset:     HuggingFace Dataset split
        tokenizer:   tokenizer to encode with
        col:         text column name
        n:           number of examples to sample
        max_seq_len: if provided, flag percentiles exceeding this limit

    Returns:
        dict with min, p25, median, p75, p95, max, mean
    """
    idx     = random.sample(range(len(dataset)), min(n, len(dataset)))
    lengths = [
        len(tokenizer(dataset[i][col], truncation=False)["input_ids"])
        for i in idx
    ]
    s = sorted(lengths)

    result = {
        "min":    s[0],
        "p25":    s[len(s) // 4],
        "median": int(statistics.median(lengths)),
        "p75":    s[3 * len(s) // 4],
        "p95":    s[int(.95 * len(s))],
        "max":    s[-1],
        "mean":   round(statistics.mean(lengths), 1),
    }

    print(f"Token-length distribution  (n={n} sample)")
    print("=" * 48)
    for k, v in result.items():
        flag = ""
        if max_seq_len and int(v) > max_seq_len:
            flag = "  ← truncated"
        print(f"  {k:<8}  {int(v):>6}{flag}")

    if max_seq_len:
        over = sum(1 for l in lengths if l > max_seq_len)
        pct  = 100 * over / len(lengths)
        print(f"\n  {pct:.1f}% of sampled examples exceed MAX_SEQ_LEN={max_seq_len}")

    return result
