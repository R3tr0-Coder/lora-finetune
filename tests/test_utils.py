"""
tests/test_utils.py

Unit tests for src/utils.py — VRAM estimation, parameter counting,
token stats.  No GPU required; no HuggingFace downloads.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils import (
    count_parameters,
    token_length_stats,
    vram_estimate,
)


# ════════════════════════════════════════════════════════════════════════════
# vram_estimate
# ════════════════════════════════════════════════════════════════════════════

class TestVramEstimate:
    def test_returns_five_keys(self):
        result = vram_estimate(1.1, bits=4, batch=4, seq_len=1024, hidden=2048)
        assert set(result.keys()) == {
            "weights_GB", "activations_GB", "grad+opt_GB", "overhead_GB", "total_GB"
        }

    def test_total_is_sum_of_parts(self):
        r = vram_estimate(1.1)
        parts_sum = (
            r["weights_GB"] + r["activations_GB"]
            + r["grad+opt_GB"] + r["overhead_GB"]
        )
        assert abs(r["total_GB"] - parts_sum) < 0.05, (
            f"total_GB={r['total_GB']} but parts sum to {parts_sum:.3f}"
        )

    def test_4bit_lighter_than_16bit(self):
        r4  = vram_estimate(7.0, bits=4)
        r16 = vram_estimate(7.0, bits=16)
        assert r4["weights_GB"] < r16["weights_GB"]

    def test_larger_batch_increases_activations(self):
        r4  = vram_estimate(1.1, batch=4)
        r16 = vram_estimate(1.1, batch=16)
        assert r16["activations_GB"] > r4["activations_GB"]

    def test_zero_params(self):
        r = vram_estimate(0.0)
        assert r["weights_GB"] == 0.0
        assert r["total_GB"] >= 0.0

    def test_values_are_reasonable_for_tinyllama(self):
        """TinyLlama 4-bit should fit in < 16 GB on a T4."""
        r = vram_estimate(1.1, bits=4, batch=4, seq_len=1024, hidden=2048)
        assert r["total_GB"] < 16.0, f"Estimate too high: {r['total_GB']:.2f} GB"

    def test_values_are_reasonable_for_7b_bf16(self):
        """7B bfloat16 weights alone should be ~14 GB."""
        r = vram_estimate(7.0, bits=16)
        assert 12 < r["weights_GB"] < 16, f"Unexpected: {r['weights_GB']}"


# ════════════════════════════════════════════════════════════════════════════
# count_parameters
# ════════════════════════════════════════════════════════════════════════════

class TestCountParameters:
    def _make_model(self, n_trainable: int, n_frozen: int):
        """
        Build a pure-Python mock model — no torch.nn required.
        count_parameters() only calls p.numel() and p.requires_grad,
        so a simple namespace object is sufficient.
        """
        class _Param:
            def __init__(self, n, grad): self._n = n; self.requires_grad = grad
            def numel(self): return self._n

        class _FakeModel:
            def parameters(self_):
                params = []
                if n_trainable: params.append(_Param(n_trainable, True))
                if n_frozen:    params.append(_Param(n_frozen, False))
                return iter(params)
            def named_parameters(self_):
                params = []
                if n_trainable: params.append(("trainable", _Param(n_trainable, True)))
                if n_frozen:    params.append(("frozen",    _Param(n_frozen, False)))
                return iter(params)

        return _FakeModel()

    def test_total_equals_trainable_plus_frozen(self):
        m = self._make_model(100, 900)
        r = count_parameters(m)
        assert r["total"] == r["trainable"] + r["frozen"]

    def test_trainable_count(self):
        m = self._make_model(128, 0)
        r = count_parameters(m)
        assert r["trainable"] == 128
        assert r["frozen"] == 0

    def test_fully_frozen(self):
        m = self._make_model(0, 1000)
        r = count_parameters(m)
        assert r["trainable"] == 0
        assert r["trainable_pct"] == 0.0

    def test_percentage_calculation(self):
        m = self._make_model(10, 90)
        r = count_parameters(m)
        assert abs(r["trainable_pct"] - 10.0) < 0.01


# ════════════════════════════════════════════════════════════════════════════
# token_length_stats
# ════════════════════════════════════════════════════════════════════════════

class TestTokenLengthStats:
    def _make_dataset(self, texts):
        """Return a simple list-based fake dataset."""
        class _FakeDataset:
            def __init__(self, data): self._data = data
            def __len__(self): return len(self._data)
            def __getitem__(self, i): return self._data[i]

        return _FakeDataset([{"text": t} for t in texts])

    def _make_tokenizer(self):
        """Tokenizer mock that returns len(text.split()) tokens."""
        tok = MagicMock()
        tok.side_effect = lambda t, **kw: {"input_ids": list(range(len(t.split())))}
        return tok

    def test_returns_expected_keys(self):
        ds  = self._make_dataset(["hello world"] * 10)
        tok = self._make_tokenizer()
        r   = token_length_stats(ds, tok, n=5)
        assert set(r.keys()) == {"min", "p25", "median", "p75", "p95", "max", "mean"}

    def test_min_leq_max(self):
        texts = ["a b c", "a b c d e f g", "x"]
        ds    = self._make_dataset(texts)
        tok   = self._make_tokenizer()
        r     = token_length_stats(ds, tok, n=3)
        assert r["min"] <= r["max"]

    def test_uniform_lengths(self):
        texts = ["one two three"] * 20     # all 3 tokens
        ds    = self._make_dataset(texts)
        tok   = self._make_tokenizer()
        r     = token_length_stats(ds, tok, n=10)
        assert r["min"] == r["max"] == 3
        assert r["mean"] == 3.0


# ════════════════════════════════════════════════════════════════════════════
# TrainConfig
# ════════════════════════════════════════════════════════════════════════════

class TestTrainConfig:
    def test_defaults(self):
        from src.trainer import TrainConfig
        cfg = TrainConfig()
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.use_4bit is True
        assert cfg.batch_size == 4

    def test_from_yaml(self, tmp_path):
        from src.trainer import TrainConfig
        config_data = {
            "base_model":   "test/model",
            "lora_r":       8,
            "lora_alpha":   16,
            "output_dir":   "/tmp/out",
            "dataset_name": "some/dataset",
        }
        p = tmp_path / "config.yaml"
        import yaml
        p.write_text(yaml.dump(config_data))

        cfg = TrainConfig.from_yaml(str(p))
        assert cfg.base_model == "test/model"
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16

    def test_from_yaml_ignores_unknown_keys(self, tmp_path):
        """Extra keys in YAML should not crash the constructor."""
        from src.trainer import TrainConfig
        import yaml
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"base_model": "x/y", "unknown_key": "value"}))
        cfg = TrainConfig.from_yaml(str(p))
        assert cfg.base_model == "x/y"

    def test_lora_scaling_rule(self):
        """By convention, alpha should equal 2 × r."""
        from src.trainer import TrainConfig
        cfg = TrainConfig()
        assert cfg.lora_alpha == 2 * cfg.lora_r, (
            f"Expected alpha={2*cfg.lora_r}, got {cfg.lora_alpha}"
        )


# ════════════════════════════════════════════════════════════════════════════
# plot_loss_curve (smoke test — no display)
# ════════════════════════════════════════════════════════════════════════════

class TestPlotLossCurve:
    def _write_state(self, path, entries):
        data = {"log_history": entries}
        Path(path).write_text(json.dumps(data))

    def test_runs_without_error(self, tmp_path):
        state_path = tmp_path / "trainer_state.json"
        self._write_state(state_path, [
            {"step": i * 10, "loss": 2.0 - i * 0.1, "epoch": 0.1 * i}
            for i in range(1, 11)
        ])
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
            from src.utils import plot_loss_curve
            plot_loss_curve(str(state_path))   # should not raise

    def test_handles_missing_file(self, tmp_path, capsys):
        from src.utils import plot_loss_curve
        plot_loss_curve(str(tmp_path / "nonexistent.json"))
        # Should log a warning and return gracefully, not raise

    def test_handles_empty_history(self, tmp_path):
        state_path = tmp_path / "trainer_state.json"
        self._write_state(state_path, [])
        with patch("matplotlib.pyplot.show"):
            from src.utils import plot_loss_curve
            plot_loss_curve(str(state_path))   # should not raise
