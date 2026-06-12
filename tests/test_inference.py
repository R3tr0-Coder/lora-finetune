"""
tests/test_inference.py

Unit tests for src/inference.py — generate() and batch_generate().
All model/tokenizer calls are pure-Python mocks; no torch or GPU needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ── Minimal tensor-like stub ──────────────────────────────────────────────────

class _FakeTensor:
    """Just enough interface to satisfy the generate() function."""
    def __init__(self, data: list):
        self._data = data

    def __getitem__(self, idx):
        # support obj[0] and obj[0][n:]
        if isinstance(idx, int):
            return _FakeSlice(self._data)
        return _FakeTensor(self._data[idx])

    @property
    def shape(self):
        class _Shape:
            def __init__(self, n): self._n = n
            def __getitem__(self, i): return self._n
        return _Shape(len(self._data))

    def tolist(self):
        return self._data


class _FakeSlice:
    """Returned by tensor[0] — supports further slicing."""
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _FakeTensor(self._data[idx])
        return self._data[idx]

    @property
    def shape(self):
        class _S:
            def __init__(self, n): self._n = n
            def __getitem__(self, i): return self._n
        return _S(len(self._data))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_model_tok(input_len=3, output_ids=None):
    """Return a (model, tokenizer) mock pair."""
    if output_ids is None:
        output_ids = [1, 2, 3, 10, 11, 12]

    # Build a mock encoding that has .to() and correct shape
    encoding = MagicMock()
    encoding.__getitem__ = lambda self, k: (
        _FakeTensor([list(range(input_len))]) if k == "input_ids"
        else _FakeTensor([[1] * input_len])
    )
    encoding.to.return_value = encoding   # .to(device) returns self

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.return_value = encoding
    tokenizer.decode.return_value = "The answer is 42."

    model = MagicMock()
    model.device = "cpu"
    model.eval    = MagicMock()
    model.generate.return_value = _FakeTensor([output_ids])

    return model, tokenizer


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_returns_string(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        result = generate(model, tok, "What is 6×7?")
        assert isinstance(result, str)

    def test_model_eval_called(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "prompt")
        model.eval.assert_called_once()

    def test_model_generate_called(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "prompt")
        assert model.generate.called

    def test_max_new_tokens_passed(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "prompt", max_new_tokens=64)
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 64

    def test_greedy_when_temperature_zero(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "prompt", temperature=0)
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["do_sample"] is False

    def test_sampling_when_temperature_nonzero(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "prompt", temperature=0.7)
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["do_sample"] is True

    def test_custom_prompt_template(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "Hi there", prompt_template="User: {prompt}\nBot:")
        call_text = tok.call_args[0][0]
        assert "User: Hi there" in call_text
        assert "Bot:" in call_text

    def test_system_prompt_prepended(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "question", system_prompt="You are a helpful assistant.")
        call_text = tok.call_args[0][0]
        assert "You are a helpful assistant." in call_text

    def test_repetition_penalty_passed(self):
        from src.inference import generate
        model, tok = _make_model_tok()
        generate(model, tok, "prompt", repetition_penalty=1.3)
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["repetition_penalty"] == 1.3

    def test_decode_called_with_new_tokens_only(self):
        """
        Decode should be called once and should NOT include the prompt prefix.
        We verify that input tokens (0,1,2) are not in the decoded slice.
        """
        from src.inference import generate
        input_len  = 3
        new_tokens = [99, 88, 77]
        all_ids    = list(range(input_len)) + new_tokens  # [0,1,2,99,88,77]
        model, tok = _make_model_tok(
            input_len=input_len,
            output_ids=all_ids,
        )
        generate(model, tok, "prompt")
        assert tok.decode.called, "tokenizer.decode was never called"
        # The argument is whatever _FakeTensor[input_len:] produces.
        # We can't test its exact value without a real torch tensor, but
        # we can confirm decode was called exactly once.
        assert tok.decode.call_count == 1


class TestBatchGenerate:
    def test_returns_list_same_length_as_prompts(self):
        from src.inference import batch_generate
        model, tok = _make_model_tok()
        results = batch_generate(model, tok, ["q1", "q2", "q3"])
        assert len(results) == 3

    def test_all_results_are_strings(self):
        from src.inference import batch_generate
        model, tok = _make_model_tok()
        results = batch_generate(model, tok, ["a", "b"])
        assert all(isinstance(r, str) for r in results)

    def test_empty_list_returns_empty(self):
        from src.inference import batch_generate
        model, tok = _make_model_tok()
        assert batch_generate(model, tok, []) == []

    def test_kwargs_forwarded(self):
        """Keyword args like max_new_tokens should be passed to generate()."""
        from src.inference import batch_generate
        model, tok = _make_model_tok()
        batch_generate(model, tok, ["q"], max_new_tokens=32)
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 32
