"""
tests/conftest.py — shared fixtures.
"""
from __future__ import annotations
import pytest


@pytest.fixture
def sample_texts():
    return [
        "### Human: What is machine learning?\n### Assistant: ML is a branch of AI...",
        "### Human: Explain LoRA.\n### Assistant: LoRA is a parameter-efficient method...",
        "### Human: Write hello world in Python.\n### Assistant: print('Hello, World!')",
    ]


@pytest.fixture
def trainer_state_entries():
    return [
        {"step": 10 * i, "loss": max(0.1, 2.5 - i * 0.12), "epoch": round(i * 0.05, 2)}
        for i in range(1, 21)
    ]
