"""Offline unit tests for the dataset adapters and schema.

No network: adapters are fed synthetic rows shaped like the real data.
Run with:  python -m pytest tests/ -q   (or: python tests/test_adapters.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orderbias.data import adapt_helpsteer2, adapt_llmbar  # noqa: E402
from orderbias.data.schema import LABEL_A, LABEL_B  # noqa: E402
from orderbias.data.splits import check_prompt_group_integrity  # noqa: E402


def _hs2_row(strength, split="train"):
    return {
        "split": split,
        "prompt": "Define signal.",
        "response_1": "A signal is energy carrying information.",
        "response_2": "A signal is a message over a channel.",
        "preference_strength": strength,
    }


def test_hs2_positive_strength_prefers_response_b():
    ex = adapt_helpsteer2(_hs2_row(2))
    assert ex is not None
    assert ex.label == LABEL_B
    assert ex.margin == 2
    assert ex.source == "helpsteer2"
    assert ex.subset == "preference"
    assert ex.split == "train"


def test_hs2_negative_strength_prefers_response_a():
    ex = adapt_helpsteer2(_hs2_row(-3))
    assert ex is not None
    assert ex.label == LABEL_A
    assert ex.margin == 3


def test_hs2_tie_is_dropped():
    assert adapt_helpsteer2(_hs2_row(0)) is None


def test_hs2_val_maps_to_validation():
    ex = adapt_helpsteer2(_hs2_row(1, split="val"))
    assert ex.split == "validation"


def test_llmbar_label_1_prefers_a_label_2_prefers_b():
    row = {"input": "Q?", "output_1": "A1", "output_2": "A2", "label": 1}
    ex = adapt_llmbar(row, "natural")
    assert ex.label == LABEL_A
    assert ex.split == "eval"
    assert ex.subset == "natural"
    assert ex.margin is None

    row2 = {"input": "Q?", "output_1": "A1", "output_2": "A2", "label": 2}
    assert adapt_llmbar(row2, "adversarial_neighbor").label == LABEL_B


def test_unified_schema_roundtrip():
    ex = adapt_helpsteer2(_hs2_row(1))
    assert ex.to_dict()["label"] == ex.label


def test_prompt_group_integrity_detects_leak():
    a = adapt_helpsteer2(_hs2_row(1, split="train"))
    b = adapt_helpsteer2(_hs2_row(1, split="val"))  # same prompt, other split
    # identical prompt -> identical prompt_group -> must be flagged
    violations = check_prompt_group_integrity([a, b])
    assert a.prompt_group == b.prompt_group
    assert violations, "expected a leakage violation for same-prompt cross-split"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
