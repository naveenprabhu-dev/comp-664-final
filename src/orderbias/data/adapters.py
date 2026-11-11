"""Minimal per-dataset adapters: raw row -> UnifiedExample.

Keep these thin (TODO #2b). All dataset-specific knowledge lives here and
nowhere else. Verified against the live datasets on 2026-11-11:

HelpSteer2 (nvidia/HelpSteer2, preference/preference.jsonl.gz):
    fields: split {"train","val"}, prompt, response_1, response_2,
            preference_strength (signed int -3..3), + justification fields.
    sign convention: preference_strength > 0 -> response_2 better,
                     < 0 -> response_1 better, == 0 -> tie (dropped).

LLMBar (princeton-nlp/LLMBar, Dataset/LLMBar/*/dataset.json):
    fields: input, output_1, output_2, label (1 -> output_1 better,
            2 -> output_2 better).
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .schema import LABEL_A, LABEL_B, UnifiedExample

_SEP = "␟"  # unlikely-to-collide separator for hashing


def _hash(*parts: str) -> str:
    return hashlib.sha1(_SEP.join(parts).encode("utf-8")).hexdigest()[:16]


def adapt_helpsteer2(row: dict) -> Optional[UnifiedExample]:
    """Adapt one HelpSteer2 preference row.

    Returns None for ties (preference_strength == 0), which the binary
    experiment removes per the project spec.
    """
    strength = int(row["preference_strength"])
    if strength == 0:
        return None  # tie -> dropped from the main binary experiment

    label = LABEL_B if strength > 0 else LABEL_A
    split = "validation" if str(row["split"]).lower() in ("val", "validation") else "train"

    prompt = row["prompt"]
    response_a = row["response_1"]
    response_b = row["response_2"]

    return UnifiedExample(
        example_id="hs2-" + _hash(prompt, response_a, response_b),
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        label=label,
        source="helpsteer2",
        split=split,
        subset="preference",
        margin=abs(strength),
        # Keep every response pair for the same prompt in the same split.
        prompt_group="hs2-" + _hash(prompt),
    )


def adapt_llmbar(row: dict, subset: str) -> UnifiedExample:
    """Adapt one LLMBar row. ``subset`` is the canonical subset name."""
    raw_label = int(row["label"])
    if raw_label not in (1, 2):
        raise ValueError(f"unexpected LLMBar label {raw_label!r} (expected 1 or 2)")
    label = LABEL_A if raw_label == 1 else LABEL_B

    prompt = row["input"]
    response_a = row["output_1"]
    response_b = row["output_2"]

    return UnifiedExample(
        example_id=f"llmbar-{subset}-" + _hash(prompt, response_a, response_b),
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        label=label,
        source="llmbar",
        split="eval",
        subset=subset,
        margin=None,  # LLMBar has no graded margin
        prompt_group=f"llmbar-{subset}-" + _hash(prompt),
    )
