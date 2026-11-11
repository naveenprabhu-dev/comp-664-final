"""The single unified example schema shared by every downstream component.

Everything after the per-dataset adapters (the training loop, the three judges,
the evaluation/consistency script) consumes ``UnifiedExample`` and never touches
raw HelpSteer2 / LLMBar fields again. That is the whole point of TODO #2b.

Label convention (applies to BOTH datasets after adaptation):
    label == 0  ->  response_a is the human/gold-preferred answer   (LABEL_A)
    label == 1  ->  response_b is the human/gold-preferred answer   (LABEL_B)

``response_a``/``response_b`` are stored in a fixed canonical order. The order
in which a judge *sees* them (and the swap augmentation / permutation eval) is a
presentation concern handled downstream, not here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

LABEL_A = 0  # response_a preferred
LABEL_B = 1  # response_b preferred

VALID_SPLITS = {"train", "validation", "eval"}
VALID_SOURCES = {"helpsteer2", "llmbar"}


@dataclass
class UnifiedExample:
    """One preference instance in the shared interface.

    Attributes:
        example_id: Stable, content-derived id (dedupe-friendly).
        prompt: The instruction / question.
        response_a: First candidate answer (canonical slot A).
        response_b: Second candidate answer (canonical slot B).
        label: 0 if response_a is preferred, 1 if response_b is preferred.
        source: "helpsteer2" or "llmbar".
        split: "train", "validation", or "eval".
        subset: Provenance tag, e.g. "preference" (HS2) or "natural",
            "adversarial_neighbor", ... (LLMBar). Used for per-subset reporting.
        margin: Human preference strength magnitude (HS2: 1-3). None when the
            source does not provide a margin (LLMBar). Enables TODO #8's
            "small preference margin" analysis.
        prompt_group: Grouping key so all pairs sharing a prompt land in the
            same split (prevents train/val leakage; see project spec).
    """

    example_id: str
    prompt: str
    response_a: str
    response_b: str
    label: int
    source: str
    split: str
    subset: str
    margin: Optional[int] = None
    prompt_group: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label not in (LABEL_A, LABEL_B):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {self.split!r}")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {self.source!r}")
        if not self.prompt or not self.response_a or not self.response_b:
            raise ValueError("prompt and both responses must be non-empty")

    @property
    def preferred_response(self) -> str:
        return self.response_a if self.label == LABEL_A else self.response_b

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedExample":
        return cls(**d)
