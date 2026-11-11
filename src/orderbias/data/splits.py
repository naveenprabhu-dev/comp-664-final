"""Split-integrity helpers.

The project spec requires that every version of a prompt stays in the same
split (so a prompt seen in training can't leak into validation). HelpSteer2's
own ``split`` field already respects this, but we verify it rather than trust
it, and expose a regrouping fallback if a violation ever appears.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set

from .schema import UnifiedExample


def check_prompt_group_integrity(
    examples: Sequence[UnifiedExample],
) -> Dict[str, Set[str]]:
    """Return {prompt_group: {splits...}} for any group spanning >1 split.

    Empty dict == clean (no leakage).
    """
    group_splits: Dict[str, Set[str]] = defaultdict(set)
    for ex in examples:
        key = ex.prompt_group or ex.prompt
        group_splits[key].add(ex.split)
    return {g: s for g, s in group_splits.items() if len(s) > 1}


def split_summary(examples: Sequence[UnifiedExample]) -> Dict[str, int]:
    """Count examples per split."""
    counts: Dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.split] += 1
    return dict(counts)


def by_split(
    examples: Sequence[UnifiedExample], split: str
) -> List[UnifiedExample]:
    return [ex for ex in examples if ex.split == split]
