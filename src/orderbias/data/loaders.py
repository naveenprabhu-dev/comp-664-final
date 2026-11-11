"""Loaders that fetch the raw datasets and return UnifiedExample lists.

HelpSteer2 is pulled from the HF Hub as a gzipped JSONL (the preference
annotations, NOT the per-attribute helpfulness ratings). LLMBar is pulled as
raw JSON from the project's GitHub repo, because its HF loader is a dataset
script that newer ``datasets`` versions no longer execute. Both are cached
under ``config.CACHE_DIR`` so repeated runs (and offline compute nodes) work.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .. import config
from .adapters import adapt_helpsteer2, adapt_llmbar
from .schema import UnifiedExample

HS2_REPO = "nvidia/HelpSteer2"
HS2_PREFERENCE_FILE = "preference/preference.jsonl.gz"

LLMBAR_BASE = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar"
# canonical subset name -> path under the LLMBar dataset dir
LLMBAR_SUBSETS = {
    "natural": "Natural",
    "adversarial_neighbor": "Adversarial/Neighbor",
    "adversarial_gptinst": "Adversarial/GPTInst",
    "adversarial_gptout": "Adversarial/GPTOut",
    "adversarial_manual": "Adversarial/Manual",
}


# --------------------------------------------------------------------------- #
# HelpSteer2
# --------------------------------------------------------------------------- #
def _iter_helpsteer2_raw(cache_dir: Optional[Path] = None) -> Iterator[dict]:
    from huggingface_hub import hf_hub_download

    cache_dir = cache_dir or config.CACHE_DIR
    path = hf_hub_download(
        HS2_REPO,
        HS2_PREFERENCE_FILE,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_helpsteer2(cache_dir: Optional[Path] = None) -> List[UnifiedExample]:
    """Load HelpSteer2 preference pairs as UnifiedExamples (train + validation).

    Ties (preference_strength == 0) are removed for the binary experiment; the
    adapter returns None for them, so they are simply skipped here.
    """
    examples: List[UnifiedExample] = []
    for row in _iter_helpsteer2_raw(cache_dir):
        ex = adapt_helpsteer2(row)
        if ex is None:
            continue  # tie
        examples.append(ex)
    return examples


# --------------------------------------------------------------------------- #
# LLMBar
# --------------------------------------------------------------------------- #
def _fetch_llmbar_subset(subset: str, cache_dir: Optional[Path] = None) -> List[dict]:
    import requests

    if subset not in LLMBAR_SUBSETS:
        raise KeyError(f"unknown LLMBar subset {subset!r}; valid: {sorted(LLMBAR_SUBSETS)}")

    cache_dir = cache_dir or config.CACHE_DIR
    local = Path(cache_dir) / "llmbar" / f"{subset}.json"
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))

    url = f"{LLMBAR_BASE}/{LLMBAR_SUBSETS[subset]}/dataset.json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(data), encoding="utf-8")
    return data


def load_llmbar(
    subsets: Optional[Iterable[str]] = None,
    cache_dir: Optional[Path] = None,
) -> List[UnifiedExample]:
    """Load LLMBar eval pairs as UnifiedExamples (all subsets by default)."""
    subsets = list(subsets) if subsets is not None else list(LLMBAR_SUBSETS)
    examples: List[UnifiedExample] = []
    for subset in subsets:
        for row in _fetch_llmbar_subset(subset, cache_dir):
            examples.append(adapt_llmbar(row, subset))
    return examples


# --------------------------------------------------------------------------- #
# Unified
# --------------------------------------------------------------------------- #
def load_unified(cache_dir: Optional[Path] = None) -> List[UnifiedExample]:
    """HelpSteer2 (train+validation) + LLMBar (eval) under one interface."""
    return load_helpsteer2(cache_dir=cache_dir) + load_llmbar(cache_dir=cache_dir)
