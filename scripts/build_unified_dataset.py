#!/usr/bin/env python3
"""Materialize the unified dataset to JSONL files (train/validation/eval).

Run anywhere with network access (laptop or Longleaf login node):

    python scripts/build_unified_dataset.py

Outputs (under ORDERBIAS_UNIFIED_DIR, default <repo>/data/unified/):
    train.jsonl        HelpSteer2 train pairs (ties removed)
    validation.jsonl   HelpSteer2 validation pairs (ties removed)
    eval.jsonl         LLMBar, all subsets, tagged by subset

Each line is a UnifiedExample dict (see orderbias.data.schema).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``src`` importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orderbias import config  # noqa: E402
from orderbias.data import (  # noqa: E402
    check_prompt_group_integrity,
    load_helpsteer2,
    load_llmbar,
)
from orderbias.data.splits import by_split, split_summary  # noqa: E402


def _write_jsonl(path: Path, examples) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    print(f"  wrote {len(examples):>6} -> {path}")


def main() -> int:
    config.ensure_dirs()

    print("Loading HelpSteer2 preference annotations (train + validation)...")
    hs2 = load_helpsteer2()
    print("Loading LLMBar (eval, all subsets)...")
    llmbar = load_llmbar()

    all_examples = hs2 + llmbar

    violations = check_prompt_group_integrity(all_examples)
    if violations:
        print(
            f"WARNING: {len(violations)} prompt group(s) span multiple splits "
            "(possible leakage). First few:",
            dict(list(violations.items())[:5]),
        )
    else:
        print("Prompt-group integrity OK (no prompt spans multiple splits).")

    print("Split sizes:", split_summary(all_examples))

    out = config.UNIFIED_DIR
    _write_jsonl(out / "train.jsonl", by_split(hs2, "train"))
    _write_jsonl(out / "validation.jsonl", by_split(hs2, "validation"))
    _write_jsonl(out / "eval.jsonl", llmbar)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
