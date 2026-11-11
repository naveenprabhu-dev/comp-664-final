"""Path configuration resolved from environment variables.

This is the single seam between the portable core and whatever machine it runs
on. Core code never hardcodes a filesystem path; instead it asks this module,
which reads env vars (with sensible local defaults). On Longleaf you export the
cluster-appropriate values (e.g. scratch/work paths, a large HF cache) in
``cluster/modules.sh`` so nothing in ``src/`` has to change.

Relevant env vars:
  ORDERBIAS_DATA_DIR     root for datasets/artifacts   (default: <repo>/data)
  ORDERBIAS_CACHE_DIR    HF/download cache             (default: <data>/cache)
  ORDERBIAS_UNIFIED_DIR  materialized unified JSONL    (default: <data>/unified)
  HF_HOME                honored by huggingface_hub if set (set on the cluster)
"""

from __future__ import annotations

import os
from pathlib import Path

# <repo>/src/orderbias/config.py -> parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _env_path("ORDERBIAS_DATA_DIR", REPO_ROOT / "data")
CACHE_DIR = _env_path("ORDERBIAS_CACHE_DIR", DATA_DIR / "cache")
UNIFIED_DIR = _env_path("ORDERBIAS_UNIFIED_DIR", DATA_DIR / "unified")


def ensure_dirs() -> None:
    """Create the data directories if they do not exist."""
    for d in (DATA_DIR, CACHE_DIR, UNIFIED_DIR):
        d.mkdir(parents=True, exist_ok=True)
