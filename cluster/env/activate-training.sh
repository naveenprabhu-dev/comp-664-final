#!/usr/bin/env bash
# Source from any directory: source /path/to/repo/cluster/env/activate-training.sh
module load anaconda/2024.02 || return 1
_orderbias_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$_orderbias_repo/.venv-training/bin/activate" || return 1
export ORDERBIAS_DATA_DIR="${ORDERBIAS_DATA_DIR:-$_orderbias_repo/data}"
export ORDERBIAS_CACHE_DIR="${ORDERBIAS_CACHE_DIR:-$_orderbias_repo/.cache}"
export ORDERBIAS_UNIFIED_DIR="${ORDERBIAS_UNIFIED_DIR:-$ORDERBIAS_DATA_DIR/unified}"
export HF_HOME="${HF_HOME:-$ORDERBIAS_CACHE_DIR/huggingface}"
unset _orderbias_repo
