#!/usr/bin/env bash
# Longleaf session setup. Source this (don't execute):  source cluster/modules.sh
#
# This is the ONE place cluster-specific paths and modules are configured.
# Edit the paths to match your Longleaf allocation; src/ picks them up via
# environment variables and never needs to change.

# --- modules (adjust to what Longleaf currently provides) --------------------
module load anaconda 2>/dev/null || true
module load cuda 2>/dev/null || true

# --- activate the training conda env (see cluster/env/environment.yml) -------
# conda activate orderbias

# --- paths: keep large data/caches off the quota-limited home dir ------------
# On Longleaf, /work/users/<n>/<o>/<onyen> is scratch-like space.
export ORDERBIAS_DATA_DIR="${ORDERBIAS_DATA_DIR:-$HOME/orderbias-data}"
export ORDERBIAS_CACHE_DIR="${ORDERBIAS_CACHE_DIR:-$ORDERBIAS_DATA_DIR/cache}"
export ORDERBIAS_UNIFIED_DIR="${ORDERBIAS_UNIFIED_DIR:-$ORDERBIAS_DATA_DIR/unified}"

# Keep the (large) HuggingFace cache out of $HOME too.
export HF_HOME="${HF_HOME:-$ORDERBIAS_CACHE_DIR/huggingface}"

echo "orderbias env configured:"
echo "  ORDERBIAS_DATA_DIR=$ORDERBIAS_DATA_DIR"
echo "  ORDERBIAS_CACHE_DIR=$ORDERBIAS_CACHE_DIR"
echo "  HF_HOME=$HF_HOME"
