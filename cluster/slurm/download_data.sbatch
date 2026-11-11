#!/usr/bin/env bash
#SBATCH --job-name=ob-download
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=cluster/logs/download-%j.out

# Build the unified dataset on a node with network access.
# Submit from the repo root:  sbatch cluster/slurm/download_data.sbatch

set -euo pipefail
source cluster/modules.sh
# conda activate orderbias
python scripts/build_unified_dataset.py
