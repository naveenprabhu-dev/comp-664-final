#!/usr/bin/env bash
#SBATCH --job-name=ob-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=cluster/logs/finetune-%j.out

# Template for the fine-tuning runs (TODO #6). Fill in the training entrypoint
# once TODOs #4/#5 land. Kept here so all SLURM/GPU specifics stay isolated
# from src/.

set -euo pipefail
source cluster/modules.sh
# conda activate orderbias

nvidia-smi || true

# Example (to be implemented by the fine-tuning task):
# python -m orderbias.train \
#   --judge paired --swap-aug false \
#   --base-model Qwen/Qwen2.5-... \
#   --data-dir "$ORDERBIAS_UNIFIED_DIR"
echo "finetune.sbatch is a template; wire up the training entrypoint in TODO #6."
