#!/usr/bin/env bash
# Submit from the repository root.
#SBATCH --job-name=ob-env-check
#SBATCH --partition=a100-gpu,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=cluster/logs/env-check-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster/env/activate-training.sh
set -u
export OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1
nvidia-smi
python cluster/env/check_gpu.py
