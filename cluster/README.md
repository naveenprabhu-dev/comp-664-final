# Cluster-facing layer (Longleaf)

## Current LoRA training environment (2026-11-15)

This checkout now has an isolated `.venv-training` using the cluster's
`anaconda/2024.02` Python 3.11.7, PyTorch 2.6.0+cu124, Transformers 5.3.0,
PEFT 0.18.1, and Accelerate 1.15.0. Activate it from the repository root:

```bash
source cluster/env/activate-training.sh
```

This activation script sets project-local data/cache defaults; existing explicit
environment overrides are preserved. It does not use `cluster/modules.sh` or
the older, unverified conda environment below. The CUDA-enabled PyTorch wheel
includes its CUDA runtime; this setup does not load a separate CUDA module.
bitsandbytes is not installed: the default ordinary LoRA configurations do not
need it. QLoRA requires a separate installation and GPU validation.

To recreate the environment in another checkout:

```bash
module load anaconda/2024.02
python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r cluster/env/requirements-cu124.txt
python -m pip install -e .
```

The installed environment passed `pip check` and the offline suite (90 passed,
1 CUDA/4-bit test skipped). The exact installed package snapshot is saved locally
in `outputs/environment/pip-freeze.txt`.

GPU visibility requires an allocation: `longleaf-login1` has no allocated GPU.
The partition named `gpu` contains GTX 1080 cards; use `a100-gpu` or `l40-gpu`
for the default BF16 configurations. This account reports `gpu_access` QoS.
Short immediate requests to both partitions found the nodes busy. An offline
GPU validation was submitted as job **9314589**; at the last check it was pending
for priority, so CUDA execution remains unverified.

```bash
# Check the existing job before submitting another.
squeue -j 9314589
cat cluster/logs/env-check-9314589.out
# To run a new check:
mkdir -p cluster/logs
sbatch cluster/slurm/check_training_env.sbatch
# For an interactive GPU session instead:
srun -p a100-gpu --qos=gpu_access --gres=gpu:1 -N 1 -n 1 --mem=16G -t 00:20:00 --pty bash
# Inside the allocation, from this checkout:
source cluster/env/activate-training.sh
python cluster/env/check_gpu.py
```

The GPU script checks BF16 CUDA forward/backward and optimizer steps for tiny
hybrid Qwen3.5 pairwise/scorer judges. It does not establish full pretrained
checkpoint compatibility, production-length memory use, or QLoRA support.
The existing `finetune.sbatch` remains a placeholder.

Everything in this directory is **specific to the Longleaf HPC cluster** and is
deliberately kept out of `src/`. The portable core (`src/orderbias/`) never
hardcodes a cluster path; it reads paths from environment variables (see
`src/orderbias/config.py`). Data/config/interface imports stay lightweight;
model and training implementations may explicitly use GPU libraries.
To move the project to a
different machine, you only change files here.

## What lives here
- `modules.sh` — `module load` lines + env-var exports (data dirs, HF cache).
- `env/environment.yml` — conda env with the CUDA training stack
  (torch/transformers/peft/bitsandbytes/trl). This is the heavy, GPU-specific
  dependency set, separate from the portable `requirements.txt` at the repo root.
  **⚠️ UNVERIFIED on Longleaf** — the CUDA/torch pins are a best-guess; run
  `CHECKLIST.md` Step 6 (GPU smoke test) before trusting it for training.
- `slurm/` — SBATCH job templates (data download, fine-tuning).

## Typical workflow on Longleaf
```bash
# 1. one-time: create the conda env
module load anaconda
conda env create -f cluster/env/environment.yml

# 2. every session: load modules + export paths
source cluster/modules.sh

# 3. build the unified dataset (login node has network; compute nodes may not)
python scripts/build_unified_dataset.py

# 4. submit a job (later TODOs)
sbatch cluster/slurm/finetune.sbatch
```

## Why "isolate the cluster-facing environment"
Teammates develop and run the data/eval logic on a laptop with the small
`requirements.txt`. Only the pieces that truly depend on the cluster — SLURM
scheduling, `module load`, the CUDA conda env, and scratch/work paths — live
here. Swap this directory out and the same `src/` runs anywhere.
