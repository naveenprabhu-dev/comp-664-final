# Cluster-facing layer (Longleaf)

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
