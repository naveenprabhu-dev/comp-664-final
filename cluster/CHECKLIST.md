# Longleaf verification checklist

One-time validation that the repo's cluster scaffold actually works on Longleaf.
Fill in the blanks (partitions, module versions, QOS, paths) with the REAL values
you discover in Step 3, then update `cluster/modules.sh` and `cluster/slurm/*.sbatch`
to match. Record results inline (✅ / ❌ + notes) so the team can see the state.

Prereq: a Longleaf account (ONYEN) and, for GPUs, access to the `gpu_access` QOS
(request via the UNC/ITS research-computing GPU access form if you don't have it).

---

## Step 1 — Connect
```bash
ssh <onyen>@longleaf.unc.edu
```
- [ ] logged in

## Step 2 — Get the code onto the cluster
```bash
cd ~
git clone https://github.com/naveenprabhu-dev/comp-664-final.git
cd comp-664-final
```
- [ ] cloned

## Step 3 — Discover the REAL cluster values (replace my placeholders)
```bash
sinfo -s                              # partitions + availability (find the GPU ones)
sinfo -p <gpu-partition> -o "%P %G %m %D"   # GPUs/mem on a specific partition
sacctmgr show qos format=Name%-20     # confirm 'gpu_access' exists
module avail 2>&1 | grep -iE "anaconda|cuda|python"   # real module names/versions
echo "/work/users/${USER:0:1}/${USER:1:1}/$USER"      # your scratch path
```
Record what you find:
- GPU partition(s): ________________   (e.g. volta-gpu / l40-gpu / a100-gpu)
- QOS: ________________                (expected: gpu_access)
- anaconda module: ________________    (e.g. anaconda/2023.03)
- cuda module: ________________
- scratch path exists? ____

> Then edit `cluster/modules.sh` and `cluster/slurm/*.sbatch` to use these values.

## Step 4 — Portable core (no GPU needed; run on the login node)
```bash
module load anaconda                  # use the real name from Step 3
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/ -q
```
- [ ] tests pass (expect 7 passed)

## Step 5 — Build the unified dataset (confirms login-node internet + HF access)
```bash
source cluster/modules.sh             # sets ORDERBIAS_* paths + HF_HOME
python scripts/build_unified_dataset.py
ls -la "$ORDERBIAS_UNIFIED_DIR"
```
- [ ] prints "Split sizes: {'train': 6766, 'validation': 352, 'eval': 419}"
- [ ] train.jsonl / validation.jsonl / eval.jsonl written under $ORDERBIAS_UNIFIED_DIR

## Step 6 — GPU smoke test (the real go/no-go for QLoRA)
Create the conda training env once:
```bash
module load anaconda
conda env create -f cluster/env/environment.yml   # pin CUDA/torch to Step 3 values if this fails
conda activate orderbias
```
Grab an interactive GPU node (use the real partition/QOS from Step 3):
```bash
srun -p <gpu-partition> --qos=gpu_access --gres=gpu:1 \
     -N 1 -n 1 --mem=16g -t 0:20:00 --pty /bin/bash
```
On the GPU node:
```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import bitsandbytes as bnb; print('bitsandbytes OK', bnb.__version__)"
```
- [ ] nvidia-smi shows a GPU
- [ ] torch.cuda.is_available() == True
- [ ] bitsandbytes imports without a CUDA error   <-- if this fails, QLoRA won't work;
      re-pin pytorch-cuda/bitsandbytes in environment.yml to match the node's CUDA.

## Step 7 — (optional) batch submission sanity
```bash
mkdir -p cluster/logs
sbatch cluster/slurm/download_data.sbatch
squeue -u $USER
# when done:
cat cluster/logs/download-*.out
```
- [ ] job completes and log shows the dataset build output

---

## Handoff
Once Steps 1-6 pass and `modules.sh` / `*.sbatch` hold the real values, the cluster
layer is verified. Training (#6, Qi), QLoRA/LoRA setup (#4, Alan), and eval (#7, Uli)
build their workloads on top of this; debugging those specific jobs is their task.
