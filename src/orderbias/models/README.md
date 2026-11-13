# Qwen3.5 judges (Phases 2 and 3)

`config.py` and `interfaces.py` remain lightweight. `backbone.py` also imports the
training stack only when a model operation is called. `heads.py`, `judges.py`,
and `checkpoints.py` explicitly import the training stack. Bare backbones produce
`last_hidden_state` with shape `[batch, length, hidden_size]`; judge wrappers
produce class logits or scalar scores. Neither computes a task loss.

## Loading

Install the portable requirements plus `requirements-training.txt` into a
training environment. Select a PyTorch build compatible with the cluster GPU
and CUDA; install bitsandbytes separately for QLoRA. The existing Longleaf
environment remains unverified. Tiny CPU integration tests passed with PyTorch
2.15.0+cpu, Transformers 5.3.0, PEFT 0.18.1, and Accelerate; actual installed
versions are written to each backbone report.

```python
from orderbias.models.config import load_config
from orderbias.models.backbone import load_backbone

config = load_config(
    "configs/models/qwen35_08b.yaml",
    "configs/experiments/pairwise.yaml",
)
loaded = load_backbone(config, device="cuda:0", local_files_only=True)
loaded.report.save_json("backbone_report.json")
# loaded.model(input_ids=..., attention_mask=...).last_hidden_state
```

`local_files_only=True` requires checkpoint files staged in the Hugging Face
cache or a local checkpoint directory. Omitting it permits downloads. The cache
defaults to `ORDERBIAS_CACHE_DIR` through the existing path config and can be
overridden explicitly. Loading uses native Transformers classes, never remote
repository code. The initial API supports a single CPU or CUDA device; a future
distributed launcher must pass the rank's explicit CUDA device. It does not use
automatic device sharding or CPU offload.

The loader reads the official multimodal config but constructs only
`Qwen3_5TextModel` using its text subconfig. It explicitly maps
`model.language_model.*` checkpoint names to the bare text model. The download
can still include the full safetensors shard, but vision, MTP, and vocabulary
heads are not instantiated. Unused known checkpoint keys are permitted;
missing/mismatched language weights and unrecognized extra keys fail. Standalone
`qwen3_5_text` checkpoints are also supported. A remote branch revision is
resolved once and reused for weight loading; local directories have no resolved
Hub commit, so their contents must be preserved separately for reproducibility.

## Adapter placement and freezing

Paths are relative to the bare text model, for example
`layers.3.self_attn.q_proj`. Matching is exact, including exclusions. Only direct
linear projections inside language decoder attention/MLP blocks are eligible.

| Preset | Projections |
| --- | --- |
| `attention` | Full attention `q_proj`, `k_proj`, `v_proj`, `o_proj`; linear attention `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` |
| `all_language_linear` | All of the above plus MLP `gate_proj`, `up_proj`, `down_proj` |

Selection inspects actual modules, not just a hardcoded suffix list. DeltaNet
convolution/state parameters, embeddings, and norms stay frozen. Vision and
heads are outside the eligible tree. Presets target both attention mechanisms;
they are not conventional full-attention-only LoRA. Unknown exclusions and
targets fail rather than silently doing nothing. Reported adapter paths and
matrix shapes show exactly what was attached.

All original parameters remain frozen. LoRA matrices train in FP32 and use
PEFT's feature-extraction wrapper rather than a generative LM wrapper. Cache use
is disabled. `backbone.gradient_checkpointing: true` enables non-reentrant
checkpointing, which is tested with frozen embeddings. `attention_implementation`
chooses `sdpa` (default) or `eager` for full attention; DeltaNet has its own
implementation. Optional optimized DeltaNet kernels are not required by the CPU
checks; their absence uses the slower PyTorch path and can increase memory use.

For QLoRA, set `quantization.mode: "4bit"` with a matching compute dtype.
The loader requires CUDA and bitsandbytes, loads 4-bit layers, then calls PEFT's
`prepare_model_for_kbit_training` before attachment. Nonquantized parameters may
be promoted to FP32 by that preparation. Head-only mode returns a fully frozen
backbone without adapters; the task head supplies the trainable parameters.
Judge construction verifies that every head parameter trains and the backbone's
trainable parameters match its prepared adapter report.

The report records adapter shapes/dtypes, trainable names/counts, logical total
parameter count, device, requested/resolved backbone revision, ignored checkpoint
keys, and dependency versions. It is not an adapter checkpoint or tokenizer
manifest. Compact judge save/reload is described below. The training runner adds
tokenizer assets/metadata; resumable training state remains unimplemented.

## Heads and judge forward passes

```python
from orderbias.models.judges import load_judge
from orderbias.models.interfaces import TokenBatch

judge = load_judge(config, device="cuda:0", local_files_only=True)
judge.eval()
# Token IDs/masks are already formatted and on the judge's device.
output = judge(TokenBatch(input_ids=input_ids, attention_mask=attention_mask))
# PairwiseJudge: output.logits [B, 2], columns A preferred then B preferred.
# SharedScorer: output.scores [B], one independent answer per input row.
```

Use the scorer experiment configuration to construct `SharedScorer`. Score both
candidates using the same instance, identical formatting, and independent text
budgets. Compare `score_A - score_B`; swapping candidates negates the difference
under deterministic inference. Eval mode disables LoRA dropout. The wrapper
does not see slot labels, generate text, compute a loss, or select a winner.
An exact score/logit tie remains available for downstream abstention.

Heads are linear, fully trainable, and initialized from a local CPU generator
using `training.seed`, independently of global RNG consumption. Weight standard
deviation and optional zero-initialized bias follow `JudgeConfig`. Both the head
parameters and head computation start in FP32, including inside an autocast
context. Pooled states are differentiably cast into the head's dtype. Place
models using the loader or device-only moves; do not cast the entire judge to
half precision if retaining this policy.

Both wrappers pool the final valid token using the attention mask rather than
`length - 1` or `mask.sum() - 1`. Inputs must be nonempty `[B, L]`, with integer
IDs and a binary mask containing a nonempty contiguous valid region per row.
Left/right padding is supported, but internal gaps are rejected. Position IDs
are derived from mask cumulative counts to keep valid text positions unchanged
by padding. All tensors must be on the same device. Inputs exceeding the
configured length fail; the wrappers never truncate complete inputs.

Use `build_judge(prepared_backbone, config)` for an already prepared model.
Backbone, LoRA, and quantization settings must match those used to prepare it;
judge/data settings can differ. `load_judge` and direct constructors start in
training mode. `judge.parameter_summary()` reports all trainable names and
separate head/adapter counts without changing model mode.

## Compact save and reload

```python
from orderbias.models.checkpoints import save_judge_checkpoint, load_judge_checkpoint

save_judge_checkpoint(judge, "outputs/run_01/judge")
restored = load_judge_checkpoint(
    "outputs/run_01/judge", device="cuda:0", local_files_only=True,
)
# restored is in eval mode; call train() before further optimization.
```

The destination must be new or empty. This project-specific format saves
`trainable.safetensors` containing every adapter and head parameter,
`resolved_config.json`, `backbone_report.json`, and a final `manifest.json` with
the format version, label convention, and parameter summary. It does not use
pickle or duplicate frozen backbone weights. A resolved Hub commit is pinned in
the saved config. Local backbone directory references are made absolute and
must retain their original contents; moving a checkpoint alone does not make
the referenced local base portable.

Reload reconstructs the referenced backbone and head, checks the exact saved
trainable names/shapes/dtypes, and copies all trainable tensors. Missing heads
or adapters fail rather than silently reinitializing parameters. The format is
intended for judge inference/restarting optimization, not exact training
resumption: tokenizer assets, optimizer/scheduler state, and RNG state are not
included by the model-only saving function. The training runner additionally
saves tokenizer assets and metadata inside each selected/final checkpoint. It
is separate from PEFT's adapter-only export format.

## Verification

With the training test stack installed:

```bash
python -m pytest tests/ -q
python scripts/check_backbone.py --device cuda:0 --local-files-only
python scripts/check_backbone.py --device cuda:0 --quantization 4bit --local-files-only
```

The script loads a pretrained checkpoint, passes eight synthetic token IDs
through the text stack, checks finite outputs and adapter gradients, and writes
resolved configuration, parameter report, and smoke result under
`outputs/backbone_check/`. Choose distinct `--output-dir` paths for separate runs.
It does not train a judge or validate realistic sequence-length memory use.
CPU checks default to FP32; GPU checks retain the configured dtype unless
`--dtype` overrides it. A BF16 CUDA request fails clearly on unsupported devices.

The offline integration tests construct tiny actual Qwen3.5 models containing
both DeltaNet and full attention, then verify gradients, target coverage,
checkpoint mapping, and frozen originals after optimizer steps. They skip if
the training stack is absent. The separate 4-bit integration test requires CUDA
and bitsandbytes. **CUDA/QLoRA and full pretrained checkpoints have not been run
on Longleaf yet.** These checks are required before treating Phase 2's GPU
completion criterion as satisfied.

Architecture/API references used for this implementation:

- [Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Transformers 5.3.0 Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Official 0.8B checkpoint config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
- [PEFT quantization preparation](https://huggingface.co/docs/peft/developer_guides/quantization)
