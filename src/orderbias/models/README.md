# Qwen3.5 backbone loading (Phase 2)

`config.py` and `interfaces.py` remain lightweight. `backbone.py` also imports the
training stack only when a model operation is called. Heads and judge forward
wrappers arrive in Phase 3; the returned backbone produces `last_hidden_state`
with shape `[batch, length, hidden_size]` and computes no task loss.

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
backbone without adapters; a later task head supplies the trainable parameters.
No head training assertion is possible until that head exists in Phase 3.

The report records adapter shapes/dtypes, trainable names/counts, logical total
parameter count, device, requested/resolved backbone revision, ignored checkpoint
keys, and dependency versions. It is not an adapter checkpoint or tokenizer
manifest. Saving/reloading complete judges and tokenizer metadata comes later.

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
