# Judge configuration

Configuration and judge protocols work without PyTorch, PEFT, CUDA, or downloads.
YAML loading requires the lightweight PyYAML dependency. Backbone construction
and adapter resolution now live in `src/orderbias/models/backbone.py` (see its
directory README). Concrete heads, judge forward passes, and compact checkpoint
save/reload are implemented. Training/tokenization now live under
`src/orderbias/training/`; see its README for the single-device entry point.

```python
from orderbias.models.config import load_config

config = load_config(
    "configs/models/qwen35_08b.yaml",
    "configs/experiments/pairwise_swap.yaml",
    overrides={"lora": {"rank": 16, "alpha": 32}},
)
config.save_json("resolved_config.json")
```

Paths here are relative to the caller's working directory. Precedence is
dataclass defaults, model file, experiment file, then explicit nested overrides.
Mappings merge recursively; scalars and arrays replace earlier values. No
implicit environment overrides or recursive file inheritance are applied.
Unknown keys and malformed types fail, even if a later layer replaces them;
cross-field constraints are checked on the final merged configuration.
The config loader accepts YAML or JSON. `ExperimentConfig.from_dict()` can restore a
saved resolved JSON mapping. Revisions default to `main` for initial exploration;
pin a commit for reproducible runs. The backbone loader records the resolved
model commit; tokenizer loading and its provenance will be added later.

## Modes and targets

- LoRA: `lora.enabled: true`, `quantization.mode: none` (default).
- QLoRA: override `quantization.mode: "4bit"`. Its compute dtype must match the
  backbone compute dtype. The model loader requires CUDA and bitsandbytes;
  configuration validation alone does not establish hardware/library support.
- Head-only probe: override `lora.enabled: false`. Adapter settings remain in the
  resolved config but are inactive. Quantization can independently remain on or
  off. All modes keep original backbone weights frozen and fully train the head.

Use either a `target_preset` (`attention` or `all_language_linear`) or explicit
`target_modules`. For example:

```python
overrides = {"lora": {
    "target_preset": None,
    "target_modules": ["actual.language.projection.path"],
    "exclude_modules": [],
}}
```

Explicit names will be exact module paths relative to the language backbone;
regex selection is not part of this initial contract. Exclusions use the same
exact-path convention. Actual names and preset coverage are resolved against
the bare Qwen3.5 text model when attaching adapters. Selection excludes vision,
vocabulary projection, and task-head modules. Empty selections, ambiguous
selections, duplicate names, and explicit target/exclusion overlap fail. Defaults
are starting values, not a tuned recipe or evidence of checkpoint compatibility.

## Inputs, outputs, and preprocessing policy

`orderbias.models.interfaces` defines generic tensor protocols; concrete models
live in `orderbias.models.judges`. Both accept
`TokenBatch(input_ids, attention_mask)` with shape `[B, L]`.
The pairwise judge returns logits `[B, 2]` (column 0 = A preferred, 1 = B).
The scorer returns scores `[B]` for a prompt plus one answer. Use the same scorer
instance for both candidates. Losses and labels remain outside these forward
interfaces. Exact equal logits/scores mean abstention, preserving slot neutrality.

Initial pooling selects the final valid token in the complete formatted input,
using the attention mask; no new special token is added. Prompt/candidate text
budgets exclude template overhead. Truncate the prompt and each answer
independently on the right before formatting, with an identical answer cap
across both slots and all conditions. The example budgets are 768 prompt tokens
and 1408 tokens per answer, within a 4096-token complete-input limit. The encoder
checks complete formatted length and fails if template overhead exceeds that
limit; it never silently truncates the assembled sequence. Tiny actual-model
padding checks and an actual Qwen tokenizer/template check have passed.

The initial contract requests non-thinking formatting and a normal-initialized,
fully trainable linear head (standard deviation 0.02, zero bias initialization
when bias is enabled). `output_dim` derives from the judge kind. The original
vocabulary head is unused. Both adapters and task heads train in FP32. Head
initialization uses a local generator seeded from `training.seed`, independent
of backbone/adapter random-number consumption. Complete-input masks must be
binary, nonempty per row, and contain no internal padding gaps. Judges compute
text positions from the mask, so left padding does not shift valid positions.

## Training settings

The config includes AdamW learning rates, weight decay, schedule, seed,
microbatch size, gradient accumulation, and a fixed optimizer-step budget.
Batch size counts preference pairs, including for the scorer's two evaluations.
The example 1000-step budget is provisional. Both-order augmentation applies
only to pairwise training; stored validation pairs remain canonical and the
training validator checks both presentation orders. Augmented sampling keeps
twins adjacent and requires an even global presentation budget. `eval_steps`,
`logging_steps`, and `max_grad_norm` control validation, progress printing, and
clipping. The trainer selects by mean validation loss and records actual
presentation/token exposure and elapsed time. It currently supports one process
and device, with no exact optimizer-state resumption.
