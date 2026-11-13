# Judge configuration (Phase 1)

Configuration and judge protocols work without PyTorch, PEFT, CUDA, or downloads.
YAML loading requires the lightweight PyYAML dependency. Model construction,
adapter target resolution, tokenization, and training are later phases.

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
The loader accepts YAML or JSON. `ExperimentConfig.from_dict()` can restore a
saved resolved JSON mapping. Revisions default to `main` for initial exploration;
pin a commit for reproducible runs. The future loader must record resolved model
and tokenizer commits, not just the requested revision.

## Modes and targets

- LoRA: `lora.enabled: true`, `quantization.mode: none` (default).
- QLoRA: override `quantization.mode: "4bit"`. Its compute dtype must match the
  backbone compute dtype. Hardware/library support is not checked in this phase.
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
exact-path convention. Actual names and preset coverage must be resolved and
validated against Qwen3.5 in Phase 2. Selection must always exclude vision,
vocabulary projection, and task-head modules. Empty selections, ambiguous
selections, duplicate names, and explicit target/exclusion overlap fail. Defaults
are starting values, not a tuned recipe or evidence of checkpoint compatibility.

## Inputs, outputs, and preprocessing policy

`orderbias.models.interfaces` defines generic tensor protocols, not runnable
models. Both accept `TokenBatch(input_ids, attention_mask)` with shape `[B, L]`.
The pairwise judge returns logits `[B, 2]` (column 0 = A preferred, 1 = B).
The scorer returns scores `[B]` for a prompt plus one answer. Use the same scorer
instance for both candidates. Losses and labels remain outside these forward
interfaces. Exact equal logits/scores mean abstention, preserving slot neutrality.

Initial pooling selects the final valid token in the complete formatted input,
using the attention mask; no new special token is added. Prompt/candidate text
budgets exclude template overhead. Truncate the prompt and each answer
independently on the right before formatting, with an identical answer cap
across both slots and all conditions. The example budgets are 768 prompt tokens
and 1408 tokens per answer, within a 4096-token complete-input limit. The future
tokenizer must verify actual formatted length and fail if template overhead
exceeds the remaining space; it must never silently truncate the assembled
sequence and thereby favor a slot. Masking/pooling behavior and checkpoint chat
template compatibility will be verified during model and tokenizer integration.

The initial contract requests non-thinking formatting and a normal-initialized,
fully trainable linear head (standard deviation 0.02, zero bias initialization
when bias is enabled). `output_dim` derives from the judge kind. The original
vocabulary head is unused. Adapter dtype handling is settled in Phase 2.

## Training settings reserved for later phases

The config includes AdamW learning rates, weight decay, schedule, seed,
microbatch size, gradient accumulation, and a fixed optimizer-step budget.
Batch size counts preference pairs, including for the scorer's two evaluations.
The example 1000-step budget is provisional. Both-order augmentation applies
only to pairwise training; validation keeps canonical pairs and evaluation
explicitly checks both orders. Sampling must ensure both orientations receive
coverage under the chosen budget. World size, actual presentations, processed
tokens, and compute will be recorded by the future training runner.
