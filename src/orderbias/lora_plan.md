# Plan: configurable LoRA judges

Status: Phase 1 configuration and interfaces implemented; model loading and
training have not started. Implement and review one phase at a time. This
document records the intended design, not verified model/library compatibility.

## Objective and experimental conditions

Study whether learned answer judges can remain consistent when candidate answers
are swapped while preserving agreement with human preferences. Start with one
Qwen3.5 checkpoint (0.8B or 2B), shared across three independently trained runs:

| Condition | Input | Head | Supervision |
| --- | --- | --- | --- |
| Standard pairwise | Prompt and both candidates in one ordering | Two-class linear head | Preference classification |
| Swap-augmented pairwise | Same format, with both candidate orderings represented in training | Identical two-class head | Preference classification with labels flipped on swaps |
| Shared scorer | Prompt and one candidate at a time | Shared scalar linear head | Pairwise ranking from independently computed scores |

Use the existing unified HelpSteer2 preference examples for all three conditions.
Do not introduce absolute attribute ratings into the main comparison: that would
change the supervision as well as the architecture. Keep LLMBar for external
evaluation. Retain the current binary convention: label 0 means A preferred,
label 1 means B preferred; ties are currently excluded by the adapter.

## Code boundaries

Reusable model, training, and evaluation logic belongs in `src/orderbias/`.
`cluster/` owns Longleaf environments, modules, SLURM resources, and launchers.
Model code must not depend on SLURM, cluster paths, or a particular GPU location.
Use the repository's existing environment-based path configuration.

Proposed files, created only as each phase needs them:

```text
src/orderbias/
  models/
    config.py          # Validated backbone, head, adapter, quantization settings
    backbone.py        # Load language backbone; resolve and attach adapters
    heads.py           # Two-class and scalar linear heads
    judges.py          # PairwiseJudge and SharedScorer
  training/
    data.py            # Presentation templates, tokenization, swap augmentation
    losses.py          # Classification and Bradley-Terry objectives
    train.py           # Reusable training entry point
  evaluation/
    evaluate.py        # Accuracy, swap consistency, per-example predictions
configs/
  models/
    qwen35_08b.yaml
    qwen35_2b.yaml
  experiments/
    pairwise.yaml
    pairwise_swap.yaml
    shared_scorer.yaml
cluster/
  env/                 # Existing training environment
  slurm/               # Existing convention for job templates
```

Allow PyTorch and training dependencies in model/training modules, but keep data
preparation lightweight. Do not import the heavy stack from the package's root
`__init__.py`. Update the existing README language about the portable core when
these modules land: portable does not mean unable to use a GPU.

## Frozen and trainable parameters

For a targeted linear projection, LoRA computes:

`output = W x + (alpha / rank) B A x`

Freeze all original backbone parameters, including embeddings, normalization,
attention, and MLP weights. Train only the injected low-rank adapter matrices and
all parameters of the new task head. The original vocabulary projection is not
used to produce judge predictions. Adapter-disabled mode trains only the new
head and serves as an optional linear-probe baseline.

Each experimental condition owns separate adapters and a separate head. Within
the shared scorer, both candidates use exactly the same backbone, adapters, and
head; there are no candidate-specific parameters.

Ordinary LoRA uses an unquantized frozen backbone, preferably BF16 if supported
and feasible. QLoRA uses a quantized frozen backbone, initially considering 4-bit
NF4 with configurable double quantization and compute dtype. Keep adapters and
the head in suitable higher precision. Choose the same precision policy across
the three main conditions. Validate support before committing to either path.

## Configuration contract

Keep configuration parsing and validation independent of heavyweight imports.
Use one explicit, documented merge order for model defaults, experiment settings,
and command-line overrides; write the fully resolved result into each run.

Expose at least:

- Backbone: checkpoint ID, revision, tokenizer/template settings, compute dtype,
  and maximum input length.
- Judge: pairwise classifier or shared scalar scorer, terminal pooling policy,
  and head initialization settings.
- LoRA: enabled, rank, alpha, dropout, target preset or explicit module selection,
  and exclusions. Start with original biases frozen.
- Quantization: disabled or supported 4-bit mode, quantization type, double
  quantization, and compute dtype.
- Training: seed, batch size, gradient accumulation, optimizer settings, separate
  adapter/head learning rates if needed, schedule, and training budget.
- Data: augmentation policy and symmetric truncation policy.

Provide candidate target presets such as `attention` and `all_language_linear`,
plus an explicit module-selection escape hatch. Presets must resolve against
the actual Qwen3.5 architecture. Its hybrid architecture means an attention
preset must document exactly which projection families it covers; do not copy
module names from a different Qwen version.

Resolve targets only within the language backbone. Exclude vision components,
the vocabulary projection, and the new task head. Fail on empty or invalid
selections. Record resolved module paths, adapter shapes, trainable parameter
names, and trainable/total parameter counts. Assert that original weights remain
frozen and that the new head is fully trainable. A rank of 8 or 16 with broad
language-linear coverage is a candidate starting point, not a fixed requirement.

## Judge behavior and objectives

Pool the final-layer hidden state at a consistently defined terminal input
position, ignoring padding. A terminal marker after the complete input is an
option; settle the exact policy after checking the checkpoint's template and
tokenization. This is the hidden state before vocabulary projection, not a
generated token. No generated labels, explanations, or reasoning are required.

For pairwise classification:

`logits = linear_head(hidden(prompt, A, B))`

Optimize two-class cross-entropy with the existing 0/1 labels. The standard and
augmented models have identical formatting, pooling, heads, and adapter settings.
Augmentation changes `(prompt, A, B, label)` to
`(prompt, B, A, 1 - label)`; it never reverses words within an answer.

For the shared scorer:

`s(prompt, answer) = scalar_head(hidden(prompt, answer))`

`loss = -log_sigmoid(s(prompt, preferred) - s(prompt, rejected))`

Use a numerically stable Bradley-Terry loss. Infer the winner from the score
difference; optionally report `sigmoid(score_A - score_B)` as the pairwise
probability. These are relative utility scores, not calibrated 0-4 ratings.
Start without preference-strength weighting or an extra margin term; preserve
the dataset's margin for stratified analysis.

The scorer must never see A/B slot labels or the competing answer. Score each
candidate with identical independent formatting and truncation. Under
deterministic evaluation, swapping candidates then negates the score difference
by construction. Define exact ties explicitly rather than always choosing the
first slot. This guarantees order consistency, not correctness or freedom from
length/style biases. Independent scoring also loses direct candidate interaction.

## Experimental controls and artifacts

- Preserve prompt-group train/validation integrity before augmentation.
- Start all conditions from the same backbone revision. Match pairwise head
  initialization across standard/augmented runs for a given seed.
- Make training exposure explicit: doubling a dataset at fixed epochs doubles
  presentations. Use a matched presentation/update budget for the primary
  pairwise comparison, ensuring both orientations receive coverage. Record
  processed pairs, tokens, optimizer steps, and compute; the scorer has a
  different input/forward-pass structure. A fixed-epoch comparison can be a
  separate ablation.
- Use candidate-independent truncation budgets so swapping does not change which
  answer text survives. Log truncation rates, including by presentation slot.
- Select checkpoints using held-out HelpSteer2, never LLMBar. Fix the selection
  criterion before comparing results, and use multiple seeds when feasible.
- Evaluate each held-out pair in both orders. Report accuracy per order and
  averaged across orders, semantic winner consistency, first-slot selection
  rate, and probability complement error. Report tie frequency separately and
  break out HelpSteer2 preference margins and LLMBar subsets.
- Save adapters, the complete new head, resolved configuration, backbone revision,
  tokenizer/template details, label mapping, dependency versions, and evaluation
  outputs. Reload must reconstruct the same judge; include optimizer/scheduler
  state when resumable training is required.

## Phased implementation

### Phase 1: configuration and interfaces

Add validated configuration structures, example model/experiment configurations,
and documented judge input/output contracts. Decide config precedence and the
initial pooling/truncation policies. Keep this phase usable without CUDA or a
model download.

Completion: configs express all three conditions and LoRA/QLoRA/head-only modes;
invalid combinations fail clearly; existing data-only imports still work.

Implemented: lightweight frozen dataclasses and strict YAML loading in
`models/config.py`, generic tensor contracts in `models/interfaces.py`, and
model/experiment YAML examples under `configs/`. See `configs/README.md` for
precedence, overrides, pooling, and truncation decisions. No concrete model,
tokenizer, adapter resolution, or CUDA execution is included in Phase 1.

### Phase 2: backbone loading and adapter attachment

Inspect Qwen3.5's actual language-backbone modules and hidden-state interface.
Implement model loading, supported quantization, target resolution, freezing,
adapter injection, and parameter reporting. Do not assume a generic sequence
classification wrapper supports this checkpoint. Verify compatible Transformers,
PEFT, PyTorch, and optional bitsandbytes versions; existing environment minimums
are not evidence of Qwen3.5 support.

Completion: a small load/forward/backward check confirms the intended modules
receive adapters, only intended parameters get gradients, and the selected
precision path works on the target GPU. Validate QLoRA separately if enabled.

### Phase 3: heads and judge forward passes

Implement terminal pooling, both linear heads, pairwise classification, and the
shared scorer. Fully train and save heads independently of adapter targeting.

Completion: correct output shapes and padding behavior; the scalar scorer uses
shared weights and reproduces scores under candidate reordering in eval mode;
a save/reload check preserves predictions within the chosen numerical tolerance.

### Phase 4: training data and objectives

Add deterministic templates, tokenization, symmetric truncation, label-flipping
augmentation, both losses, and a training entry point with explicit budgets.
Keep canonical unified examples unchanged.

Completion: inspect actual formatted examples; verify swap labels and split
integrity; verify the ranking loss favors the preferred answer; run a tiny
training smoke check that exercises adapters and heads.

### Phase 5: Longleaf integration

Validate/update `cluster/env/environment.yml` and the GPU checks in the existing
cluster checklist. Add thin SLURM launchers that select configs and supply
resource/path overrides. Plan checkpoint staging and offline loading if compute
nodes cannot access the model host. Run a short GPU job before full training.

Completion: a cluster run saves a reloadable judge, resolved configuration, and
training logs; record the environment and observed memory use.

### Phase 6: evaluation and controlled experiment runs

Implement both-order evaluation, the metrics above, and subset/margin summaries.
Confirm the shared scorer's structural consistency, then compare human agreement
and compute under the declared budgets. Launch full runs only after the smoke
checks and experimental controls are settled.

Completion: comparable results for all three conditions, including per-example
outputs, seeds, training exposure, tie handling, and any compatibility or
truncation limitations.

## Design references

- Qwen3.5 checkpoint cards:
  https://huggingface.co/Qwen/Qwen3.5-0.8B and
  https://huggingface.co/Qwen/Qwen3.5-2B
- PEFT LoRA: https://huggingface.co/docs/peft/package_reference/lora
- PEFT quantization: https://huggingface.co/docs/peft/developer_guides/quantization
- PEFT head saving considerations:
  https://huggingface.co/docs/peft/developer_guides/troubleshooting
- Bradley-Terry reward modeling: https://huggingface.co/docs/trl/reward_trainer
- HelpSteer2 supervision: https://huggingface.co/datasets/nvidia/HelpSteer2
