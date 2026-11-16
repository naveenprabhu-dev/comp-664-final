# Judge training (Phase 4)

`data.py` prepares immutable presentations of unified examples, `losses.py`
implements preference objectives, and `train.py` runs fixed-budget single-device
training. Dataset preparation/config/interface imports remain free of the GPU
stack. Training itself requires the portable and training requirements.

## Running

First materialize the unified data using `scripts/build_unified_dataset.py` and
stage the model/tokenizer assets. GPU/QLoRA execution still needs verification on
Longleaf. From the repository root:

```bash
python scripts/train_judge.py --model-config configs/models/qwen35_08b.yaml --experiment-config configs/experiments/pairwise.yaml --device cuda:0 --output-dir outputs/plain --local-files-only
python scripts/train_judge.py --model-config configs/models/qwen35_08b.yaml --experiment-config configs/experiments/pairwise_swap.yaml --device cuda:0 --output-dir outputs/swap --local-files-only
python scripts/train_judge.py --model-config configs/models/qwen35_08b.yaml --experiment-config configs/experiments/shared_scorer.yaml --device cuda:0 --output-dir outputs/scorer --local-files-only
```

Alternatively, use `python -m orderbias.training.train` after installing the
package. `--train-data` and `--validation-data` default to the unified directory
configured by environment variables. Both must contain nonempty HelpSteer2
examples with the matching split labels. Prompt-group overlap, identical prompt
text across splits, duplicate example IDs, and external evaluation data fail
before training. No dataset downloads happen inside the trainer.

`--overrides path.yaml` applies nested settings after the model and experiment
files. For example, a short CPU smoke run uses:

```yaml
backbone:
  compute_dtype: float32
training:
  max_steps: 2
  eval_steps: 1
  logging_steps: 1
  batch_size: 2
  gradient_accumulation_steps: 1
```

Choose `--device cpu` with that override. CPU tests use tiny locally saved actual
Qwen3.5 models, not full pretrained checkpoints. To request QLoRA, override
`quantization.mode: "4bit"` and select CUDA; bitsandbytes is required. Setting
`lora.enabled: false` trains the task head alone. The trainer supports one process
and device, rejecting `WORLD_SIZE > 1`; it does not implement DDP or resumption.

## Formatting and truncation

The versioned `judge_v1` template puts judging criteria in a system message and
the request/candidate text in a user message. Pairwise input contains both answer
slots; scorer input contains one answer without A/B identity or competitor text.
No preference labels or human justifications appear in model inputs. Plain-text
formatting is available through `backbone.use_chat_template: false`.

Fast-tokenizer offsets cap the prompt and each answer independently before
formatting. Clipping retains an original Unicode-safe text prefix and checks its
token count again. Identical candidate caps apply across slots and conditions.
Swapping happens after clipping. The complete formatted input is then tokenized
without adding duplicate special tokens and checked against `max_input_length`.
Overflow fails with a request to lower content budgets; it never truncates the
assembled sequence. Collation right-pads with an existing pad token, falling
back to an existing EOS token when needed; it never resizes the vocabulary.

Chat formatting sets `add_generation_prompt=False` and `enable_thinking=False`.
These judges pool a terminal input hidden state rather than generate an answer.
The actual Qwen3.5-0.8B tokenizer/chat template was checked at Hub commit
`2fc06364715b967f1860aea9cf38778875588b17`: a simple arithmetic pair rendered
as a system/user conversation, without a generated assistant or thinking prefix.
No model weights were needed for that check. The run stores actual formatted
input previews, template text/hash, tokenizer assets, and revision metadata.

## Training exposure and objectives

Standard pairwise training cycles shuffled canonical pairs. Augmented training
visits the same shuffled pair cycles but presents both orders adjacently, with
a seeded random choice of which orientation comes first. It flips labels as
`(A, B, y) -> (B, A, 1-y)` without changing canonical examples. The global
presentation budget must be even in this condition, so every visited pair gets
both orientations before training finishes. A tiny budget may visit only part
of the dataset; exposure files make that explicit.

Every optimizer step consumes exactly `batch_size * gradient_accumulation_steps`
preference presentations. Full microbatches can cross dataset-cycle boundaries;
there are no discarded tail samples or unequal partial-batch weights. Defaults
give 1000 optimizer steps and 16000 presentations for each condition. At that
budget, augmentation trades unique-pair exposure for a second ordering. This
controls presentations/updates, not equal compute; processed input tokens and
wall time are recorded. The scorer batches inputs as A0,B0,A1,B1,..., scores each
answer independently, and reconstructs pairwise score differences for the loss.

Pairwise models minimize two-class cross-entropy. The scorer minimizes
`-logsigmoid((score_A-score_B) * (1-2*label))`, where labels 0/1 mean A/B preferred.
Losses compute in FP32 with stable functions, without weighting by preference
strength or adding a numeric margin. Only adapters and the complete head enter
AdamW, with separate configurable learning rates. The loop supports linear,
cosine, or constant schedules with warmup, gradient accumulation, and norm
clipping. CUDA FP16 uses gradient scaling. Nonfinite losses/gradients fail.

## Validation and saved runs

Every `eval_steps` and at the final step, evaluate held-out HelpSteer2. Pairwise
validation averages NLL and accuracy over both orientations for every canonical
pair, regardless of training augmentation. The scorer evaluates canonical pairs
once because independent scoring enforces complementary probabilities. Exact
ties count as incorrect and are reported separately. Evaluation disables dropout
and restores the previous mode afterward.

The lowest **mean validation NLL** selects the best checkpoint (earliest wins an
exact tie). This criterion is fixed in the entry point. LLMBar never selects a
checkpoint. Comprehensive both-order/subset/margin evaluation remains Phase 6.

The output directory must be new or empty. It contains:

- `requested_config.json`, `backbone_report.json`, saved `tokenizer/`, and
  `tokenizer_manifest.json` with revision, versions, template text/hash, and flags.
- `data_manifest.json`: counts, hashes of ordered canonical examples, and static
  truncation counts by canonical answer and reversed presentation slot.
- `input_preview.json`: the first canonical input's exact token IDs and decoded
  model input, with its label stored separately for review.
- `metrics.jsonl`: every optimizer step's loss, learning rates, cumulative
  presentations/tokens, and validation metrics when computed.
- `checkpoints/step_XXXXXX/`: each validation improvement, including compact judge
  tensors and tokenizer assets. `summary.json` identifies the best checkpoint.
- `final/`: final-step judge plus tokenizer assets, saved even if a previous
  checkpoint achieved better validation loss.
- `exposure.json` and `summary.json`: per-pair orientation exposure, total
  processed tokens, actual training truncations, parameter counts, training-loop
  elapsed time, and peak CUDA allocation when applicable.

Judge checkpoint configs pin the resolved backbone revision. Local backbone
directories must retain their original contents. Load a selected judge with
`load_judge_checkpoint`, its tokenizer with
`AutoTokenizer.from_pretrained(checkpoint / "tokenizer")`, and use `JudgeEncoder`
with the restored judge's config. Compact checkpoints omit optimizer/scheduler
and RNG state; they support inference or starting fresh optimization, not exact
resumption of this run.
