# comp-664-final — Answer-Order Bias in Learned Response Evaluators

Can a learned pairwise judge stay **consistent when the two candidate answers
are swapped** without losing **agreement with human preferences**? We compare
three judges built from the same base model (Qwen3.5-2B / 0.8B variant):

1. **Standard pairwise judge** — one ordering per pair.
2. **Swap-augmented pairwise judge** — both orders, label flipped (fix bias via
   *data augmentation*).
3. **Shared scorer** — scores each answer independently and compares; order
   invariance is enforced *architecturally*.

Training: **HelpSteer2** preference annotations. External eval: **LLMBar**
(Natural + Adversarial).

## Repo layout
```
src/orderbias/          Portable core (no cluster paths; lightweight data/config imports)
  config.py             Paths resolved from env vars
  data/
    schema.py           UnifiedExample (shared interface)
    adapters.py         adapt_helpsteer2(), adapt_llmbar()  (minimal adapters)
    loaders.py          load_helpsteer2(), load_llmbar(), load_unified()
    splits.py           prompt-group integrity / split helpers
  models/               Validated experiment config and judge protocols (no torch)
configs/                Model and experiment YAML files (see configs/README.md)
scripts/                build_unified_dataset.py  (run anywhere)
cluster/                ISOLATED Longleaf layer: SLURM, modules, conda env
tests/                  offline adapter tests
requirements.txt        portable, data-only deps
cluster/env/            heavy CUDA training deps (separate on purpose)
```

## Unified data interface (TODO #2)
Every downstream component consumes `UnifiedExample`:

| field | meaning |
|---|---|
| `prompt`, `response_a`, `response_b` | the pair (fixed canonical order) |
| `label` | `0` → A preferred, `1` → B preferred |
| `margin` | HS2 preference strength 1–3 (None for LLMBar) |
| `source` | `helpsteer2` / `llmbar` |
| `split` | `train` / `validation` (HS2) · `eval` (LLMBar) |
| `subset` | `preference`, or `natural` / `adversarial_*` |

- HelpSteer2: `preference_strength` sign sets `label` (>0 → B, <0 → A); ties
  (`0`) are dropped. Original train/val designation is honored; all pairs for a
  prompt stay in one split.
- LLMBar: `label` 1→A, 2→B; everything tagged `eval`.

## Quickstart (local)
```bash
python -m pip install -r requirements.txt
python -m pytest tests/ -q               # offline adapter tests
python scripts/build_unified_dataset.py  # writes data/unified/{train,validation,eval}.jsonl
```

## On Longleaf
See `cluster/README.md`. In short: `source cluster/modules.sh` to set
cluster paths + modules, then run the same scripts.
