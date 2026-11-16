"""Single-device fixed-budget training for pairwise judges and shared scorers.

Run `python -m orderbias.training.train --help` after installing the package,
or use scripts/train_judge.py from the repository root. No distributed training
or exact optimizer-state resumption is implemented in this phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from orderbias.models.checkpoints import save_judge_checkpoint
from orderbias.models.config import load_config
from orderbias.models.judges import PairwiseJudge, load_judge
from .data import (
    JudgeEncoder, collate_views, describe_tokenizer, load_tokenizer, presentation_stream, read_examples,
    truncation_summary, validate_training_splits,
)
from .losses import classification_loss, ranking_loss


def _forward(judge, views, tokenizer, device, *, reduction="mean"):
    inputs, labels = collate_views(views, tokenizer.pad_token_id, device)
    if isinstance(judge, PairwiseJudge):
        logits = judge(inputs).logits
        loss = classification_loss(logits, labels, reduction=reduction)
        gap = logits[:, 0] - logits[:, 1]
    else:
        scores = judge(inputs).scores.reshape(len(views), 2)
        loss = ranking_loss(scores[:, 0], scores[:, 1], labels, reduction=reduction)
        gap = scores[:, 0] - scores[:, 1]
    predicted = torch.where(gap > 0, 0, torch.where(gap < 0, 1, -1))
    return loss, predicted, labels, int(inputs.attention_mask.sum())


def evaluate_validation(judge, rows, tokenizer) -> dict:
    """Mean NLL selects checkpoints; pairwise validation checks both orders.

    Shared scores are evaluated once per canonical pair because their probability
    complement is structural. Ties count as incorrect and are also reported.
    """
    mode = judge.training
    judge.eval()
    views = [view for row in rows for view in (row.canonical, row.reversed) if view is not None]
    count = correct = ties = tokens = 0
    total_loss = 0.0
    device = judge.head.weight.device
    try:
        with torch.no_grad():
            for start in range(0, len(views), judge.config.training.batch_size):
                chunk = views[start:start + judge.config.training.batch_size]
                loss, predicted, labels, used = _forward(judge, chunk, tokenizer, device, reduction="sum")
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss")
                total_loss += float(loss)
                count += len(chunk)
                correct += int((predicted == labels).sum())
                ties += int((predicted == -1).sum())
                tokens += used
    finally:
        judge.train(mode)
    if not count:
        raise ValueError("Validation requires nonempty encoded examples")
    return {"loss": total_loss / count, "accuracy": correct / count, "ties": ties,
            "pairs": len(rows), "presentations": count, "input_tokens": tokens}


def build_optimizer(judge):
    judge.validate_trainable_parameters()
    settings = judge.config.training
    groups = [{"params": list(judge.head.parameters()), "lr": settings.head_learning_rate, "name": "head"}]
    adapters = [p for p in judge.backbone.parameters() if p.requires_grad]
    if adapters:
        groups.append({"params": adapters, "lr": settings.adapter_learning_rate, "name": "adapters"})
    return torch.optim.AdamW(groups, weight_decay=settings.weight_decay)


def build_scheduler(optimizer, settings):
    warmup = int(settings.max_steps * settings.warmup_ratio)

    def factor(step):
        if warmup and step < warmup:
            return step / warmup
        if settings.scheduler == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / (settings.max_steps - warmup)))
        return 1.0 - progress if settings.scheduler == "linear" else 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _fingerprint(examples):
    digest = hashlib.sha256()
    for example in examples:
        digest.update((json.dumps(example.to_dict(), sort_keys=True, ensure_ascii=False) + "\n").encode())
    return digest.hexdigest()


def train_judge(judge, tokenizer, train_examples, validation_examples, output_dir,
                *, tokenizer_manifest: dict | None = None) -> dict:
    """Train an already initialized judge; injectable for offline integration.

    Microbatches are always full, cycling shuffled canonical pairs. The overall
    budget is max_steps * accumulation * batch_size preference presentations.
    Both-order presentations remain adjacent, even across batch boundaries.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This training entry point supports one process/device only")
    validate_training_splits(train_examples, validation_examples)
    judge.validate_trainable_parameters()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Training output directory must be new or empty: {output_dir}")
    config = judge.config
    settings = config.training
    torch.manual_seed(settings.seed)
    encoder = JudgeEncoder(tokenizer, config)
    both = config.data.augmentation == "both_orders"
    train_rows = [encoder.encode(ex, include_reversed=both) for ex in train_examples]
    val_rows = [encoder.encode(ex, include_reversed=config.judge.kind == "pairwise") for ex in validation_examples]
    vocab_size = judge.backbone.get_input_embeddings().num_embeddings
    for row in [*train_rows, *val_rows]:
        for view in (row.canonical, row.reversed):
            if view is not None and any(token < 0 or token >= vocab_size for seq in view.sequences for token in seq):
                raise ValueError("Tokenizer IDs exceed the frozen backbone vocabulary; use its matching tokenizer")
    if not 0 <= tokenizer.pad_token_id < vocab_size:
        raise ValueError("Padding ID is outside the frozen backbone vocabulary")

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(output_dir / "requested_config.json")
    judge.backbone_report.save_json(output_dir / "backbone_report.json")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    manifest = {**describe_tokenizer(tokenizer, config), **(tokenizer_manifest or {})}
    manifest["saved_assets"] = "tokenizer/"
    (output_dir / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    preprocessing = {
        "train": truncation_summary(train_rows), "validation": truncation_summary(val_rows),
        "train_sha256": _fingerprint(train_examples), "validation_sha256": _fingerprint(validation_examples),
        "training_reversed_truncated_a": sum(row.truncated[2] for row in train_rows) if both else 0,
        "training_reversed_truncated_b": sum(row.truncated[1] for row in train_rows) if both else 0,
    }
    (output_dir / "data_manifest.json").write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    # Decode one actual model input for review; retain exact IDs as well.
    preview = {"example_id": train_rows[0].example_id, "label": train_rows[0].canonical.label,
               "input_ids": train_rows[0].canonical.sequences,
               "formatted_inputs": [tokenizer.decode(ids, skip_special_tokens=False)
                                    for ids in train_rows[0].canonical.sequences]}
    (output_dir / "input_preview.json").write_text(json.dumps(preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    device = judge.head.weight.device
    optimizer = build_optimizer(judge)
    scheduler = build_scheduler(optimizer, settings)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.backbone.compute_dtype == "float16")
    stream = presentation_stream(train_rows, both_orders=both, seed=settings.seed)
    counts = [[0, 0] for _ in train_rows]
    presentations = tokens = 0
    best_loss = float("inf")
    best_path = None
    best_step = 0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    judge.train()

    def save_checkpoint(name):
        path = output_dir / name
        save_judge_checkpoint(judge, path)
        # Each selected checkpoint has the preprocessing assets needed to score
        # raw text later, not only the tensor forward-pass weights.
        tokenizer.save_pretrained(path / "tokenizer")
        (path / "tokenizer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return name

    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, settings.max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
            for _ in range(settings.gradient_accumulation_steps):
                selected = [next(stream) for _ in range(settings.batch_size)]
                views = [view for _, view in selected]
                loss, _, _, used = _forward(judge, views, tokenizer, device)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss / settings.gradient_accumulation_steps).backward()
                step_loss += float(loss.detach()) / settings.gradient_accumulation_steps
                tokens += used
                presentations += len(views)
                for index, view in selected:
                    counts[index][int(view.swapped)] += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in judge.parameters() if p.requires_grad], settings.max_grad_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            record = {"step": step, "train_loss": step_loss, "learning_rates": lrs,
                      "training_presentations": presentations, "training_input_tokens": tokens}
            if step % settings.eval_steps == 0 or step == settings.max_steps:
                metrics = evaluate_validation(judge, val_rows, tokenizer)
                record["validation"] = metrics
                if metrics["loss"] < best_loss:
                    best_loss, best_step = metrics["loss"], step
                    best_path = save_checkpoint(f"checkpoints/step_{step:06d}")
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % settings.logging_steps == 0 or "validation" in record:
                print(json.dumps(record), flush=True)
        final_path = save_checkpoint("final")
    summary = {
        "status": "complete", "optimizer_steps": settings.max_steps, "world_size": 1,
        "training_presentations": presentations, "training_input_tokens": tokens,
        "canonical_presentations": sum(c[0] for c in counts),
        "reversed_presentations": sum(c[1] for c in counts),
        "unique_training_pairs_seen": sum(sum(c) > 0 for c in counts),
        "training_truncations": {
            "prompt": sum(sum(c) * row.truncated[0] for row, c in zip(train_rows, counts)),
            "candidate_a": sum(c[0] * row.truncated[1] + c[1] * row.truncated[2] for row, c in zip(train_rows, counts)),
            "candidate_b": sum(c[0] * row.truncated[2] + c[1] * row.truncated[1] for row, c in zip(train_rows, counts)),
        },
        "selection_metric": "validation_loss", "best_validation_loss": best_loss,
        "best_step": best_step, "best_checkpoint": best_path, "final_checkpoint": final_path,
        "training_elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "parameter_counts": {key: value for key, value in judge.parameter_summary().items()
                             if key != "trainable_parameter_names"},
    }
    (output_dir / "exposure.json").write_text(json.dumps({row.example_id: {"canonical": c[0], "reversed": c[1]}
                                                        for row, c in zip(train_rows, counts)}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    from orderbias.config import UNIFIED_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/models/qwen35_08b.yaml")
    parser.add_argument("--experiment-config", default="configs/experiments/pairwise.yaml")
    parser.add_argument("--overrides", type=Path, help="YAML/JSON nested override mapping")
    parser.add_argument("--train-data", type=Path, default=UNIFIED_DIR / "train.jsonl")
    parser.add_argument("--validation-data", type=Path, default=UNIFIED_DIR / "validation.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    overrides = None
    if args.overrides:
        import yaml
        overrides = yaml.safe_load(args.overrides.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("Override file must contain a mapping")
    config = load_config(args.model_config, args.experiment_config, overrides=overrides)
    train_examples, val_examples = read_examples(args.train_data), read_examples(args.validation_data)
    validate_training_splits(train_examples, val_examples)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Distributed training is not implemented")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Training output directory must be new or empty")
    torch.manual_seed(config.training.seed)
    judge = load_judge(config, device=args.device, local_files_only=args.local_files_only)
    tokenizer, manifest = load_tokenizer(config, backbone_revision=judge.backbone_report.resolved_revision,
                                         local_files_only=args.local_files_only)
    summary = train_judge(judge, tokenizer, train_examples, val_examples, args.output_dir,
                          tokenizer_manifest=manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
