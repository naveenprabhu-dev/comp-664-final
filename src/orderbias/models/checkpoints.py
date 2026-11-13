"""Compact judge checkpoints: trainable tensors plus frozen-backbone provenance.

This project-specific format stores LoRA matrices and the complete task head in
one safetensors file. It does not duplicate the backbone, store tokenizer assets,
or contain optimizer/scheduler/RNG state for exact training resumption.
"""

from dataclasses import replace
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import ExperimentConfig
from .judges import PairwiseJudge, SharedScorer, load_judge


def save_judge_checkpoint(judge: PairwiseJudge | SharedScorer, directory: str | Path) -> None:
    """Save to a new/empty directory; no overwrite of an existing checkpoint."""
    judge.validate_trainable_parameters()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Checkpoint directory must be empty: {directory}")
    config = judge.config
    if judge.backbone_report.resolved_revision is not None:
        config = replace(config, backbone=replace(
            config.backbone, revision=judge.backbone_report.resolved_revision,
        ))
    elif Path(config.backbone.checkpoint).is_dir():
        config = replace(config, backbone=replace(
            config.backbone, checkpoint=str(Path(config.backbone.checkpoint).resolve()),
        ))
    tensors = {
        name: p.detach().cpu().contiguous().clone()
        for name, p in judge.named_parameters() if p.requires_grad
    }
    save_file(tensors, str(directory / "trainable.safetensors"))
    config.save_json(directory / "resolved_config.json")
    judge.backbone_report.save_json(directory / "backbone_report.json")
    manifest = {
        "format_version": 1,
        "judge_kind": config.judge.kind,
        "label_mapping": {"0": "A_preferred", "1": "B_preferred"},
        "parameters": judge.parameter_summary(),
    }
    # Manifest written last marks a completed checkpoint.
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_judge_checkpoint(
    directory: str | Path, *, device: str = "cpu",
    cache_dir: str | Path | None = None, local_files_only: bool = False,
) -> PairwiseJudge | SharedScorer:
    """Recreate the referenced backbone and strictly restore adapters and head.

    Returns eval mode. Local backbone directories must still contain the same
    weights used at save time. Device can change, but architecture/precision
    configuration is retained. No arbitrary pickled Python objects are loaded.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported judge checkpoint format")
    config = ExperimentConfig.from_dict(json.loads(
        (directory / "resolved_config.json").read_text(encoding="utf-8"),
    ))
    if manifest.get("judge_kind") != config.judge.kind:
        raise ValueError("Checkpoint judge kind and configuration disagree")
    tensors = load_file(str(directory / "trainable.safetensors"), device="cpu")
    judge = load_judge(config, device=device, cache_dir=cache_dir, local_files_only=local_files_only)
    expected = {name: p for name, p in judge.named_parameters() if p.requires_grad}
    if set(tensors) != set(expected):
        raise ValueError("Checkpoint must contain exactly the expected adapter and head parameters")
    for name, parameter in expected.items():
        tensor = tensors[name]
        if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
            raise ValueError(f"Checkpoint parameter shape/dtype mismatch: {name}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(tensors[name].to(device=parameter.device))
    judge.validate_trainable_parameters()
    judge.eval()
    return judge
