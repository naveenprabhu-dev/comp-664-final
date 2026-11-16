"""Concrete pairwise and independent-scoring judge forward passes.

Import this module only in the training environment. Inputs are already
tokenized TokenBatch objects; templates, augmentation, and losses live outside
the judges. Score both candidates with one SharedScorer instance in eval mode.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .backbone import LoadedBackbone, load_backbone
from .config import ExperimentConfig
from .heads import ClassificationHead, ScoringHead, last_token_indices
from .interfaces import PairwiseOutput, ScorerOutput, TokenBatch


class _Judge(nn.Module):
    def __init__(self, loaded: LoadedBackbone, config: ExperimentConfig) -> None:
        super().__init__()
        for section in ("backbone", "lora", "quantization"):
            if getattr(loaded.config, section) != getattr(config, section):
                raise ValueError(f"Prepared backbone and judge {section} configuration disagree")
        self.config = config
        self.backbone = loaded.model
        self.backbone_report = loaded.report
        self._adapter_names = frozenset(loaded.report.trainable_parameter_names)
        self._validate_backbone()

    def _validate_backbone(self) -> None:
        actual = {name for name, p in self.backbone.named_parameters() if p.requires_grad}
        if actual != self._adapter_names or any("lora_" not in name for name in actual):
            raise ValueError("Backbone trainables must match the prepared adapter report")
        if bool(self.backbone_report.adapters) != self.config.lora.enabled:
            raise ValueError("Backbone adapter mode does not match the judge configuration")
        if self.backbone_report.quantization != self.config.quantization.mode:
            raise ValueError("Backbone quantization does not match the judge configuration")

    def validate_trainable_parameters(self) -> None:
        """Check that original weights are frozen and every head parameter trains."""
        self._validate_backbone()
        if not all(p.requires_grad for p in self.head.parameters()):
            raise ValueError("All task-head parameters must be trainable")
        expected = {f"backbone.{name}" for name in self._adapter_names}
        expected.update(f"head.{name}" for name, _ in self.head.named_parameters())
        actual = {name for name, p in self.named_parameters() if p.requires_grad}
        if actual != expected:
            raise ValueError("Unexpected trainable judge parameters")

    def _pool(self, inputs: TokenBatch[Tensor]) -> Tensor:
        ids, mask = inputs.input_ids, inputs.attention_mask
        if ids.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("input_ids and attention_mask must have matching shape [batch, length]")
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer token IDs")
        if ids.device != mask.device or ids.device != self.head.weight.device:
            raise ValueError("Inputs, attention_mask, and judge must be on the same device")
        if ids.shape[1] > self.config.backbone.max_input_length:
            raise ValueError("Input exceeds configured max_input_length; truncate before formatting")
        indices = last_token_indices(mask)  # Validate before the expensive backbone call
        # Padding must not change a candidate's text positions. Qwen accepts
        # these 2-D text positions and expands its multimodal RoPE internally.
        position_ids = mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(~mask.bool(), 0)
        hidden = self.backbone(
            input_ids=ids, attention_mask=mask, position_ids=position_ids,
            use_cache=False, return_dict=True,
        ).last_hidden_state
        if hidden.shape != (*ids.shape, self.backbone_report.hidden_size):
            raise RuntimeError("Backbone returned an unexpected hidden-state shape")
        return hidden[torch.arange(ids.shape[0], device=ids.device), indices]

    def parameter_summary(self) -> dict:
        self.validate_trainable_parameters()
        head_count = sum(p.numel() for p in self.head.parameters())
        adapter_count = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        return {
            "head_parameters": head_count,
            "adapter_parameters": adapter_count,
            "trainable_parameters": head_count + adapter_count,
            "total_parameters": self.backbone_report.total_parameters + head_count,
            "trainable_parameter_names": [name for name, p in self.named_parameters() if p.requires_grad],
        }


class PairwiseJudge(_Judge):
    """Prompt + both candidates -> logits [B, 2]; no generation or loss."""

    def __init__(self, loaded: LoadedBackbone, config: ExperimentConfig) -> None:
        if config.judge.kind != "pairwise":
            raise ValueError("PairwiseJudge requires judge.kind='pairwise'")
        super().__init__(loaded, config)
        self.head = ClassificationHead(
            loaded.report.hidden_size, config.judge, seed=config.training.seed,
            device=next(self.backbone.parameters()).device,
        )
        self.validate_trainable_parameters()
        self.train()

    def forward(self, inputs: TokenBatch[Tensor]) -> PairwiseOutput[Tensor]:
        return PairwiseOutput(logits=self.head(self._pool(inputs)))


class SharedScorer(_Judge):
    """Prompt + one candidate -> scores [B]; no candidate-slot input."""

    def __init__(self, loaded: LoadedBackbone, config: ExperimentConfig) -> None:
        if config.judge.kind != "shared_scorer":
            raise ValueError("SharedScorer requires judge.kind='shared_scorer'")
        super().__init__(loaded, config)
        self.head = ScoringHead(
            loaded.report.hidden_size, config.judge, seed=config.training.seed,
            device=next(self.backbone.parameters()).device,
        )
        self.validate_trainable_parameters()
        self.train()

    def forward(self, inputs: TokenBatch[Tensor]) -> ScorerOutput[Tensor]:
        return ScorerOutput(scores=self.head(self._pool(inputs)))


def build_judge(loaded: LoadedBackbone, config: ExperimentConfig) -> PairwiseJudge | SharedScorer:
    """Wrap an already prepared backbone; useful for offline tiny-model checks."""
    cls = PairwiseJudge if config.judge.kind == "pairwise" else SharedScorer
    return cls(loaded, config)


def load_judge(
    config: ExperimentConfig, *, device: str = "cpu",
    cache_dir: str | Path | None = None, local_files_only: bool = False,
) -> PairwiseJudge | SharedScorer:
    """Load a frozen/adapted backbone and initialize its fully trainable head.

    Returns a model in training mode; call eval() for deterministic evaluation.
    Restoring trained parameters is handled by checkpoints.load_judge_checkpoint.
    """
    loaded = load_backbone(config, device=device, cache_dir=cache_dir, local_files_only=local_files_only)
    return build_judge(loaded, config)
