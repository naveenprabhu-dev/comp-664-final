"""Fully trainable linear judge heads and padding-aware terminal pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import JudgeConfig


def last_token_indices(attention_mask: Tensor) -> Tensor:
    """Find the last valid token per row for left or right contiguous padding.

    Empty rows, nonbinary masks, and internal padding gaps are rejected. An
    internal gap is not ordinary padding for Qwen's recurrent attention path.
    """
    if attention_mask.ndim != 2 or 0 in attention_mask.shape:
        raise ValueError("attention_mask must have nonempty shape [batch, length]")
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError("attention_mask must contain only 0 and 1")
    valid = attention_mask.bool()
    counts = valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every input row must contain at least one valid token")
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    first = positions.masked_fill(~valid, valid.shape[1]).amin(dim=1)
    last = positions.masked_fill(~valid, -1).amax(dim=1)
    if ((last - first + 1) != counts).any():
        raise ValueError("attention_mask must not contain internal padding gaps")
    return last


def last_non_padding(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool [B, L, H] to [B, H], preserving gradients at the selected tokens."""
    if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states [B, L, H] must match attention_mask [B, L]")
    if hidden.device != attention_mask.device:
        raise ValueError("hidden states and attention_mask must be on the same device")
    indices = last_token_indices(attention_mask)
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class _LinearHead(nn.Module):
    def __init__(
        self, hidden_size: int, output_dim: int, config: JudgeConfig,
        *, seed: int, device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        # A local CPU generator makes initialization independent of adapter RNG
        # consumption and of whether the final model resides on CPU or CUDA.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weight = torch.empty(output_dim, hidden_size, dtype=torch.float32)
        nn.init.normal_(weight, std=config.head_init_std, generator=generator)
        self.weight = nn.Parameter(weight.to(device))
        if config.head_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32, device=device))
        else:
            self.register_parameter("bias", None)

    def _project(self, pooled: Tensor) -> Tensor:
        if pooled.ndim != 2 or pooled.shape[1] != self.weight.shape[1]:
            raise ValueError("Head input must have shape [batch, hidden_size]")
        # Retain FP32 head computation even inside a mixed-precision backbone
        # autocast context. Casting is differentiable back into the adapters.
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            return F.linear(pooled.to(dtype=self.weight.dtype), self.weight, self.bias)


class ClassificationHead(_LinearHead):
    """Two unnormalized logits per input; column 0 = A preferred, 1 = B."""

    def __init__(self, hidden_size: int, config: JudgeConfig, *, seed: int, device="cpu"):
        if config.kind != "pairwise":
            raise ValueError("ClassificationHead requires judge.kind='pairwise'")
        super().__init__(hidden_size, 2, config, seed=seed, device=device)

    def forward(self, pooled: Tensor) -> Tensor:
        return self._project(pooled)


class ScoringHead(_LinearHead):
    """One unbounded scalar per input; always returns [batch], including B=1."""

    def __init__(self, hidden_size: int, config: JudgeConfig, *, seed: int, device="cpu"):
        if config.kind != "shared_scorer":
            raise ValueError("ScoringHead requires judge.kind='shared_scorer'")
        super().__init__(hidden_size, 1, config, seed=seed, device=device)

    def forward(self, pooled: Tensor) -> Tensor:
        return self._project(pooled).squeeze(-1)
