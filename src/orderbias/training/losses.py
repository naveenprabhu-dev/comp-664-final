"""Preference objectives with the shared label convention 0=A, 1=B."""

import torch
from torch import Tensor
from torch.nn import functional as F


def _labels(labels: Tensor, size: int) -> None:
    if labels.shape != (size,) or labels.dtype != torch.long or size == 0:
        raise ValueError("labels must be a nonempty int64 [batch] tensor")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must be 0 (A preferred) or 1 (B preferred)")


def classification_loss(logits: Tensor, labels: Tensor, *, reduction="mean") -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("classification logits must have shape [batch, 2]")
    _labels(labels, logits.shape[0])
    return F.cross_entropy(logits.float(), labels, reduction=reduction)


def ranking_loss(scores_a: Tensor, scores_b: Tensor, labels: Tensor, *, reduction="mean") -> Tensor:
    if scores_a.ndim != 1 or scores_a.shape != scores_b.shape:
        raise ValueError("scores_a and scores_b must have matching shape [batch]")
    _labels(labels, scores_a.shape[0])
    gap = scores_a.float() - scores_b.float()
    preferred_gap = torch.where(labels == 0, gap, -gap)
    loss = -F.logsigmoid(preferred_gap)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError("reduction must be mean, sum, or none")
    return loss.mean()
