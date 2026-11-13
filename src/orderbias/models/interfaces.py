"""Backend-independent judge contracts. No concrete models or losses yet.

The tensor type is generic so these protocols can later be specialized to
torch.Tensor without importing torch during configuration or data preparation.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

Tensor = TypeVar("Tensor")


@dataclass(frozen=True)
class TokenBatch(Generic[Tensor]):
    """Tokenized complete inputs with IDs and 0/1 masks of shape [batch, length].

    Masks identify valid tokens regardless of padding side. Each row must have
    at least one valid token. Pooling selects its last valid token. A candidate
    pair occupies one row for a pairwise judge; one candidate occupies one row
    for a scorer. Tokenization validates budgets before these inputs are built.
    """

    input_ids: Tensor
    attention_mask: Tensor


@dataclass(frozen=True)
class PairwiseOutput(Generic[Tensor]):
    """Unnormalized logits [batch, 2]: column 0 = A, column 1 = B preferred.

    A/B refer to presentation slots, not canonical dataset identity. Training
    supplies labels separately; the forward pass does not calculate a loss.
    Equal logits represent an abstention at evaluation, not an A-slot win.
    """

    logits: Tensor


@dataclass(frozen=True)
class ScorerOutput(Generic[Tensor]):
    """Unbounded relative utility scores [batch], one per prompt/answer input.

    Compare two calls to the same scorer via score_A - score_B. A positive
    difference favors A (label 0); zero is an abstention. The scalar head's
    trailing singleton dimension is removed by the future implementation.
    """

    scores: Tensor


class PairwiseJudge(Protocol[Tensor]):
    """Consumes prompt + A + B; standard and augmented conditions share this API."""

    def __call__(self, inputs: TokenBatch[Tensor]) -> PairwiseOutput[Tensor]: ...


class SharedScorer(Protocol[Tensor]):
    """Consumes prompt + one answer, without slot identity or competitor context.

    Both answers use this same instance and identical independent preprocessing.
    Swap consistency assumes deterministic eval behavior. It is not guaranteed
    by these type contracts alone; later implementation checks must enforce it.
    """

    def __call__(self, inputs: TokenBatch[Tensor]) -> ScorerOutput[Tensor]: ...
