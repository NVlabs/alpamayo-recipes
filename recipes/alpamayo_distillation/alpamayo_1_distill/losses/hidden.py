"""Feature-space distillation losses on VLM hidden states."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_normalized_hidden_distance(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Average squared L2 distance between normalized hidden states.

    For unit vectors, the feature-wise sum is ``2 - 2 * cosine_similarity``.
    Summing, rather than averaging, over the hidden dimension keeps the loss scale
    independent of hidden width. The final reduction is a mean over valid tokens.
    """
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError(
            "student_hidden and teacher_hidden must have identical shapes, got "
            f"{student_hidden.shape} and {teacher_hidden.shape}"
        )
    if valid_mask.shape != student_hidden.shape[:-1]:
        raise ValueError(
            "valid_mask must match the hidden-state token dimensions, got "
            f"{valid_mask.shape} for hidden states {student_hidden.shape}"
        )

    student_normalized = F.normalize(student_hidden.float(), dim=-1)
    teacher_normalized = F.normalize(teacher_hidden.float(), dim=-1)
    token_distance = (student_normalized - teacher_normalized).square().sum(dim=-1)

    mask = valid_mask.to(device=token_distance.device, dtype=token_distance.dtype)
    return (token_distance * mask).sum() / mask.sum().clamp_min(1.0)
