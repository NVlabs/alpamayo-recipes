"""Token-level divergences between a student and a teacher next-token distribution.

Two layers:

* :func:`token_divergence` works on flat ``(N, V)`` logits — one row per supervised
  token — and implements the actual divergence math.
* :func:`sequence_distillation_loss` handles the causal shift and masking for
  ``(B, S, V)`` sequence logits and delegates to the former.

The split keeps the math testable without constructing sequences, and lets the
off-policy and on-policy paths share one implementation.

Memory note: ``V`` is ~155k here, so every ``(N, V)`` intermediate is large — at
N=3000 that is ~0.9 GB in bf16 *per tensor*. Logits are therefore left in their
incoming dtype (bf16 has fp32's exponent range, so log_softmax is stable) and the
modes are ordered below by how many such intermediates they need:
``forward_kl`` (2) < ``reverse_kl`` (4) < ``jsd`` (6). Prefer restricting the mask
to the positions you actually want supervised rather than upcasting or chunking.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FORWARD_KL = "forward_kl"
REVERSE_KL = "reverse_kl"
JSD = "jsd"
DIVERGENCE_MODE_NAMES = (FORWARD_KL, REVERSE_KL, JSD)


def token_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    mode: str = FORWARD_KL,
    beta: float = 0.5,
) -> torch.Tensor:
    """Mean per-token divergence between two distributions over the vocabulary.

    Args:
        student_logits: ``(N, V)`` student logits. Gradients flow through these.
        teacher_logits: ``(N, V)`` teacher logits. Treated as constants.
        temperature: Softmax temperature applied to both sides. Changes target
            softness only — no Hinton ``T**2`` rescaling — so magnitude stays under
            the caller's control via an explicit loss weight.
        mode: One of

            * ``forward_kl`` — ``KL(teacher || student)``, mode-covering. The classic
              distillation objective and what the off-policy path uses.
            * ``reverse_kl`` — ``KL(student || teacher)``, mode-seeking. Pulls the
              student's mass onto the teacher's dominant mode; the natural choice when
              single-sample (greedy) quality matters more than sample coverage.
            * ``jsd`` — generalized Jensen-Shannon, interpolating between the two.

        beta: Mixture weight on the teacher for ``jsd``. ``0.5`` is standard JSD;
            ``beta -> 1`` approaches forward KL, ``beta -> 0`` reverse KL.

    Returns:
        Scalar loss, averaged over the ``N`` supervised tokens.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have identical shapes, got "
            f"{student_logits.shape} and {teacher_logits.shape}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if mode not in DIVERGENCE_MODE_NAMES:
        raise ValueError(f"unknown divergence mode {mode!r}; expected one of {DIVERGENCE_MODE_NAMES}")

    student_logp = F.log_softmax(student_logits / temperature, dim=-1)

    if mode == FORWARD_KL:
        # F.kl_div takes log-probabilities as `input` and probabilities as `target`;
        # "batchmean" divides by the row count, giving a per-token mean.
        with torch.no_grad():
            teacher_p = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(student_logp, teacher_p, reduction="batchmean")

    with torch.no_grad():
        teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)

    if mode == REVERSE_KL:
        # KL(student || teacher), written out rather than routed through F.kl_div so
        # the gradient path through the student is explicit.
        student_p = student_logp.exp()
        return (student_p * (student_logp - teacher_logp)).sum(dim=-1).mean()

    # Generalized JSD against the mixture m = beta * teacher + (1 - beta) * student,
    # built in log space so low-probability tokens don't underflow.
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {beta}")
    log_beta = float(torch.log(torch.tensor(max(beta, 1e-12))))
    log_one_minus_beta = float(torch.log(torch.tensor(max(1.0 - beta, 1e-12))))
    mixture_logp = torch.logaddexp(teacher_logp + log_beta, student_logp + log_one_minus_beta)

    teacher_term = (teacher_logp.exp() * (teacher_logp - mixture_logp)).sum(dim=-1)
    student_p = student_logp.exp()
    student_term = (student_p * (student_logp - mixture_logp)).sum(dim=-1)
    return (beta * teacher_term + (1.0 - beta) * student_term).mean()


def sequence_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    mode: str = FORWARD_KL,
    beta: float = 0.5,
    ce_weight: float = 0.0,
) -> torch.Tensor:
    """Distillation loss over a causal sequence.

    Applies the next-token shift, selects supervised positions, then calls
    :func:`token_divergence` on the surviving rows. Flattening before the softmax
    matters: materialising probabilities for unsupervised positions is the dominant
    cost at this vocabulary size.

    Args:
        student_logits: ``(B, S, V)``.
        teacher_logits: ``(B, S, V)``.
        labels: Optional ``(B, S)`` with ``-100`` marking ignored positions. When
            given, only non-ignored *target* positions are supervised.
        attention_mask: Optional ``(B, S)`` boolean/int mask over positions; entries
            that are falsy are dropped. Both this and ``labels`` are applied after the
            causal shift.
        temperature: See :func:`token_divergence`.
        mode: See :func:`token_divergence`.
        beta: See :func:`token_divergence`.
        ce_weight: When non-zero, blends in hard cross-entropy against ``labels`` as
            ``ce_weight * CE + (1 - ce_weight) * divergence``; requires ``labels``.
            The distillation path keeps this at ``0`` and tracks CE separately so the
            two terms can be weighted independently.

    Returns:
        Scalar loss. Returns a graph-connected zero when nothing is supervised, so
        DDP still sees a gradient path for every parameter.
    """
    student_shifted = student_logits[:, :-1, :]
    teacher_shifted = teacher_logits[:, :-1, :]

    if labels is not None:
        targets = labels[:, 1:]
        supervised = targets != -100
    else:
        targets = None
        supervised = torch.ones(
            student_shifted.shape[:2], device=student_shifted.device, dtype=torch.bool
        )

    if attention_mask is not None:
        supervised = supervised & (attention_mask[:, 1:] > 0)

    if not supervised.any():
        return student_shifted.sum() * 0.0

    divergence = token_divergence(
        student_shifted[supervised],
        teacher_shifted[supervised],
        temperature=temperature,
        mode=mode,
        beta=beta,
    )

    if ce_weight == 0.0:
        return divergence
    if targets is None:
        raise ValueError("ce_weight is non-zero but no labels were provided")

    cross_entropy = F.cross_entropy(
        student_shifted.reshape(-1, student_shifted.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return ce_weight * cross_entropy + (1.0 - ce_weight) * divergence
