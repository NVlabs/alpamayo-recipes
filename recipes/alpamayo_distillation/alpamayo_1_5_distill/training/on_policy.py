# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduling and divergence objectives for student-sampled distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

FORWARD_KL = "forward_kl"
REVERSE_KL = "reverse_kl"
JSD = "jsd"
DIVERGENCE_MODES = (FORWARD_KL, REVERSE_KL, JSD)


@dataclass(frozen=True)
class OnPolicyDistillConfig:
    """Control when and how reasoning is sampled from the student."""

    ratio: float = 0.0
    start_step: int = 0
    divergence: str = FORWARD_KL
    beta: float = 0.5
    temperature: float = 1.0
    sample_temperature: float = 0.6
    sample_top_p: float = 0.98
    kl_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"ratio must lie in [0, 1], got {self.ratio}")
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if self.divergence not in DIVERGENCE_MODES:
            raise ValueError(
                f"divergence must be one of {DIVERGENCE_MODES}, got {self.divergence!r}"
            )
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must lie in [0, 1], got {self.beta}")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.sample_temperature <= 0.0:
            raise ValueError("sample_temperature must be positive")
        if not 0.0 < self.sample_top_p <= 1.0:
            raise ValueError("sample_top_p must lie in (0, 1]")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.ratio > 0.0

    @property
    def stride(self) -> int:
        if not self.enabled:
            return 0
        return max(1, round(1.0 / self.ratio))

    @property
    def effective_ratio(self) -> float:
        return 0.0 if not self.enabled else 1.0 / self.stride

    def is_on_policy_step(self, global_step: int) -> bool:
        if not self.enabled or global_step < self.start_step:
            return False
        return (global_step - self.start_step) % self.stride == 0

    def describe(self) -> str:
        if not self.enabled:
            return "on-policy distillation: disabled"
        beta = f" beta={self.beta}" if self.divergence == JSD else ""
        return (
            f"on-policy distillation: every {self.stride} step(s) "
            f"(effective ratio {self.effective_ratio:.3f}) from step "
            f"{self.start_step}, divergence={self.divergence}{beta}, "
            f"T={self.temperature}, sampling(temp={self.sample_temperature}, "
            f"top_p={self.sample_top_p}), kl_weight={self.kl_weight}"
        )


def token_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    mode: str = FORWARD_KL,
    beta: float = 0.5,
) -> torch.Tensor:
    """Return the mean divergence over a flattened collection of tokens."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have identical shapes, got "
            f"{student_logits.shape} and {teacher_logits.shape}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if mode not in DIVERGENCE_MODES:
        raise ValueError(f"unknown divergence mode {mode!r}; expected one of {DIVERGENCE_MODES}")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must lie in [0, 1], got {beta}")

    student_logp = F.log_softmax(student_logits / temperature, dim=-1)
    if mode == FORWARD_KL:
        with torch.no_grad():
            teacher_p = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(student_logp, teacher_p, reduction="batchmean")

    with torch.no_grad():
        teacher_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    if mode == REVERSE_KL:
        student_p = student_logp.exp()
        return (student_p * (student_logp - teacher_logp)).sum(dim=-1).mean()

    log_beta = student_logits.new_tensor(max(beta, 1e-12)).log()
    log_one_minus_beta = student_logits.new_tensor(max(1.0 - beta, 1e-12)).log()
    mixture_logp = torch.logaddexp(
        teacher_logp + log_beta,
        student_logp + log_one_minus_beta,
    )
    teacher_term = (teacher_logp.exp() * (teacher_logp - mixture_logp)).sum(dim=-1)
    student_p = student_logp.exp()
    student_term = (student_p * (student_logp - mixture_logp)).sum(dim=-1)
    return (beta * teacher_term + (1.0 - beta) * student_term).mean()


def sequence_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    mode: str = FORWARD_KL,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute a causal next-token divergence over selected target tokens."""
    if student_logits.ndim != 3 or teacher_logits.ndim != 3:
        raise ValueError("sequence logits must have shape [batch, sequence, vocabulary]")
    student_shifted = student_logits[:, :-1, :]
    teacher_shifted = teacher_logits[:, :-1, :]
    valid = torch.ones(
        student_shifted.shape[:2],
        dtype=torch.bool,
        device=student_shifted.device,
    )
    if attention_mask is not None:
        if attention_mask.shape != student_logits.shape[:2]:
            raise ValueError("attention_mask must match the logit token dimensions")
        valid &= attention_mask[:, 1:].to(device=valid.device, dtype=torch.bool)
    if not bool(valid.any()):
        return student_shifted.sum() * 0.0
    return token_divergence(
        student_shifted[valid],
        teacher_shifted[valid],
        temperature=temperature,
        mode=mode,
        beta=beta,
    )
