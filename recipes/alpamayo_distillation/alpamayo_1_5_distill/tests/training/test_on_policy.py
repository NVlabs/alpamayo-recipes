# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from alpamayo1_5_distill.training.on_policy import (
    FORWARD_KL,
    JSD,
    REVERSE_KL,
    OnPolicyDistillConfig,
    sequence_distillation_loss,
    token_divergence,
)


def test_on_policy_schedule_matches_integer_stride() -> None:
    disabled = OnPolicyDistillConfig()
    assert not disabled.enabled
    assert disabled.stride == 0
    assert not disabled.is_on_policy_step(100)

    config = OnPolicyDistillConfig(ratio=0.25, start_step=3)
    assert config.stride == 4
    assert config.effective_ratio == 0.25
    assert [step for step in range(12) if config.is_on_policy_step(step)] == [3, 7, 11]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ratio": -0.1}, "ratio"),
        ({"start_step": -1}, "start_step"),
        ({"divergence": "bad"}, "divergence"),
        ({"temperature": 0.0}, "temperature"),
        ({"sample_top_p": 1.1}, "sample_top_p"),
    ],
)
def test_on_policy_config_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        OnPolicyDistillConfig(**kwargs)


@pytest.mark.parametrize("mode", [FORWARD_KL, REVERSE_KL, JSD])
def test_token_divergence_is_zero_for_equal_logits(mode: str) -> None:
    logits = torch.randn(5, 11)
    loss = token_divergence(logits, logits, mode=mode)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)


@pytest.mark.parametrize("mode", [FORWARD_KL, REVERSE_KL, JSD])
def test_token_divergence_only_backpropagates_to_student(mode: str) -> None:
    student = torch.randn(4, 7, requires_grad=True)
    teacher = torch.randn(4, 7, requires_grad=True)
    loss = token_divergence(student, teacher, mode=mode)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_sequence_distillation_uses_causal_target_mask() -> None:
    student = torch.zeros(1, 4, 3, requires_grad=True)
    teacher = torch.zeros_like(student)
    teacher[:, 1, 0] = 8.0
    teacher[:, 2, 1] = 8.0
    target_mask = torch.tensor([[False, False, True, False]])

    masked = sequence_distillation_loss(
        student,
        teacher,
        attention_mask=target_mask,
    )
    expected = token_divergence(student[:, 1, :], teacher[:, 1, :])
    torch.testing.assert_close(masked, expected)


def test_sequence_distillation_empty_mask_returns_connected_zero() -> None:
    student = torch.randn(1, 3, 5, requires_grad=True)
    teacher = torch.randn(1, 3, 5)
    loss = sequence_distillation_loss(
        student,
        teacher,
        attention_mask=torch.zeros(1, 3, dtype=torch.bool),
    )
    loss.backward()
    assert loss.item() == 0.0
    assert student.grad is not None
