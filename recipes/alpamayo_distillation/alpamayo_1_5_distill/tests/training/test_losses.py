# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.nn.functional as F
from alpamayo1_5_distill.training.losses import DistillationLoss, DistillationLossConfig
from alpamayo1_5_distill.training.on_policy import OnPolicyDistillConfig


def test_masked_cross_entropy_uses_only_selected_tokens():
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]], requires_grad=True)
    labels = torch.tensor([[0, 0, 1]])
    mask = torch.tensor([[True, False, True]])
    objective = DistillationLoss(
        DistillationLossConfig(ce_weight=1.0, kd_weight=0.0, temperature=1.0)
    )

    losses = objective({"logits": logits}, {"labels": labels, "loss_mask": mask})
    expected = F.cross_entropy(logits[:, 0], labels[:, 1])

    torch.testing.assert_close(losses["ce_loss"], expected)
    torch.testing.assert_close(losses["loss"], expected)


def test_language_losses_filter_masked_tokens_before_vocabulary_ops():
    student = torch.tensor(
        [[[2.0, 0.0], [float("nan"), float("nan")], [0.0, 2.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[1.0, 0.0], [float("nan"), float("nan")], [0.0, 1.0]]],
        dtype=torch.bfloat16,
    )
    batch = {
        "labels": torch.tensor([[0, 1, -100]]),
        "loss_mask": torch.tensor([[True, False, False]]),
        "teacher_logits": teacher,
    }
    objective = DistillationLoss(
        DistillationLossConfig(ce_weight=1.0, kd_weight=1.0, temperature=1.0)
    )

    losses = objective({"logits": student}, batch)

    assert losses["ce_loss"].isfinite()
    assert losses["kd_loss"].isfinite()
    losses["loss"].backward()
    assert student.grad is not None
    assert student.grad[:, 0].isfinite().all()


def test_precomputed_language_terms_are_reused_without_logits():
    objective = DistillationLoss(DistillationLossConfig(ce_weight=2.0, kd_weight=3.0))
    ce = torch.tensor(1.5, requires_grad=True)
    kd = torch.tensor(2.0, requires_grad=True)

    losses = objective({"ce_loss": ce, "kd_loss": kd})

    torch.testing.assert_close(losses["loss"], torch.tensor(9.0))
    losses["loss"].backward()
    torch.testing.assert_close(ce.grad, torch.tensor(2.0))
    torch.testing.assert_close(kd.grad, torch.tensor(3.0))


def test_dense_and_sparse_kd_are_finite_and_backpropagate():
    student = torch.randn(2, 3, 7, requires_grad=True)
    teacher = torch.randn(2, 3, 7)
    config = DistillationLossConfig(ce_weight=0.0, kd_weight=1.0, temperature=2.0)
    dense = DistillationLoss(config)(
        {"logits": student},
        {"teacher_logits": teacher, "loss_mask": torch.ones(2, 3, dtype=torch.bool)},
    )["loss"]
    topk_logits, topk_indices = teacher.topk(3, dim=-1)
    sparse = DistillationLoss(config)(
        {"logits": student},
        {
            "teacher_topk_logits": topk_logits,
            "teacher_topk_indices": topk_indices,
            "teacher_residual_mass": torch.full((2, 3), 0.2),
        },
    )["loss"]

    assert dense.isfinite()
    assert sparse.isfinite()
    (dense + sparse).backward()
    assert student.grad is not None
    assert student.grad.isfinite().all()


def test_kd_can_optionally_include_prompt_positions() -> None:
    student = torch.zeros(1, 4, 2)
    teacher = torch.tensor([[[4.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    batch = {
        "teacher_logits": teacher,
        "loss_mask": torch.tensor([[False, True, False, False]]),
        "prompt_mask": torch.tensor([[True, True, False, False]]),
    }

    reasoning_only = DistillationLoss(
        DistillationLossConfig(
            ce_weight=0.0,
            kd_weight=1.0,
            temperature=1.0,
            kd_include_prompt=False,
        )
    )({"logits": student}, batch)["kd_loss"]
    prompt_and_reasoning = DistillationLoss(
        DistillationLossConfig(
            ce_weight=0.0,
            kd_weight=1.0,
            temperature=1.0,
            kd_include_prompt=True,
        )
    )({"logits": student}, batch)["kd_loss"]

    teacher_probs = F.softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    per_token = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(dim=-1)
    torch.testing.assert_close(reasoning_only, per_token[:, 1].mean())
    torch.testing.assert_close(
        prompt_and_reasoning,
        per_token[:, :2].mean(),
    )


def test_ce_only_batch_skips_kd_prompt_validation() -> None:
    student = torch.randn(1, 4, 3, requires_grad=True)
    batch = {
        "labels": torch.tensor([[-100, -100, 1, 2]]),
        "loss_mask": torch.tensor([[False, True, True, False]]),
    }
    objective = DistillationLoss(
        DistillationLossConfig(
            ce_weight=1.0,
            kd_weight=1.0,
            temperature=1.0,
            kd_include_prompt=True,
        )
    )

    language = objective.language_terms({"student_logits": student}, batch)

    assert set(language) == {"ce_loss"}
    assert language["ce_loss"].isfinite()
    with pytest.raises(ValueError, match="kd_include_prompt requires prompt_mask"):
        objective.language_terms(
            {"student_logits": student},
            {**batch, "teacher_logits": torch.randn_like(student)},
        )


def test_projected_hidden_flow_and_teacher_vector_losses():
    outputs = {
        "student_hidden": torch.randn(2, 4, 3, requires_grad=True),
        "student_vector": torch.zeros(2, 4, 2, requires_grad=True),
    }
    batch = {
        "teacher_hidden": torch.randn(2, 4, 6),
        "hidden_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
        "flow_target": torch.ones(2, 4, 2),
        "flow_mask": torch.ones(2, 4, dtype=torch.bool),
        "teacher_vector": torch.ones(2, 4, 2),
    }
    objective = DistillationLoss(
        DistillationLossConfig(
            ce_weight=0.0,
            kd_weight=0.0,
            hidden_weight=0.5,
            flow_weight=2.0,
            teacher_vector_weight=3.0,
            student_hidden_dim=3,
            teacher_hidden_dim=6,
        )
    )

    losses = objective(outputs, batch)

    assert set(losses) == {"loss", "hidden_loss", "flow_loss", "teacher_vector_loss"}
    assert objective.student_projection.out_features == 6
    torch.testing.assert_close(losses["flow_loss"], torch.tensor(1.0))
    torch.testing.assert_close(losses["teacher_vector_loss"], torch.tensor(1.0))
    losses["loss"].backward()
    assert objective.student_projection.weight.grad is not None


def test_hidden_loss_sums_normalized_feature_distance_like_reference() -> None:
    objective = DistillationLoss(
        DistillationLossConfig(
            ce_weight=0.0,
            kd_weight=0.0,
            hidden_weight=1.0,
            student_hidden_dim=2,
            teacher_hidden_dim=2,
        )
    )
    outputs = {
        "student_hidden": torch.tensor([[[1.0, 0.0]]], requires_grad=True),
    }
    batch = {
        "teacher_hidden": torch.tensor([[[0.0, 1.0]]]),
        "hidden_mask": torch.tensor([[True]]),
    }

    losses = objective(outputs, batch)

    torch.testing.assert_close(losses["hidden_loss"], torch.tensor(2.0))
    losses["loss"].backward()
    assert outputs["student_hidden"].grad is not None


def test_hidden_kd_prompt_range_follows_hidden_include_prompt() -> None:
    outputs = {
        "student_hidden": torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
    }
    batch = {
        "teacher_hidden": torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]),
        "hidden_mask": torch.tensor([[False, False, True]]),
        "prompt_mask": torch.tensor([[True, True, False]]),
    }

    reasoning_only = DistillationLoss(
        DistillationLossConfig(
            ce_weight=0.0,
            kd_weight=0.0,
            hidden_weight=1.0,
            student_hidden_dim=2,
            teacher_hidden_dim=2,
            kd_include_prompt=True,
            hidden_include_prompt=False,
        )
    )(outputs, batch)["hidden_loss"]
    prompt_and_reasoning = DistillationLoss(
        DistillationLossConfig(
            ce_weight=0.0,
            kd_weight=0.0,
            hidden_weight=1.0,
            student_hidden_dim=2,
            teacher_hidden_dim=2,
            kd_include_prompt=False,
            hidden_include_prompt=True,
        )
    )(outputs, batch)["hidden_loss"]

    torch.testing.assert_close(reasoning_only, torch.tensor(0.0))
    torch.testing.assert_close(prompt_and_reasoning, torch.tensor(4.0 / 3.0))


def test_future_trajectory_vlm_loss_is_weighted() -> None:
    auxiliary = torch.tensor(2.5, requires_grad=True)
    objective = DistillationLoss(
        DistillationLossConfig(
            ce_weight=0.0,
            kd_weight=0.0,
            future_traj_vlm_weight=0.4,
        )
    )

    losses = objective({"future_traj_vlm_loss": auxiliary})

    torch.testing.assert_close(losses["future_traj_vlm_loss"], auxiliary)
    torch.testing.assert_close(losses["loss"], torch.tensor(1.0))


def test_on_policy_language_terms_disable_ce_and_off_policy_kd() -> None:
    student = torch.zeros(1, 4, 3, requires_grad=True)
    teacher = torch.zeros_like(student)
    teacher[:, 1, 0] = 6.0
    teacher[:, 2, 1] = 6.0
    batch = {
        "on_policy": True,
        "labels": torch.tensor([[-100, 0, 1, 2]]),
        "loss_mask": torch.ones(1, 4, dtype=torch.bool),
        "on_policy_kl_mask": torch.tensor([[False, False, True, False]]),
        "teacher_logits": teacher,
    }
    objective = DistillationLoss(
        DistillationLossConfig(ce_weight=3.0, kd_weight=4.0),
        OnPolicyDistillConfig(kl_weight=2.0),
    )

    language = objective.language_terms({"student_logits": student}, batch)
    losses = objective(language, batch)

    torch.testing.assert_close(language["ce_loss"], torch.tensor(0.0))
    torch.testing.assert_close(language["kd_loss"], torch.tensor(0.0))
    assert language["on_policy_kl_loss"] > 0
    torch.testing.assert_close(
        losses["loss"],
        2.0 * language["on_policy_kl_loss"],
    )
    losses["loss"].backward()
    assert student.grad is not None
