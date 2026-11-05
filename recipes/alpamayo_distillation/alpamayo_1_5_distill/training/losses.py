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

"""Hybrid objectives for language, representation, and trajectory distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from alpamayo1_5_distill.training.on_policy import (
    OnPolicyDistillConfig,
    sequence_distillation_loss,
)


@dataclass(frozen=True)
class DistillationLossConfig:
    """Weights and dimensions for :class:`DistillationLoss`."""

    ce_weight: float = 1.0
    kd_weight: float = 1.0
    hidden_weight: float = 0.0
    flow_weight: float = 0.0
    teacher_vector_weight: float = 0.0
    future_traj_vlm_weight: float = 0.0
    temperature: float = 2.0
    kd_scale_by_temperature_squared: bool = True
    kd_include_prompt: bool = False
    hidden_include_prompt: bool = False
    ignore_index: int = -100
    student_hidden_dim: int | None = None
    teacher_hidden_dim: int | None = None


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, values.shape)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class DistillationLoss(nn.Module):
    """Compute a weighted collection of independently optional losses.

    Inputs are a model-output mapping plus an optional batch mapping. Values are
    read from outputs first and then from the batch. Language losses use
    ``logits``/``student_logits``, ``labels``, ``loss_mask``, ``teacher_logits``,
    ``teacher_topk_indices``, ``teacher_topk_logits``, and
    ``teacher_residual_mass``. Hidden-state distillation uses
    ``hidden_states``/``student_hidden``,
    ``teacher_hidden_states``/``teacher_hidden``, and ``hidden_mask``. Action
    losses compare ``student_vector`` with ``flow_target`` and
    ``teacher_vector``, optionally using ``flow_mask`` and
    ``teacher_vector_mask``. ``future_traj_vlm_loss`` is accepted as an
    already-normalized scalar produced by the student model.
    """

    def __init__(
        self,
        config: DistillationLossConfig,
        on_policy_config: OnPolicyDistillConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.on_policy_config = on_policy_config or OnPolicyDistillConfig()
        if config.hidden_weight == 0.0:
            self.student_projection: nn.Module = nn.Identity()
        else:
            if config.student_hidden_dim is None or config.teacher_hidden_dim is None:
                raise ValueError(
                    "student_hidden_dim and teacher_hidden_dim are required "
                    "when hidden_weight is nonzero"
                )
            if config.student_hidden_dim == config.teacher_hidden_dim:
                self.student_projection = nn.Identity()
            else:
                self.student_projection = nn.Linear(
                    config.student_hidden_dim,
                    config.teacher_hidden_dim,
                    bias=False,
                )
                nn.init.normal_(
                    self.student_projection.weight,
                    std=config.student_hidden_dim**-0.5,
                )

    @staticmethod
    def _get(outputs: dict[str, Any], batch: dict[str, Any], key: str) -> Any:
        return outputs[key] if key in outputs else batch.get(key)

    @classmethod
    def _get_any(cls, outputs: dict[str, Any], batch: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = cls._get(outputs, batch, key)
            if value is not None:
                return value
        return None

    def _ce(self, outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor | None:
        logits = self._get_any(outputs, batch, "logits", "student_logits")
        labels = self._get(outputs, batch, "labels")
        if logits is None or labels is None:
            return None
        if logits.shape[-2] != labels.shape[-1]:
            raise ValueError(
                f"logit sequence length {logits.shape[-2]} does not match labels {labels.shape[-1]}"
            )
        # Causal logits at position i predict the token at position i + 1.
        shifted_logits = logits[..., :-1, :]
        shifted_labels = labels[..., 1:]
        valid = shifted_labels.ne(self.config.ignore_index)
        loss_mask = self._get(outputs, batch, "loss_mask")
        if loss_mask is not None:
            if loss_mask.shape == labels.shape:
                loss_mask = loss_mask[..., :-1]
            valid = valid & loss_mask.to(device=valid.device, dtype=torch.bool)
        if not bool(valid.any()):
            return logits.sum() * 0.0

        # Select supervised positions before the vocabulary-sized operation. Model
        # logits are bf16 under autocast; keeping that dtype avoids a full
        # [batch, sequence, vocabulary] fp32 copy.
        selected_logits = shifted_logits[valid]
        selected_labels = shifted_labels[valid]
        return F.cross_entropy(
            selected_logits,
            selected_labels.long(),
            ignore_index=self.config.ignore_index,
            reduction="mean",
        )

    def _kd(self, outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor | None:
        student_logits = self._get_any(outputs, batch, "logits", "student_logits")
        if student_logits is None:
            return None
        teacher_logits = self._get(outputs, batch, "teacher_logits")
        topk_indices = self._get(outputs, batch, "teacher_topk_indices")
        topk_logits = self._get(outputs, batch, "teacher_topk_logits")
        if teacher_logits is None and topk_indices is None and topk_logits is None:
            # Some tasks, such as off-policy VQA, supervise language with CE
            # labels only. Skip KD before validating masks that are meaningful
            # only when teacher distribution targets are present.
            return None
        temperature = self.config.temperature
        cached_temperature = self._get(outputs, batch, "teacher_temperature")
        if cached_temperature is not None:
            cached_temperature = torch.as_tensor(cached_temperature).float()
            expected = torch.full_like(cached_temperature, temperature)
            if not torch.allclose(cached_temperature, expected):
                raise ValueError(
                    "loss temperature differs from the temperature used for "
                    "cached teacher residual mass"
                )
        mask = self._get(outputs, batch, "loss_mask")
        if self.config.kd_include_prompt:
            prompt_mask = self._get(outputs, batch, "prompt_mask")
            if prompt_mask is None:
                raise ValueError("kd_include_prompt requires prompt_mask in the batch")
            prompt_mask = prompt_mask.to(
                device=student_logits.device,
                dtype=torch.bool,
            )
            if prompt_mask.shape != student_logits.shape[:-1]:
                raise ValueError("prompt_mask must match the student logit token dimensions")
            # Logit position i predicts token i + 1. Shift the prompt token mask
            # back one position; loss_mask already covers the first reasoning token.
            prompt_prediction_mask = torch.zeros_like(prompt_mask)
            prompt_prediction_mask[..., :-1] = prompt_mask[..., 1:]
            mask = (
                prompt_prediction_mask if mask is None else (mask.bool() | prompt_prediction_mask)
            )
        if mask is None:
            valid = torch.ones(
                student_logits.shape[:-1],
                dtype=torch.bool,
                device=student_logits.device,
            )
        else:
            valid = mask.to(device=student_logits.device, dtype=torch.bool)
            if valid.shape != student_logits.shape[:-1]:
                raise ValueError("loss_mask must match the student logit token dimensions")
        if not bool(valid.any()):
            return student_logits.sum() * 0.0

        # Filter tokens before log-softmax so padding and ignored positions never
        # create vocabulary-sized temporaries. bf16 retains fp32's exponent range;
        # only the final vocabulary reduction is accumulated in fp32.
        selected_student_logits = student_logits[valid]
        student_log_probs = F.log_softmax(
            selected_student_logits / temperature,
            dim=-1,
        )
        if teacher_logits is not None:
            teacher_log_probs = F.log_softmax(
                teacher_logits[valid] / temperature,
                dim=-1,
            )
            teacher_probs = teacher_log_probs.exp()
            token_loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(
                dim=-1, dtype=torch.float32
            )
        else:
            if topk_indices is None or topk_logits is None:
                return None
            selected_topk_indices = topk_indices[valid].long()
            gathered_log_probs = student_log_probs.gather(
                -1,
                selected_topk_indices,
            )
            residual = self._get(outputs, batch, "teacher_residual_mass")
            topk_probs = F.softmax(topk_logits[valid] / temperature, dim=-1)
            if residual is not None:
                residual = residual[valid].float().clamp(0.0, 1.0)
                topk_probs = topk_probs * (1.0 - residual.unsqueeze(-1))
            topk_log_probs = topk_probs.clamp_min(1e-12).log()
            token_loss = (topk_probs * (topk_log_probs - gathered_log_probs)).sum(
                dim=-1, dtype=torch.float32
            )
            if residual is not None:
                student_topk_mass = (
                    gathered_log_probs.exp().sum(dim=-1, dtype=torch.float32).clamp(max=1.0)
                )
                student_residual = (1.0 - student_topk_mass).clamp_min(1e-12)
                token_loss = token_loss + residual * (
                    residual.clamp_min(1e-12).log() - student_residual.log()
                )
        loss = token_loss.mean()
        if self.config.kd_scale_by_temperature_squared:
            loss = loss * (temperature**2)
        return loss

    @staticmethod
    def _is_on_policy(batch: dict[str, Any]) -> bool:
        value = batch.get("on_policy", False)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("on_policy batch flag must be scalar")
            return bool(value.item())
        return bool(value)

    def _on_policy_kl(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        student_logits = self._get_any(outputs, batch, "logits", "student_logits")
        teacher_logits = self._get(outputs, batch, "teacher_logits")
        mask = self._get(outputs, batch, "on_policy_kl_mask")
        if student_logits is None or teacher_logits is None or mask is None:
            raise ValueError(
                "on-policy distillation requires student_logits, dense "
                "teacher_logits, and on_policy_kl_mask"
            )
        return sequence_distillation_loss(
            student_logits,
            teacher_logits,
            attention_mask=mask,
            temperature=self.on_policy_config.temperature,
            mode=self.on_policy_config.divergence,
            beta=self.on_policy_config.beta,
        )

    def language_terms(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Compute language losses early so full logits can be released."""
        if self._is_on_policy(batch):
            student_logits = self._get_any(
                outputs,
                batch,
                "logits",
                "student_logits",
            )
            if student_logits is None:
                raise ValueError("on-policy distillation requires student logits")
            zero = student_logits[..., 0].sum() * 0.0
            return {
                "ce_loss": zero,
                "kd_loss": zero,
                "on_policy_kl_loss": self._on_policy_kl(outputs, batch),
            }
        terms = {
            "ce_loss": self._ce(outputs, batch),
            "kd_loss": self._kd(outputs, batch),
        }
        return {name: value for name, value in terms.items() if value is not None}

    def _hidden(self, outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor | None:
        student = self._get_any(outputs, batch, "hidden_states", "student_hidden")
        teacher = self._get_any(outputs, batch, "teacher_hidden_states", "teacher_hidden")
        if student is None or teacher is None:
            return None
        student_parameter = next(self.student_projection.parameters(), None)
        if student_parameter is not None:
            student = student.to(dtype=student_parameter.dtype)
        student = F.normalize(self.student_projection(student).float(), dim=-1)
        teacher = F.normalize(teacher.float(), dim=-1)
        if student.shape != teacher.shape:
            raise ValueError(
                "projected student and teacher hidden states must have identical "
                f"shapes, got {student.shape} and {teacher.shape}"
            )
        # Match the reference objective: for normalized vectors this is
        # 2 - 2 * cosine_similarity, independent of hidden width.
        per_item = (student - teacher).square().sum(dim=-1)
        hidden_mask = self._get(outputs, batch, "hidden_mask")
        if self.config.hidden_include_prompt:
            prompt_mask = self._get(outputs, batch, "prompt_mask")
            if prompt_mask is None:
                raise ValueError("hidden_include_prompt requires prompt_mask in the batch")
            prompt_mask = prompt_mask.to(device=per_item.device, dtype=torch.bool)
            hidden_mask = prompt_mask if hidden_mask is None else (hidden_mask.bool() | prompt_mask)
        return _masked_mean(per_item, hidden_mask)

    def _mse(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        prediction_key: str,
        target_key: str,
        mask_key: str,
    ) -> torch.Tensor | None:
        prediction = self._get(outputs, batch, prediction_key)
        target = self._get(outputs, batch, target_key)
        if prediction is None or target is None:
            return None
        error = (prediction.float() - target.float()).square()
        return _masked_mean(error, self._get(outputs, batch, mask_key))

    def forward(
        self, outputs: dict[str, Any], batch: dict[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        batch = batch or {}
        is_on_policy = self._is_on_policy(batch)
        ce = self._get(outputs, batch, "ce_loss")
        kd = self._get(outputs, batch, "kd_loss")
        on_policy_kl = self._get(outputs, batch, "on_policy_kl_loss")
        if is_on_policy:
            student_logits = self._get_any(
                outputs,
                batch,
                "logits",
                "student_logits",
            )
            if student_logits is not None:
                zero = student_logits[..., 0].sum() * 0.0
                ce = zero if ce is None else ce
                kd = zero if kd is None else kd
            if on_policy_kl is None:
                on_policy_kl = self._on_policy_kl(outputs, batch)
        terms = {
            "ce": ce if ce is not None else self._ce(outputs, batch),
            "kd": kd if kd is not None else self._kd(outputs, batch),
            "on_policy_kl": on_policy_kl,
            "hidden": self._hidden(outputs, batch),
            "flow": self._mse(
                outputs,
                batch,
                "student_vector",
                "flow_target",
                "flow_mask",
            ),
            "teacher_vector": self._mse(
                outputs,
                batch,
                "student_vector",
                "teacher_vector",
                "teacher_vector_mask",
            ),
            "future_traj_vlm": self._get(outputs, batch, "future_traj_vlm_loss"),
        }
        weights = {
            "ce": self.config.ce_weight,
            "kd": self.config.kd_weight,
            "on_policy_kl": self.on_policy_config.kl_weight,
            "hidden": self.config.hidden_weight,
            "flow": self.config.flow_weight,
            "teacher_vector": self.config.teacher_vector_weight,
            "future_traj_vlm": self.config.future_traj_vlm_weight,
        }
        active = {name: value for name, value in terms.items() if value is not None}
        if not active:
            raise ValueError("no compatible distillation loss inputs were provided")
        total = sum(value * weights[name] for name, value in active.items())
        return {"loss": total, **{f"{name}_loss": value for name, value in active.items()}}
