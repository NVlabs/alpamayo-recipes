# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Configuration classes for Alpamayo R1 release models."""

from collections.abc import Mapping
from typing import Any

from alpamayo1_distill.models.base_model import ReasoningVLAConfig
from alpamayo1_distill.defaults import (
    SUPPORTED_ACTION_EXPERT_ARCHITECTURE,
    SUPPORTED_STUDENT_BACKEND,
    SUPPORTED_STUDENT_TEXT_ARCHITECTURE,
    SUPPORTED_STUDENT_VISION_ARCHITECTURE,
)


_LEGACY_NAMESPACE = "alpamayo_r1."
_CANONICAL_NAMESPACE = "alpamayo1_distill."


def _validate_architecture_fields(
    source: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    mismatches = []
    for field, expected_value in expected.items():
        actual_value = (
            source.get(field) if isinstance(source, Mapping) else getattr(source, field, None)
        )
        if actual_value != expected_value:
            mismatches.append(
                f"{field}: expected={expected_value!r}, found={actual_value!r}"
            )
    if mismatches:
        raise ValueError(
            f"Unsupported {label}; this recipe only supports Cosmos-Reason2-2B "
            f"with the approximately 100M action expert ({'; '.join(mismatches)})"
        )


def validate_supported_student_vlm_config(vlm_config: Any) -> None:
    """Require the Cosmos-Reason2-2B/Qwen3-VL-2B backbone dimensions."""

    if getattr(vlm_config, "model_type", None) != "qwen3_vl":
        raise ValueError(
            "Unsupported student VLM model_type; this recipe only supports "
            "Cosmos-Reason2-2B (qwen3_vl)"
        )
    text_config = getattr(vlm_config, "text_config", None)
    if text_config is None:
        raise ValueError("Cosmos-Reason2-2B config is missing text_config")
    _validate_architecture_fields(
        text_config,
        SUPPORTED_STUDENT_TEXT_ARCHITECTURE,
        label="student VLM text architecture",
    )
    vision_config = getattr(vlm_config, "vision_config", None)
    if vision_config is None:
        raise ValueError("Cosmos-Reason2-2B config is missing vision_config")
    _validate_architecture_fields(
        vision_config,
        SUPPORTED_STUDENT_VISION_ARCHITECTURE,
        label="student VLM vision architecture",
    )


def canonicalize_legacy_targets(value: Any) -> Any:
    """Translate legacy Hydra ``_target_`` values in checkpoint configuration.

    Only Hydra target fields are migrated.  Other strings are preserved even if
    they happen to begin with ``alpamayo_r1.``; they may be user metadata or a
    path-like value rather than an import target.
    """

    if isinstance(value, Mapping):
        migrated = {}
        for key, item in value.items():
            if (
                key == "_target_"
                and isinstance(item, str)
                and item.startswith(_LEGACY_NAMESPACE)
            ):
                item = _CANONICAL_NAMESPACE + item[len(_LEGACY_NAMESPACE) :]
            else:
                item = canonicalize_legacy_targets(item)
            migrated[key] = item
        return migrated
    if isinstance(value, list):
        return [canonicalize_legacy_targets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(canonicalize_legacy_targets(item) for item in value)
    return value


class AlpamayoR1Config(ReasoningVLAConfig):
    """Configuration for the Alpamayo R1 release model."""

    # This serialized identifier belongs to released Alpamayo-R1 checkpoints.
    # Keep it stable even though the recipe's Python namespace is now
    # ``alpamayo1_distill``.
    model_type = "alpamayo_r1"

    def __init__(
        self,
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs = canonicalize_legacy_targets(kwargs)
        super().__init__(**kwargs)
        self.diffusion_cfg = canonicalize_legacy_targets(diffusion_cfg)
        self.action_space_cfg = canonicalize_legacy_targets(action_space_cfg)
        self.action_in_proj_cfg = canonicalize_legacy_targets(action_in_proj_cfg)
        self.action_out_proj_cfg = canonicalize_legacy_targets(action_out_proj_cfg)
        self.expert_cfg = canonicalize_legacy_targets(expert_cfg)
        self.keep_same_dtype = keep_same_dtype
        self.expert_non_causal_attention = expert_non_causal_attention


class ExpertModelConfig(ReasoningVLAConfig):
    """Configuration for the ExpertModel model."""

    model_type = "alpamayo_reasoning_vla_expert_distill"

    def __init__(
        self,
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        expert_hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        traj_loss_weight: float = 1.0,
        future_traj_vlm_loss_weight: float = 1.0,
        cotrain_vlm: bool = True,
        stop_grad_from_vlm: bool = True,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        hidden_distill_weight: float = 0.0,
        reasoning_max_generation_length: int = 256,
        **kwargs: Any,
    ) -> None:
        kwargs = canonicalize_legacy_targets(kwargs)
        super().__init__(**kwargs)
        self.diffusion_cfg = canonicalize_legacy_targets(diffusion_cfg)
        self.action_space_cfg = canonicalize_legacy_targets(action_space_cfg)
        self.action_in_proj_cfg = canonicalize_legacy_targets(action_in_proj_cfg)
        self.action_out_proj_cfg = canonicalize_legacy_targets(action_out_proj_cfg)
        self.expert_cfg = canonicalize_legacy_targets(expert_cfg)
        # Early Cosmos checkpoints inherited these two dimensions from the
        # 2B backbone instead of serializing them in expert_cfg. Materialize the
        # same values so those checkpoints retain the identical ~100M expert.
        if self.expert_cfg is not None:
            self.expert_cfg.setdefault("num_hidden_layers", 28)
            self.expert_cfg.setdefault("num_key_value_heads", 8)
        self.expert_hist_traj_tokenizer_cfg = canonicalize_legacy_targets(
            expert_hist_traj_tokenizer_cfg
        )
        self.traj_loss_weight = traj_loss_weight
        self.future_traj_vlm_loss_weight = future_traj_vlm_loss_weight
        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm
        self.keep_same_dtype = keep_same_dtype
        self.expert_non_causal_attention = expert_non_causal_attention
        self.hidden_distill_weight = hidden_distill_weight
        if reasoning_max_generation_length <= 0:
            raise ValueError("reasoning_max_generation_length must be positive")
        self.reasoning_max_generation_length = reasoning_max_generation_length
        self.validate_supported_architecture()

    def validate_supported_architecture(self) -> None:
        """Enforce the recipe's single Cosmos-2B + ~100M-expert contract."""

        if self.vlm_backend != SUPPORTED_STUDENT_BACKEND:
            raise ValueError(
                f"Unsupported student VLM backend {self.vlm_backend!r}; this recipe "
                "only supports Cosmos-Reason2-2B through the qwenvl3 backend"
            )
        if self.expert_cfg is None:
            raise ValueError("The supported action expert requires expert_cfg")
        _validate_architecture_fields(
            self.expert_cfg,
            SUPPORTED_ACTION_EXPERT_ARCHITECTURE,
            label="action expert architecture",
        )
        allowed_expert_fields = set(SUPPORTED_ACTION_EXPERT_ARCHITECTURE) | {"dtype"}
        extra_expert_fields = sorted(set(self.expert_cfg) - allowed_expert_fields)
        if extra_expert_fields:
            raise ValueError(
                "Unsupported action expert overrides for the fixed approximately "
                f"100M architecture: {extra_expert_fields}"
            )
        if self.expert_cfg.get("dtype") != "bfloat16":
            raise ValueError("The supported action expert requires dtype='bfloat16'")
        if self.expert_hist_traj_tokenizer_cfg is not None:
            raise ValueError(
                "The supported action expert does not use an additional expert history encoder"
            )
        if self.hist_traj_embed_cfg is not None:
            raise ValueError(
                "The supported Cosmos student does not use hist_traj_embed_cfg"
            )

        expected_action_in = {
            "_target_": "alpamayo1_distill.models.action_in_proj.PerWaypointActionInProjV2",
            "hidden_size": 512,
            "num_enc_layers": 2,
            "num_fourier_feats": 20,
            "max_freq": 100.0,
        }
        if self.action_in_proj_cfg is None:
            raise ValueError("The supported action expert requires action_in_proj_cfg")
        _validate_architecture_fields(
            self.action_in_proj_cfg,
            expected_action_in,
            label="action-input projection architecture",
        )
        if self.action_out_proj_cfg != {"_target_": "torch.nn.Linear"}:
            raise ValueError(
                "Unsupported action-output projection; this recipe only supports "
                "torch.nn.Linear"
            )
        if self.action_space_cfg is None:
            raise ValueError("The supported action expert requires action_space_cfg")
        _validate_architecture_fields(
            self.action_space_cfg,
            {
                "_target_": (
                    "alpamayo1_distill.action_space."
                    "UnicycleAccelCurvatureActionSpace"
                ),
                "n_waypoints": 64,
                "dt": 0.1,
            },
            label="action space",
        )
        if self.diffusion_cfg is None:
            raise ValueError("The supported action expert requires diffusion_cfg")
        _validate_architecture_fields(
            self.diffusion_cfg,
            {
                "_target_": "alpamayo1_distill.diffusion.flow_matching.FlowMatching",
                "int_method": "euler",
            },
            label="diffusion architecture",
        )
