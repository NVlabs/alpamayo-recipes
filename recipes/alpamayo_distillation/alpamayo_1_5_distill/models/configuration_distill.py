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

"""Configuration for the Alpamayo 1.5 distillation model."""

import copy
from typing import Any

from alpamayo1_5.models.base_model import ReasoningVLAConfig

COSMOS_REASON2_2B = "nvidia/Cosmos-Reason2-2B"
SMALL_ACTION_EXPERT_CFG: dict[str, int] = {
    # Fixed to match Cosmos-Reason2-2B so the expert can reuse the VLM KV cache.
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_attention_heads": 16,
    "head_dim": 128,
}


class DistillAlpamayo1_5Config(ReasoningVLAConfig):
    """Configuration for the Alpamayo 1.5 release model."""

    model_type = "alpamayo1_5_distill"

    def __init__(
        self,
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        stop_action_grad_to_vlm: bool = True,
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.diffusion_cfg = copy.deepcopy(diffusion_cfg)
        if self.diffusion_cfg is not None:
            self.diffusion_cfg["_target_"] = (
                "alpamayo1_5_distill.diffusion.flow_matching.FlowMatching"
            )
        self.action_space_cfg = action_space_cfg
        self.action_in_proj_cfg = action_in_proj_cfg
        self.action_out_proj_cfg = action_out_proj_cfg
        self.expert_cfg = expert_cfg
        self.keep_same_dtype = keep_same_dtype
        self.expert_non_causal_attention = expert_non_causal_attention
        self.stop_action_grad_to_vlm = stop_action_grad_to_vlm
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums

    @classmethod
    def cosmos_reason2_2b_student(
        cls,
        base_config: ReasoningVLAConfig | None = None,
        **kwargs: Any,
    ) -> "DistillAlpamayo1_5Config":
        """Build the supported 2B VLM + compact action-expert configuration.

        ``base_config`` should normally be the AM1.5 teacher configuration so
        the action space, diffusion, and trajectory tokenizer settings remain
        identical. Explicit caller values take precedence.
        """
        config_values = base_config.to_dict() if base_config is not None else {}
        config_values["vlm_name_or_path"] = kwargs.pop("vlm_name_or_path", COSMOS_REASON2_2B)
        config_values["expert_cfg"] = kwargs.pop("expert_cfg", SMALL_ACTION_EXPERT_CFG.copy())
        config_values.update(kwargs)
        return cls(**config_values)
