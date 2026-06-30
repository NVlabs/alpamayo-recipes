# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ExpertModel configuration extending the recipe's RL-wrapper config."""

from typing import Any

from alpamayo1_x_rl.models.reasoning_vla.config import RLWrapperReasoningVLAConfig


class ExpertModelConfig(RLWrapperReasoningVLAConfig):
    """Configuration for ExpertModel.

    Inherits the recipe's RL-wrapper config (which already provides the
    vLLM/Cosmos-RL helpers: `_patch_llm_config_vocab`, `get_llm_config`,
    and the `_cached_llm_config`-aware `to_dict`) and adds the expert-head
    + alpagym-specific knobs on top. Also restores a kwargs-accepting
    `get_text_config` so transformers utilities that call it with
    `decoder=True`/`encoder=True` keep working.
    """

    model_type = "alpamayo_reasoning_vla_expert"

    def __init__(
        self,
        diffusion_cfg: dict[str, Any] | None = None,
        action_space_cfg: dict[str, Any] | None = None,
        action_in_proj_cfg: dict[str, Any] | None = None,
        action_out_proj_cfg: dict[str, Any] | None = None,
        expert_cfg: dict[str, Any] | None = None,
        expert_hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        traj_loss_weight: float = 1.0,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        keep_same_dtype: bool = True,
        expert_non_causal_attention: bool = True,
        legacy_inference_image_input_format: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ExpertModel configuration.

        Args:
            diffusion_cfg: Configuration for the diffusion head.
            action_space_cfg: Configuration for the action space.
            action_in_proj_cfg: Configuration for the action input projection.
            action_out_proj_cfg: Configuration for the action output projection.
            expert_cfg: Configuration for the expert head.
            expert_hist_traj_tokenizer_cfg: Configuration for the expert's
                history trajectory tokenizer, when separate from the main
                trajectory tokenizer.
            traj_loss_weight: Loss weight applied to the trajectory term.
            cotrain_vlm: When True, train the VLM parameters alongside the
                expert head. When False, freeze the VLM.
            stop_grad_from_vlm: When True, detach VLM outputs before they
                feed the expert head.
            keep_same_dtype: When True, cast added submodules to the expert
                head's dtype after construction.
            expert_non_causal_attention: When True, the expert head uses
                non-causal attention (additive mask) over its own tokens.
            legacy_inference_image_input_format: When True, inference inputs
                arrive in ``[-1, 1]`` and are rescaled to ``[0, 1]`` before
                tokenization. When False (default), inputs are assumed to
                already be in ``[0, 1]``.
            **kwargs: Forwarded to `RLWrapperReasoningVLAConfig` (e.g.
                `vlm_name_or_path`, `traj_vocab_size`, `model_dtype`).
        """
        # alpagym ckpts are always built with the full SPECIAL_TOKENS set
        # (28 entries); the recipe defaults to TRAJ_TOKEN-only (6) and would
        # produce a smaller vocab that fails to load existing weights.
        kwargs["add_special_tokens"] = True
        super().__init__(**kwargs)
        self.diffusion_cfg = diffusion_cfg
        self.action_space_cfg = action_space_cfg
        self.action_in_proj_cfg = action_in_proj_cfg
        self.action_out_proj_cfg = action_out_proj_cfg
        self.expert_cfg = expert_cfg
        self.expert_hist_traj_tokenizer_cfg = expert_hist_traj_tokenizer_cfg
        self.traj_loss_weight = traj_loss_weight
        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm
        self.keep_same_dtype = keep_same_dtype
        self.expert_non_causal_attention = expert_non_causal_attention
        self.legacy_inference_image_input_format = legacy_inference_image_input_format

    def get_text_config(self, **kwargs: Any) -> Any:
        """Forward kwargs to the backing VLM text config.

        Transformers utilities (KV cache, ``_tie_weights``, quantizers) call
        ``config.get_text_config(decoder=True)``; the recipe override has
        signature ``(self)`` and would raise ``TypeError`` on that kwarg.
        Mirror the base ``PretrainedConfig`` contract by accepting and
        forwarding kwargs to the Qwen-VL text-config resolver.
        """
        llm_cfg = self.get_llm_config()
        if hasattr(llm_cfg, "get_text_config"):
            return llm_cfg.get_text_config(**kwargs)
        return llm_cfg
