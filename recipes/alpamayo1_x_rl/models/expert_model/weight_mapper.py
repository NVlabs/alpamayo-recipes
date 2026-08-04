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

"""Weight mapper for ExpertModel (Policy <-> Rollout key normalization)."""

from __future__ import annotations

import torch
from alpamayo1_x_rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper
from cosmos_rl.utils import util


class ExpertModelWeightMapper(ReasoningVLAWeightMapper):
    """Weight mapper that short-circuits Expert-specific keys.

    Expert keys (expert.*, action_in_proj.*, action_out_proj.*, diffusion.*)
    bypass the VLM key mapping chain to avoid corruption by the parent mapper.
    """

    _EXPERT_PREFIXES = (
        "expert.",
        "action_in_proj.",
        "action_out_proj.",
        "expert_hist_traj_tokenizer.",
        "diffusion.",
    )

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        """Map policy ExpertModel parameter names to HF checkpoint key-space."""
        name = util.clear_weight_name(name)

        for wrapper_prefix in ("expert_model.", "reasoning_vla."):
            if name.startswith(wrapper_prefix):
                name = name[len(wrapper_prefix) :]
                break

        for prefix in self._EXPERT_PREFIXES:
            if name.startswith(prefix):
                return name

        stripped = name
        if stripped.startswith("vlm."):
            stripped = stripped[len("vlm.") :]
        stripped = stripped.replace("model.language_model.", "model.", 1)
        stripped = stripped.replace("model.visual.", "visual.", 1)

        return super(ReasoningVLAWeightMapper, self).policy_map_local_key_to_hf_key(stripped)

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Guard against rollout TP>1 — recipe trim only covers single-rank shards.

        The recipe ``ReasoningVLAWeightMapper`` trims vocab padding only when
        ``t.shape[0] > vocab_size``, which holds for TP=1 (one rank owns the
        full padded vocab) but not for TP>1: each TP rank's local view is
        ``ceil(padded_vocab / tp_size)`` rows, smaller than the unpadded
        vocab, so the `>` check is False and padded rows leak through the
        weight-sync into the HF policy embed/lm_head — corrupting the
        embedding table.

        alpagym's ``host/config_validation.py`` rejects rollout ``tp_size != 1``
        today, so this guard is belt-and-suspenders: if that enforcement is
        ever loosened without first upstreaming a TP-aware trim to the recipe,
        the rollout fails here instead of silently shipping corrupt embeddings.
        """
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size

            tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            tp_size = 1
        if tp_size != 1:
            raise AssertionError(
                f"ExpertModelWeightMapper requires rollout tp_size=1 (got {tp_size}); "
                "recipe ReasoningVLAWeightMapper does not trim vocab-padding for TP shards."
            )
        return super().rollout_split_local_key_n_param_to_hf_key_n_param(param_name, param)

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        """Map rollout ExpertModel parameter names to HF checkpoint key-space."""
        name = rollout_weight_name

        for prefix in self._EXPERT_PREFIXES:
            if name.startswith(prefix):
                return name

        if name.startswith("vlm."):
            name = name[len("vlm.") :]
        if name.startswith("model.language_model."):
            name = name.replace("model.language_model.", "model.", 1)
        elif name.startswith("model.visual."):
            name = name.replace("model.visual.", "visual.", 1)

        return super().policy_map_local_key_to_hf_key("reasoning_vla.vlm." + name)
