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

"""Alpamayo Model for AlpaGym"""

from typing import Any

import einops
import hydra.utils as hyu
import torch
from alpamayo_r1.common.logging import RankedLogger
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from alpamayo1_x_rl.models.expert_model.config import ExpertModelConfig

logger = RankedLogger(__name__, rank_zero_only=True)


class ExpertModel(AlpamayoR1):
    """Expert model for reasoning VLA.

    Unlike the alpagym variant, this recipe ExpertModel does not ship an online
    tokenization fallback: callers must pre-tokenize their batches (e.g. via a
    packer) and place the result under `data["tokenized_data"]` before invoking
    `sample_trajectories_from_data`.
    """

    config_class: type[ExpertModelConfig] = ExpertModelConfig
    _supports_sdpa: bool = True
    # FA2 misreads the expert head's 4-D additive attention_mask as a 2-D padding mask.
    _supports_flash_attn_2: bool = False

    def __init__(
        self,
        config: ExpertModelConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ) -> None:
        """Build the AlpamayoR1 base plus the optional expert history-trajectory
        tokenizer, and freeze the VLM unless `cotrain_vlm` is set.

        Args:
            config: ExpertModel configuration.
            pretrained_modules: Optional pretrained submodules to reuse instead of
                rebuilding from scratch.
            original_vocab_size: Original VLM vocab size before alpamayo's
                trajectory/special tokens were appended.
        """
        super().__init__(config, pretrained_modules, original_vocab_size)

        expert_hidden_size = self.expert.config.hidden_size
        if config.expert_hist_traj_tokenizer_cfg is not None:
            self.expert_hist_traj_tokenizer = hyu.instantiate(
                config.expert_hist_traj_tokenizer_cfg, outdim=expert_hidden_size
            )
            if self.config.keep_same_dtype:
                self.expert_hist_traj_tokenizer = self.expert_hist_traj_tokenizer.to(
                    dtype=self.expert.dtype
                )
            # Skip re-init of pretrained submodule weights on a later post_init() call.
            self.expert_hist_traj_tokenizer._is_hf_initialized = True
        else:
            self.expert_hist_traj_tokenizer = None

        if not self.config.cotrain_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False

    def _process_position_ids_qwen2_5_vl(
        self,
        vlm_outputs: Any,
        batch_size: int,
        num_expert_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build expert-side Qwen-VL RoPE position_ids that continue the VLM prefill.

        Qwen-VL uses 3-axis RoPE; expert tokens get a per-batch arange shifted by
        `vlm_outputs.rope_deltas + past_key_values.get_seq_length()` so they stitch
        on cleanly to the VLM's prefill cache.

        Args:
            vlm_outputs: Output from the VLM forward, carrying `rope_deltas` and
                `past_key_values`.
            batch_size: Batch dimension to broadcast position_ids over.
            num_expert_tokens: Number of expert tokens that need positions.
            device: Device to allocate position_ids on.

        Returns:
            Position_ids of shape `(3, batch_size, num_expert_tokens)`.
        """
        position_ids = torch.arange(num_expert_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
        position_ids += delta.to(position_ids.device)
        return position_ids

    def sample_trajectories_from_data(  # type: ignore[override]
        self,
        data: dict[str, Any],
        with_vlm_rollout: bool = True,
        last_component: str = "traj_future",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Dispatcher for the two concrete samplers defined on `ExpertModelRL`.

        Base `ExpertModel` exists for HF `AutoModel.register` and shared
        construction; it is not directly samplable. The `ExpertModelRL`
        subclass owns both `_sample_trajectories_from_data_with_vlm_rollout`
        and `_sample_trajectories_from_data_without_vlm_rollout`, including
        SDE log_prob bookkeeping that the alpagym `inference_model.py`
        contract requires.

        Args:
            data: Batched model input. Must contain a pre-tokenized
                `tokenized_data` entry; the recipe ExpertModel does not
                tokenize on the fly.
            with_vlm_rollout: When True, run the VLM CoT/text rollout before
                expert diffusion sampling. When False, skip straight to expert
                sampling (the production path for `last_component='traj_future'`).
            last_component: Conversation cut-off; retained on the signature for
                parity with the alpagym variant but unused here since callers
                pre-tokenize.
            *args: Forwarded to the chosen branch.
            **kwargs: Forwarded to the chosen branch.

        Returns:
            `(pred_xyz, pred_rot, logprob[, extra])` as produced by the
            `ExpertModelRL` sampler implementations.
        """
        assert "tokenized_data" in data, (
            "ExpertModel.sample_trajectories_from_data requires the caller to "
            "pre-tokenize and populate data['tokenized_data']."
        )
        if with_vlm_rollout:
            return self._sample_trajectories_from_data_with_vlm_rollout(data, *args, **kwargs)
        return self._sample_trajectories_from_data_without_vlm_rollout(data, *args, **kwargs)
