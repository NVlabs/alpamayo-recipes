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

"""Cosmos-RL BaseModel wrapper for ExpertModel with CFM log_prob support."""

from __future__ import annotations

import json
import os

import torch
from cosmos_rl.policy.model.base import ModelRegistry
from cosmos_rl.utils.logging import logger
from transformers import AutoConfig, AutoModel

from alpamayo1_x_rl.base_cosmos_wrapper import BaseCosmosWrapper
from alpamayo1_x_rl.utils.fsdp import pad_linear_for_fsdp, shard_lm_layers, shard_visual_tower
from alpamayo1_x_rl.models.expert_model.model import ExpertModelRL
from alpamayo1_x_rl.models.expert_model.config import ExpertModelConfig
from alpamayo1_x_rl.models.expert_model.expert_model import ExpertModel
from alpamayo1_x_rl.utils.weight_loading import copy_state_into_dtensor_shards, detect_fsdp2_active
from alpamayo1_x_rl.models.expert_model.weight_mapper import ExpertModelWeightMapper

AutoConfig.register("alpamayo_reasoning_vla_expert", ExpertModelConfig, exist_ok=True)
AutoModel.register(ExpertModelConfig, ExpertModel, exist_ok=True)


class ExpertModelCosmos(BaseCosmosWrapper):
    """Cosmos BaseModel wrapper for ExpertModel with CFM log_prob support."""

    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        self.expert_model = ExpertModelRL(hf_config)

        self.expert_model.config.cotrain_vlm = False

        vlm_model = self.expert_model.vlm.model
        lm = getattr(vlm_model, "language_model", None) or getattr(vlm_model, "model", None)
        if lm is not None:
            ct = 0
            for param in lm.parameters():
                param.requires_grad = False
                ct += 1
            logger.info(f"[ExpertModelCosmos] Froze {ct} LM params ({type(lm).__name__})")
        else:
            logger.warning("[ExpertModelCosmos] Could not find language_model to freeze!")

        visual = getattr(vlm_model, "visual", None)
        if visual is not None:
            for param in visual.parameters():
                param.requires_grad = False

        if "trainable_params" in self.__dict__:
            del self.__dict__["trainable_params"]

        trainable = sum(1 for p in self.expert_model.parameters() if p.requires_grad)
        logger.info(f"[ExpertModelCosmos] Total trainable params: {trainable}")

    @staticmethod
    def supported_model_types():
        """Return the HF model types this wrapper handles."""
        return ["alpamayo_reasoning_vla_expert"]

    def forward(self, teacher_model=None, **kwargs):
        """Forward pass for GRPO training.

        Pops SDE-specific fields from kwargs, calls cfm_logprob_sde,
        returns {"logits": log_probs [B, 1], "kl_div": kl_div [B] or None}.
        """
        samples_list = kwargs.pop("samples_list")
        timesteps = kwargs.pop("timesteps")
        noise_level = kwargs.pop("noise_level", 0.7)
        if isinstance(noise_level, torch.Tensor):
            noise_level = noise_level.flatten()[0].item()

        vlm_generated_ids = kwargs.pop("vlm_generated_ids", None)
        if vlm_generated_ids is not None and isinstance(vlm_generated_ids, torch.Tensor):
            if vlm_generated_ids.numel() == 0:
                vlm_generated_ids = None
            else:
                vlm_generated_ids = vlm_generated_ids.to(samples_list.device)
                if vlm_generated_ids.dim() == 1:
                    vlm_generated_ids = vlm_generated_ids.unsqueeze(0)

        data = {
            "tokenized_data": kwargs.get("tokenized_data", {}),
            "ego_history_xyz": kwargs.get("ego_history_xyz"),
            "ego_history_rot": kwargs.get("ego_history_rot"),
        }

        teacher = None
        if teacher_model is not None:
            teacher = getattr(teacher_model, "expert_model", teacher_model)

        log_probs, kl_div = self.expert_model.cfm_logprob_sde(
            data,
            samples_list,
            timesteps,
            noise_level,
            teacher,
            vlm_generated_ids=vlm_generated_ids,
        )
        return {"logits": log_probs.unsqueeze(1), "kl_div": kl_div}

    def _apply_fsdp2(self, dp_mesh, fsdp_config: dict, reshard_fn) -> None:
        """Apply FSDP2 sharding to the ExpertModel sub-modules."""
        from torch.distributed.fsdp import fully_shard

        em = self.expert_model

        shard_visual_tower(em, fsdp_config, reshard_fn, model_name="ExpertModel")
        shard_lm_layers(em, fsdp_config, reshard_fn, model_name="ExpertModel")

        if hasattr(em, "expert") and hasattr(em.expert, "layers"):
            items = list(enumerate(em.expert.layers))
            for idx, blk in items:
                fully_shard(blk, **fsdp_config, reshard_after_forward=reshard_fn(idx, len(items)))
            fully_shard(em.expert, **fsdp_config, reshard_after_forward=True)
            logger.info(f"[ExpertModel][FSDP] Sharded {len(items)} expert layers")

        dp_world = dp_mesh.size()
        if isinstance(em.action_out_proj, torch.nn.Linear):
            pad_linear_for_fsdp(em.action_out_proj, min_out=dp_world)

        for pname, p in em.named_parameters():
            if p.numel() > 0 and p.numel() < dp_world:
                logger.warning(
                    f"[ExpertModel][FSDP] param {pname} has numel={p.numel()} "
                    f"< dp_shard_size={dp_world}; may produce empty DTensor shards"
                )

        fully_shard(self, **fsdp_config, reshard_after_forward=True)
        logger.info("[ExpertModel][FSDP] Applied FSDP2 to full model")

    def load_hf_weights(
        self, model_name_or_path: str, parallel_dims=None, device=None, revision=None
    ):
        """Load ExpertModel weights from checkpoint."""
        from safetensors.torch import load_file

        model_path = model_name_or_path

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            ckpt_state = {}
            for shard in shard_files:
                ckpt_state.update(load_file(os.path.join(model_path, shard)))
        else:
            ckpt_state = load_file(os.path.join(model_path, "model.safetensors"))

        if not detect_fsdp2_active(self.expert_model):
            self.expert_model.load_state_dict(ckpt_state, strict=False)
            if device is not None:
                self.expert_model = self.expert_model.to(device)
            logger.info(f"[ExpertModel] Loaded weights (non-sharded) from {model_path}")
            return

        logger.info(f"[ExpertModel] Loading weights into FSDP2 DTensor shards from {model_path}")
        copy_state_into_dtensor_shards(
            self.expert_model,
            ckpt_state,
            strict=False,
            pad_to_match=True,
        )
        logger.info(f"[ExpertModel] Loaded weights into FSDP2 shards from {model_path}")

    def post_to_empty_hook(self, cosmos_config):
        """Warm up HF auto-class registry after model is moved to meta device."""
        model_path = (
            getattr(cosmos_config.policy, "model_name_or_path", None)
            if cosmos_config is not None
            else None
        ) or getattr(self.hf_config, "_name_or_path", None)
        assert model_path is not None, "model_path is None"
        ref = ExpertModel.from_pretrained(model_path, trust_remote_code=True).to("cpu")
        del ref

    @classmethod
    def from_pretrained(
        cls, hf_config, model_name_or_path: str = None, max_position_embeddings=None, **kwargs
    ):
        """Construct an ExpertModelCosmos from its HF config."""
        return cls(hf_config)


if "alpamayo_reasoning_vla_expert" not in ModelRegistry._MODEL_REGISTRY:
    ModelRegistry.register(ExpertModelWeightMapper)(ExpertModelCosmos)
