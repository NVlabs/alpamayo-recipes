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

import copy
import logging
from functools import partial
from typing import Any

import einops
import hydra.utils as hyu
import numpy as np
import torch
import torch.nn.functional as F
from alpamayo1_5.action_space import ActionSpace
from alpamayo1_5.diffusion.base import BaseDiffusion
from alpamayo1_5.models.base_model import ReasoningVLA
from alpamayo1_5.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo1_5.nav_utils import remove_nav_text
from transformers import (
    AutoConfig,
    AutoModel,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)

from alpamayo1_5_distill.models.configuration_distill import DistillAlpamayo1_5Config
from alpamayo1_5_distill.models.expert_context import (
    build_expert_attention_mask,
    build_expert_position_ids,
    extend_prompt_attention_mask,
    find_future_start_offsets,
)
from alpamayo1_5_distill.models.trajectory_tokens import tokenize_future_trajectory

logger = logging.getLogger(__name__)


def _remove_synthesized_collation_masks(
    tokenized_data: dict[str, Any],
) -> dict[str, Any]:
    """Drop ``recursive_collate`` masks while preserving real model masks."""

    return {
        key: value
        for key, value in tokenized_data.items()
        if not (key.endswith("_mask") and key.removesuffix("_mask") in tokenized_data)
    }


class ExpertLogitsProcessor(LogitsProcessor):
    """Masks out the logits for discrete trajectory tokens."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        """Initialize the ExpertLogitsProcessor.

        Args:
            traj_token_offset: The offset of the trajectory tokens.
            traj_vocab_size: The vocabulary size of the trajectory tokens.
        """
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call the ExpertLogitsProcessor to mask out the logits for discrete trajectory tokens.

        The discrete trajectory tokens are not used for the expert model thus masking them out for
        better CoC generation.

        Args:
            input_ids: The input IDs.
            scores: The scores.

        Returns:
            torch.FloatTensor: The scores with trajectory-token positions
                masked out (set to ``-inf``).
        """
        # Directly assign -inf to the trajectory token positions in the scores tensor
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float(
            "-inf"
        )
        return scores


class DistillAlpamayo1_5(ReasoningVLA):
    """Expert model for reasoning VLA."""

    config_class: type[DistillAlpamayo1_5Config] = DistillAlpamayo1_5Config
    base_model_prefix = "vlm"

    @classmethod
    def config_with_vlm_override(
        cls,
        name_or_path: str,
        vlm_name_or_path: str | None = None,
    ) -> DistillAlpamayo1_5Config:
        """Load a config after replacing a checkpoint's embedded VLM path."""
        if vlm_name_or_path is None:
            return cls.config_class.from_pretrained(name_or_path)
        config_dict, remaining_kwargs = cls.config_class.get_config_dict(name_or_path)
        config_dict["vlm_name_or_path"] = vlm_name_or_path
        return cls.config_class.from_dict(config_dict, **remaining_kwargs)

    @classmethod
    def from_distillation_student(
        cls,
        teacher_name_or_path: str = "nvidia/Alpamayo-1.5-10B",
        teacher_vlm_name_or_path: str | None = None,
        train_timestep_sampler: str | None = None,
        **config_overrides: Any,
    ) -> "DistillAlpamayo1_5":
        """Initialize the compact student while retaining AM1.5 action settings."""
        teacher_config = cls.config_with_vlm_override(
            teacher_name_or_path,
            teacher_vlm_name_or_path,
        )
        if train_timestep_sampler is not None:
            if train_timestep_sampler not in {"uniform", "beta"}:
                raise ValueError("train_timestep_sampler must be 'uniform' or 'beta'")
            diffusion_cfg = copy.deepcopy(teacher_config.diffusion_cfg)
            if diffusion_cfg is None:
                raise ValueError("teacher configuration does not define diffusion_cfg")
            diffusion_cfg["train_timestep_sampler"] = train_timestep_sampler
            config_overrides["diffusion_cfg"] = diffusion_cfg
        student_config = DistillAlpamayo1_5Config.cosmos_reason2_2b_student(
            base_config=teacher_config,
            **config_overrides,
        )
        if teacher_config.traj_token_ids != student_config.traj_token_ids:
            raise ValueError(
                "Teacher and student trajectory special-token IDs differ; "
                "direct token/logit distillation is unsafe"
            )
        if teacher_config.traj_token_start_idx != student_config.traj_token_start_idx:
            raise ValueError("Teacher and student discrete trajectory token ranges are not aligned")
        if teacher_config.vocab_size != student_config.vocab_size:
            raise ValueError("Teacher and student token vocabularies have different sizes")
        return cls.from_pretrained_submodules(student_config)

    def __init__(
        self,
        config: DistillAlpamayo1_5Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # we only need the text config for the expert model
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self._validate_expert_cache_compatibility(self.vlm.config.text_config, expert_config)
        # The diffusion expert does not support FlashAttention 2.
        # Force sdpa for the expert in that case.
        if expert_config._attn_implementation == "flash_attention_2":
            expert_config._attn_implementation = "sdpa"
        self.expert = AutoModel.from_config(expert_config)
        # we don't need the embed_tokens of the expert model
        del self.expert.embed_tokens

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()

    def generate_text(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_samples: int = 1,
        max_generation_length: int = 256,
        stopping_criteria: StoppingCriteriaList | None = None,
        bad_words_ids: list[list[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate structured text with optional answer-specific stopping."""
        data = copy.deepcopy(data)
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")

        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        if stopping_criteria is not None:
            generation_config.eos_token_id = None
            generation_config.forced_eos_token_id = None

        generated = self.vlm.generate(
            input_ids=input_ids,
            **tokenized_data,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            bad_words_ids=bad_words_ids,
        )
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        extra = extract_text_tokens(self.tokenizer, generated_tokens)
        for key in extra:
            extra[key] = np.array(extra[key]).reshape([input_ids.shape[0], num_samples])
        return extra

    @staticmethod
    def _validate_expert_cache_compatibility(vlm_config: Any, expert_config: Any) -> None:
        """Validate dimensions that make a VLM KV cache reusable by the expert."""
        cache_fields = ("num_key_value_heads", "head_dim")
        mismatches = [
            f"{field}: VLM={getattr(vlm_config, field, None)}, "
            f"expert={getattr(expert_config, field, None)}"
            for field in cache_fields
            if getattr(vlm_config, field, None) != getattr(expert_config, field, None)
        ]
        if mismatches:
            raise ValueError(
                "The action expert cannot consume this VLM's KV cache; " + "; ".join(mismatches)
            )
        if expert_config.num_hidden_layers > vlm_config.num_hidden_layers:
            raise ValueError(
                "The action expert has more layers than the VLM KV cache: "
                f"VLM={vlm_config.num_hidden_layers}, "
                f"expert={expert_config.num_hidden_layers}"
            )
        if expert_config.hidden_size <= 0 or expert_config.intermediate_size <= 0:
            raise ValueError("Expert hidden_size and intermediate_size must be positive")
        if expert_config.num_attention_heads <= 0:
            raise ValueError("Expert num_attention_heads must be positive")

    @staticmethod
    def _detach_kv_cache(cache: Any) -> Any:
        """Detach cache tensors so action loss does not backpropagate into the VLM."""
        if hasattr(cache, "layers"):
            for layer in cache.layers:
                for attribute in ("keys", "values"):
                    tensor = getattr(layer, attribute, None)
                    if isinstance(tensor, torch.Tensor):
                        setattr(layer, attribute, tensor.detach())
            return cache
        for attribute in ("key_cache", "value_cache"):
            tensors = getattr(cache, attribute, None)
            if tensors is not None:
                setattr(
                    cache,
                    attribute,
                    [
                        tensor.detach() if isinstance(tensor, torch.Tensor) else tensor
                        for tensor in tensors
                    ],
                )
        return cache

    def _build_gt_future_traj_token_suffix(
        self, traj_data: dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        """Tokenize the GT future trajectory and append its closing marker."""
        if self.traj_tokenizer is None:
            raise RuntimeError("future trajectory VLM loss requires config.traj_tokenizer_cfg")
        future_tokens = tokenize_future_trajectory(
            self.traj_tokenizer,
            traj_data,
            self.future_token_start_idx,
        ).to(device=device, dtype=torch.long)
        future_end = torch.full(
            (future_tokens.shape[0], 1),
            self.special_token_ids["traj_future_end"],
            dtype=future_tokens.dtype,
            device=device,
        )
        return torch.cat((future_tokens, future_end), dim=1)

    def _compute_future_traj_vlm_loss_from_prefix(
        self,
        vlm_outputs: Any,
        seq: torch.Tensor,
        attention_mask: torch.Tensor,
        future_start_offsets: torch.LongTensor,
        gt_future_suffix: torch.Tensor,
        tokenized_data: dict[str, Any],
    ) -> torch.Tensor:
        """Compute GT future-token CE while reusing the multimodal prefix cache."""
        batch_size, suffix_len = gt_future_suffix.shape
        device = seq.device
        if suffix_len <= 0:
            return vlm_outputs.logits.sum() * 0.0

        batch_idx = torch.arange(batch_size, device=device)
        future_start_offsets = future_start_offsets.to(device=device)
        marker_positions = future_start_offsets - 1
        prev_positions = marker_positions - 1
        if bool((prev_positions < 0).any()):
            bad = (prev_positions < 0).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                f"future trajectory aux loss: marker has no previous token for batch indices {bad}"
            )

        marker_id = self.special_token_ids["traj_future_start"]
        has_marker = seq[batch_idx, marker_positions].eq(marker_id)
        if not bool(has_marker.all()):
            return vlm_outputs.logits.sum() * 0.0

        marker_labels = torch.full((batch_size,), marker_id, dtype=torch.long, device=device)
        marker_logits = vlm_outputs.logits[batch_idx, prev_positions, :]
        loss_sum = F.cross_entropy(marker_logits, marker_labels, reduction="sum")

        first_future_labels = gt_future_suffix[:, 0].to(device=device, dtype=torch.long)
        first_future_logits = vlm_outputs.logits[batch_idx, marker_positions, :]
        loss_sum = loss_sum + F.cross_entropy(
            first_future_logits, first_future_labels, reduction="sum"
        )
        token_count = torch.tensor(batch_size * 2, device=device, dtype=loss_sum.dtype)
        if suffix_len == 1:
            return loss_sum / token_count.clamp_min(1.0)

        kv_cache = vlm_outputs.past_key_values
        prefill_seq_len = kv_cache.get_seq_length()
        suffix_input_ids = gt_future_suffix[:, :-1].to(device=device, dtype=torch.long)
        suffix_labels = gt_future_suffix[:, 1:].to(device=device, dtype=torch.long)
        suffix_input_len = suffix_input_ids.shape[1]

        cache_positions = torch.arange(prefill_seq_len, device=device)
        visible_cache = cache_positions.unsqueeze(0) < future_start_offsets.unsqueeze(1)
        visible_cache &= attention_mask[:, :prefill_seq_len].to(device=device, dtype=torch.bool)
        suffix_visible = torch.ones((batch_size, suffix_input_len), dtype=torch.bool, device=device)
        suffix_attention_mask = torch.cat((visible_cache, suffix_visible), dim=1).to(
            dtype=attention_mask.dtype
        )
        suffix_position_ids = build_expert_position_ids(
            future_start_offsets,
            vlm_outputs.rope_deltas,
            query_seq_len=suffix_input_len,
            device=device,
        )

        suffix_data = dict(tokenized_data)
        suffix_data["attention_mask"] = suffix_attention_mask
        suffix_data["position_ids"] = suffix_position_ids
        for key in (
            "cache_position",
            "rope_deltas",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            suffix_data.pop(key, None)
        try:
            suffix_outputs = self.vlm(
                input_ids=suffix_input_ids,
                past_key_values=kv_cache,
                use_cache=False,
                **suffix_data,
            )
            suffix_logits = suffix_outputs.logits
            loss_sum = loss_sum + F.cross_entropy(
                suffix_logits.reshape(-1, suffix_logits.shape[-1]),
                suffix_labels.reshape(-1),
                reduction="sum",
            )
        finally:
            if kv_cache.get_seq_length() != prefill_seq_len:
                kv_cache.crop(prefill_seq_len)
        token_count = token_count + suffix_labels.numel()
        return loss_sum / token_count.clamp_min(1.0)

    def forward_distill(
        self,
        batch: dict[str, Any],
        language_loss_fn: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a teacher-forced student VLM and one action flow-training step.

        Expected inputs are ``tokenized_data``, trajectory history, and either
        ``target_action`` or future XYZ/rotation tensors. Cached ``teacher_*``
        tensors are copied to the result for the hybrid loss module.
        """
        tokenized_data = _remove_synthesized_collation_masks(batch["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        labels = batch.get("labels", tokenized_data.pop("labels", None))
        input_ids = self.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": batch["ego_history_xyz"],
                "ego_history_rot": batch["ego_history_rot"],
            },
        )

        need_hidden_states = bool(batch.get("return_hidden_states", True))
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            **tokenized_data,
            use_cache=True,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        student_hidden = (
            vlm_outputs.hidden_states[-1]
            if need_hidden_states and vlm_outputs.hidden_states
            else None
        )
        # Only the final hidden state participates in distillation. Drop the
        # output tuple's references to every intermediate layer immediately.
        vlm_outputs.hidden_states = None
        prompt_cache = vlm_outputs.past_key_values
        cache_seq_len = prompt_cache.get_seq_length()
        batch_size = input_ids.shape[0]
        if cache_seq_len != input_ids.shape[1]:
            logger.warning(
                "distillation training: regenerated VLM cache length does not match "
                f"the sequence ({cache_seq_len} != {input_ids.shape[1]}); "
                "clamping expert context"
            )
            cache_seq_len = min(cache_seq_len, input_ids.shape[1])
            if hasattr(prompt_cache, "crop"):
                prompt_cache.crop(cache_seq_len)

        attention_mask_1d = tokenized_data.get("attention_mask")
        if attention_mask_1d is None:
            attention_mask_1d = torch.ones_like(input_ids)
        future_start_id = self.special_token_ids["traj_future_start"]
        prompt_lengths = batch.get("_teacher_prompt_length")
        search_start = (
            prompt_lengths.to(device=input_ids.device, dtype=torch.long)
            if isinstance(prompt_lengths, torch.Tensor)
            else 0
        )
        future_start_offsets = find_future_start_offsets(
            input_ids,
            future_start_id,
            search_start=search_start,
            cache_seq_len=cache_seq_len,
            allow_missing=False,
            require_trailing_token=True,
            context="distillation training",
        )
        rope_deltas = getattr(vlm_outputs, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = getattr(self.vlm.model, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = torch.zeros((batch_size, 1), dtype=torch.long, device=input_ids.device)
        vlm_outputs.rope_deltas = rope_deltas

        future_traj_vlm_loss = None
        if bool(batch.get("compute_future_traj_vlm_loss", False)):
            gt_future_suffix = self._build_gt_future_traj_token_suffix(
                {
                    "ego_history_xyz": batch["ego_history_xyz"],
                    "ego_history_rot": batch["ego_history_rot"],
                    "ego_future_xyz": batch["ego_future_xyz"],
                    "ego_future_rot": batch["ego_future_rot"],
                },
                device=input_ids.device,
            )
            future_traj_vlm_loss = self._compute_future_traj_vlm_loss_from_prefix(
                vlm_outputs,
                input_ids,
                attention_mask_1d,
                future_start_offsets,
                gt_future_suffix,
                tokenized_data,
            )

        language_losses: dict[str, torch.Tensor] = {}
        if language_loss_fn is not None:
            language_losses = language_loss_fn(
                {"student_logits": vlm_outputs.logits},
                batch,
            )
            # The language losses now own everything required for backward.
            # Release full-vocabulary outputs and frozen teacher targets before
            # constructing action-expert activations.
            vlm_outputs.logits = None
            for key in (
                "teacher_logits",
                "teacher_topk_indices",
                "teacher_topk_logits",
            ):
                batch.pop(key, None)

        if self.config.stop_action_grad_to_vlm:
            prompt_cache = self._detach_kv_cache(prompt_cache)

        if "target_action" in batch:
            target_action = batch["target_action"]
        else:
            target_action = self.action_space.traj_to_action(
                batch["ego_history_xyz"][:, -1],
                batch["ego_history_rot"][:, -1],
                batch["ego_future_xyz"][:, -1],
                batch["ego_future_rot"][:, -1],
            )
        target_action = target_action.to(device=input_ids.device, dtype=self.expert.dtype)

        flow_noise = batch.get("flow_noise")
        if flow_noise is not None:
            flow_noise = flow_noise.to(device=target_action.device, dtype=target_action.dtype)
        flow_timestep = batch.get("flow_timestep")
        if flow_timestep is not None:
            flow_timestep = flow_timestep.to(device=target_action.device).reshape(batch_size)
        noisy_action, flow_timestep, flow_target = self.diffusion.sample_training_pair(
            target_action,
            noise=flow_noise,
            timesteps=flow_timestep,
        )

        n_action_tokens = target_action.shape[-2]
        position_ids = build_expert_position_ids(
            future_start_offsets,
            rope_deltas,
            query_seq_len=n_action_tokens,
            device=input_ids.device,
        )
        expert_attention_mask = build_expert_attention_mask(
            future_start_offsets.to(input_ids.device),
            cache_seq_len=cache_seq_len,
            query_seq_len=n_action_tokens,
            attention_implementation=(
                getattr(self.expert.config, "_attn_implementation", None)
                or self.config.attn_implementation
            ),
            cache_valid_mask=attention_mask_1d[:, :cache_seq_len],
            dtype=self.expert.dtype,
        )

        future_token_embeds = self.action_in_proj(
            noisy_action,
            flow_timestep.view(batch_size, 1, 1),
        )
        expert_forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            expert_forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=expert_attention_mask,
            use_cache=True,
            output_hidden_states=need_hidden_states,
            return_dict=True,
            **expert_forward_kwargs,
        )
        prompt_cache.crop(cache_seq_len)
        action_hidden = expert_outputs.last_hidden_state[:, -n_action_tokens:]
        student_vector = self.action_out_proj(action_hidden).reshape_as(target_action)

        result = {
            **language_losses,
            "student_vector": student_vector,
            "flow_target": flow_target,
            "target_action": target_action,
            "flow_noise": noisy_action - flow_timestep.view(batch_size, 1, 1) * flow_target,
            "flow_timestep": flow_timestep,
            "student_action_hidden": action_hidden,
        }
        if language_loss_fn is None:
            result["student_logits"] = vlm_outputs.logits
        if labels is not None:
            result["labels"] = labels
        if student_hidden is not None:
            result["student_hidden"] = student_hidden
        if future_traj_vlm_loss is not None:
            result["future_traj_vlm_loss"] = future_traj_vlm_loss
        for key, value in batch.items():
            if key.startswith("teacher_") and isinstance(value, torch.Tensor):
                result[key] = value
        return result

    def forward(
        self,
        batch: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Dispatch the trainable forward through the module wrapper.

        Calling ``model(batch)`` instead of a custom method is required for
        DDP/FSDP forward hooks and gradient synchronization.
        """
        if batch is None:
            batch = kwargs
        elif kwargs:
            raise ValueError("Pass either a batch dictionary or keyword inputs, not both")
        return self.forward_distill(batch)

    def sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        allow_missing_future_start: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories from the data with VLM rollout.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            allow_missing_future_start: If true, use the end of the available
                KV cache when the VLM does not emit the future-start marker.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) run autoregressive generation for the VLM
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = copy.deepcopy(self.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        do_sample = bool(kwargs.get("do_sample", True))
        generation_config.do_sample = do_sample
        generation_config.num_beams = 1
        generation_config.num_return_sequences = n_samples_total
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        generation_config.max_new_tokens = max_generation_length + 1
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # use custom stopping criteria to stop after EOS token + one more token,
        # because the KV cache is updated after the next token is generated
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        generation_inputs = dict(tokenized_data)
        generation_inputs.pop("input_ids", None)
        generation_inputs.update(
            do_sample=do_sample,
            num_beams=1,
            num_return_sequences=n_samples_total,
            max_new_tokens=max_generation_length + 1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None,
            forced_eos_token_id=None,
            output_logits=False,
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **generation_inputs,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        offset = find_future_start_offsets(
            vlm_outputs.sequences,
            eos_token_id,
            search_start=input_ids.shape[1],
            cache_seq_len=prefill_seq_len,
            require_trailing_token=True,
            allow_missing=allow_missing_future_start,
            context="student VLM rollout",
        )
        prefix_mask = generation_inputs.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        cache_valid_mask = extend_prompt_attention_mask(
            prefix_mask,
            batch_size=b_star,
            prompt_seq_len=input_ids.shape[1],
            total_seq_len=prefill_seq_len,
            device=device,
        )
        position_ids = build_expert_position_ids(
            offset,
            vlm_outputs.rope_deltas,
            query_seq_len=n_diffusion_tokens,
            device=device,
        )
        attention_mask = build_expert_attention_mask(
            offset.to(device),
            cache_seq_len=prefill_seq_len,
            query_seq_len=n_diffusion_tokens,
            attention_implementation=(
                getattr(self.expert.config, "_attn_implementation", None)
                or self.config.attn_implementation
            ),
            cache_valid_mask=cache_valid_mask,
            dtype=self.expert.dtype,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # Run expert with cached prefill, only on the future tokens
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # Repeat history to align with num_traj_samples
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # return the text tokens generated by the VLM
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot

    @torch.no_grad()
    def sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample trajectories from the data with VLM rollout.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        data = copy.deepcopy(data)
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # 1) run autoregressive generation for the VLM
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = copy.deepcopy(self.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        do_sample = bool(kwargs.get("do_sample", True))
        generation_config.do_sample = do_sample
        generation_config.num_beams = 1
        generation_config.num_return_sequences = n_samples_total
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        generation_config.max_new_tokens = max_generation_length + 1
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # use custom stopping criteria to stop after EOS token + one more token,
        # because the KV cache is updated after the next token is generated
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        generation_inputs = dict(tokenized_data)
        generation_inputs.pop("input_ids", None)
        generation_inputs.update(
            do_sample=do_sample,
            num_beams=1,
            num_return_sequences=n_samples_total,
            max_new_tokens=max_generation_length + 1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None,
            forced_eos_token_id=None,
            output_logits=False,
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **generation_inputs,
        )
        # Generation logits are disabled; only sequences and the KV cache are retained.
        torch.cuda.empty_cache()
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prompt_cache_seq_len = prompt_cache.get_seq_length()

        b_star = vlm_outputs.sequences.shape[0]
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        offset = find_future_start_offsets(
            vlm_outputs.sequences,
            eos_token_id,
            search_start=input_ids.shape[1],
            cache_seq_len=prompt_cache_seq_len,
            require_trailing_token=True,
            context="guided student VLM rollout",
        )
        prefix_mask = generation_inputs.get("attention_mask")
        if prefix_mask is not None:
            prefix_mask = torch.repeat_interleave(prefix_mask, n_samples_total, dim=0)
        cache_valid_mask = extend_prompt_attention_mask(
            prefix_mask,
            batch_size=b_star,
            prompt_seq_len=input_ids.shape[1],
            total_seq_len=prompt_cache_seq_len,
            device=device,
        )
        position_ids = build_expert_position_ids(
            offset,
            vlm_outputs.rope_deltas,
            query_seq_len=n_diffusion_tokens,
            device=device,
        )
        attention_mask = build_expert_attention_mask(
            offset.to(device),
            cache_seq_len=prompt_cache_seq_len,
            query_seq_len=n_diffusion_tokens,
            attention_implementation=(
                getattr(self.expert.config, "_attn_implementation", None)
                or self.config.attn_implementation
            ),
            cache_valid_mask=cache_valid_mask,
            dtype=self.expert.dtype,
        )

        # 2) construct unguided kv cache
        # Build unguided input_ids by removing <|route_start|>...<|route_end|> span
        unguided_input_ids = []
        for i in range(input_ids.shape[0]):
            unguided_input_ids.append(remove_nav_text(input_ids, self.tokenizer, i)[0])
        unguided_input_ids = torch.nn.utils.rnn.pad_sequence(
            unguided_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="left",
        ).to(device)
        unguided_prefix_mask = unguided_input_ids.ne(self.tokenizer.pad_token_id).long()

        # Step 1: Prefill unguided prefix ONCE with original batch (B samples).
        # Vision encoder runs only once — no pixel_values repetition needed.
        unguided_prefill_outputs = self.vlm(
            input_ids=unguided_input_ids,
            attention_mask=unguided_prefix_mask,
            image_grid_thw=tokenized_data.get("image_grid_thw"),
            pixel_values=tokenized_data.get("pixel_values"),
            use_cache=True,
            logits_to_keep=1,
        )

        # Step 2: Repeat KV cache for n_samples_total (cheap memory copy, no recomputation)
        # Free the prefill outputs first — we only need the KV cache, not the logits
        unguided_prompt_cache = unguided_prefill_outputs.past_key_values
        del unguided_prefill_outputs
        torch.cuda.empty_cache()
        unguided_prompt_cache.batch_repeat_interleave(n_samples_total)

        # Step 3: Forward generated_tokens (which differ per sample) using the repeated
        # KV cache. No pixel_values needed — images are already encoded in the cache.
        generated_tokens = vlm_outputs.sequences[:, input_ids.shape[1] :]
        unguided_prefix_len = unguided_input_ids.shape[1]
        gen_len = generated_tokens.shape[1]

        prefix_mask_repeated = unguided_prefix_mask.repeat_interleave(n_samples_total, dim=0)
        gen_mask = generated_tokens.ne(self.tokenizer.pad_token_id).long()
        full_attention_mask = torch.cat([prefix_mask_repeated, gen_mask], dim=1)

        cache_position = torch.arange(
            unguided_prefix_len,
            unguided_prefix_len + gen_len,
            device=device,
            dtype=torch.long,
        )

        unguided_vlm_outputs = self.vlm(
            input_ids=generated_tokens,
            attention_mask=full_attention_mask,
            past_key_values=unguided_prompt_cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        unguided_prompt_cache = unguided_vlm_outputs.past_key_values
        unguided_rope_deltas = getattr(unguided_vlm_outputs, "rope_deltas", None)
        if unguided_rope_deltas is None:
            unguided_rope_deltas = getattr(self.vlm.model, "rope_deltas", None)
        if unguided_rope_deltas is None:
            unguided_rope_deltas = torch.zeros((b_star, 1), dtype=torch.long, device=device)
        del unguided_vlm_outputs.logits
        torch.cuda.empty_cache()

        full_unguided_tokens = torch.cat(
            [torch.repeat_interleave(unguided_input_ids, n_samples_total, dim=0), generated_tokens],
            dim=1,
        )
        unguided_cache_seq_len = unguided_prompt_cache.get_seq_length()
        unguided_offset = find_future_start_offsets(
            full_unguided_tokens,
            eos_token_id,
            search_start=unguided_prefix_len,
            cache_seq_len=unguided_cache_seq_len,
            require_trailing_token=True,
            context="unguided student VLM rollout",
        )
        unguided_position_ids = build_expert_position_ids(
            unguided_offset,
            unguided_rope_deltas,
            query_seq_len=n_diffusion_tokens,
            device=device,
        )
        unguided_attention_mask = build_expert_attention_mask(
            unguided_offset.to(device),
            cache_seq_len=unguided_cache_seq_len,
            query_seq_len=n_diffusion_tokens,
            attention_implementation=(
                getattr(self.expert.config, "_attn_implementation", None)
                or self.config.attn_implementation
            ),
            cache_valid_mask=full_attention_mask[:, :unguided_cache_seq_len],
            dtype=self.expert.dtype,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 3) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            position_ids: torch.Tensor,
            past_key_values: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # Run expert with cached prefill, only on the future tokens
            prefill_seq_len = past_key_values.get_seq_length()
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            past_key_values.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 4) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=partial(
                step_fn,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ),
            unguided_step_fn=partial(
                step_fn,
                past_key_values=unguided_prompt_cache,
                attention_mask=unguided_attention_mask,
                position_ids=unguided_position_ids,
            ),
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # Repeat history to align with num_traj_samples
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 5) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # return the text tokens generated by the VLM
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot


AutoConfig.register("alpamayo1_5_distill", DistillAlpamayo1_5Config)
AutoModel.register(DistillAlpamayo1_5Config, DistillAlpamayo1_5)
