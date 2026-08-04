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

"""RL-specific ExpertModel subclass with SDE log_prob support.

Extends ExpertModel with:
- SDE mode for diffusion sampling (return_all_steps, log_prob computation)
- VLM prefill with gradient checkpointing toggle for RL training
- cfm_logprob_sde for GRPO policy gradient computation
- Parallel denoise function for efficient multi-timestep Expert forward pass
"""

from __future__ import annotations

import copy
from typing import Any

import einops
import numpy as np
import torch
from alpamayo_r1.common.logging import RankedLogger
from alpamayo_r1.models.base_model import SPECIAL_TOKENS
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
)
from alpamayo1_x_rl.models.expert_model.expert_model import ExpertModel
from alpamayo1_x_rl.utils.logit_processor import MaskDiscreteTrajectoryLogitsProcessor
from transformers import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

logger = RankedLogger(__name__, rank_zero_only=True)


class ExpertModelRL(ExpertModel):
    """ExpertModel with RL/SDE extensions for GRPO training and rollout.

    Adds SDE-aware sampling (with log_prob, diffusion path tracking),
    cfm_logprob_sde for policy training, and VLM prefill utilities that
    handle gradient checkpointing and generated-token replay.
    """

    # ------------------------------------------------------------------
    # VLM prefill utilities (shared by sampling and cfm_logprob_sde)
    # ------------------------------------------------------------------

    def _vlm_prefill(
        self,
        data: dict[str, Any],
    ) -> tuple[Any, Any, torch.Tensor | None, int]:
        """Run VLM prefill and build action history embeddings.

        Extracts common prefill logic shared by sampling and cfm_logprob_sde.
        Uses a deep copy of tokenized_data to avoid destroying the caller's dict.

        Returns:
            prompt_cache: DynamicCache, VLM KV cache.
            vlm_outputs: VLM model outputs (carries rope_deltas).
            action_history_embeds: [B, n_hist, hidden] or None.
            prefill_seq_len: int, number of VLM prefill tokens.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]

        tokenized_data = copy.deepcopy(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        tokenized_data.pop("attention_mask", None)
        tokenized_data.pop("text", None)

        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)

        vlm_outputs = self._run_vlm_prefill_with_cache(
            input_ids=input_ids,
            tokenized_data=tokenized_data,
        )
        prompt_cache = vlm_outputs.past_key_values
        if prompt_cache is None:
            raise RuntimeError(
                "VLM prefill returned no past_key_values. "
                "This path requires KV cache for cfm_logprob_sde/get_parallel_denoise_func. "
                "If gradient checkpointing is enabled on the VLM LM, disable it for prefill "
                "or keep the temporary toggle in _run_vlm_prefill_with_cache."
            )
        prefill_seq_len = input_ids.shape[1]

        if self.expert_hist_traj_tokenizer is not None:
            action_history_embeds = self.expert_hist_traj_tokenizer(
                xyz=ego_history_xyz.squeeze(1),
                rot=ego_history_rot.squeeze(1),
            ).unsqueeze(1)
        else:
            action_history_embeds = None

        return prompt_cache, vlm_outputs, action_history_embeds, prefill_seq_len

    def _run_vlm_prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        tokenized_data: dict[str, Any],
    ) -> Any:
        """Run VLM prefill with ``use_cache=True``, handling GC/cache incompatibility.

        HF decoder stacks disable ``use_cache`` during training when gradient
        checkpointing is enabled, which would make ``past_key_values=None``.
        This helper temporarily disables LM checkpointing for the prefill call,
        then restores it.
        """
        vlm_model = self.vlm.model
        lm = getattr(vlm_model, "language_model", None) or getattr(vlm_model, "model", None)

        gc_toggled = False
        if (
            self.training
            and lm is not None
            and getattr(lm, "is_gradient_checkpointing", False)
            and hasattr(lm, "gradient_checkpointing_disable")
        ):
            lm.gradient_checkpointing_disable()
            gc_toggled = True

        try:
            return vlm_model(
                input_ids=input_ids,
                **tokenized_data,
                use_cache=True,
            )
        finally:
            if gc_toggled and lm is not None and hasattr(lm, "gradient_checkpointing_enable"):
                try:
                    lm.gradient_checkpointing_enable({"use_reentrant": False})
                except TypeError:
                    lm.gradient_checkpointing_enable()

    def _vlm_prefill_with_generated_ids(
        self,
        data: dict[str, Any],
        vlm_generated_ids: torch.Tensor,
    ) -> tuple[Any, Any, torch.Tensor | None, int]:
        """Run VLM prefill with the rollout's generated token IDs appended.

        Reconstructs the full sequence that the rollout VLM saw:
        [fused_prompt (without trailing <traj_future_start>)] + [vlm_generated_ids]

        vlm_generated_ids is the raw VLM output after the prompt, which already
        includes <cot_start>, CoT content, <cot_end>, and <traj_future_start>.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]

        tokenized_data = copy.deepcopy(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        tokenized_data.pop("attention_mask", None)
        tokenized_data.pop("text", None)

        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)

        future_start_id = self.config.traj_token_ids["future_start"]
        if input_ids[0, -1] == future_start_id:
            input_ids = input_ids[:, :-1]

        input_ids = torch.cat([input_ids, vlm_generated_ids], dim=1)

        vlm_outputs = self._run_vlm_prefill_with_cache(
            input_ids=input_ids,
            tokenized_data=tokenized_data,
        )
        prompt_cache = vlm_outputs.past_key_values
        if prompt_cache is None:
            raise RuntimeError(
                "VLM prefill with generated IDs returned no past_key_values. "
                "This path requires KV cache for cfm_logprob_sde/get_parallel_denoise_func."
            )
        prefill_seq_len = input_ids.shape[1]

        if self.expert_hist_traj_tokenizer is not None:
            action_history_embeds = self.expert_hist_traj_tokenizer(
                xyz=ego_history_xyz.squeeze(1),
                rot=ego_history_rot.squeeze(1),
            ).unsqueeze(1)
        else:
            action_history_embeds = None

        return prompt_cache, vlm_outputs, action_history_embeds, prefill_seq_len

    # ------------------------------------------------------------------
    # Parallel denoise for cfm_logprob_sde
    # ------------------------------------------------------------------

    def get_parallel_denoise_func(
        self,
        prompt_cache: Any,
        vlm_outputs: Any,
        action_history_embeds: torch.Tensor | None,
        n_traj_chunks: int,
    ) -> Any:
        """Build a closure that denoises all timestep chunks in a single Expert forward pass.

        Used by cfm_logprob_sde() to replay a saved diffusion path under the current policy.

        Args:
            prompt_cache: VLM KV cache from prefill.
            vlm_outputs: VLM outputs (carries rope_deltas).
            action_history_embeds: [B, n_hist, hidden] or None.
            n_traj_chunks: Number of timestep chunks (= T diffusion steps).

        Returns:
            parallel_denoise: callable
                (x [B, n_chunks, Tf, D], t [B, n_chunks]) -> [B, n_chunks, Tf, D]
        """
        prefill_seq_len = prompt_cache.get_seq_length()
        n_diff = self.action_space.get_action_space_dims()[0]
        n_hist = action_history_embeds.shape[1] if action_history_embeds is not None else 0
        tpc = n_hist + n_diff  # tokens per chunk
        B = vlm_outputs.rope_deltas.shape[0]
        device = vlm_outputs.rope_deltas.device
        action_dims = self.action_space.get_action_space_dims()
        hidden_size = self.expert.config.hidden_size

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # Precompute 3D RoPE position_ids [3, B, n_chunks * tpc]
        chunk_pos = torch.arange(tpc, device=device)
        pos_flat = chunk_pos.repeat(n_traj_chunks)
        position_ids = einops.repeat(pos_flat, "l -> 3 b l", b=B).clone()
        delta = vlm_outputs.rope_deltas + prefill_seq_len
        position_ids += delta.to(position_ids.device)

        # Precompute block-diagonal attention mask
        if self.expert.config._attn_implementation != "flash_attention_2":
            _kv_dtype = prompt_cache.layers[0].keys.dtype
            block_diag = torch.kron(
                torch.eye(n_traj_chunks, device=device),
                torch.ones(tpc, tpc, device=device),
            ).to(dtype=_kv_dtype)
            block_diag.masked_fill_(block_diag == 0, float("-inf"))
            block_diag.masked_fill_(block_diag == 1, 0.0)
            attention_mask = block_diag[None, None].expand(B, 1, -1, -1)
            left = torch.zeros(
                B,
                1,
                n_traj_chunks * tpc,
                prefill_seq_len,
                device=device,
                dtype=_kv_dtype,
            )
            attention_mask = torch.cat([left, attention_mask], dim=3)
        else:
            attention_mask = None

        if action_history_embeds is not None:
            hist_tiled = action_history_embeds.unsqueeze(1).expand(-1, n_traj_chunks, -1, -1)
        else:
            hist_tiled = None

        model_dtype = next(self.action_in_proj.parameters()).dtype

        def parallel_denoise(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            Bx = x.shape[0]
            x = x.to(dtype=model_dtype)
            t = t.to(dtype=model_dtype)

            if t.ndim == 1:
                t = t.view(1, -1, 1, 1).expand(Bx, -1, 1, 1)
            elif t.ndim == 2:
                t = t.unsqueeze(-1).unsqueeze(-1)

            embeds = self.action_in_proj(x.flatten(0, 1), t.flatten(0, 1)).reshape(
                Bx, n_traj_chunks, n_diff, hidden_size
            )

            if hist_tiled is not None:
                embeds = torch.cat([hist_tiled[:Bx], embeds], dim=2)

            embeds = embeds.flatten(1, 2)

            out = self.expert(
                inputs_embeds=embeds,
                position_ids=position_ids[:, :Bx],
                past_key_values=prompt_cache,
                use_cache=False,
                attention_mask=(attention_mask[:Bx] if attention_mask is not None else None),
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)

            hidden_out = out.last_hidden_state.reshape(Bx, n_traj_chunks, tpc, -1)
            hidden_out = hidden_out[:, :, -n_diff:]

            pred = self.action_out_proj(hidden_out).reshape(Bx, n_traj_chunks, *action_dims)
            return pred

        return parallel_denoise

    # ------------------------------------------------------------------
    # cfm_logprob_sde (GRPO policy training)
    # ------------------------------------------------------------------

    def cfm_logprob_sde(
        self,
        data: dict[str, Any],
        samples_list: torch.Tensor,
        timesteps: torch.Tensor,
        noise_level: float = 0.7,
        teacher: ExpertModelRL | ExpertModel | None = None,
        vlm_generated_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Recompute log_prob for a saved diffusion path under the current policy.

        Used during GRPO training to get new_logprob for the importance ratio.

        Args:
            data: Input data dict (tokenized_data, ego_history_xyz/rot, etc.)
            samples_list: [B, T+1, Tf, D] full diffusion path from rollout.
            timesteps: [B, T+1] time grid.
            noise_level: SDE noise level (must match rollout).
            teacher: Optional reference model for KL divergence.
            vlm_generated_ids: [B, L_gen] VLM-generated token IDs from rollout.

        Returns:
            log_prob: [B] scalar per sample.
            kl_div: [B] or None (KL divergence from teacher).
        """
        assert hasattr(self.diffusion, "_batched_sde_logprob"), (
            "SDE log_prob requires a FlowMatching-style diffusion class exposing "
            "`_batched_sde_logprob`; the configured diffusion class "
            f"({type(self.diffusion).__module__}.{type(self.diffusion).__qualname__}) "
            "does not provide it."
        )

        if vlm_generated_ids is not None:
            prompt_cache, vlm_outputs, action_history_embeds, prefill_seq_len = (
                self._vlm_prefill_with_generated_ids(data, vlm_generated_ids)
            )
        else:
            prompt_cache, vlm_outputs, action_history_embeds, prefill_seq_len = self._vlm_prefill(
                data
            )

        n_traj_chunks = samples_list.shape[1] - 1

        denoise_fn = self.get_parallel_denoise_func(
            prompt_cache,
            vlm_outputs,
            action_history_embeds,
            n_traj_chunks,
        )
        model_output = denoise_fn(samples_list[:, :-1], timesteps[:, :-1])

        ref_model_output = None
        if teacher is not None:
            teacher_denoise = teacher.get_parallel_denoise_func(
                prompt_cache,
                vlm_outputs,
                action_history_embeds,
                n_traj_chunks,
            )
            ref_model_output = teacher_denoise(samples_list[:, :-1], timesteps[:, :-1])

        log_prob, kl_div = self.diffusion._batched_sde_logprob(
            model_output,
            timesteps,
            samples_list,
            ref_model_output,
            noise_level=noise_level,
        )
        return log_prob, kl_div

    # ------------------------------------------------------------------
    # Override sampling methods with SDE support
    # ------------------------------------------------------------------

    def _sample_trajectories_from_data_without_vlm_rollout(
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
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories without VLM autoregressive rollout (SDE-aware)."""
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."

        prompt_cache, vlm_outputs, action_history_embeds, prefill_seq_len = self._vlm_prefill(data)
        device = prompt_cache.layers[0].keys.device
        _attn_dtype = prompt_cache.layers[0].keys.dtype

        prompt_cache.batch_repeat_interleave(n_samples_total)
        if action_history_embeds is not None:
            action_history_embeds = torch.repeat_interleave(
                action_history_embeds, n_samples_total, dim=0
            )
        vlm_outputs.rope_deltas = torch.repeat_interleave(
            vlm_outputs.rope_deltas, n_samples_total, dim=0
        )
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        if action_history_embeds is not None:
            position_ids = self._process_position_ids_qwen2_5_vl(
                vlm_outputs,
                n_samples_total * B,
                action_history_embeds.shape[1] + n_diffusion_tokens,
                action_history_embeds.device,
            )
            attention_mask = torch.zeros(
                (
                    n_samples_total * B,
                    1,
                    n_diffusion_tokens,
                    prompt_cache.get_seq_length()
                    + action_history_embeds.shape[1]
                    + n_diffusion_tokens,
                ),
                dtype=_attn_dtype,
                device=device,
            )
        else:
            position_ids = self._process_position_ids_qwen2_5_vl(
                vlm_outputs,
                n_samples_total * B,
                n_diffusion_tokens,
                device,
            )
            attention_mask = torch.zeros(
                (
                    n_samples_total * B,
                    1,
                    n_diffusion_tokens,
                    prompt_cache.get_seq_length() + n_diffusion_tokens,
                ),
                dtype=_attn_dtype,
                device=device,
            )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            b_star = x.shape[0]
            _mdtype = next(self.action_in_proj.parameters()).dtype
            future_token_embeds = self.action_in_proj(x.to(dtype=_mdtype), t.to(dtype=_mdtype))
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            future_token_embeds = (
                torch.cat([action_history_embeds, future_token_embeds], dim=1)
                if action_history_embeds is not None
                else future_token_embeds
            )

            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )
            return pred

        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        use_sde = diffusion_kwargs.get("int_method") == "sde" and diffusion_kwargs.get(
            "return_info", False
        )

        if use_sde:
            result = self.diffusion.sample(
                batch_size=total_batch,
                step_fn=step_fn,
                device=device,
                return_all_steps=True,
                **diffusion_kwargs,
            )
            sampled_action = result["x"]
            sde_log_prob = result["log_prob"]
            sde_info = {
                "samples_list": result["all_steps"],
                "timesteps": result["timesteps"],
                "noise_level": diffusion_kwargs.get("noise_level", 0.7),
            }
        else:
            sampled_action = self.diffusion.sample(
                batch_size=total_batch,
                step_fn=step_fn,
                device=device,
                return_all_steps=False,
                **diffusion_kwargs,
            )
            sde_log_prob = None
            sde_info = None

        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        if use_sde:
            logprob = sde_log_prob.reshape(B, num_traj_sets, num_traj_samples)
            assert sde_info is not None
            return pred_xyz, pred_rot, logprob, sde_info
        else:
            logprob = torch.zeros_like(pred_xyz[..., 0])
            return pred_xyz, pred_rot, logprob

    def _sample_trajectories_from_data_with_vlm_rollout(
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
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories with VLM autoregressive rollout (SDE-aware)."""
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."

        tokenized_data = copy.deepcopy(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        tokenized_data.pop("text", None)
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        logits_processor = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )

        eos_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future_start"])
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        if self.expert_hist_traj_tokenizer is not None:
            action_history_embeds = self.expert_hist_traj_tokenizer(
                xyz=ego_history_xyz.squeeze(1),
                rot=ego_history_rot.squeeze(1),
            ).unsqueeze(1)
        else:
            action_history_embeds = None

        if action_history_embeds is not None:
            action_history_embeds = torch.repeat_interleave(
                action_history_embeds, n_samples_total, dim=0
            )

        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        for i in range(b_star):
            if not has_traj_future_start[i]:
                logger.warning(
                    f"No <traj_future_start> token found in the generated sequences for seq {i}."
                    f"sequences: {self.tokenizer.decode(vlm_outputs.sequences[i])}"
                )
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        offset = valid_token_pos_id + 1

        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
            dtype=prompt_cache.layers[0].keys.dtype,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            b_star = x.shape[0]
            _mdtype = next(self.action_in_proj.parameters()).dtype
            future_token_embeds = self.action_in_proj(x.to(dtype=_mdtype), t.to(dtype=_mdtype))
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            future_token_embeds = (
                torch.cat([action_history_embeds, future_token_embeds], dim=1)
                if action_history_embeds is not None
                else future_token_embeds
            )

            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )
            return pred

        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        use_sde = diffusion_kwargs.get("int_method") == "sde" and diffusion_kwargs.get(
            "return_info", False
        )

        if use_sde:
            result = self.diffusion.sample(
                batch_size=total_batch,
                step_fn=step_fn,
                device=device,
                return_all_steps=True,
                **diffusion_kwargs,
            )
            sampled_action = result["x"]
            sde_log_prob = result["log_prob"]
            sde_info = {
                "samples_list": result["all_steps"],
                "timesteps": result["timesteps"],
                "noise_level": diffusion_kwargs.get("noise_level", 0.7),
            }
        else:
            sampled_action = self.diffusion.sample(
                batch_size=total_batch,
                step_fn=step_fn,
                device=device,
                return_all_steps=False,
                **diffusion_kwargs,
            )
            sde_log_prob = None
            sde_info = None

        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        if use_sde:
            logprob = sde_log_prob.reshape(B, num_traj_sets, num_traj_samples)
            assert sde_info is not None
            if kwargs.get("return_extra", False):
                extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
                for text_tokens in extra.keys():
                    extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                        [input_ids.shape[0], num_traj_sets, num_traj_samples]
                    )
                sde_info.update(extra)

                prompt_len = input_ids.shape[0 if input_ids.dim() == 1 else 1]
                generated_seqs = vlm_outputs.sequences
                logger.info(
                    f"[vlm_gen_ids] prompt_len={prompt_len}, "
                    f"sequences_shape={generated_seqs.shape}, "
                    f"gen_len={generated_seqs.shape[1] - prompt_len}"
                )
                vlm_gen_ids_list = []
                pad_id = self.tokenizer.pad_token_id or 0
                for i in range(generated_seqs.shape[0]):
                    gen = generated_seqs[i, prompt_len:]
                    non_pad = gen != pad_id
                    if non_pad.any():
                        last = non_pad.nonzero(as_tuple=True)[0][-1].item() + 1
                        vlm_gen_ids_list.append(gen[:last].cpu())
                    else:
                        vlm_gen_ids_list.append(torch.tensor([], dtype=torch.long))

                sde_info["vlm_generated_ids_list"] = vlm_gen_ids_list

            return pred_xyz, pred_rot, logprob, sde_info
        else:
            logprob = torch.zeros_like(pred_xyz[..., 0])
            if kwargs.get("return_extra", False):
                extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
                for text_tokens in extra.keys():
                    extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                        [input_ids.shape[0], num_traj_sets, num_traj_samples]
                    )
                return pred_xyz, pred_rot, logprob, extra
            return pred_xyz, pred_rot, logprob
