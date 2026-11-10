# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-only evaluation using the public Edge model's expert sampler."""

from typing import Any

import torch
from alpamayo2_edge.models.expert_utils import build_expert_pos_ids_and_attn_mask

from alpamayo2edge_sft.models.utils import fuse_traj_tokens


@torch.inference_mode()
def sample_anchored_trajectories(
    model,
    data: dict[str, Any],
    top_p: float = 0.98,
    top_k: int | None = None,
    temperature: float = 0.6,
    num_traj_samples: int = 1,
    num_traj_sets: int = 1,
    max_generation_length: int | None = None,
    diffusion_kwargs: dict[str, Any] | None = None,
    return_extra: bool = False,
) -> tuple:
    """Prefill a history-only prompt through its existing future-start anchor.

    The recipe evaluates trajectories without generating reasoning text. Text
    sampling arguments are accepted for the shared evaluator API but have no
    effect here. Diffusion options, including a seeded ``generator``, pass through
    to the public sampler. Inputs can be reused across candidate chunks.
    The third return value is a zero placeholder, not a trajectory log probability;
    it must not be used for candidate ranking or likelihood metrics.
    """
    del top_p, top_k, temperature, max_generation_length
    if model.training:
        raise ValueError("Call model.eval() before sampling trajectories")
    if not model.config.enable_expert:
        raise ValueError("Trajectory inference requires an expert checkpoint")
    for name, value in (("num_traj_sets", num_traj_sets), ("num_traj_samples", num_traj_samples)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    tokenized = dict(data["tokenized_data"])
    input_ids = tokenized["input_ids"]
    prompt_mask = tokenized.get("attention_mask")
    if (
        input_ids.ndim != 2
        or input_ids.shape[1] == 0
        or prompt_mask is None
        or prompt_mask.shape != input_ids.shape
    ):
        raise ValueError("Generation requires input_ids and attention_mask shaped [B, L]")
    anchor = model.config.traj_ids["future_start"]
    if (
        not torch.all(input_ids[:, -1] == anchor)
        or not torch.all(input_ids.eq(anchor).sum(dim=-1) == 1)
        or not torch.all(prompt_mask[:, -1] == 1)
    ):
        raise ValueError("Each generation prompt must end with exactly one future-start token")
    future_ids = (input_ids >= model.config.traj_ids["future_id0"]) & (
        input_ids < model.config.traj_ids["future_id0"] + model.config.future_vocab_size
    )
    if future_ids.any() or input_ids.eq(model.config.traj_ids["future_pad"]).any():
        raise ValueError("Generation prompts must not contain future trajectory targets")

    history = {key: data[key].float() for key in ("ego_history_xyz", "ego_history_rot")}
    tokenized["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        input_ids.clone(),
        history,
        model.config.traj_ids,
    )
    has_vision = any(tokenized.get(key) is not None for key in ("image_grid_thw", "video_grid_thw"))
    if has_vision and "mm_token_type_ids" not in tokenized:
        token_types = torch.zeros_like(input_ids)
        token_types[input_ids == model.config.vlm_config.image_token_id] = 1
        token_types[input_ids == model.config.vlm_config.video_token_id] = 2
        tokenized["mm_token_type_ids"] = token_types
    if not has_vision:
        # Direct HF5 text forwards do not initialize RoPE deltas or account for
        # left padding as generate() does. Also discard deltas from earlier images.
        positions = prompt_mask.long().cumsum(-1) - 1
        tokenized["position_ids"] = positions.masked_fill(prompt_mask == 0, 0)
        model.vlm.model.rope_deltas = input_ids.new_zeros((input_ids.shape[0], 1))
    outputs = model.vlm.model(**tokenized, use_cache=True)
    cache = outputs.past_key_values
    del outputs
    if cache is None:
        raise RuntimeError("The VLM did not return the KV cache needed by the expert")
    rope_deltas = model.vlm.model.rope_deltas
    if rope_deltas is None:
        raise RuntimeError("The VLM did not produce multimodal rotary position offsets")

    batch_size, prefix_length = input_ids.shape
    n_candidates = num_traj_sets * num_traj_samples
    expert_batch = batch_size * n_candidates
    cache.batch_repeat_interleave(n_candidates)
    # HF5 offsets refer to unpadded positions; expert offsets count cache columns.
    rope_deltas = (rope_deltas - prompt_mask.eq(0).sum(dim=-1, keepdim=True)).repeat_interleave(
        n_candidates, dim=0
    )
    prefix_mask = prompt_mask.repeat_interleave(n_candidates, dim=0)
    action_dims = model.expert.action_space.get_action_space_dims()
    position_ids, attention_mask = build_expert_pos_ids_and_attn_mask(
        offset=input_ids.new_full((expert_batch,), prefix_length),
        rope_deltas=rope_deltas,
        kv_cache_seq_len=prefix_length,
        n_diffusion_tokens=action_dims[0],
        b_star=expert_batch,
        device=input_ids.device,
        prefix_mask=prefix_mask,
    )
    if model.expert.expert.config._attn_implementation == "flash_attention_2":
        attention_mask = attention_mask[:, 0, 0].eq(0).to(prefix_mask.dtype)
    sampled_action = model._sample_actions(
        cache,
        prefix_length,
        position_ids,
        attention_mask,
        expert_batch,
        input_ids.device,
        diffusion_kwargs,
    )
    pred_xyz, pred_rot = model.expert.action_space.action_to_traj(
        sampled_action,
        history["ego_history_xyz"][:, -1].repeat_interleave(n_candidates, dim=0),
        history["ego_history_rot"][:, -1].repeat_interleave(n_candidates, dim=0),
    )
    shape = (batch_size, num_traj_sets, num_traj_samples)
    pred_xyz = pred_xyz.reshape(*shape, *pred_xyz.shape[1:])
    pred_rot = pred_rot.reshape(*shape, *pred_rot.shape[1:])
    result = (pred_xyz, pred_rot, torch.zeros_like(pred_xyz[..., 0]))
    if return_extra:
        return (*result, {"generated_ids": input_ids.new_empty((expert_batch, 0))})
    return result
