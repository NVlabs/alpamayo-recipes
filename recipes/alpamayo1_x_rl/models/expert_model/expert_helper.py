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

import logging

import einops
import torch

logger = logging.getLogger(__name__)


def find_eos_offset(
    sequences: torch.Tensor,
    eos_token_id: int,
    device: torch.device,
    warn: bool = True,
) -> torch.Tensor:
    """Find the first eos_token_id position in each sequence and return offset = pos + 1.

    Falls back to the last token position when eos_token_id is not found.
    The returned offset marks the boundary between VLM-generated tokens and
    the region where expert diffusion tokens will be appended.
    """
    b_star = sequences.shape[0]
    mask = sequences == eos_token_id
    has_eos = mask.any(dim=1)  # [b_star]
    if warn:
        for i in range(b_star):
            if not has_eos[i]:
                logger.warning(
                    f"No <traj_future_start> token found in generated sequences for sequence {i}"
                )
    eos_positions = mask.int().argmax(dim=1)  # [b_star], first occurrence
    last_positions = torch.full((b_star,), sequences.shape[1] - 1, device=device)
    return torch.where(has_eos, eos_positions, last_positions) + 1


def build_expert_pos_ids_and_attn_mask(
    offset: torch.Tensor,
    rope_deltas: torch.Tensor,
    kv_cache_seq_len: int,
    n_diffusion_tokens: int,
    b_star: int,
    device: torch.device,
    prefix_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build position IDs and 4D attention mask for the expert denoiser.

    Args:
        offset: [b_star] — token position right after <traj_future_start>.
        rope_deltas: [b_star, 1] — RoPE delta from the VLM.
        kv_cache_seq_len: sequence length already in the KV cache.
        n_diffusion_tokens: number of expert diffusion tokens to append.
        b_star: batch size (B * num_return_sequences).
        device: torch device.
        prefix_mask: [b_star, L] optional 1D attention mask (already repeated
            to match b_star); zeros mark padding positions that should be
            masked in the expert's cross-attention to the KV cache.

    Returns:
        position_ids: [3, b_star, n_diffusion_tokens] — Qwen2.5-VL RoPE ids.
        attention_mask: [b_star, 1, n_diffusion_tokens, KV] — 4D float mask
            (0 = attend, -inf = masked).
    """
    # Qwen2.5-VL uses 3-component (temporal, height, width) RoPE
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
    position_ids += (rope_deltas + offset[:, None]).to(position_ids.device)

    # [b_star, H, Q, KV] — mask the gap between offset and diffusion tokens
    attention_mask = torch.zeros(
        (b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens),
        dtype=torch.float32,
        device=device,
    )
    for i in range(b_star):
        attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
            attention_mask.dtype
        ).min

    # Propagate input padding mask (left-padding) into the KV prefix region
    if prefix_mask is not None:
        # [b_star, H, Q, KV]
        input_mask = prefix_mask[:, None, None, :]
        attention_mask[:, :, :, : input_mask.shape[-1]] = torch.where(
            input_mask == 0,
            torch.finfo(attention_mask.dtype).min,
            attention_mask[:, :, :, : input_mask.shape[-1]],
        )

    return position_ids, attention_mask
