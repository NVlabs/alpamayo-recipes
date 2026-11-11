# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal RoPE (M-RoPE) for the Cosmos3-Edge engines, in the exact layout Edge-LLM's kernels produce.

- ``mrope_positions``: the Qwen2-VL ``get_rope_index`` rule as the stock runner implements it
  (``QwenViTRunner::getMRopePositionIds``): text tokens advance all three (t, h, w) components together, an image
  block gets (t, h, w) = block start + (0, row, column) and advances the counter by max(rows, columns), and
  every token after the prompt (generated ones) is text at ``rope_delta + sequence index``.
- ``cos_sin``: the ``initializeMRopeCosSin`` kernel: row = [cos(64) | sin(64)], pair ``i`` (frequency
  theta^(2i/128)) rotates the H component when i % 3 == 1 and i < 3 * section_h, the W component when i % 3 == 2
  and i < 3 * section_w, else T (interleaved mrope_section [24, 20, 20]).
"""

from __future__ import annotations

import torch

HEAD_DIM = 128


def mrope_positions(
    token_ids: list[int], images: list[dict], merge: int = 2
) -> tuple[torch.Tensor, int]:
    """``images``: in prompt order, each with ``token_start`` (first ``<|image_pad|>`` row), ``height``/``width``
    in pixels and ``tokens``. Returns ([3, L] int64 positions, rope_delta) for the prompt."""
    length = len(token_ids)
    pos = torch.empty(3, length, dtype=torch.int64)
    start, start_idx = 0, 0
    for image in images:
        first = image["token_start"]
        rows, cols = image["height"] // (16 * merge), image["width"] // (16 * merge)
        if rows * cols != image["tokens"]:
            raise ValueError(f"image grid {rows}x{cols} does not give {image['tokens']} tokens")
        text_len = first - start  # includes the <|vision_start|> token
        pos[:, start:first] = torch.arange(start_idx, start_idx + text_len)
        base = start_idx + text_len
        pos[0, first : first + rows * cols] = base
        pos[1, first : first + rows * cols] = (
            (base + torch.arange(rows))[:, None].expand(rows, cols).reshape(-1)
        )
        pos[2, first : first + rows * cols] = (
            (base + torch.arange(cols))[None, :].expand(rows, cols).reshape(-1)
        )
        start = first + rows * cols
        start_idx += max(rows, cols) + text_len
    pos[:, start:] = torch.arange(start_idx, start_idx + length - start)
    rope_delta = int(pos.max()) + 1 - length
    return pos, rope_delta


def cos_sin(
    pos: torch.Tensor,
    theta: float,
    section: tuple[int, int, int] = (24, 20, 20),
    head_dim: int = HEAD_DIM,
) -> torch.Tensor:
    """[n, head_dim] float32 for [3, n] positions (float32 arithmetic like the kernel)."""
    half = head_dim // 2
    i = torch.arange(half, dtype=torch.int64, device=pos.device)
    group = torch.zeros(half, dtype=torch.int64, device=pos.device)
    group[(i % 3 == 1) & (i < 3 * section[1])] = 1
    group[(i % 3 == 2) & (i < 3 * section[2])] = 2
    freq = torch.pow(
        torch.tensor(theta, dtype=torch.float32, device=pos.device), (2 * i).float() / head_dim
    )  # [half]
    angle = pos.float()[group] / freq[:, None]  # [half, n]
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=0).t().contiguous()


def text_cos_sin(first: int, count: int, theta: float, **kw) -> torch.Tensor:
    """Rows for text positions first .. first + count - 1 (all three components equal)."""
    pos = torch.arange(first, first + count, dtype=torch.int64)[None].expand(3, -1)
    return cos_sin(pos, theta, **kw)
