# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-label masking independent of the VLM's conversation vocabulary."""

import torch


def get_trajectory_mask(
    tokens: torch.Tensor,
    start_id: int,
    end_id: int,
    eos_token_id: int | list[int] | None = None,
) -> torch.Tensor:
    """Supervise one future trajectory, its markers, and an immediate closing EOS."""
    starts = (tokens == start_id).nonzero(as_tuple=True)[0]
    ends = (tokens == end_id).nonzero(as_tuple=True)[0]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise ValueError("Training samples require exactly one complete future trajectory")
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    last = int(ends[0])
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    if last + 1 < len(tokens) and int(tokens[last + 1]) in eos_ids:
        last += 1
    mask[starts[0] : last + 1] = True
    return mask
