# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation-only trajectory tokenization helpers."""

from typing import Any

import einops
import torch


def tokenize_future_trajectory(
    tokenizer: Any, traj_data: dict[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize future trajectories with input shape ``[B, n_traj, T, ...]``."""
    if "ego_future_xyz" not in traj_data:
        raise KeyError("traj_data must contain ego_future_xyz")
    if traj_data["ego_future_xyz"].ndim != 4:
        raise ValueError("ego_future_xyz must be 4D [B, n_traj, T, 3]")

    batch_size = traj_data["ego_future_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    fut_xyz = traj_data["ego_future_xyz"].flatten(start_dim=0, end_dim=1)
    fut_rot = traj_data["ego_future_rot"].flatten(start_dim=0, end_dim=1)
    future_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz,
            hist_rot=hist_rot,
            fut_xyz=fut_xyz,
            fut_rot=fut_rot,
        )
        + start_idx
    )
    return einops.rearrange(future_idx, "(b n_traj) n -> b (n_traj n)", b=batch_size)
