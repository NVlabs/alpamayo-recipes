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

"""Collision cost helpers for Alpamayo Cosmos-RL.

Returns a **binary** collision indicator (1 if any collision happens, else 0).
"""

from __future__ import annotations

from typing import Any

import torch
from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

from alpamayo.metrics.metric_api import CollisionMetrics


def _as_tensor(x: Any, *, name: str) -> torch.Tensor:
    """Convert to tensor, raising KeyError if None."""
    if x is None:
        raise KeyError(f"Missing required key {name!r} for collision metric.")
    return x if torch.is_tensor(x) else torch.as_tensor(x)


def _ensure_batch_lwh(ego_lwh: Any, *, batch_size: int) -> torch.Tensor:
    """Normalize ego LWH to ``[B, 3]``."""
    ego_lwh_t = _as_tensor(ego_lwh, name="ego_lwh").float()
    if ego_lwh_t.ndim == 1:
        ego_lwh_t = ego_lwh_t.unsqueeze(0)
    if ego_lwh_t.shape[0] == 1 and batch_size > 1:
        ego_lwh_t = ego_lwh_t.expand(batch_size, -1)
    return ego_lwh_t


def _ensure_batch_offset(ego_length_offset: Any, *, batch_size: int) -> torch.Tensor:
    """Normalize ego length offset to ``[B]``."""
    off_t = _as_tensor(ego_length_offset, name="ego_length_offset").float()
    if off_t.ndim == 0:
        off_t = off_t.view(1)
    if off_t.ndim > 1:
        off_t = off_t.reshape(-1)
    if off_t.shape[0] == 1 and batch_size > 1:
        off_t = off_t.expand(batch_size)
    return off_t


def _ensure_batched_sample_tensor(
    x: Any, *, batch_size: int, desired_rank: int, name: str
) -> torch.Tensor:
    """Normalize decoded tensors to ``[B, S, ...]`` with a known rank.

    Typical decoded sample shapes are ``[S, ...]`` (no batch dim). Metrics
    expect ``[B, S, ...]``. If already batched as ``[B, ...]``, we insert
    the sample dimension.
    """
    t = _as_tensor(x, name=name)

    if t.ndim == desired_rank - 1 and t.shape[0] == batch_size:
        t = t.unsqueeze(1)
    else:
        while t.ndim < desired_rank:
            t = t.unsqueeze(0)

    if t.shape[0] == 1 and batch_size > 1:
        t = t.expand(batch_size, *t.shape[1:])

    return t


def compute_collision_cost(
    *,
    reference: dict[str, Any],
    predicted_fut_xyz: torch.Tensor,
    predicted_fut_rot: torch.Tensor,
    timestep_horizons: list[int] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute a binary collision indicator for the predicted trajectory.

    Args:
        reference: Dict with keys ego_lwh, ego_length_offset, ego_history_xyz/rot,
            ego_future_xyz/rot, obstacle_bbox_history/future.
        predicted_fut_xyz: [B, Tf, 3] predicted future positions.
        predicted_fut_rot: [B, Tf, 3, 3] predicted future rotations.
        timestep_horizons: Future timestep indices to evaluate.

    Returns:
        Tuple of (collision_cost, collision_by_t dict).
        collision_cost is 1.0 if any collision, else 0.0.
    """
    collision_by_t: dict[str, float] = {}

    try:
        if not torch.is_tensor(predicted_fut_xyz):
            predicted_fut_xyz = torch.as_tensor(predicted_fut_xyz)
        if not torch.is_tensor(predicted_fut_rot):
            predicted_fut_rot = torch.as_tensor(predicted_fut_rot)

        batch_size = int(predicted_fut_xyz.shape[0])

        data_batch = {
            "ego_lwh": _ensure_batch_lwh(reference.get("ego_lwh"), batch_size=batch_size),
            "ego_length_offset": _ensure_batch_offset(
                reference.get("ego_length_offset"), batch_size=batch_size
            ),
            "ego_history_xyz": _ensure_batched_sample_tensor(
                reference.get("ego_history_xyz"),
                batch_size=batch_size,
                desired_rank=4,
                name="ego_history_xyz",
            ),
            "ego_history_rot": _ensure_batched_sample_tensor(
                reference.get("ego_history_rot"),
                batch_size=batch_size,
                desired_rank=5,
                name="ego_history_rot",
            ),
            "ego_future_xyz": _ensure_batched_sample_tensor(
                reference.get("ego_future_xyz"),
                batch_size=batch_size,
                desired_rank=4,
                name="ego_future_xyz",
            ),
            "ego_future_rot": _ensure_batched_sample_tensor(
                reference.get("ego_future_rot"),
                batch_size=batch_size,
                desired_rank=5,
                name="ego_future_rot",
            ),
            "obstacle_bbox_history": _ensure_batched_sample_tensor(
                reference.get("obstacle_bbox_history"),
                batch_size=batch_size,
                desired_rank=5,
                name="obstacle_bbox_history",
            ),
            "obstacle_bbox_future": _ensure_batched_sample_tensor(
                reference.get("obstacle_bbox_future"),
                batch_size=batch_size,
                desired_rank=5,
                name="obstacle_bbox_future",
            ),
        }

        Tf = int(predicted_fut_xyz.shape[1])
        horizons = timestep_horizons if timestep_horizons is not None else [Tf]

        metric = CollisionMetrics(prefix="", timestep_horizons=horizons)
        out = metric.evaluate(
            model=None,
            data_batch=data_batch,
            output_batch={
                "pred_xyz": predicted_fut_xyz[:, None, None, ...],
                "pred_rot": predicted_fut_rot[:, None, None, ...],
            },
        )
        if not out:
            return 0.0, {}

        for k, v in out.items():
            if torch.is_tensor(v):
                collision_by_t[k] = float(v.float().mean().item())

        vals = [v for v in out.values() if torch.is_tensor(v)]
        if not vals:
            return 0.0, collision_by_t
        any_collision = torch.stack([vv.float() for vv in vals], dim=0).max(dim=0).values
        collision_cost = float((any_collision > 0.0).float().mean().item())

        return collision_cost, collision_by_t
    except Exception as e:
        logger.warning(f"[compute_collision_cost] Collision metric failed: {e}")
        return 0.0, {}
