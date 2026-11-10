# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory token fusion and supervised losses owned by the SFT recipe."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def compute_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return float32 mean CE for aligned supervised predictors and targets."""
    if logits.ndim != 2 or labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Expected logits [N, V] and labels [N]")
    if labels_mask is not None:
        if labels_mask.shape != labels.shape:
            raise ValueError("labels_mask must have the same shape as labels")
        active = labels_mask.to(device=labels.device, dtype=torch.bool)
        logits, labels = logits[active], labels[active]
    if labels.numel() == 0:
        return logits.sum() * 0.0
    logits = logits.contiguous().float()
    labels = labels.contiguous().to(logits.device)
    if token_mask is not None:
        if token_mask.shape != (logits.shape[-1],):
            raise ValueError("token_mask must have shape [V]")
        logits = logits.masked_fill(
            ~token_mask.to(device=logits.device, dtype=torch.bool),
            torch.finfo(torch.float32).min,
        )
    # Match the native objective, including an entirely ignored target bucket.
    return torch.nan_to_num(F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX), nan=0.0)


def compute_next_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return mean next-token CE over supervised positions in float32.

    Args:
        logits: (B, L, V) model predictions.
        labels: (B, L) target token IDs; -100 marks ignored positions.
        labels_mask: Optional (B, L) subset of target positions to supervise.
        token_mask: Optional (V,) mask restricting the prediction vocabulary.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B, L, V] and labels [B, L]")
    active = labels[:, 1:].ne(IGNORE_INDEX)
    if labels_mask is not None:
        if labels_mask.shape != labels.shape:
            raise ValueError("labels_mask must have the same shape as labels")
        active = active & labels_mask[:, 1:].bool()
    selected_logits = logits[:, :-1][active]
    selected_labels = labels[:, 1:][active]
    return compute_token_loss(selected_logits, selected_labels, token_mask=token_mask)


def future_traj_loss_mask(
    labels: torch.Tensor,
    traj_ids: Mapping[str, int],
    future_vocab_size: int,
    eos_token_id: int | list[int] | None,
) -> torch.Tensor:
    """Select future tokens, their delimiters, and EOS immediately after the end delimiter."""
    mask = (
        ((labels >= traj_ids["future_id0"]) & (labels < traj_ids["future_id0"] + future_vocab_size))
        | labels.eq(traj_ids["future_start"])
        | labels.eq(traj_ids["future_end"])
    )
    if eos_token_id is not None:
        eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        is_eos = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in eos_ids:
            is_eos |= labels.eq(token_id)
        mask[:, 1:] |= labels[:, :-1].eq(traj_ids["future_end"]) & is_eos[:, 1:]
    return mask


def _replace_trajectory_tokens(
    input_ids: torch.Tensor,
    new_ids: torch.Tensor,
    placeholder_id: int,
) -> torch.Tensor:
    """Replace complete per-row trajectory blocks without crossing sample boundaries."""
    if new_ids.ndim != 2 or new_ids.shape[0] != input_ids.shape[0]:
        raise ValueError("Trajectory tokenizer must return a [B, K] token tensor")
    mask = input_ids.eq(placeholder_id)
    counts = mask.sum(dim=1)
    expected = new_ids.shape[1]
    if ((counts != 0) & (counts != expected)).any():
        raise ValueError(
            "Trajectory placeholder counts must be zero or match the encoded trajectory "
            f"exactly: got {counts.tolist()}, expected {expected}. "
            "Check the history length and avoid truncating trajectory blocks."
        )
    replacements = new_ids[counts != 0].to(device=input_ids.device, dtype=input_ids.dtype)
    return input_ids.masked_scatter(mask, replacements)


@torch.no_grad()
def fuse_traj_tokens(
    history_traj_tokenizer: Any,
    future_traj_tokenizer: Any,
    input_ids: torch.Tensor,
    traj_data: Mapping[str, torch.Tensor] | list[Mapping[str, torch.Tensor] | None] | None,
    traj_ids: Mapping[str, int],
) -> torch.Tensor:
    """Replace history/future placeholders using checkpoint trajectory tokenizers.

    Trajectories have shape (B, n_traj, T, 3) for XYZ and
    (B, n_traj, T, 3, 3) for rotations. Tokenization runs in float32 with
    autocast disabled because the action-space solvers require that precision.
    Input tensors are not modified.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, L]")
    if isinstance(traj_data, list):
        if len(traj_data) != input_ids.shape[0]:
            raise ValueError("Per-row trajectory data must match the input batch size")
        rows = [index for index, value in enumerate(traj_data) if value is not None]
        if rows:
            stacked = {key: torch.stack([traj_data[index][key] for index in rows])
                       for key in traj_data[rows[0]]}
            fused = input_ids.clone()
            fused[rows] = fuse_traj_tokens(
                history_traj_tokenizer, future_traj_tokenizer, input_ids[rows], stacked, traj_ids
            )
            return fused
        traj_data = None
    has_history = bool(input_ids.eq(traj_ids["history_pad"]).any())
    has_future = bool(input_ids.eq(traj_ids["future_pad"]).any())
    if not has_history and not has_future:
        return input_ids
    if traj_data is None:
        raise ValueError("Trajectory placeholders require trajectory data")

    history_xyz = traj_data["ego_history_xyz"]
    history_rot = traj_data["ego_history_rot"]
    if (
        history_xyz.ndim != 4
        or history_xyz.shape[0] != input_ids.shape[0]
        or history_xyz.shape[-1] != 3
        or history_xyz.shape[-2] == 0
        or history_rot.shape != (*history_xyz.shape[:-1], 3, 3)
    ):
        raise ValueError(
            "Expected history XYZ [B, n_traj, T, 3] and rotations [B, n_traj, T, 3, 3]"
        )
    batch_size = input_ids.shape[0]
    history_xyz = history_xyz.flatten(0, 1).float()
    history_rot = history_rot.flatten(0, 1).float()
    with torch.autocast(device_type=history_xyz.device.type, enabled=False):
        if has_history:
            history_ids = history_traj_tokenizer.encode(
                hist_xyz=history_xyz[:, :1],
                hist_rot=history_rot[:, :1],
                fut_xyz=history_xyz,
                fut_rot=history_rot,
            )
            history_ids = history_ids.reshape(batch_size, -1) + traj_ids["history_id0"]
            input_ids = _replace_trajectory_tokens(input_ids, history_ids, traj_ids["history_pad"])
        if has_future:
            future_xyz = traj_data["ego_future_xyz"]
            future_rot = traj_data["ego_future_rot"]
            if (
                future_xyz.ndim != 4
                or future_xyz.shape[:2] != traj_data["ego_history_xyz"].shape[:2]
                or future_xyz.shape[-1] != 3
                or future_rot.shape != (*future_xyz.shape[:-1], 3, 3)
            ):
                raise ValueError(
                    "Future XYZ/rotation shapes must match the history batch and trajectories"
                )
            future_ids = future_traj_tokenizer.encode(
                hist_xyz=history_xyz,
                hist_rot=history_rot,
                fut_xyz=future_xyz.flatten(0, 1).float(),
                fut_rot=future_rot.flatten(0, 1).float(),
            )
            future_ids = future_ids.reshape(batch_size, -1) + traj_ids["future_id0"]
            input_ids = _replace_trajectory_tokens(input_ids, future_ids, traj_ids["future_pad"])
    return input_ids
