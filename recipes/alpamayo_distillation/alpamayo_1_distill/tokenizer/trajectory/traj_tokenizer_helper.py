from typing import Any, Mapping

import einops
import torch

from alpamayo1_distill.tokenizer.trajectory.base_trajectory_tokenizer import BaseTrajectoryTokenizer


def tokenize_history_trajectory(
    tokenizer: BaseTrajectoryTokenizer, traj_data: Mapping[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the history trajectory with prefix shape of (B, n_traj, ...).

    Args:
        tokenizer: BaseTrajectoryTokenizer
        traj_data: dict[str, Any], needs to at least contain "ego_history_xyz" and "ego_history_rot"
        start_idx: int, start of token index of the history trajectory tokens

    Returns:
        torch.Tensor: [B, n_traj *  tokens_per_history_traj]
    """
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4, "ego_history_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,  # note hist_xyz is passed to fut_xyz as it's encoding history.
            fut_rot=hist_rot,
        )
        + start_idx
    )  # [B*n_traj, tokens_per_history_traj]
    hist_idx = einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)
    #print("in tokenize history traj, fut_idx shape", hist_idx.shape)
    return hist_idx


def tokenize_future_trajectory(
    tokenizer: BaseTrajectoryTokenizer, traj_data: Mapping[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the future trajectory with prefix shape of (B, n_traj, ...).

    Args:
        tokenizer: BaseTrajectoryTokenizer
        traj_data: dict[str, Any], needs to at least contain:
            - "ego_history_xyz"
            - "ego_history_rot"
            - "ego_future_xyz"
            - "ego_future_rot"
        start_idx: int, start of token index of the future trajectory tokens

    Returns:
        torch.Tensor: [B, n_traj * tokens_per_future_traj]
    """
    assert "ego_future_xyz" in traj_data
    assert traj_data["ego_future_xyz"].ndim == 4, "ego_future_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_future_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    fut_xyz = traj_data["ego_future_xyz"].flatten(start_dim=0, end_dim=1)
    fut_rot = traj_data["ego_future_rot"].flatten(start_dim=0, end_dim=1)

    fut_idx = (
        tokenizer.encode(hist_xyz=hist_xyz, hist_rot=hist_rot, fut_xyz=fut_xyz, fut_rot=fut_rot)
        + start_idx
    )
    fut_idx = einops.rearrange(fut_idx, "(b n_traj) n -> b (n_traj n)", b=B)
    #print("in tokenize future traj, fut_idx shape", fut_idx.shape)
    return fut_idx
