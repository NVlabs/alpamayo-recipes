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

"""Obstacle decoder for PAI (obstacle.offline) parquet format.

Converts PAI obstacle labels from rig-frame-per-timestamp to ego-frame-at-t0,
producing ``obstacle_bbox_history`` and ``obstacle_bbox_future`` tensors
compatible with :class:`alpamayo.metrics.metric_api.CollisionMetrics`.

PAI obstacle.offline columns:
    timestamp_us, track_id, center_x/y/z, size_x/y/z,
    orientation_x/y/z/w (quaternion), label_class, reference_frame,
    reference_frame_timestamp_us.

Coordinate transform:
    Each obstacle observation is in the ``ego`` (rig) frame at its own timestamp.
    We build a :class:`physical_ai_av.utils.tf.TransformTree` rooted at ``anchor``
    with ``ego`` as a time-varying child driven by the egomotion interpolator, then
    look up the transform between ``FrameInfo("ego", t0_us)`` and ``FrameInfo("ego",
    reference_frame_timestamp_us)`` to obtain a vectorized rig@ts → ego@t0
    ``RigidTransform``. Raw
    obstacle boxes are transformed into the common ego@t0 frame before temporal
    interpolation; interpolating directly in per-timestamp rig frames is invalid
    because those frames move and rotate with the ego vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from physical_ai_av.utils import tf as pai_tf
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=False)
logger.setLevel("INFO")


@dataclass(frozen=True)
class ObstacleDecoderConfig:
    """Configuration for obstacle decoder.

    Attributes:
        distance_filtering_x_min: Min x (rear) in ego frame to keep obstacle.
        distance_filtering_x_max: Max x (front) in ego frame to keep obstacle.
        distance_filtering_y_min: Min y (left) in ego frame to keep obstacle.
        distance_filtering_y_max: Max y (right) in ego frame to keep obstacle.
        max_objects: Pad/truncate to this many objects.
    """

    distance_filtering_x_min: float = -20.0
    distance_filtering_x_max: float = 20.0
    distance_filtering_y_min: float = -20.0
    distance_filtering_y_max: float = 20.0
    max_objects: int = 128


def _quat_to_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert scalar-last quaternions [..., 4] to yaw angle (rad) around z-axis."""
    # 'zyx' Euler with degrees=False: [yaw, pitch, roll].
    return Rotation.from_quat(quat_xyzw).as_euler("zyx", degrees=False)[..., 0]


def _valid_obstacle(xyz_disp_from_ego: np.ndarray, config: ObstacleDecoderConfig) -> bool:
    """Return True if obstacle is within distance thresholds relative to ego at any future step."""
    x = xyz_disp_from_ego[..., 0]
    y = xyz_disp_from_ego[..., 1]
    return bool(
        np.any(
            np.logical_and(
                np.logical_and(config.distance_filtering_x_min < x, x < config.distance_filtering_x_max),
                np.logical_and(config.distance_filtering_y_min < y, y < config.distance_filtering_y_max),
            )
        )
    )


def _remove_theta_outliers(
    raw_tvals: np.ndarray,
    raw_xyz_lwh_theta: np.ndarray,
    window_size: int = 5,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove rows with outlier theta values from raw track data.

    Uses ``np.unwrap`` to handle ±pi discontinuities, then applies a median
    filter to compute a robust local estimate. Points deviating from the local
    median by more than the threshold are removed.
    """
    if len(raw_tvals) < 3:
        return raw_tvals, raw_xyz_lwh_theta

    raw_xyz_lwh_theta = np.asarray(raw_xyz_lwh_theta, dtype=np.float64)
    theta = raw_xyz_lwh_theta[:, 6]

    sort_idx = np.argsort(raw_tvals)
    theta_sorted = theta[sort_idx]

    theta_unwrapped = np.unwrap(theta_sorted)
    median_vals = median_filter(
        theta_unwrapped, size=min(window_size, len(theta_unwrapped)), mode="reflect"
    )
    deviation = np.abs(theta_unwrapped - median_vals)

    if threshold is None:
        mad = np.median(deviation)
        threshold = max(5.0 * mad, 0.5)

    valid_mask_sorted = deviation <= threshold

    valid_mask = np.ones(len(theta), dtype=bool)
    valid_mask[sort_idx] = valid_mask_sorted

    return raw_tvals[valid_mask], raw_xyz_lwh_theta[valid_mask]


def _angle_wrap_np(radians: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval [-pi, pi)."""
    return (radians + np.pi) % (2 * np.pi) - np.pi


def _transform_to_ego_t0(
    raw_xyz_lwh_theta: np.ndarray,
    tf_ego_t0_from_rig_ts: spt.RigidTransform,
) -> np.ndarray:
    """Transform raw obstacle boxes from rig@ts into the common ego@t0 frame.

    Args:
        raw_xyz_lwh_theta: [M, 7] obstacle data (x, y, z, l, w, h, yaw) in rig@ts.
        tf_ego_t0_from_rig_ts: Per-row ``RigidTransform`` mapping rig@ts to ego@t0.

    Returns:
        [M, 7] obstacle boxes in ego@t0 frame.
    """
    xyz_rig = raw_xyz_lwh_theta[:, :3]
    lwh = raw_xyz_lwh_theta[:, 3:6]
    theta_rig = raw_xyz_lwh_theta[:, 6]

    xyz_ego = tf_ego_t0_from_rig_ts.apply(xyz_rig)

    forward_rig = np.stack(
        [np.cos(theta_rig), np.sin(theta_rig), np.zeros_like(theta_rig)], axis=-1
    )
    forward_ego = tf_ego_t0_from_rig_ts.rotation.apply(forward_rig)
    yaw_ego = np.arctan2(forward_ego[:, 1], forward_ego[:, 0])

    return np.concatenate([xyz_ego, lwh, yaw_ego[:, None]], axis=-1)


def _interpolate_in_ego_t0(
    raw_tvals: np.ndarray,
    raw_xyz_lwh_theta: np.ndarray,
    requested_tvals: np.ndarray,
) -> np.ndarray:
    """Interpolate a track in the common ego@t0 frame.

    Args:
        raw_tvals: [M] relative time in seconds for raw observations.
        raw_xyz_lwh_theta: [M, 7] obstacle data (x, y, z, l, w, h, yaw) in ego@t0.
        requested_tvals: [T] relative time in seconds at which to sample.

    Returns:
        [T, 7] obstacle bbox (x, y, z, l, w, h, yaw) in ego@t0 frame.
    """
    # clean theta outliers
    clean_tvals, clean_data = _remove_theta_outliers(raw_tvals, raw_xyz_lwh_theta)
    if len(clean_tvals) >= 2:
        raw_tvals, raw_xyz_lwh_theta = clean_tvals, clean_data

    if len(raw_tvals) < 2:
        return np.full((len(requested_tvals), 7), np.nan, dtype=np.float64)

    sort_idx = np.argsort(raw_tvals)
    raw_tvals = raw_tvals[sort_idx]
    raw_xyz_lwh_theta = raw_xyz_lwh_theta[sort_idx].copy()
    raw_xyz_lwh_theta[:, 6] = np.unwrap(raw_xyz_lwh_theta[:, 6])

    # temporal interpolation in the common ego@t0 frame
    interp_fn = interp1d(
        x=raw_tvals,
        y=raw_xyz_lwh_theta,
        axis=0,
        assume_sorted=True,
        fill_value=np.nan,
        bounds_error=False,
    )
    interped = interp_fn(requested_tvals).astype(np.float64)  # [T, 7]
    interped[:, 6] = _angle_wrap_np(interped[:, 6])

    # preserve NaN from interpolation
    nan_mask = np.isnan(interped[:, 0])
    interped[nan_mask] = np.nan

    return interped  # [T, 7]


def decode_obstacles_pai(
    obstacle_df: pd.DataFrame,
    egomotion_interp: object,
    t0_us: int,
    history_timestamps_us: np.ndarray,
    future_timestamps_us: np.ndarray,
    config: ObstacleDecoderConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Decode PAI obstacle.offline parquet into obstacle bbox tensors.

    Args:
        obstacle_df: DataFrame from PAI obstacle.offline parquet for one clip.
            Expected columns: timestamp_us, track_id, center_x/y/z, size_x/y/z,
            orientation_x/y/z/w, label_class.
        egomotion_interp: Egomotion interpolator (callable on int64 timestamps).
        t0_us: Prediction base timestamp in microseconds.
        history_timestamps_us: [Th] absolute timestamps for history steps.
        future_timestamps_us: [Tf] absolute timestamps for future steps.
        config: Decoder configuration.

    Returns:
        Dict with:
            ``obstacle_bbox_history``: [N, Th, 7] tensor (xyz, lwh, yaw) in ego@t0 frame.
            ``obstacle_bbox_future``: [N, Tf, 7] tensor (xyz, lwh, yaw) in ego@t0 frame.
            ``num_obstacles``: [1] tensor with actual obstacle count before padding.
    """
    if config is None:
        config = ObstacleDecoderConfig()

    Th = len(history_timestamps_us)
    Tf = len(future_timestamps_us)

    # Build a TransformTree with `ego` as a time-varying child of `anchor` driven by the
    # egomotion interpolator. Looking up FrameInfo("ego", t0_us) vs FrameInfo("ego",
    # ts_array) returns a vectorized rig@ts → ego@t0 RigidTransform.
    tf_tree = pai_tf.TransformTree(root_frame_id="anchor")
    tf_tree.add_transform("anchor", "ego", egomotion_interp)
    target = pai_tf.FrameInfo("ego", int(t0_us))

    # use an absolute reference time for converting to relative seconds
    timestamp_offset = t0_us
    history_tvals = (history_timestamps_us - timestamp_offset) * 1e-6  # [Th] seconds
    future_tvals = (future_timestamps_us - timestamp_offset) * 1e-6  # [Tf] seconds

    # parse obstacle DataFrame
    if obstacle_df is None or len(obstacle_df) == 0:
        return {
            "obstacle_bbox_history": torch.full((config.max_objects, Th, 7), float("nan")),
            "obstacle_bbox_future": torch.full((config.max_objects, Tf, 7), float("nan")),
            "num_obstacles": torch.tensor([0], dtype=torch.long),
        }

    # quaternion → yaw via scipy
    quat_xyzw = obstacle_df[
        ["orientation_x", "orientation_y", "orientation_z", "orientation_w"]
    ].to_numpy(dtype=np.float64)
    yaw = _quat_to_yaw(quat_xyzw)

    timestamps_us = obstacle_df["timestamp_us"].to_numpy(dtype=np.int64)
    if "reference_frame_timestamp_us" in obstacle_df.columns:
        frame_timestamps_us = (
            obstacle_df["reference_frame_timestamp_us"]
            .where(obstacle_df["reference_frame_timestamp_us"].notna(), obstacle_df["timestamp_us"])
            .to_numpy(dtype=np.int64)
        )
    else:
        frame_timestamps_us = timestamps_us

    # build standardized columns: x, y, z, length, width, height, theta, timestamp_us
    obs_data = pd.DataFrame(
        {
            "track_id": obstacle_df["track_id"].values,
            "x": obstacle_df["center_x"].to_numpy(dtype=np.float64),
            "y": obstacle_df["center_y"].to_numpy(dtype=np.float64),
            "z": obstacle_df["center_z"].to_numpy(dtype=np.float64),
            "length": obstacle_df["size_x"].to_numpy(dtype=np.float64),
            "width": obstacle_df["size_y"].to_numpy(dtype=np.float64),
            "height": obstacle_df["size_z"].to_numpy(dtype=np.float64),
            "theta": yaw,
            "timestamp_us": timestamps_us,
            "frame_timestamp_us": frame_timestamps_us,
        }
    )
    obs_xyz_lwh_theta_ego_t0 = _transform_to_ego_t0(
        raw_xyz_lwh_theta=obs_data[
            ["x", "y", "z", "length", "width", "height", "theta"]
        ].to_numpy(dtype=np.float64),
        tf_ego_t0_from_rig_ts=tf_tree.lookup_transform(
            target,
            pai_tf.FrameInfo("ego", obs_data["frame_timestamp_us"].to_numpy(dtype=np.int64)),
        ).tf_target_source,
    )
    obs_data[["x", "y", "z", "length", "width", "height", "theta"]] = obs_xyz_lwh_theta_ego_t0

    obstacle_bbox_history_list: list[np.ndarray] = []
    obstacle_bbox_future_list: list[np.ndarray] = []
    obstacle_distances: list[float] = []

    grouped = obs_data.groupby("track_id")
    for _track_id, track_df in grouped:
        ts_us = track_df["timestamp_us"].to_numpy(dtype=np.int64)

        # skip tracks with duplicate timestamps
        if len(ts_us) != len(np.unique(ts_us)):
            logger.debug(f"Skipping track {_track_id} due to duplicate timestamps.")
            continue

        # convert to relative seconds
        raw_tvals = (ts_us - timestamp_offset) * 1e-6

        if len(raw_tvals) < 2:
            continue

        raw_xyz_lwh_theta = track_df[
            ["x", "y", "z", "length", "width", "height", "theta"]
        ].to_numpy(dtype=np.float64)

        # interpolate in the common ego@t0 frame for history
        hist_bbox = _interpolate_in_ego_t0(
            raw_tvals=raw_tvals,
            raw_xyz_lwh_theta=raw_xyz_lwh_theta,
            requested_tvals=history_tvals,
        )  # [Th, 7]

        # interpolate in the common ego@t0 frame for future
        fut_bbox = _interpolate_in_ego_t0(
            raw_tvals=raw_tvals,
            raw_xyz_lwh_theta=raw_xyz_lwh_theta,
            requested_tvals=future_tvals,
        )  # [Tf, 7]

        # Distance filtering in the common ego@t0 frame.
        fut_xyz_from_ego = fut_bbox[:, :3]  # [Tf, 3]
        if not _valid_obstacle(fut_xyz_from_ego, config):
            continue

        obstacle_bbox_history_list.append(hist_bbox)
        obstacle_bbox_future_list.append(fut_bbox)

        # record distance at current time (last history step) for sorting
        current_xyz = hist_bbox[-1, :3]
        obstacle_distances.append(float(np.linalg.norm(current_xyz[:2])))

    num_obstacles = len(obstacle_bbox_history_list)

    # pad/truncate to max_objects
    all_hist = np.full((config.max_objects, Th, 7), np.nan, dtype=np.float64)
    all_fut = np.full((config.max_objects, Tf, 7), np.nan, dtype=np.float64)

    if num_obstacles > 0:
        indices = slice(None)
        if num_obstacles > config.max_objects:
            indices = np.argsort(obstacle_distances)[: config.max_objects]
            num_obstacles = config.max_objects

        hist_stack = np.stack(obstacle_bbox_history_list, axis=0)[indices]  # [N, Th, 7]
        fut_stack = np.stack(obstacle_bbox_future_list, axis=0)[indices]  # [N, Tf, 7]
        all_hist[: len(hist_stack)] = hist_stack
        all_fut[: len(fut_stack)] = fut_stack

    return {
        "obstacle_bbox_history": torch.from_numpy(all_hist).float(),
        "obstacle_bbox_future": torch.from_numpy(all_fut).float(),
        "num_obstacles": torch.tensor([num_obstacles], dtype=torch.long),
    }
