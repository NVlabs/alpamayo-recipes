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

"""Load data from physical_ai_av.PhysicalAIAVDatasetInterface for model inference."""

from typing import Any

import numpy as np
import physical_ai_av
import scipy.spatial.transform as spt
import torch
from einops import rearrange


class InfeasibleClipError(RuntimeError):
    """Raised when a clip cannot provide the requested sample window."""


def _sample_window_us(
    num_history_steps: int,
    num_future_steps: int,
    num_frames: int,
    time_step: float,
) -> tuple[int, int, int]:
    step_us = int(time_step * 1_000_000)
    return (
        (num_history_steps - 1) * step_us,
        num_future_steps * step_us,
        (num_frames - 1) * step_us,
    )


def _feasible_t0_range(
    *,
    ego_time_range: tuple[int, int],
    camera_time_ranges: list[tuple[int, int]],
    num_history_steps: int,
    num_future_steps: int,
    num_frames: int,
    time_step: float,
    margin_us: int,
) -> tuple[int, int]:
    """Return the inclusive t0 range supported by ego and every camera."""

    history_back_us, future_forward_us, image_back_us = _sample_window_us(
        num_history_steps,
        num_future_steps,
        num_frames,
        time_step,
    )
    ego_t_min, ego_t_max = (int(value) for value in ego_time_range)
    camera_t_min = max(int(bounds[0]) for bounds in camera_time_ranges)
    camera_t_max = min(int(bounds[1]) for bounds in camera_time_ranges)
    strict_history_floor = int(num_history_steps * time_step * 1_000_000) + 1
    lower = (
        max(
            ego_t_min + history_back_us,
            camera_t_min + image_back_us,
            strict_history_floor,
        )
        + margin_us
    )
    upper = (
        min(
            ego_t_max - future_forward_us,
            camera_t_max,
        )
        - margin_us
    )
    return lower, upper


def _sample_t0_from_range(
    lower: int,
    upper: int,
    random_bits: int,
    quantum_us: int | None,
) -> int:
    """Map deterministic random bits into an optional timestamp grid."""
    if quantum_us is None:
        return lower + int(random_bits) % (upper - lower + 1)
    quantum = int(quantum_us)
    first = ((lower + quantum - 1) // quantum) * quantum
    last = (upper // quantum) * quantum
    if last < first:
        raise InfeasibleClipError(
            f"no feasible t0 on the {quantum}-us grid inside [{lower}, {upper}]"
        )
    grid_size = (last - first) // quantum + 1
    return first + int(random_bits) % grid_size * quantum


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int | None = 5_100_000,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    maybe_stream: bool = True,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
    local_dir: str | None = None,
    t0_margin_us: int = 0,
    t0_random_bits: int | None = None,
    t0_quantum_us: int | None = None,
) -> dict[str, Any]:
    """Load data from physical_ai_av for model inference.

    This function loads a sample from the physical_ai_av dataset and converts it
    to the format expected by Alpamayo1_5 model inference.

    Args:
        clip_id: The clip ID to load data from. Can be obtained from vla_golden.parquet.
        t0_us: The timestamp (in microseconds) at which to sample the trajectory.
            If None, select from the clip-specific feasible sensor window.
        avdi: Optional pre-initialized PhysicalAIAVDatasetInterface. If None, creates one.
        maybe_stream: Whether to stream data from HuggingFace (if not downloaded locally).
        num_history_steps: Number of history trajectory steps (default: 16 for 1.6s at 10Hz).
        num_future_steps: Number of future trajectory steps (default: 64 for 6.4s at 10Hz).
        time_step: Time step between trajectory points in seconds (default: 0.1s = 10Hz).
        camera_features: List of camera features to load. If None, uses 4 cameras:
            [CAMERA_FRONT_WIDE_120FOV, CAMERA_FRONT_TELE_30FOV,
             CAMERA_CROSS_LEFT_120FOV, CAMERA_CROSS_RIGHT_120FOV].
        num_frames: Number of frames per camera to load (default: 4).
        local_dir: Optional local PhysicalAI-AV dataset directory.
        t0_margin_us: Margin removed from both ends of the feasible t0 range.
        t0_random_bits: Deterministic random bits mapped into the feasible range.
        t0_quantum_us: If set, random timestamps are sampled on this exact grid.

    Returns:
        A dictionary with the following keys:
            - image_frames: torch.Tensor of shape (N_cameras, num_frames, 3, H, W)
            - camera_indices: torch.Tensor of shape (N_cameras,)
            - ego_history_xyz: torch.Tensor of shape (1, 1, num_history_steps, 3)
            - ego_history_rot: torch.Tensor of shape (1, 1, num_history_steps, 3, 3)
            - ego_future_xyz: torch.Tensor of shape (1, 1, num_future_steps, 3)
            - ego_future_rot: torch.Tensor of shape (1, 1, num_future_steps, 3, 3)
            - relative_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - absolute_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - t0_us: The t0 timestamp used
            - clip_id: The clip ID
    """
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=local_dir)
    if t0_margin_us < 0:
        raise ValueError("t0_margin_us must be non-negative")
    if t0_quantum_us is not None and t0_quantum_us <= 0:
        raise ValueError("t0_quantum_us must be positive when provided")

    if camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    egomotion = avdi.get_clip_feature(
        clip_id,
        avdi.features.LABELS.EGOMOTION,
        maybe_stream=maybe_stream,
    )
    cameras: list[Any | None] = []
    try:
        for feature in camera_features:
            cameras.append(
                avdi.get_clip_feature(
                    clip_id,
                    feature,
                    maybe_stream=maybe_stream,
                )
            )

        camera_ranges = [
            (
                int(np.asarray(camera.timestamps).min()),
                int(np.asarray(camera.timestamps).max()),
            )
            for camera in cameras
        ]
        lower, upper = _feasible_t0_range(
            ego_time_range=egomotion.time_range,
            camera_time_ranges=camera_ranges,
            num_history_steps=num_history_steps,
            num_future_steps=num_future_steps,
            num_frames=num_frames,
            time_step=time_step,
            margin_us=t0_margin_us,
        )
        if upper < lower:
            raise InfeasibleClipError(
                f"clip {clip_id!r} has no feasible t0 range: "
                f"[{lower}, {upper}], ego={egomotion.time_range}, "
                f"cameras={camera_ranges}"
            )
        if t0_us is None:
            if t0_random_bits is None:
                t0_random_bits = int.from_bytes(
                    np.random.bytes(8),
                    byteorder="little",
                )
            try:
                t0_us = _sample_t0_from_range(
                    lower,
                    upper,
                    int(t0_random_bits),
                    t0_quantum_us,
                )
            except InfeasibleClipError as error:
                raise InfeasibleClipError(f"clip {clip_id!r} {error}") from error
        elif not lower <= int(t0_us) <= upper:
            raise InfeasibleClipError(
                f"clip {clip_id!r} t0={t0_us} is outside feasible range [{lower}, {upper}]"
            )

        assert t0_us > num_history_steps * time_step * 1_000_000, (
            "t0_us must be greater than the history time range"
        )

        # History ends at t0; future starts one step after t0.
        history_offsets_us = np.arange(
            -(num_history_steps - 1) * time_step * 1_000_000,
            time_step * 1_000_000 / 2,
            time_step * 1_000_000,
        ).astype(np.int64)
        history_timestamps = t0_us + history_offsets_us
        future_offsets_us = np.arange(
            time_step * 1_000_000,
            (num_future_steps + 0.5) * time_step * 1_000_000,
            time_step * 1_000_000,
        ).astype(np.int64)
        future_timestamps = t0_us + future_offsets_us

        ego_history = egomotion(history_timestamps)
        ego_history_xyz = ego_history.pose.translation
        ego_history_quat = ego_history.pose.rotation.as_quat()
        ego_future = egomotion(future_timestamps)
        ego_future_xyz = ego_future.pose.translation
        ego_future_quat = ego_future.pose.rotation.as_quat()
        ego_arrays = (
            ego_history_xyz,
            ego_history_quat,
            ego_future_xyz,
            ego_future_quat,
        )
        if not all(np.isfinite(array).all() for array in ego_arrays):
            raise InfeasibleClipError(
                f"clip {clip_id!r} produced non-finite egomotion at t0={t0_us}"
            )

        t0_xyz = ego_history_xyz[-1].copy()
        t0_quat = ego_history_quat[-1].copy()
        t0_rot_inv = spt.Rotation.from_quat(t0_quat).inv()
        ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
        ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
        ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
        ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

        ego_history_xyz_tensor = (
            torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
        )
        ego_history_rot_tensor = (
            torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
        )
        ego_future_xyz_tensor = (
            torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
        )
        ego_future_rot_tensor = (
            torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)
        )

        image_frames_list = []
        camera_indices_list = []
        timestamps_list = []
        step_us = int(time_step * 1_000_000)
        image_timestamps = np.array(
            [t0_us - (num_frames - 1 - index) * step_us for index in range(num_frames)],
            dtype=np.int64,
        )

        for index, (cam_feature, camera) in enumerate(zip(camera_features, cameras, strict=True)):
            assert camera is not None
            frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
            close = getattr(camera, "close", None)
            if close is not None:
                close()
            cameras[index] = None
            frames_tensor = rearrange(
                torch.from_numpy(frames),
                "t h w c -> t c h w",
            )
            if not isinstance(cam_feature, str):
                raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
            cam_name = cam_feature.rsplit("/", 1)[-1].lower()
            image_frames_list.append(frames_tensor)
            camera_indices_list.append(camera_name_to_index.get(cam_name, 0))
            timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

        image_frames = torch.stack(image_frames_list, dim=0)
        camera_indices = torch.tensor(
            camera_indices_list,
            dtype=torch.int64,
        )
        all_timestamps = torch.stack(timestamps_list, dim=0)
        sort_order = torch.argsort(camera_indices)
        image_frames = image_frames[sort_order]
        camera_indices = camera_indices[sort_order]
        all_timestamps = all_timestamps[sort_order]
        camera_tmin = all_timestamps.min()
        relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

        return {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "ego_history_xyz": ego_history_xyz_tensor,
            "ego_history_rot": ego_history_rot_tensor,
            "ego_future_xyz": ego_future_xyz_tensor,
            "ego_future_rot": ego_future_rot_tensor,
            "relative_timestamps": relative_timestamps,
            "absolute_timestamps": all_timestamps,
            "t0_us": t0_us,
            "clip_id": clip_id,
        }
    finally:
        for camera in cameras:
            if camera is not None:
                close = getattr(camera, "close", None)
                if close is not None:
                    close()
        close_egomotion = getattr(egomotion, "close", None)
        if close_egomotion is not None:
            close_egomotion()
