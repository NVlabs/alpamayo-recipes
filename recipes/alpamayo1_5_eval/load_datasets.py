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

"""Single-clip dataset loader for Alpamayo 1.5 minADE evaluation."""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.spatial.transform as spt
import torch
from einops import rearrange

import physical_ai_av

logger = logging.getLogger(__name__)


def _ensure_hf_cache_from_local_dir(local_dir: str | pathlib.Path, revision: str) -> None:
    """Ensure the HF hub cache refs and symlinks are set up so offline mode works.

    When ``local_dir`` has been populated by a previous online download, HuggingFace
    Hub stores the actual files there but may be missing the ``refs/<revision>`` pointer
    inside the hub cache (``~/.cache/huggingface/hub/``).  Without that pointer,
    ``try_to_load_from_cache(revision=revision)`` cannot locate the files and offline
    mode raises ``OfflineModeIsEnabled`` even though the data is present on disk.

    This function:
    1. Finds the existing snapshot directory inside the hub cache.
    2. Creates ``refs/<revision>`` (mapping branch name → commit hash) if missing.
    3. Creates symlinks inside the snapshot directory for any file that exists in
       ``local_dir`` but is not yet represented in the snapshot, so that
       ``try_to_load_from_cache`` returns a valid path for every local file.
    """
    try:
        import huggingface_hub

        repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"
        repo_type = "dataset"
        cache_dir_name = f"{repo_type}s--{repo_id.replace('/', '--')}"
        repo_cache = pathlib.Path(huggingface_hub.constants.HF_HUB_CACHE) / cache_dir_name
    except Exception:
        logger.debug("Could not determine HF cache path; skipping cache setup.", exc_info=True)
        return

    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.exists():
        logger.debug("No snapshots dir found in HF cache; skipping cache setup.")
        return

    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        logger.debug("No snapshot subdirectories found; skipping cache setup.")
        return

    snapshot_dir = snapshot_dirs[0]
    snapshot_hash = snapshot_dir.name

    refs_dir = repo_cache / "refs"
    refs_dir.mkdir(exist_ok=True)
    refs_file = refs_dir / revision
    if not refs_file.exists():
        refs_file.write_text(snapshot_hash)
        logger.debug("Created HF cache refs/%s -> %s", revision, snapshot_hash)

    local_path = pathlib.Path(local_dir).resolve()
    created = 0
    for src in local_path.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(local_path)
        dst = snapshot_dir / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                dst.symlink_to(src)
                created += 1
            except OSError:
                logger.debug("Could not create symlink %s -> %s", dst, src)

    if created:
        logger.debug("Created %d HF cache symlink(s) from local_dir.", created)


@dataclass
class SingleClipDatasetConfig:
    """Configuration for loading a single PhysicalAI-AV clip.

    Args:
        clip_id: PhysicalAI-AV clip identifier (UUID string).
        t0_us: Evaluation timestamp in microseconds. Must be greater than
            ``num_history_steps * time_step * 1e6``.
        maybe_stream: If True, allow streaming from the HuggingFace dataset
            cache instead of requiring a full local download.
        num_history_steps: Number of past ego-pose steps to load.
        num_future_steps: Number of future ego-pose steps used as ground truth.
        time_step: Time interval between consecutive steps in seconds.
        num_frames: Number of image frames per camera to load (most-recent
            frames up to ``t0_us``).
        dataset_local_dir: Optional path to a locally downloaded copy of the
            PhysicalAI-AV dataset.  When provided, the interface reads files
            from this directory and avoids unnecessary downloads.  A valid HF
            ``revision`` must also be supplied (``"main"`` works for the
            official dataset branch).
        dataset_revision: HuggingFace git revision (branch name, tag, or commit
            hash).  Pass a non-``None`` value to skip the ``list_repo_refs``
            network call that resolves the default ``main`` commit hash.
    """

    clip_id: str
    t0_us: int = 5_100_000
    maybe_stream: bool = True
    num_history_steps: int = 16
    num_future_steps: int = 64
    time_step: float = 0.1
    num_frames: int = 4
    dataset_local_dir: str | None = None
    dataset_revision: str | None = None


class SingleClipDatasetLoader:
    """Load a single PhysicalAI-AV clip for Alpamayo 1.5 inference.

    Retrieves ego-motion history and future ground-truth trajectories,
    transforms them into the t0 ego frame, and decodes images from the four
    default cameras.  The returned dictionary is ready to be consumed by
    :class:`AlpamayoSingleClipModel`.
    """

    _DEFAULT_CAMERAS: tuple[str, ...] = (
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_front_tele_30fov",
    )

    _CAMERA_NAME_TO_INDEX: dict[str, int] = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    def __init__(self, config: SingleClipDatasetConfig) -> None:
        self.config = config
        if config.dataset_local_dir is not None and config.dataset_revision is not None:
            _ensure_hf_cache_from_local_dir(config.dataset_local_dir, config.dataset_revision)
        self.avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
            revision=config.dataset_revision,
            local_dir=config.dataset_local_dir,
        )

    def _get_camera_features(self) -> list[str]:
        features = self.avdi.features.CAMERA
        return [
            features.CAMERA_CROSS_LEFT_120FOV,
            features.CAMERA_FRONT_WIDE_120FOV,
            features.CAMERA_CROSS_RIGHT_120FOV,
            features.CAMERA_FRONT_TELE_30FOV,
        ]

    def _build_timestamps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute history, future, and image timestamps relative to t0_us."""
        cfg = self.config
        min_t0 = cfg.num_history_steps * cfg.time_step * 1_000_000
        if cfg.t0_us <= min_t0:
            raise ValueError(
                f"t0_us={cfg.t0_us} is too small; "
                f"it must be greater than {int(min_t0)} us."
            )

        step_us = int(cfg.time_step * 1_000_000)

        history_offsets_us = np.arange(
            -(cfg.num_history_steps - 1) * step_us,
            step_us // 2,
            step_us,
        ).astype(np.int64)

        future_offsets_us = np.arange(
            step_us,
            (cfg.num_future_steps + 0.5) * step_us,
            step_us,
        ).astype(np.int64)

        image_timestamps = np.array(
            [cfg.t0_us - (cfg.num_frames - 1 - i) * step_us for i in range(cfg.num_frames)],
            dtype=np.int64,
        )

        history_timestamps = cfg.t0_us + history_offsets_us
        future_timestamps = cfg.t0_us + future_offsets_us
        return history_timestamps, future_timestamps, image_timestamps

    def _load_egomotion(
        self,
        history_timestamps: np.ndarray,
        future_timestamps: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Load ego-motion and transform poses into the t0 ego frame.

        Returns tensors shaped ``(1, 1, T, ...)`` to match the batch/group
        dimensions expected by the model and metrics.
        """
        cfg = self.config
        egomotion = self.avdi.get_clip_feature(
            cfg.clip_id,
            self.avdi.features.LABELS.EGOMOTION,
            maybe_stream=cfg.maybe_stream,
        )

        ego_history = egomotion(history_timestamps)
        ego_future = egomotion(future_timestamps)

        ego_history_xyz: np.ndarray = ego_history.pose.translation
        ego_future_xyz: np.ndarray = ego_future.pose.translation
        ego_history_quat: np.ndarray = ego_history.pose.rotation.as_quat()
        ego_future_quat: np.ndarray = ego_future.pose.rotation.as_quat()

        t0_xyz = ego_history_xyz[-1].copy()
        t0_rot = spt.Rotation.from_quat(ego_history_quat[-1].copy())
        t0_rot_inv = t0_rot.inv()

        ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
        ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

        ego_history_rot_local = (
            t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)
        ).as_matrix()
        ego_future_rot_local = (
            t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)
        ).as_matrix()

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)

        return {
            "ego_history_xyz": _to_tensor(ego_history_xyz_local),
            "ego_history_rot": _to_tensor(ego_history_rot_local),
            "ego_future_xyz": _to_tensor(ego_future_xyz_local),
            "ego_future_rot": _to_tensor(ego_future_rot_local),
        }

    def _load_camera_frames(
        self,
        image_timestamps: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Decode frames from the four default cameras and sort by camera index.

        Returns:
            image_frames: ``(num_cameras, num_frames, C, H, W)`` uint8 tensor.
            camera_indices: ``(num_cameras,)`` int64 index tensor.
            absolute_timestamps: ``(num_cameras, num_frames)`` int64 tensor.
            relative_timestamps: ``(num_cameras, num_frames)`` float32 tensor
                in seconds, relative to the earliest frame across all cameras.
        """
        cfg = self.config
        image_frames_list: list[torch.Tensor] = []
        camera_indices_list: list[int] = []
        timestamps_list: list[torch.Tensor] = []

        for cam_feature in self._get_camera_features():
            camera = self.avdi.get_clip_feature(
                cfg.clip_id,
                cam_feature,
                maybe_stream=cfg.maybe_stream,
            )
            frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)

            frames_tensor = rearrange(torch.from_numpy(frames), "t h w c -> t c h w")

            if not isinstance(cam_feature, str):
                raise TypeError(f"Unexpected camera feature type: {type(cam_feature)}")

            cam_name = cam_feature.split("/")[-1].lower()
            cam_idx = self._CAMERA_NAME_TO_INDEX.get(cam_name, 0)

            image_frames_list.append(frames_tensor)
            camera_indices_list.append(cam_idx)
            timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

        image_frames = torch.stack(image_frames_list, dim=0)
        camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
        absolute_timestamps = torch.stack(timestamps_list, dim=0)

        sort_order = torch.argsort(camera_indices)
        image_frames = image_frames[sort_order]
        camera_indices = camera_indices[sort_order]
        absolute_timestamps = absolute_timestamps[sort_order]

        camera_tmin = absolute_timestamps.min()
        relative_timestamps = (absolute_timestamps - camera_tmin).float() * 1e-6

        return {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "absolute_timestamps": absolute_timestamps,
            "relative_timestamps": relative_timestamps,
        }

    def load(self) -> dict[str, Any]:
        """Load the clip and return all tensors needed for model inference.

        Returns:
            A dictionary containing ego-motion tensors, camera image tensors,
            and metadata keys ``clip_id`` and ``t0_us``.
        """
        history_ts, future_ts, image_ts = self._build_timestamps()
        ego_data = self._load_egomotion(history_ts, future_ts)
        camera_data = self._load_camera_frames(image_ts)
        return {
            **ego_data,
            **camera_data,
            "clip_id": self.config.clip_id,
            "t0_us": self.config.t0_us,
        }
