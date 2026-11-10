# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PhysicalAI-AV sampling with the Edge camera and trajectory schema."""

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import get_method, instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from alpamayo.data.pai_metadata import validate_navigation

logger = logging.getLogger(__name__)

CAMERA_NAMES_TO_INDICES = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}
DEFAULT_CAMERAS = [name for name, index in CAMERA_NAMES_TO_INDICES.items() if index != 4]


def _interpolate_pose_samples(
    times: np.ndarray, values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Keep linear pose interpolation stable across SciPy versions."""
    order = np.argsort(times, kind="mergesort")
    times = np.asarray(times, dtype=np.float64)[order]
    values = np.asarray(values, dtype=np.float64)[order]
    if np.any(query < times[0]) or np.any(query > times[-1]):
        raise ValueError("Requested pose times are outside the interpolation range")
    hi = np.searchsorted(times, query).clip(1, len(times) - 1)
    lo = hi - 1
    # Preserve slope-times-offset arithmetic before the eventual float32 cast.
    # Weighted endpoint interpolation can round to different float32 targets.
    slope = (values[hi] - values[lo]) / (times[hi] - times[lo])[:, None]
    return slope * (query - times[lo])[:, None] + values[lo]


def sample_linear_poses(
    egomotion: Any,
    anchor_us: int,
    offsets_us: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    """Use linear XYZ and normalized linear quaternion pose interpolation.

    The PAI reader exposes the original pose samples on its interpolator. Reuse
    those samples instead of invoking its cubic-translation / SLERP defaults.
    Interpolate using relative times, centered positions, and sign-aligned
    quaternions to preserve numerical precision and rotation continuity.
    """
    timestamps = np.asarray(egomotion.timestamps)
    pose = egomotion.values.pose
    positions = np.asarray(pose.translation, dtype=np.float64)
    quaternions = pose.rotation.as_quat(scalar_first=True).astype(np.float32)
    if (
        timestamps.ndim != 1
        or len(timestamps) < 2
        or not np.issubdtype(timestamps.dtype, np.integer)
        or np.any(np.diff(timestamps) <= 0)
        or positions.shape != (len(timestamps), 3)
        or quaternions.shape != (len(timestamps), 4)
        or not np.isfinite(positions).all()
        or not np.isfinite(quaternions).all()
    ):
        raise ValueError("Egomotion requires finite poses and increasing integer timestamps")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(np.abs(norms - 1) >= 1e-3):
        raise ValueError("Egomotion quaternions must be normalized")
    quaternions /= norms
    signs = np.ones(len(quaternions), dtype=np.float32)
    signs[1:] = np.where((quaternions[:-1] * quaternions[1:]).sum(axis=1) >= 0, 1, -1)
    quaternions *= np.cumprod(signs)[:, None]
    centered = (positions - positions.mean(axis=0, keepdims=True)).astype(np.float32)
    times = ((timestamps - timestamps[0]) * 1e-6).astype(np.float32)
    values = np.concatenate((centered, quaternions), axis=1).astype(np.float64)
    query = np.float32((anchor_us - timestamps[0]) * 1e-6) + (
        np.asarray(offsets_us) * 1e-6
    ).astype(np.float32)
    interpolated = _interpolate_pose_samples(times, values, query)
    rotations = interpolated[:, 3:]
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)
    return interpolated[:, :3].astype(np.float32), Rotation.from_quat(
        rotations.astype(np.float32), scalar_first=True
    )


def _create_public_dataset_interface(**kwargs: Any) -> Any:
    """Load the shared reader only when the public backend needs it."""
    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface

    return PhysicalAIAVDatasetLocalInterface(**kwargs)


def validate_camera_pixel_bounds(bounds: Mapping[int, Sequence[int]]) -> dict[int, tuple[int, int]]:
    """Validate per-camera minimum/maximum image areas in pixels."""
    if OmegaConf.is_config(bounds):
        bounds = OmegaConf.to_container(bounds, resolve=True)
    if not isinstance(bounds, Mapping):
        raise TypeError("camera_pixel_bounds must map integer camera IDs to [min, max] pixels")
    validated = {}
    for camera_id, pair in bounds.items():
        if type(camera_id) is not int or camera_id not in CAMERA_NAMES_TO_INDICES.values():
            raise ValueError("camera_pixel_bounds keys must be integer camera IDs from 0 through 6")
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) != 2
            or any(type(value) is not int or value <= 0 for value in pair)
            or pair[0] > pair[1]
        ):
            raise ValueError("camera_pixel_bounds values must be positive integer [min, max] pairs")
        validated[camera_id] = tuple(pair)
    return validated


class PhysicalAIDataset(Dataset):
    """Sample local clips as flat images and trajectories in the current ego frame."""

    DEFAULT_T0_US = 5_100_000

    def __init__(
        self,
        local_dir: str,
        chunk_ids: list[int] | str | None = None,
        model_config: Any | None = None,
        preprocess: dict[str, Any] | None = None,
        use_default_keyframe: bool = False,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        reasoning_metadata: str | None = None,
        num_history_steps: int | None = None,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        camera_features: list[str] | None = None,
        frame_offsets: list[float] | None = None,
        front_frame_offsets: list[float] | None = None,
        camera_pixel_bounds: dict[int, list[int]] | None = None,
        include_calibration: bool = False,
        dataset_interface_factory: str | None = None,
        split: str | None = None,
        random_keyframes: bool = False,
        data_seed: int = 42,
    ) -> None:
        if random_keyframes and use_default_keyframe:
            raise ValueError("random_keyframes and use_default_keyframe are mutually exclusive")
        if reasoning_metadata is not None and (random_keyframes or use_default_keyframe):
            raise ValueError("reasoning_metadata requires annotated event times: set random_keyframes=false and use_default_keyframe=false")
        if type(data_seed) is not int or data_seed < 0:
            raise ValueError("data_seed must be a nonnegative integer")
        self.random_keyframes = random_keyframes
        self.data_seed = data_seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        if isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if model_config is not None:
            history_tokens = model_config.tokens_per_history_traj
            if history_tokens <= 0 or history_tokens % 3:
                raise ValueError("Edge history must contain three delta tokens per pose pair")
            expected_steps = history_tokens // 3 + 1
            if num_history_steps is not None and num_history_steps != expected_steps:
                raise ValueError("num_history_steps does not match checkpoint history tokens")
            num_history_steps = expected_steps
        if num_history_steps is None or num_history_steps < 2:
            raise ValueError("Provide model_config or num_history_steps >= 2")
        if num_future_steps < 1 or time_step <= 0:
            raise ValueError("num_future_steps and time_step must be positive")
        self.camera_pixel_bounds = (
            validate_camera_pixel_bounds(camera_pixel_bounds)
            if camera_pixel_bounds is not None
            else None
        )
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step_us = round(time_step * 1_000_000)
        self.camera_features = list(DEFAULT_CAMERAS if camera_features is None else camera_features)
        if not self.camera_features or len(set(self.camera_features)) != len(self.camera_features):
            raise ValueError("camera_features must be nonempty and unique")
        self.camera_ids = [
            CAMERA_NAMES_TO_INDICES[feature.rsplit("/", maxsplit=1)[-1].lower()]
            for feature in self.camera_features
        ]
        self.frame_offsets_us = self._offsets_us(
            [-1.0, -0.1, 0.0] if frame_offsets is None else frame_offsets
        )
        self.front_frame_offsets_us = self._offsets_us(
            [-2.0, -1.0, -0.5, -0.1, 0.0] if front_frame_offsets is None else front_frame_offsets
        )
        interface_factory = (
            get_method(dataset_interface_factory)
            if dataset_interface_factory is not None
            else _create_public_dataset_interface
        )
        self.dataset = interface_factory(
            local_dir=local_dir,
            chunk_ids=chunk_ids,
            features_metadata=features_metadata,
            clip_index_metadata=clip_index_metadata,
            reasoning_metadata=reasoning_metadata,
            split=split,
            inclusive_start=True,
            start_safe_margin_seconds=max(
                -self.frame_offsets_us[0],
                -self.front_frame_offsets_us[0],
                (num_history_steps - 1) * self.time_step_us,
            )
            / 1_000_000,
            end_safe_margin_seconds=num_future_steps * time_step,
        )
        self.clip_ids = self.dataset.get_all_clip_ids()
        if not self.clip_ids:
            raise ValueError("No clips remain in the selected dataset")
        self._reasoning_events = []
        if self.dataset.reasoning_db is not None:
            for clip_id in self.clip_ids:
                if hasattr(self.dataset, "clip_index") and "event_t0s" in self.dataset.clip_index.columns:
                    timestamps = self.dataset.clip_index.at[clip_id, "event_t0s"]
                else:
                    timestamps = self.dataset.reasoning_db.get(clip_id, {}).get("event_t0s", [])
                self._reasoning_events.extend((clip_id, int(t0)) for t0 in timestamps)
        self.use_default_keyframe = use_default_keyframe
        self.include_calibration = include_calibration
        self.preprocess = (
            instantiate(preprocess, _partial_=True)(model_config=model_config)
            if preprocess is not None
            else None
        )

    @staticmethod
    def _offsets_us(offsets: list[float]) -> np.ndarray:
        """Require chronological image offsets ending at the current frame."""
        values = np.rint(np.asarray(offsets) * 1_000_000).astype(np.int64)
        if values.ndim != 1 or not values.size or values[-1] != 0:
            raise ValueError("Frame offsets must be a nonempty list ending at zero")
        if np.any(np.diff(values) <= 0):
            raise ValueError("Frame offsets must be strictly increasing")
        return values

    def __len__(self) -> int:
        """Return the number of indexed clips."""
        return len(self._reasoning_events) if self._reasoning_events else len(self.clip_ids)

    def set_epoch(self, epoch: int) -> None:
        """Update deterministic timestamps, including persistent DataLoader workers."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._epoch.fill_(epoch)

    def _random_t0_us(self, index: int) -> int:
        margin = max(-int(self.frame_offsets_us[0]), -int(self.front_frame_offsets_us[0]),
                     (self.num_history_steps - 1) * self.time_step_us)
        first = (margin + 99_999) // 100_000
        last = (20_000_000 - self.num_future_steps * self.time_step_us) // 100_000
        if first > last:
            raise ValueError("No valid 0.1-second keyframes for the requested history/future window")
        rng = np.random.default_rng(np.random.SeedSequence([self.data_seed, int(self._epoch), index]))
        return int(rng.integers(first, last + 1)) * 100_000

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load selected image times and trajectories for one clip."""
        if self._reasoning_events:
            return self._load_sample(*self._reasoning_events[index])
        clip_id = self.clip_ids[index]
        if self.random_keyframes:
            return self._load_sample(clip_id, self._random_t0_us(index))
        requested_t0_us = int(
            self.DEFAULT_T0_US
            if self.use_default_keyframe
            else self.dataset.get_clip_key_frame(clip_id)
        )
        return self._load_sample(clip_id, requested_t0_us)

    def _load_sample(
        self,
        clip_id: str,
        requested_t0_us: int,
        text_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load one event, adding optional text before tokenization."""
        images, camera_ids, timestamps = [], [], []
        for camera_id, feature in sorted(zip(self.camera_ids, self.camera_features)):
            offsets = self.front_frame_offsets_us if camera_id in {1, 6} else self.frame_offsets_us
            camera = self.dataset.get_clip_feature(clip_id, feature)
            try:
                frames, actual_times = camera.decode_images_from_timestamps(requested_t0_us + offsets)
            finally:
                camera.close()
            actual_times = np.asarray(actual_times)
            if (
                actual_times.shape != (len(offsets),)
                or not np.issubdtype(actual_times.dtype, np.integer)
                or np.any(np.diff(actual_times) < 0)
            ):
                raise ValueError("Captured frames require chronological integer timestamps")
            images.append(torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(1))
            camera_ids.append(torch.full((len(offsets),), camera_id, dtype=torch.int64))
            timestamps.append(torch.as_tensor(actual_times, dtype=torch.int64))
        # Anchor poses at the latest captured image,
        # which can precede the requested event time and need not belong to camera 6.
        absolute_timestamps = torch.cat(timestamps)
        t0_us = int(absolute_timestamps.max())
        history_offsets = np.arange(1 - self.num_history_steps, 1) * self.time_step_us
        future_offsets = np.arange(1, self.num_future_steps + 1) * self.time_step_us
        # Recorded camera timestamps can straddle the nominal clip start:
        # requesting the first frame at 0 us may return a capture around -33 ms.
        # The video reader already selected an available frame. Check the pose
        # window here, preserving the actual capture times used by the model.
        earliest_pose_us = t0_us + history_offsets[0]
        if earliest_pose_us < 0 or t0_us + future_offsets[-1] > 20_000_000:
            raise ValueError("Selected history or future falls outside the 20-second PAI clip")
        egomotion = self.dataset.get_clip_feature(clip_id, "egomotion")
        positions, rotations = sample_linear_poses(
            egomotion, t0_us, np.concatenate((history_offsets, future_offsets))
        )
        origin = positions[self.num_history_steps - 1]
        inverse_rotation = rotations[self.num_history_steps - 1].inv()
        sample: dict[str, Any] = {
            "clip_id": clip_id,
            "requested_t0_us": requested_t0_us,
            "t0_us": t0_us,
        }
        for name, selected in (
            ("history", slice(None, self.num_history_steps)),
            ("future", slice(self.num_history_steps, None)),
        ):
            sample[f"ego_{name}_xyz"] = (
                torch.from_numpy(inverse_rotation.apply(positions[selected] - origin))
                .float()
                .unsqueeze(0)
            )
            sample[f"ego_{name}_rot"] = (
                torch.from_numpy((inverse_rotation * rotations[selected]).as_matrix())
                .float()
                .unsqueeze(0)
            )
        sample["image_frames"] = torch.cat(images)
        sample["camera_indices"] = torch.cat(camera_ids)
        sample["absolute_timestamps"] = absolute_timestamps
        # Native Edge frame labels use the final camera-major image as their
        # display reference, separately from the latest-capture ego pose anchor.
        sample["relative_timestamps"] = (
            sample["absolute_timestamps"] - sample["absolute_timestamps"][-1]
        ).float() / 1_000_000
        if self.camera_pixel_bounds is not None:
            sample["camera_pixel_bounds"] = self.camera_pixel_bounds
        if self.include_calibration:
            sample["intr"] = self.dataset.get_clip_feature(clip_id, "camera_intrinsics")
            sample["extr"] = self.dataset.get_clip_feature(clip_id, "sensor_extrinsics")
        if self.dataset.reasoning_db is not None:
            # Annotation keys refer to the requested event, not its video capture time.
            sample.update(self.dataset.get_reasoning_data(clip_id, requested_t0_us))
        if text_fields is not None:
            sample.update(text_fields)
        if self.preprocess is not None:
            sample["tokenized_data"] = self.preprocess(data=sample)
        return sample


class PhysicalAIAnnotatedDataset(PhysicalAIDataset):
    """Load one PAI event per JSON row, with navigation and/or reasoning labels.

    Rows require ``clip_id`` and ``t0_relative`` (integer clip-relative
    microseconds). Optional ``nav_text`` is a nonempty string or single-string
    list; optional ``cot`` is a nonempty string. The corresponding ``require_*``
    flags require those labels on every row. Separate annotations for the same
    clip remain separate samples, in file order. Only clips available through
    the selected chunk index are kept.
    """

    def __init__(
        self,
        annotations_path: str,
        require_navigation: bool = False,
        require_cot: bool = False,
        validate_navigation_format: bool = True,
        **kwargs: Any,
    ) -> None:
        if type(require_navigation) is not bool or type(require_cot) is not bool:
            raise TypeError("require_navigation and require_cot must be booleans")
        if kwargs.get("random_keyframes", False):
            raise ValueError("Annotated samples require random_keyframes=false")
        if kwargs.get("reasoning_metadata") is not None:
            raise ValueError("Annotated samples cannot be combined with reasoning_metadata")
        if kwargs.get("use_default_keyframe", False):
            raise ValueError("Annotated samples require use_default_keyframe=false")
        path = Path(annotations_path).expanduser()
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{path}: annotations must be a nonempty JSON list")

        samples = []
        for index, row in enumerate(rows):
            context = f"{path}: annotation {index + 1}"
            if not isinstance(row, dict):
                raise ValueError(f"{context} must be a JSON object")
            clip_id = row.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id.strip():
                raise ValueError(f"{context}: clip_id must be a nonempty string")
            timestamp = row.get("t0_relative")
            if type(timestamp) is not int:
                raise ValueError(
                    f"{context}: t0_relative must be integer clip-relative microseconds"
                )
            sample = {"clip_id": clip_id, "t0_relative": timestamp}
            if "nav_text" in row:
                nav = row["nav_text"]
                if isinstance(nav, str):
                    nav = [nav]
                if (
                    not isinstance(nav, list)
                    or len(nav) != 1
                    or not isinstance(nav[0], str)
                    or not nav[0].strip()
                ):
                    raise ValueError(
                        f"{context}: nav_text must be a nonempty string or single-string list"
                    )
                if validate_navigation_format:
                    validate_navigation(nav[0], context)
                sample["nav_text"] = list(nav)
            elif require_navigation:
                raise ValueError(f"{context}: nav_text is required for navigation training")
            if "cot" in row:
                cot = row["cot"]
                if not isinstance(cot, str) or not cot.strip():
                    raise ValueError(f"{context}: cot must be a nonempty string")
                sample["cot"] = cot
            elif require_cot:
                raise ValueError(f"{context}: cot is required for reasoning training")
            samples.append(sample)

        # Delay processor construction until annotations pass metadata validation.
        preprocess = kwargs.pop("preprocess", None)
        if isinstance(kwargs.get("model_config"), dict):
            kwargs["model_config"] = OmegaConf.create(kwargs["model_config"])
        super().__init__(preprocess=None, **kwargs)
        if self.dataset.reasoning_db is not None:
            raise ValueError("Annotated samples cannot use a reasoning metadata backend")
        history_margin = max(
            -int(self.frame_offsets_us[0]),
            -int(self.front_frame_offsets_us[0]),
            (self.num_history_steps - 1) * self.time_step_us,
        )
        latest_t0 = 20_000_000 - self.num_future_steps * self.time_step_us
        for index, sample in enumerate(samples):
            if not history_margin <= sample["t0_relative"] <= latest_t0:
                raise ValueError(
                    f"{path}: annotation {index + 1}: t0_relative must be between "
                    f"{history_margin} and {latest_t0} microseconds to keep history, "
                    "camera frames, and future inside the 20-second PAI clip"
                )
        allowed_clip_ids = set(self.clip_ids)
        self.samples = [sample for sample in samples if sample["clip_id"] in allowed_clip_ids]
        dropped = len(samples) - len(self.samples)
        if not self.samples:
            raise ValueError(
                f"{path}: no annotated samples remain in the selected clip index "
                f"(chunk_ids={kwargs.get('chunk_ids')!r}); check annotation clip IDs and chunk_ids"
            )
        if dropped:
            logger.warning(
                "%s: filtered out %d/%d annotations outside the selected clip index "
                "(chunk_ids=%r); keeping %d",
                path, dropped, len(samples), kwargs.get("chunk_ids"), len(self.samples),
            )
        self.clip_ids = [sample["clip_id"] for sample in self.samples]
        self.preprocess = (
            instantiate(preprocess, _partial_=True)(model_config=kwargs.get("model_config"))
            if preprocess is not None
            else None
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load the row's event timestamp and pass its labels to preprocessing."""
        row = self.samples[index]
        text_fields = {}
        if "nav_text" in row:
            text_fields["nav_text"] = list(row["nav_text"])
        if "cot" in row:
            text_fields["cot"] = row["cot"]
        return self._load_sample(row["clip_id"], row["t0_relative"], text_fields)
