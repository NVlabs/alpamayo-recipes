# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-only ONCE and Tartan Drive 2.0 adapters for the public Edge recipe.

Both adapters require a recording-level split manifest; they never split windows
randomly. Split IDs alone do not establish geographic or temporal separation.
Images must be real, causal captures. No missing cameras, navigation instructions
or reasoning labels are synthesized. Positions are in meters, poses use XYZW
quaternions, and returned trajectories are X-forward/Y-left/Z-up in the anchor
body frame. Per-recording odometry origins are not global geographic coordinates.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import tarfile
import tempfile
import warnings
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation
from PIL import Image
from torch.utils.data import Dataset

from alpamayo2edge_sft.dataset import sample_linear_poses, validate_camera_pixel_bounds
from alpamayo2edge_sft.driving_splits import MIN_TIME_GAP_SECONDS, audit_split, load_split_manifest

logger = logging.getLogger(__name__)

_ONCE_AXES = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _seconds_us(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be positive seconds")
    if not np.isfinite(value) or value <= 0 or value * 1_000_000 > np.iinfo(np.int64).max:
        raise ValueError(f"{label} must be finite positive seconds")
    converted = round(value * 1_000_000)
    if converted <= 0:
        raise ValueError(f"{label} must be at least one microsecond")
    return converted


def _safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path.as_posix()


class _Images:
    """Read an extracted directory or uncompressed tar without extracting files.

    Tar offsets may be cached only at an explicitly supplied location. A stat
    fingerprint invalidates the index when an archive changes. File handles are
    opened per read so the reader can be shared by DataLoader worker processes.
    """

    def __init__(self, path: Path, cache_dir: str | Path | None = None) -> None:
        self.path = path
        self.is_directory = path.is_dir()
        self.timestamp_entries = {}
        if self.is_directory:
            self.entries = {
                file.relative_to(path).as_posix(): None
                for file in sorted(path.rglob("*"))
                if file.is_file() and file.suffix.lower() in _IMAGE_SUFFIXES
            }
        else:
            stat = path.stat()
            self._archive_stat = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            # Version the cache so old image-only indices cannot hide a clock.
            identity = f"v2:{path.resolve()}:{self._archive_stat}"
            cache = None
            if cache_dir is not None:
                directory = Path(cache_dir).expanduser()
                directory.mkdir(parents=True, exist_ok=True)
                cache = directory / (hashlib.sha256(identity.encode()).hexdigest() + ".json")
            if cache is not None and cache.exists():
                cached = json.loads(cache.read_text(encoding="utf-8"))
                self.entries = cached["images"]
                self.timestamp_entries = cached["timestamps"]
            else:
                self.entries = {}
                # r: deliberately rejects compressed archives; byte offsets must
                # point into the original file for os.pread to be correct.
                with tarfile.open(path, "r:") as archive:
                    for member in archive:
                        name = _safe_member(member.name)
                        if member.isfile() and PurePosixPath(name).suffix.lower() in _IMAGE_SUFFIXES:
                            if name in self.entries:
                                raise ValueError(f"Duplicate image archive member: {name}")
                            self.entries[name] = [member.offset_data, member.size]
                        elif member.isfile() and PurePosixPath(name).name == "timestamps.txt":
                            if name in self.timestamp_entries:
                                raise ValueError(f"Duplicate timestamp archive member: {name}")
                            self.timestamp_entries[name] = [member.offset_data, member.size]
                if cache is not None:
                    with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, delete=False) as out:
                        temporary = Path(out.name)
                        json.dump({"images": self.entries, "timestamps": self.timestamp_entries}, out)
                    os.replace(temporary, cache)
        if not self.entries:
            raise ValueError(f"No image files found in {path}")

    def read_camera_timestamps(self) -> tuple[str, bytes] | None:
        """Read the clock bundled with the images, never another modality's clock."""
        if not self.is_directory:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != self._archive_stat:
                raise ValueError(f"Image archive changed after its index was loaded: {self.path}")
        parents = {PurePosixPath(name).parent for name in self.entries}
        if len(parents) != 1:
            raise ValueError("Tartan images must share one directory within the camera export")
        sibling = (next(iter(parents)) / "timestamps.txt").as_posix()
        candidates = {sibling, "timestamps.txt"}
        found = sorted(name for name in candidates if (
            (self.path / name).is_file() if self.is_directory else name in self.timestamp_entries
        ))
        if len(found) > 1:
            raise ValueError("Ambiguous camera timestamps.txt files in the image export")
        if not found:
            return None
        name = found[0]
        if self.is_directory:
            return name, (self.path / name).read_bytes()
        offset, size = self.timestamp_entries[name]
        with self.path.open("rb", buffering=0) as stream:
            data = os.pread(stream.fileno(), size, offset)
        if len(data) != size:
            raise OSError(f"Short timestamp read from {self.path}: {name}")
        return name, data

    def read(self, name: str) -> np.ndarray:
        if self.is_directory:
            with Image.open(self.path / name) as image:
                return np.asarray(image.convert("RGB")).copy()
        offset, size = self.entries[name]
        with self.path.open("rb", buffering=0) as stream:
            data = os.pread(stream.fileno(), size, offset)
        if len(data) != size:
            raise OSError(f"Short image read from {self.path}: {name}")
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB")).copy()


def _pose_container(times: np.ndarray, xyz: np.ndarray, quaternions: np.ndarray):
    times = np.asarray(times)
    xyz = np.asarray(xyz, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or times.dtype != np.int64
            or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or xyz.shape != (len(times), 3) or quaternions.shape != (len(times), 4)
            or not np.isfinite(xyz).all() or not np.isfinite(quaternions).all()):
        raise ValueError("Poses require finite XYZ/XYZW values and increasing int64 microseconds")
    if np.any(np.abs(np.linalg.norm(quaternions, axis=1) - 1) >= 1e-3):
        raise ValueError("Pose XYZW quaternions must be normalized")
    return SimpleNamespace(timestamps=times, values=SimpleNamespace(pose=SimpleNamespace(
        translation=xyz, rotation=Rotation.from_quat(quaternions))))


class _DrivingDataset(Dataset):
    def __init__(
        self, data_root: str, split_manifest: str, split: str, *, dataset: str,
        model_config: Any | None = None, preprocess: Any | None = None,
        num_history_steps: int | None = None, num_future_steps: int = 64,
        time_step: float = 0.1, frame_offsets: list[float] | None = None,
        camera_pixel_bounds: dict[int, list[int]] | None = None,
        max_samples: int | None = None, data_seed: int = 42,
        sample_stride_seconds: float = 2.0, max_pose_gap_seconds: float,
        max_image_age_seconds: float, cache_dir: str | None = None,
        split_audit: Mapping[str, Any] | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        self._split_manifest_path = Path(split_manifest).expanduser()
        self.split_manifest, self.split_manifest_sha256 = load_split_manifest(
            self._split_manifest_path, dataset, return_sha256=True)
        self.split = split
        self.root = Path(data_root).expanduser()
        self.cache_dir = cache_dir
        self.selected_recordings = list(self.split_manifest[split])
        if model_config is not None:
            count = (model_config["tokens_per_history_traj"] if isinstance(model_config, Mapping)
                     else model_config.tokens_per_history_traj)
            if type(count) is not int or count <= 0 or count % 3:
                raise ValueError("Edge history requires three delta tokens per pose pair")
            expected = count // 3 + 1
            if num_history_steps is not None and num_history_steps != expected:
                raise ValueError("num_history_steps does not match checkpoint history tokens")
            num_history_steps = expected
        if num_history_steps is None or type(num_history_steps) is not int or num_history_steps < 2:
            raise ValueError("Provide model_config or num_history_steps >= 2")
        self.num_history_steps = num_history_steps
        self.num_future_steps = _positive_integer(num_future_steps, "num_future_steps")
        self.time_step_us = _seconds_us(time_step, "time_step")
        self.sample_stride_us = _seconds_us(sample_stride_seconds, "sample_stride_seconds")
        self.max_pose_gap_us = _seconds_us(max_pose_gap_seconds, "max_pose_gap_seconds")
        self.max_image_age_us = _seconds_us(max_image_age_seconds, "max_image_age_seconds")
        offsets = [-2., -1., -.5, -.1, 0.] if frame_offsets is None else frame_offsets
        values = np.asarray(offsets)
        if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all() or np.any(values > 0)
                or np.any(np.diff(values) <= 0) or values[-1] != 0):
            raise ValueError("frame_offsets must increase, be nonpositive seconds, and end at zero")
        self.frame_offsets_us = np.rint(values * 1_000_000).astype(np.int64)
        if np.any(np.diff(self.frame_offsets_us) <= 0):
            raise ValueError("frame_offsets must be distinct at microsecond precision")
        self.camera_pixel_bounds = validate_camera_pixel_bounds(
            {1: [163840, 196608]} if camera_pixel_bounds is None else camera_pixel_bounds)
        if set(self.camera_pixel_bounds) != {1}:
            raise ValueError("Single-front-camera adapters require camera_pixel_bounds for camera 1 only")
        if max_samples is not None:
            _positive_integer(max_samples, "max_samples")
        if type(data_seed) is not int or data_seed < 0:
            raise ValueError("data_seed must be a nonnegative integer")
        policy = {"temporal_only": True, "min_time_gap_seconds": 86400,
                  "min_location_gap_m": 250}
        if split_audit is not None:
            if not isinstance(split_audit, Mapping):
                raise ValueError("split_audit must be a mapping of temporal_only/min_time_gap_seconds/min_location_gap_m")
            unknown = set(split_audit) - policy.keys()
            if unknown:
                raise ValueError(f"Unknown split_audit options: {sorted(map(str, unknown))}")
            policy.update(split_audit)
        # Recompute from the exact manifest bytes loaded above; a saved report
        # or manifest metadata cannot authorize skipping the runtime audit.
        report = audit_split(self.split_manifest, **policy)
        context_us = (max((self.num_history_steps - 1) * self.time_step_us,
                          -int(self.frame_offsets_us[0]))
                      + self.num_future_steps * self.time_step_us)
        required_gap = max(MIN_TIME_GAP_SECONDS, context_us / 1e6)
        if policy["min_time_gap_seconds"] < required_gap:
            raise ValueError(
                f"split_audit.min_time_gap_seconds must be at least {required_gap:g} seconds "
                "for the configured requested image/history/future context"
            )
        report.update(manifest_sha256=self.split_manifest_sha256,
                      policy_source="runtime_configuration",
                      requested_context_span_seconds=context_us / 1e6,
                      minimum_required_gap_seconds=required_gap)
        self.split_audit_report = report
        if not report["passed"]:
            failed = [pair for pair in report["pairs"] if not pair["temporal_passed"] or (
                not policy["temporal_only"] and pair["spatial"]["status"] != "passed")]
            details = "; ".join(
                f"{pair['train']}/{pair['val']}: gap={pair['time_gap_seconds']:g}s, "
                f"spatial={pair['spatial']['status']}" for pair in failed[:3]
            )
            raise ValueError(f"{dataset} split audit failed ({details}); policy={policy}")
        logger.info("%s split audit: %s; minimum gap=%gs; spatial_passed=%s",
                    dataset, report["status"], report["min_time_gap_seconds"], report["spatial_passed"])
        self.max_samples, self.data_seed = max_samples, data_seed
        self.preprocess = (instantiate(preprocess, _partial_=True)(model_config=model_config)
                           if preprocess is not None else None)
        self._samples = []
        self.index_statistics = {}
        self.source_intervals = {}

    def _split_audit_protocol(self) -> dict[str, Any]:
        """Return the enforced audit scope, bound to the manifest actually loaded."""
        if hashlib.sha256(self._split_manifest_path.read_bytes()).hexdigest() != self.split_manifest_sha256:
            raise ValueError("Split manifest changed after the dataset was loaded and audited")
        # Pairwise details stay on the dataset report. Keep evaluation metadata
        # bounded in size even for manifests with many cross-split pairs.
        keys = ("schema_version", "dataset", "status", "passed", "full_validation_passed",
                "temporal_passed", "spatial_passed", "temporal_only", "policy", "policy_source",
                "manifest_sha256", "recording_counts", "min_time_gap_seconds",
                "requested_context_span_seconds", "minimum_required_gap_seconds", "limitations")
        return {key: deepcopy(self.split_audit_report[key]) for key in keys}

    def _index_sequence(self, sequence: dict[str, Any]) -> None:
        times, pose_times = sequence["image_times"], sequence["poses"].timestamps
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("Image timestamps must strictly increase in microseconds")
        start, end = int(min(times[0], pose_times[0])), int(max(times[-1], pose_times[-1]))
        metadata = self.split_manifest["recordings"][sequence["name"]]
        if metadata["start_time_us"] > start or metadata["end_time_us"] < end:
            raise ValueError(f"Manifest interval does not cover raw image/pose clocks for {sequence['name']}")
        self.source_intervals[sequence["name"]] = {"start_time_us": start, "end_time_us": end}
        last_selected = None
        count = len(self._samples)
        for index, anchor in enumerate(times):
            if last_selected is not None and anchor - last_selected < self.sample_stride_us:
                continue
            lower = anchor - (self.num_history_steps - 1) * self.time_step_us
            upper = anchor + self.num_future_steps * self.time_step_us
            if lower < pose_times[0] or upper > pose_times[-1]:
                continue
            lo = max(0, np.searchsorted(pose_times, lower, side="right") - 1)
            hi = min(len(pose_times), np.searchsorted(pose_times, upper, side="left") + 1)
            if np.any(np.diff(pose_times[lo:hi]) > self.max_pose_gap_us):
                continue
            requested = anchor + self.frame_offsets_us
            image_indices = np.searchsorted(times, requested, side="right") - 1
            if np.any(image_indices < 0):
                continue
            if np.any(requested - times[image_indices] > self.max_image_age_us):
                continue
            if any(int(i) not in sequence["image_names"] for i in image_indices):
                continue
            self._samples.append((sequence, index, image_indices))
            last_selected = anchor
        self.index_statistics[sequence["name"]] = {"usable_windows": len(self._samples) - count}
        if len(self._samples) == count:
            raise ValueError(f"No complete image/history/future windows for {sequence['name']}")

    def _finish(self) -> None:
        if self.max_samples is not None and len(self._samples) > self.max_samples:
            indices = sorted(random.Random(self.data_seed).sample(range(len(self._samples)), self.max_samples))
            self._samples = [self._samples[i] for i in indices]
        if not self._samples:
            raise ValueError("No complete image/history/future windows remain")
        self.clip_ids = [item[0]["name"] for item in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence, anchor_index, image_indices = self._samples[index]
        anchor = int(sequence["image_times"][anchor_index])
        image_times = sequence["image_times"][image_indices]
        images = [sequence["images"].read(sequence["image_names"][int(i)]) for i in image_indices]
        if len({image.shape for image in images}) != 1:
            raise ValueError("Image dimensions change inside a temporal window")
        history_offsets = np.arange(1 - self.num_history_steps, 1) * self.time_step_us
        future_offsets = np.arange(1, self.num_future_steps + 1) * self.time_step_us
        # Bound interpolation to the local window: long recordings can lose
        # sub-frame precision when their entire time span is cast to float32.
        poses = sequence["poses"]
        lo = max(0, np.searchsorted(poses.timestamps, anchor + history_offsets[0], side="right") - 1)
        hi = min(len(poses.timestamps), np.searchsorted(poses.timestamps, anchor + future_offsets[-1], side="left") + 1)
        local = _pose_container(poses.timestamps[lo:hi], poses.values.pose.translation[lo:hi],
                                poses.values.pose.rotation[lo:hi].as_quat())
        xyz, rotation = sample_linear_poses(local, anchor, np.r_[history_offsets, future_offsets])
        reference = self.num_history_steps - 1
        inverse = rotation[reference].inv()
        xyz = inverse.apply(xyz - xyz[reference]) @ sequence["axes"].T
        rot = sequence["axes"][None] @ (inverse * rotation).as_matrix() @ sequence["axes"].T[None]
        sample = {
            "clip_id": sequence["name"], "requested_t0_us": anchor, "t0_us": anchor,
            "image_frames": torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(1),
            "camera_indices": torch.ones(len(images), dtype=torch.int64),
            "absolute_timestamps": torch.from_numpy(image_times.copy()),
            "relative_timestamps": torch.tensor((image_times - anchor) / 1e6, dtype=torch.float32),
            "camera_pixel_bounds": self.camera_pixel_bounds,
        }
        for name, selected in (("history", slice(None, reference + 1)),
                               ("future", slice(reference + 1, None))):
            sample[f"ego_{name}_xyz"] = torch.from_numpy(xyz[selected].astype(np.float32)).unsqueeze(0)
            sample[f"ego_{name}_rot"] = torch.from_numpy(rot[selected].astype(np.float32)).unsqueeze(0)
        if self.preprocess is not None:
            sample["tokenized_data"] = self.preprocess(data=sample)
        return sample


class ONCEDataset(_DrivingDataset):
    """Read ONCE cam01 with millisecond frame IDs and XYZW + XYZ poses.

    Layout: extracted ``data/ID/ID.json`` and ``data/ID/cam01/*.jpg``, or original
    ``PARTITION_anno/PARTITION_infos.tar`` and ``PARTITION_camera/PARTITION_cam01.tar``.
    Manifest ``recordings.ID.source_partition`` locates a tar bundle independently
    of whether that sequence is assigned to training or validation. Without it,
    conventional bundles are searched and ambiguous duplicate IDs are rejected.

    ONCE poses are relative to each sequence origin. The sensor origin is LiDAR,
    not a known rear axle. Sparse poses are linearly interpolated; this does not
    create high-frequency measured ground truth. Source conventions:
    https://once-for-auto-driving.github.io/documentation.html
    """

    def __init__(self, data_root: str, split_manifest: str, split: str = "train", *,
                 model_config=None, preprocess=None, num_history_steps=None, num_future_steps=64,
                 time_step=0.1, frame_offsets=None, camera_pixel_bounds=None, max_samples=None,
                 data_seed=42, sample_stride_seconds=2.0, max_pose_gap_seconds=1.0,
                 max_image_age_seconds=0.55, cache_dir=None, split_audit=None):
        super().__init__(data_root, split_manifest, split, dataset="once", model_config=model_config,
                         preprocess=preprocess, num_history_steps=num_history_steps,
                         num_future_steps=num_future_steps, time_step=time_step, frame_offsets=frame_offsets,
                         camera_pixel_bounds=camera_pixel_bounds, max_samples=max_samples, data_seed=data_seed,
                         sample_stride_seconds=sample_stride_seconds, max_pose_gap_seconds=max_pose_gap_seconds,
                         max_image_age_seconds=max_image_age_seconds, cache_dir=cache_dir,
                         split_audit=split_audit)
        bundle_docs = {}
        readers = {}
        for name in self.selected_recordings:
            directory = self.root / "data" / name
            if not directory.exists():
                directory = self.root / name
            annotation = directory / f"{name}.json"
            if annotation.is_file() and (directory / "cam01").is_dir():
                document = json.loads(annotation.read_text(encoding="utf-8"))
                images = _Images(directory / "cam01")
                prefix = ""
            else:
                partition = self.split_manifest.get("recordings", {}).get(name, {}).get("source_partition")
                bundles = ([self.root / f"{partition}_anno" / f"{partition}_infos.tar"]
                           if partition else sorted(self.root.glob("*_anno/*_infos.tar")))
                candidates = []
                for bundle in bundles:
                    if bundle not in bundle_docs:
                        docs = {}
                        with tarfile.open(bundle, "r:") as archive:
                            for member in archive:
                                if member.isfile() and member.name.endswith(".json"):
                                    member_name = _safe_member(member.name)
                                    key = PurePosixPath(member_name).stem
                                    if key in docs:
                                        raise ValueError(f"Duplicate sequence annotation: {key}")
                                    docs[key] = json.load(archive.extractfile(member))
                        bundle_docs[bundle] = docs
                    if name in bundle_docs[bundle]:
                        candidates.append(bundle)
                if len(candidates) != 1:
                    raise ValueError(f"Expected exactly one annotation for {name}; found {len(candidates)}")
                bundle = candidates[0]
                partition = bundle.name.removesuffix("_infos.tar")
                camera = self.root / f"{partition}_camera" / f"{partition}_cam01.tar"
                if camera not in readers:
                    readers[camera] = _Images(camera, cache_dir)
                document, images = bundle_docs[bundle][name], readers[camera]
                prefix = f"data/{name}/cam01/"
            frames = document.get("frames")
            if not isinstance(frames, list) or len(frames) < 2:
                raise ValueError(f"ONCE {name} requires at least two frames")
            raw_times = [frame.get("frame_id") for frame in frames]
            if any(not isinstance(value, str) or not value.isdigit() for value in raw_times):
                raise ValueError("ONCE frame_id must be a decimal millisecond timestamp string")
            times = np.array([int(value) for value in raw_times], dtype=np.int64)
            if np.any(times > np.iinfo(np.int64).max // 1000):
                raise ValueError("ONCE millisecond timestamps exceed int64 microsecond range")
            times *= 1000
            pose = np.asarray([frame["pose"] for frame in frames], dtype=np.float64)
            if pose.shape != (len(times), 7):
                raise ValueError("ONCE poses must contain quaternion XYZW then translation XYZ")
            # Input annotation order is checked, not silently sorted or repaired.
            poses = _pose_container(times, pose[:, 4:7], pose[:, :4])
            if any(frame.get("sequence_id", name) != name for frame in frames):
                raise ValueError(f"ONCE frame sequence_id disagrees with manifest ID {name}")
            names = {i: f"{prefix}{value}.jpg" for i, value in enumerate(raw_times)
                     if f"{prefix}{value}.jpg" in images.entries}
            self._index_sequence({"name": name, "images": images, "image_names": names,
                                  "image_times": times, "poses": poses, "axes": _ONCE_AXES})
        self._finish()

    def evaluation_sampling_protocol(self) -> dict[str, Any]:
        """Describe the resolved adapter semantics, not measured source rates."""
        return {
            "adapter_variant": "once_causal_images_interpolated_lidar_poses_v1",
            "split_audit": self._split_audit_protocol(),
            "source": {
                "documentation": "https://once-for-auto-driving.github.io/documentation.html",
                "nominal_camera_hz": 2,
                "nominal_pose_hz": 2,
                "rate_basis": "provider release specification; actual intervals may vary",
            },
            "images": {
                "requested_offsets_seconds": (self.frame_offsets_us / 1e6).tolist(),
                "selection": "latest annotated frame at or before request; reject missing selected images",
                "duplicate_frames": "retained",
                "timestamp_labels": "actual capture times relative to anchor",
                "max_age_from_request_seconds": self.max_image_age_us / 1e6,
            },
            "trajectory": {
                "source_origin": "LiDAR sensor center",
                "target_frame": "anchor-relative X-forward/Y-left/Z-up; no rear-axle translation",
                "translation_interpolation": "piecewise linear",
                "rotation_interpolation": "sign-aligned normalized linear quaternion interpolation (NLERP)",
                "labels": "interpolated from released poses, not measured high-frequency rear-axle ground truth",
                "time_step_seconds": self.time_step_us / 1e6,
                "history_steps": self.num_history_steps,
                "future_steps": self.num_future_steps,
                "max_source_gap_seconds": self.max_pose_gap_us / 1e6,
            },
            "minimum_anchor_stride_seconds": self.sample_stride_us / 1e6,
        }


class TartanDriveDataset(_DrivingDataset):
    """Read Tartan Drive 2.0 camera exports and estimated body-frame odometry.

    The camera directory/archive must contain timestamps.txt and contiguous
    zero-based numeric image filenames, one image per strictly increasing clock
    row. This uses the camera modality's own clock; it assumes an unmodified,
    correctly associated provider export, not independently verified physical
    sensor synchronization. Odometry is super_odom/{timestamps.txt,odometry.npy}
    with XYZ metres then XYZW quaternions in X-forward/Y-left/Z-up coordinates.

    A missing camera clock fails by default. Explicit legacy opt-in may use
    tartanvo_odom/timestamps.txt by numeric image index, with unverified timing
    recorded in the evaluation protocol. No malformed camera clock falls back.
    Source: https://github.com/castacks/tartan_drive_2.0
    """

    def __init__(self, data_root: str, split_manifest: str, split: str = "train", *,
                 model_config=None, preprocess=None, num_history_steps=None, num_future_steps=64,
                 time_step=0.1, frame_offsets=None, camera_pixel_bounds=None, max_samples=None,
                 data_seed=42, sample_stride_seconds=2.0, max_pose_gap_seconds=0.1,
                 max_image_age_seconds=0.15, cache_dir=None,
                 allow_unverified_image_timestamps=False, legacy_image_timestamp_evidence=None,
                 split_audit=None):
        if type(allow_unverified_image_timestamps) is not bool:
            raise ValueError("allow_unverified_image_timestamps must be a boolean")
        if legacy_image_timestamp_evidence is not None and (
            not isinstance(legacy_image_timestamp_evidence, str) or not legacy_image_timestamp_evidence.strip()
        ):
            raise ValueError("legacy_image_timestamp_evidence must be a nonempty description")
        if allow_unverified_image_timestamps and legacy_image_timestamp_evidence is None:
            raise ValueError("Legacy image timing requires legacy_image_timestamp_evidence describing the export and assumption")
        super().__init__(data_root, split_manifest, split, dataset="tartan", model_config=model_config,
                         preprocess=preprocess, num_history_steps=num_history_steps,
                         num_future_steps=num_future_steps, time_step=time_step, frame_offsets=frame_offsets,
                         camera_pixel_bounds=camera_pixel_bounds, max_samples=max_samples, data_seed=data_seed,
                         sample_stride_seconds=sample_stride_seconds, max_pose_gap_seconds=max_pose_gap_seconds,
                         max_image_age_seconds=max_image_age_seconds, cache_dir=cache_dir,
                         split_audit=split_audit)
        self.clock_cleanup = {}
        self.image_timestamp_sources = {}
        self._timestamp_checks = {}
        self._pose_clock_checks = {}
        for name in self.selected_recordings:
            directory = self.root / name
            pose_path = directory / "super_odom" / "timestamps.txt"
            pose_bytes = pose_path.read_bytes()
            pose_times = _parse_seconds_clock(pose_bytes, str(pose_path))
            pose_digest = hashlib.sha256(pose_bytes).hexdigest()
            self._pose_clock_checks[name] = (pose_path, pose_digest)
            odometry = np.load(directory / "super_odom" / "odometry.npy", allow_pickle=False)
            if (odometry.ndim != 2 or odometry.shape[1] < 7 or len(odometry) != len(pose_times)
                    or not np.isfinite(odometry[:, :7]).all()):
                raise ValueError("Tartan odometry requires finite XYZ/XYZW rows matching its clock")
            duplicates = np.flatnonzero(np.diff(pose_times) == 0)
            if duplicates.size and not np.allclose(odometry[duplicates, :7], odometry[duplicates + 1, :7], atol=1e-8, rtol=0):
                raise ValueError("Duplicate pose timestamp has conflicting pose labels")
            pose_keep = np.r_[True, np.diff(pose_times) > 0]
            poses = _pose_container(pose_times[pose_keep], odometry[pose_keep, :3], odometry[pose_keep, 3:7])
            image_path = directory / "image_left_color"
            if not image_path.is_dir():
                image_path = directory / "image_left_color.tar"
            images = _Images(image_path, cache_dir)
            indexed = {}
            for key in images.entries:
                stem = PurePosixPath(key).stem
                if not stem.isdigit():
                    raise ValueError("Tartan image filenames must be zero-based decimal indices")
                index = int(stem)
                if index in indexed:
                    raise ValueError(f"Ambiguous duplicate Tartan image index {index}")
                indexed[index] = key
            camera_clock = images.read_camera_timestamps()
            if camera_clock is not None:
                clock_name, clock_bytes = camera_clock
                image_times = _parse_seconds_clock(clock_bytes, f"{image_path}:{clock_name}")
                if (len(indexed) != len(image_times)
                        or sorted(indexed) != list(range(len(image_times)))):
                    raise ValueError("Camera timestamps.txt requires exactly one image per row with contiguous zero-based indices")
                if np.any(np.diff(image_times) <= 0):
                    raise ValueError("Camera timestamps.txt must be strictly increasing at microsecond precision")
                mode = "camera_modality_timestamps"
                source = f"image_left_color/{clock_name}" if images.is_directory else f"image_left_color.tar:{clock_name}"
                evidence = "Camera modality clock and numeric image indices from the same export"
                self._timestamp_checks[name] = (images, clock_name, None, hashlib.sha256(clock_bytes).hexdigest())
            else:
                if not allow_unverified_image_timestamps:
                    raise ValueError(
                        f"{name}: missing camera timestamps.txt in image_left_color/ or image_left_color.tar; "
                        "provide the original camera export, or explicitly set allow_unverified_image_timestamps=true "
                        "and legacy_image_timestamp_evidence for an unverified legacy demonstration"
                    )
                clock_path = directory / "tartanvo_odom" / "timestamps.txt"
                clock_bytes = clock_path.read_bytes()
                image_times = _parse_seconds_clock(clock_bytes, str(clock_path))
                if min(indexed) != 0 or max(indexed) >= len(image_times):
                    raise ValueError("Tartan image indices do not match the supplied image clock")
                mode = "unverified_tartanvo_index"
                source = "tartanvo_odom/timestamps.txt"
                evidence = legacy_image_timestamp_evidence
                self._timestamp_checks[name] = (None, None, clock_path, hashlib.sha256(clock_bytes).hexdigest())
                warnings.warn(
                    f"{name}: unverified Tartan image timing uses TartanVO rows by image index; "
                    "counts and numeric filenames do not establish camera synchronization",
                    RuntimeWarning, stacklevel=2,
                )
            index_bytes = json.dumps(sorted(indexed.items()), separators=(",", ":")).encode()
            self.image_timestamp_sources[name] = {
                "mode": mode, "source": source, "source_evidence": evidence,
                "clock_sha256": hashlib.sha256(clock_bytes).hexdigest(),
                "pose_clock_sha256": pose_digest,
                "image_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "timestamp_unit": "seconds converted to integer microseconds",
                "image_count": len(indexed), "clock_rows": len(image_times),
            }
            image_keep = np.flatnonzero(np.r_[True, np.diff(image_times) > 0])
            names = {new: indexed[int(old)] for new, old in enumerate(image_keep) if int(old) in indexed}
            self.clock_cleanup[name] = {
                "identical_pose_duplicates_removed": int((~pose_keep).sum()),
                "repeated_image_timestamp_rows_removed": int(len(image_times) - len(image_keep)),
            }
            self._index_sequence({"name": name, "images": images, "image_names": names,
                                  "image_times": image_times[image_keep], "poses": poses, "axes": np.eye(3)})
        self._finish()

    def evaluation_sampling_protocol(self) -> dict[str, Any]:
        """Record the clock actually loaded and reject later metadata replacement."""
        for name, (images, clock_name, path, digest) in self._timestamp_checks.items():
            if images is not None:
                camera_clock = images.read_camera_timestamps()
                if camera_clock is None or camera_clock[0] != clock_name:
                    raise ValueError(f"{name}: camera timestamp source changed after the dataset was loaded")
                data = camera_clock[1]
            else:
                data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"{name}: image timestamp source changed after the dataset was loaded")
            pose_path, pose_digest = self._pose_clock_checks[name]
            if hashlib.sha256(pose_path.read_bytes()).hexdigest() != pose_digest:
                raise ValueError(f"{name}: pose clock changed after the dataset was loaded")
        return {
            "adapter_variant": "tartan_camera_clock_v1",
            "split_audit": self._split_audit_protocol(),
            "contains_unverified_timing": any(
                row["mode"] == "unverified_tartanvo_index" for row in self.image_timestamp_sources.values()
            ),
            "recordings": {name: dict(row) for name, row in self.image_timestamp_sources.items()},
            "assurance": "Validates export clock/index structure, not independent physical sensor synchronization or image payload identity",
            "requested_offsets_seconds": (self.frame_offsets_us / 1e6).tolist(),
            "time_step_seconds": self.time_step_us / 1e6,
            "history_steps": self.num_history_steps, "future_steps": self.num_future_steps,
            "max_image_age_seconds": self.max_image_age_us / 1e6,
            "max_pose_gap_seconds": self.max_pose_gap_us / 1e6,
            "minimum_anchor_stride_seconds": self.sample_stride_us / 1e6,
        }


def _load_seconds_clock(path: Path) -> np.ndarray:
    return _parse_seconds_clock(path.read_bytes(), str(path))


def _parse_seconds_clock(data: bytes, context: str) -> np.ndarray:
    values = np.loadtxt(io.BytesIO(data), ndmin=1)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > np.iinfo(np.int64).max / 1e6)
            or np.any(np.diff(values) < 0)):
        raise ValueError(f"Expected finite nondecreasing second timestamps in {context}")
    return np.rint(values * 1_000_000).astype(np.int64)
