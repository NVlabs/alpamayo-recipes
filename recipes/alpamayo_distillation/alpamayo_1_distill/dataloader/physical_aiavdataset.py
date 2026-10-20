import torch
import torch.distributed as dist
from torch.utils.data import Dataset

import numpy as np
import scipy.spatial.transform as spt
from transformers import AutoTokenizer

from einops import rearrange
from alpamayo1_distill.defaults import (
    DEFAULT_GOLD_CLIP_IDS_PATH,
    PAI_FULL_VAL_SIZE,
    PAI_GOLD_EVAL_SIZE,
    PAI_GOLD_TRAIN_OVERLAP,
    PAI_TRAIN_SIZE,
    PAI_TRAIN_SPLIT_SIZE,
)
import alpamayo1_distill.dataloader.physical_ai_av.src.physical_ai_av.dataset as physical_ai_av


class _InfeasibleClipError(RuntimeError):
    """A clip cannot supply a valid sample (too short, or its egomotion/video ranges don't overlap)."""


# Stride used to walk to a substitute sample when a clip turns out to be infeasible.
# Coprime with most dataset sizes so retries spread out instead of hitting neighbours
# of the same clip (which share the same feasibility verdict).
_RETRY_STRIDE = 9973


class physicalAiAvDataset(Dataset):
    def __init__(self,
                t0_us: int = 5_100_000,
                avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
                maybe_stream: bool = True,
                num_history_steps: int = 16,
                num_future_steps: int = 64,
                time_step: float = 0.1,
                camera_features: list | None = None,
                num_frames: int = 4,
                split: str = 'train',
                tokenizer: AutoTokenizer = None,
                local_files_only: bool = False,
                local_dir: str | None = None,
                gold_eval: bool = False,
                t0_us_list: list[int] | None = None,
                random_t0: bool = False,
                num_samples_per_clip: int = 1,
                t0_margin_us: int = 0,
                t0_seed: int | None = None,
                max_clip_retries: int = 8,
    ):
        if avdi is None:
            self.avdi = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=local_dir, local_files_only=local_files_only)
        else:
            self.avdi = avdi

        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if gold_eval and split != "val":
            raise ValueError("gold_eval is only valid for split='val'")

        with DEFAULT_GOLD_CLIP_IDS_PATH.open() as f:
            gold_clip_ids = [line.strip() for line in f if line.strip()]
        if (
            len(gold_clip_ids) != PAI_GOLD_EVAL_SIZE
            or len(set(gold_clip_ids)) != PAI_GOLD_EVAL_SIZE
        ):
            raise RuntimeError(
                "The packaged gold evaluation list must contain exactly "
                f"{PAI_GOLD_EVAL_SIZE} unique clip IDs"
            )

        df = self.avdi.clip_index
        if gold_eval or split == "train":
            available_clip_ids = set(str(value) for value in df.index.tolist())
            missing_gold_ids = [
                clip_id for clip_id in gold_clip_ids if clip_id not in available_clip_ids
            ]
            if missing_gold_ids:
                raise ValueError(
                    f"{len(missing_gold_ids)}/{PAI_GOLD_EVAL_SIZE} packaged gold "
                    "clips are absent from the local PAI clip index; first missing "
                    f"IDs: {missing_gold_ids[:10]}"
                )
        if gold_eval:
            self.clip_ids = gold_clip_ids
        else:
            self.clip_ids = df.index[df["split"] == split].tolist()

        if split == "train":
            before = len(self.clip_ids)
            if before != PAI_TRAIN_SPLIT_SIZE:
                raise ValueError(
                    "The local PAI train split has the wrong cardinality: "
                    f"expected={PAI_TRAIN_SPLIT_SIZE}, found={before}. Verify that "
                    "the local dataset is the supported snapshot and is complete."
                )
            gold_clip_id_set = set(gold_clip_ids)
            self.clip_ids = [c for c in self.clip_ids if c not in gold_clip_id_set]
            excluded = before - len(self.clip_ids)
            if excluded != PAI_GOLD_TRAIN_OVERLAP or len(self.clip_ids) != PAI_TRAIN_SIZE:
                raise ValueError(
                    "The local PAI train/gold split relationship does not match the "
                    "supported snapshot: "
                    f"expected_overlap={PAI_GOLD_TRAIN_OVERLAP}, found_overlap={excluded}, "
                    f"expected_train_after_holdout={PAI_TRAIN_SIZE}, "
                    f"found_train_after_holdout={len(self.clip_ids)}"
                )
            print(f"[physicalAiAvDataset] Excluded {before - len(self.clip_ids)} eval clips from {split} split "
                  f"({len(self.clip_ids)} remaining)")
        elif not gold_eval and len(self.clip_ids) != PAI_FULL_VAL_SIZE:
            raise ValueError(
                "The local PAI val split has the wrong cardinality: "
                f"expected={PAI_FULL_VAL_SIZE}, found={len(self.clip_ids)}. Verify "
                "that the local dataset is the supported snapshot and is complete."
            )
        self.t0_us = t0_us
        self.maybe_stream = maybe_stream
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.camera_features = camera_features
        self.num_frames = num_frames
        self.split = split
        self.tokenizer = tokenizer
        self.local_files_only = local_files_only

        self.random_t0 = random_t0
        self.t0_margin_us = int(t0_margin_us)
        self.t0_seed = t0_seed
        self.max_clip_retries = int(max_clip_retries)
        self._rng: np.random.Generator | None = None
        self._warned_clips: set[str] = set()
        self._rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

        if random_t0:
            # t0 is drawn per __getitem__ call, so a sample is just a slot on a clip.
            # `num_samples_per_clip` slots per clip => that many independent draws per epoch.
            n_per_clip = max(1, int(num_samples_per_clip))
            self.samples = [(clip_id, None) for clip_id in self.clip_ids for _ in range(n_per_clip)]
            print(f"[physicalAiAvDataset] random t0 sampling enabled for split={split}: "
                  f"{len(self.clip_ids)} clips x {n_per_clip} draws = {len(self.samples)} samples/epoch")
        else:
            _t0_list = t0_us_list if t0_us_list is not None else [t0_us]
            self.samples = [(clip_id, t0) for clip_id in self.clip_ids for t0 in _t0_list]

    def __len__(self):
        return len(self.samples)

    # ---------------- random t0 helpers ----------------
    def _get_rng(self) -> np.random.Generator:
        """Per-worker RNG, created lazily so each dataloader worker draws its own stream.

        With ``t0_seed=None`` the generator is seeded from OS entropy: t0 differs across
        workers, ranks and epochs, at the cost of exact data reproducibility on resume
        (weights/optimizer resume is unaffected). With ``t0_seed`` set the stream is
        reproducible for a fixed (rank, worker) layout; the generator lives in the worker,
        so with ``persistent_workers=True`` it keeps advancing across epochs rather than
        replaying the same t0s.
        """
        if self._rng is None:
            if self.t0_seed is None:
                self._rng = np.random.default_rng()
            else:
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else 0
                self._rng = np.random.default_rng([int(self.t0_seed), int(self._rank), int(worker_id)])
        return self._rng

    def _sample_window_us(self) -> tuple[int, int, int]:
        """Time the sample spans around t0, in microseconds: (history_back, image_back, future_forward)."""
        step_us = self.time_step * 1_000_000
        history_back_us = int((self.num_history_steps - 1) * step_us)
        image_back_us = int((self.num_frames - 1) * step_us)
        future_forward_us = int(self.num_future_steps * step_us)
        return history_back_us, image_back_us, future_forward_us

    def _feasible_t0_range(self, ego_t_min: int, ego_t_max: int,
                           cam_t_min: int, cam_t_max: int) -> tuple[int, int]:
        """Inclusive [lo, hi] range of t0 values fully covered by egomotion *and* every camera.

        Egomotion interpolation has extrapolation disabled (it silently yields NaN, and
        Slerp raises), and the video readers raise when a requested timestamp falls outside
        the decoded frame range — so every history/future/image timestamp must sit inside
        both ranges. That is exactly what this window guarantees.
        """
        history_back_us, image_back_us, future_forward_us = self._sample_window_us()

        lo = max(
            ego_t_min + history_back_us,              # oldest history pose must exist
            cam_t_min + image_back_us,                # oldest image frame must exist
            int(self.num_history_steps * self.time_step * 1_000_000) + 1,  # legacy t0 lower bound
        ) + self.t0_margin_us
        hi = min(
            ego_t_max - future_forward_us,            # furthest future pose must exist
            cam_t_max,                                # image at t0 must exist
        ) - self.t0_margin_us
        return int(lo), int(hi)

    def __getitem__(self, idx):
        if not self.random_t0:
            clip_id, t0_us = self.samples[idx]
            return self._build_sample(clip_id, t0_us)

        # Random mode: a clip whose usable window is empty (clip shorter than the
        # 1.5s history + 6.4s future span, or misaligned egomotion/video coverage)
        # cannot yield a valid sample, so substitute another clip instead of
        # crashing the run or emitting NaN trajectories.
        n = len(self.samples)
        last_err: Exception | None = None
        for attempt in range(self.max_clip_retries):
            clip_id, _ = self.samples[(idx + attempt * _RETRY_STRIDE) % n]
            try:
                return self._build_sample(clip_id, None)
            except _InfeasibleClipError as e:
                last_err = e
                # Once per clip per worker: the same clip is revisited every epoch and
                # the message is only useful the first time.
                if clip_id not in self._warned_clips:
                    self._warned_clips.add(clip_id)
                    print(f"[physicalAiAvDataset] skipping clip {clip_id}: {e}")
        raise RuntimeError(
            f"No feasible clip found for idx={idx} after {self.max_clip_retries} attempts; "
            f"last error: {last_err}"
        )

    def _build_sample(self, clip_id: str, t0_us: int | None):
        """Load one sample. ``t0_us=None`` draws t0 uniformly from the clip's valid window."""

        if self.camera_features is None:
            self.camera_features = [
                self.avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
                self.avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
                self.avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
                self.avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
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

        # Load egomotion data
        egomotion = self.avdi.get_clip_feature(
            clip_id,
            self.avdi.features.LABELS.EGOMOTION,
            maybe_stream=self.maybe_stream,
        )

        # Open the camera readers up front: their frame timestamps bound how late a
        # sample may start, so t0 can only be drawn once every camera is known.
        # Readers hold the compressed video in memory, so each one is closed as soon
        # as its frames are decoded below.
        cameras = [
            (
                cam_feature,
                self.avdi.get_clip_feature(clip_id, cam_feature, maybe_stream=self.maybe_stream),
            )
            for cam_feature in self.camera_features
        ]

        try:
            if t0_us is None:
                ego_t_min, ego_t_max = egomotion.time_range
                cam_t_min = max(int(cam.timestamps.min()) for _, cam in cameras)
                cam_t_max = min(int(cam.timestamps.max()) for _, cam in cameras)
                lo, hi = self._feasible_t0_range(int(ego_t_min), int(ego_t_max), cam_t_min, cam_t_max)
                if hi < lo:
                    history_back_us, _, future_forward_us = self._sample_window_us()
                    raise _InfeasibleClipError(
                        f"no valid t0 (feasible range [{lo}, {hi}] is empty); "
                        f"egomotion=[{int(ego_t_min)}, {int(ego_t_max)}] us, "
                        f"cameras=[{cam_t_min}, {cam_t_max}] us, "
                        f"sample needs {history_back_us + future_forward_us} us of coverage"
                    )
                # Inclusive on both ends: hi itself is a fully covered t0.
                t0_us = int(self._get_rng().integers(lo, hi + 1))
            else:
                assert t0_us > self.num_history_steps * self.time_step * 1_000_000, (
                    "t0_us must be greater than the history time range"
                )

            return self._build_sample_at_t0(clip_id, int(t0_us), egomotion, cameras, camera_name_to_index)
        finally:
            for _, cam in cameras:
                close = getattr(cam, "close", None)
                if close is not None:
                    close()

    def _build_sample_at_t0(self, clip_id, t0_us, egomotion, cameras, camera_name_to_index):
        """Assemble the sample for a t0 that is already known to be inside the clip."""

        # Compute timestamps for trajectory sampling
        # History: [..., t0-0.2s, t0-0.1s, t0] (num_history_steps points ending at t0)
        # Future: [t0+0.1s, t0+0.2s, ..., t0+6.4s] (num_future_steps points after t0)
        history_offsets_us = np.arange(
            -(self.num_history_steps - 1) * self.time_step * 1_000_000,
            self.time_step * 1_000_000 / 2,
            self.time_step * 1_000_000,
        ).astype(np.int64)
        history_timestamps = t0_us + history_offsets_us

        future_offsets_us = np.arange(
            self.time_step * 1_000_000,
            (self.num_future_steps + 0.5) * self.time_step * 1_000_000,
            self.time_step * 1_000_000,
        ).astype(np.int64)
        future_timestamps = t0_us + future_offsets_us

        # Get egomotion at history and future timestamps
        ego_history = egomotion(history_timestamps)
        ego_history_xyz = ego_history.pose.translation  # (num_history_steps, 3)
        ego_history_quat = ego_history.pose.rotation.as_quat()  # (num_history_steps, 4)

        ego_future = egomotion(future_timestamps)
        ego_future_xyz = ego_future.pose.translation  # (num_future_steps, 3)
        ego_future_quat = ego_future.pose.rotation.as_quat()  # (num_future_steps, 4)

        # Egomotion interpolation has extrapolation disabled, so a query past the end of
        # the clip comes back as NaN rather than raising. Catch it here: a NaN trajectory
        # would silently poison the loss.
        for name, arr in (
            ("ego_history_xyz", ego_history_xyz), ("ego_history_quat", ego_history_quat),
            ("ego_future_xyz", ego_future_xyz), ("ego_future_quat", ego_future_quat),
        ):
            if not np.isfinite(arr).all():
                raise _InfeasibleClipError(
                    f"{name} has non-finite values at t0_us={t0_us} "
                    f"(egomotion covers {egomotion.time_range} us)"
                )

        # Transform to local frame (relative to t0 pose)
        # The model expects trajectories in the ego frame at t0.
        # Transformation: xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)
        t0_xyz = ego_history_xyz[-1].copy()  # Position at t0
        t0_quat = ego_history_quat[-1].copy()  # Orientation at t0
        t0_rot = spt.Rotation.from_quat(t0_quat)
        t0_rot_inv = t0_rot.inv()

        # Transform history positions to local frame
        ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)

        # Transform future positions to local frame
        ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

        # Transform rotations to local frame
        ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
        ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

        # Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
        ego_history_xyz_tensor = (
            torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
        )
        ego_history_rot_tensor = (
            torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
        )
        ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
        ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

        # Load camera images
        image_frames_list = []
        camera_indices_list = []
        timestamps_list = []

        # Image timestamps: if num_frames=4, load at [t0-0.3s, t0-0.2s, t0-0.1s, t0]
        image_timestamps = np.array(
            [t0_us - (self.num_frames - 1 - i) * int(self.time_step * 1_000_000) for i in range(self.num_frames)],
            dtype=np.int64,
        )

        for cam_feature, camera in cameras:
            # frames: (num_frames, H, W, 3) uint8
            frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
            # Release the decoder (and its in-memory video buffer) as soon as we have
            # the frames; the caller closes whatever is left on the way out.
            camera.close()

            # Convert to (num_frames, 3, H, W) for model input
            frames_tensor = torch.from_numpy(frames)
            frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

            # Extract camera name from feature path
            if isinstance(cam_feature, str):
                cam_name = cam_feature.split("/")[-1] if "/" in cam_feature else cam_feature
                cam_name = cam_name.lower()
            else:
                raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
            cam_idx = camera_name_to_index.get(cam_name, 0)

            image_frames_list.append(frames_tensor)
            camera_indices_list.append(cam_idx)
            timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

        # Stack and sort by camera index for consistent ordering
        image_frames = torch.stack(image_frames_list, dim=0)  # (N_cameras, num_frames, 3, H, W)
        camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)  # (N_cameras,)
        all_timestamps = torch.stack(timestamps_list, dim=0)  # (N_cameras, num_frames)

        # Sort by camera index to ensure consistent ordering [0, 1, 2, 6] instead of arbitrary order
        sort_order = torch.argsort(camera_indices)
        image_frames = image_frames[sort_order]
        camera_indices = camera_indices[sort_order]
        all_timestamps = all_timestamps[sort_order]

        # Compute relative timestamps in seconds
        camera_tmin = all_timestamps.min()
        relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6  # (N_cameras, num_frames)

        return {
            "image_frames": image_frames,  # (N_cameras, num_frames, 3, H, W)
            "camera_indices": camera_indices,  # (N_cameras,)
            "ego_history_xyz": ego_history_xyz_tensor,  # (1, 1, num_history_steps, 3)
            "ego_history_rot": ego_history_rot_tensor,  # (1, 1, num_history_steps, 3, 3)
            "ego_future_xyz": ego_future_xyz_tensor,  # (1, 1, num_future_steps, 3)
            "ego_future_rot": ego_future_rot_tensor,  # (1, 1, num_future_steps, 3, 3)
            "relative_timestamps": relative_timestamps,  # (N_cameras, num_frames)
            "absolute_timestamps": all_timestamps,  # (N_cameras, num_frames)
            "t0_us": t0_us,
            "clip_id": clip_id,
    }
