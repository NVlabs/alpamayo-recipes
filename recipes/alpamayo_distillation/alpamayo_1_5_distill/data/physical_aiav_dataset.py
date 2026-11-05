# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PhysicalAI-AV dataset with deterministic timestamp sampling."""

from __future__ import annotations

import hashlib
import importlib
import importlib.resources
import logging
import math
import os
from collections.abc import Sequence
from fractions import Fraction
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, get_worker_info

from alpamayo1_5_distill.data.nav_labels import NavigationLabelStore
from alpamayo1_5_distill.data.sample_index import SampleIndex, unpack_sample_index

_DEFAULT_T0_US = (5_100_000,)
logger = logging.getLogger(__name__)


class InfeasibleClipError(RuntimeError):
    """Dataset-level signal that a clip has no valid sensor window."""


def load_physical_aiavdataset(**kwargs: Any) -> dict[str, Any]:
    """Lazily call the repository's PhysicalAI-AV sample loader."""

    module = importlib.import_module("alpamayo1_5_distill.data.physical_aiav_loader")
    try:
        return module.load_physical_aiavdataset(**kwargs)
    except module.InfeasibleClipError as error:
        raise InfeasibleClipError(str(error)) from error


def identity_collate(samples: Sequence[Any]) -> list[Any]:
    """Keep raw samples separate for main-process/GPU teacher preprocessing."""

    return list(samples)


def _read_clip_ids(path: str | os.PathLike[str]) -> list[str]:
    path_text = os.fspath(path)
    if path_text.startswith("pkg://"):
        package, separator, relative = path_text.removeprefix("pkg://").partition("/")
        if not package or not separator or not relative:
            raise ValueError(f"invalid package resource URI: {path_text}")
        clip_path = Path(importlib.resources.files(package).joinpath(relative))
    else:
        clip_path = Path(path)
    if not clip_path.is_file():
        raise FileNotFoundError(f"clip-ID file does not exist: {clip_path}")
    if clip_path.suffix == ".parquet":
        import pandas as pd

        frame = pd.read_parquet(clip_path, columns=["key"])
        values = frame["key"].dropna().astype(str).tolist()
    else:
        values = [
            line.strip()
            for line in clip_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return list(dict.fromkeys(values))


class OnlinePhysicalAIAVDataset(Dataset[dict[str, Any]]):
    """Load raw model samples directly from PhysicalAI-AV.

    Each dataset index identifies a clip and a sample slot. Random timestamps
    are derived from that stable pair plus ``seed``, ``epoch``, and an optional
    sampler-supplied visit ID. Repeated training visits get fresh draws without
    depending on mutable RNG state, worker assignment, or distributed rank.
    Integer access retains the original per-epoch, per-slot behavior.
    """

    def __init__(
        self,
        *,
        local_dir: str | os.PathLike[str] | None = None,
        token: str | None = None,
        split: str = "train",
        samples_per_clip: int = 1,
        t0_us_list: Sequence[int] = _DEFAULT_T0_US,
        random_t0: bool = False,
        margin_us: int = 0,
        seed: int = 42,
        max_clip_retries: int = 8,
        loader_kwargs: dict[str, Any] | None = None,
        clip_allowlist: Sequence[str] | None = None,
        clip_allowlist_path: str | os.PathLike[str] | None = None,
        exclude_clip_ids: Sequence[str] | None = None,
        exclude_clip_ids_path: str | os.PathLike[str] | None = None,
        nav_config: dict[str, Any] | None = None,
        mixture_config: dict[str, Any] | None = None,
    ) -> None:
        if samples_per_clip <= 0:
            raise ValueError("samples_per_clip must be positive")
        if margin_us < 0:
            raise ValueError("margin_us must be non-negative")
        if max_clip_retries < 0:
            raise ValueError("max_clip_retries must be non-negative")

        self.local_dir = None if local_dir is None else str(local_dir)
        self.token = token
        self.split = str(split)
        self.samples_per_clip = int(samples_per_clip)
        self.random_t0 = bool(random_t0)
        self.margin_us = int(margin_us)
        self.seed = int(seed)
        self.max_clip_retries = int(max_clip_retries)
        self.loader_kwargs = dict(loader_kwargs or {})
        self.mixture_config = dict(mixture_config or {})
        self.mixture_enabled = bool(self.mixture_config.get("enabled", False))
        self.nav_store = NavigationLabelStore(nav_config or {})
        self.basic_ratio = float(self.mixture_config.get("basic_ratio", 0.5))
        self.nav_ratio = float(self.mixture_config.get("nav_ratio", 0.5))
        self.nav_t0_retries = int(self.mixture_config.get("nav_t0_retries", 3))
        self.nav_sample_retries = int(self.mixture_config.get("nav_sample_retries", 3))
        self.fallback_nav_to_basic = bool(self.mixture_config.get("fallback_nav_to_basic", True))
        if self.mixture_enabled:
            if not self.nav_store.enabled:
                raise ValueError("explicit basic/nav mixture requires navigation enabled")
            if self.basic_ratio < 0.0 or self.nav_ratio < 0.0:
                raise ValueError("mixture ratios must be non-negative")
            if self.basic_ratio + self.nav_ratio <= 0.0:
                raise ValueError("at least one mixture ratio must be positive")
            if self.nav_t0_retries <= 0:
                raise ValueError("nav_t0_retries must be positive")
            if self.nav_sample_retries <= 0:
                raise ValueError("nav_sample_retries must be positive")
            nav_fraction = self.nav_ratio / (self.basic_ratio + self.nav_ratio)
            fraction = Fraction(nav_fraction).limit_denominator(1000)
            self._nav_slots = fraction.numerator
            self._mixture_period = fraction.denominator
        reserved = {
            "clip_id",
            "t0_us",
            "t0_margin_us",
            "t0_random_bits",
            "avdi",
            "local_dir",
        }
        overlap = reserved.intersection(self.loader_kwargs)
        if overlap:
            raise ValueError(f"loader_kwargs cannot override {sorted(overlap)}")

        self.t0_us_list = tuple(int(value) for value in t0_us_list)
        if not self.random_t0 and not self.t0_us_list:
            raise ValueError("t0_us_list must not be empty in fixed mode")

        # Shared storage lets set_epoch() affect already-running persistent
        # DataLoader workers as well as workers created after the call.
        self._epoch = get_context("spawn").Value("q", 0)
        self._avdi: Any = None
        self._avdi_owner: tuple[int, int | None] | None = None
        self._logged_infeasible_clip_ids: set[str] = set()
        clip_index = self._interface().clip_index
        if clip_allowlist is not None and clip_allowlist_path is not None:
            raise ValueError("pass only one of clip_allowlist and clip_allowlist_path")
        if clip_allowlist_path is not None:
            selected_ids = _read_clip_ids(clip_allowlist_path)
        elif clip_allowlist is not None:
            selected_ids = list(dict.fromkeys(str(value) for value in clip_allowlist))
        else:
            if "split" not in clip_index:
                raise ValueError(f"'split' is not in clip index columns {list(clip_index.columns)}")
            selected = clip_index.index[clip_index["split"] == self.split]
            selected_ids = selected.dropna().astype(str).drop_duplicates().tolist()

        excluded = {str(value) for value in (exclude_clip_ids or ())}
        excluded_from_path: list[str] = []
        if exclude_clip_ids_path is not None:
            excluded_from_path = _read_clip_ids(exclude_clip_ids_path)
            excluded.update(excluded_from_path)
        self.clip_ids = [clip_id for clip_id in selected_ids if clip_id not in excluded]
        self.exclusion_file_clip_count = len(excluded_from_path)
        self.excluded_selected_clip_count = len(selected_ids) - len(self.clip_ids)
        self.nav_clip_ids: list[str] = []
        if self.nav_store.enabled:
            # Mixture navigation samples select aligned t0s directly from the
            # sidecar. Basic samples and navigation fallbacks need no shared grid.
            if not self.mixture_enabled:
                if self.random_t0:
                    if self.loader_kwargs.get("t0_quantum_us") != 100_000:
                        raise ValueError(
                            "navigation with random_t0 requires loader_kwargs.t0_quantum_us=100000"
                        )
                elif any(value % 100_000 for value in self.t0_us_list):
                    raise ValueError(
                        "navigation requires every fixed t0_us value on the 100000-us grid"
                    )
            available_nav_clip_ids = self.nav_store.available_clip_ids()
            self.nav_clip_ids = [
                clip_id
                for clip_id in self.clip_ids
                if clip_id.removeprefix("clipgt-") in available_nav_clip_ids
            ]
        if not self.clip_ids:
            raise ValueError(f"clip selection contains no clips for split {self.split!r}")
        if (
            self.mixture_enabled
            and self._nav_slots > 0
            and not self.nav_clip_ids
            and not self.fallback_nav_to_basic
        ):
            raise ValueError(
                "navigation mixture has no sidecar clips and fallback_nav_to_basic is false"
            )
        # Data-loading interfaces are process-local; workers create their own.
        self._avdi = None
        self._avdi_owner = None

    def __len__(self) -> int:
        return len(self.clip_ids) * self.samples_per_clip

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch used for deterministic random timestamps."""

        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        """Return the epoch currently used for deterministic random timestamps."""

        with self._epoch.get_lock():
            return int(self._epoch.value)

    def _random_bits(self, clip_index: int, sample_slot: int, visit_id: int | None = None) -> int:
        return self._stable_bits(clip_index, sample_slot, visit_id=visit_id)

    def _stable_bits(self, *parts: object, visit_id: int | None = None) -> int:
        if visit_id is not None:
            parts = (*parts, "visit", visit_id)
        payload = ":".join(
            (str(self.seed), str(self.epoch), *(str(part) for part in parts))
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def _requests_navigation(self, index: int) -> bool:
        if not self.mixture_enabled:
            return False
        slot = (index + self.epoch * self._nav_slots) % self._mixture_period
        return slot < self._nav_slots

    def _interface(self) -> Any:
        worker = get_worker_info()
        owner = (os.getpid(), None if worker is None else worker.id)
        if self._avdi is None or self._avdi_owner != owner:
            kwargs: dict[str, Any] = {"local_dir": self.local_dir}
            if self.token is not None:
                kwargs["token"] = self.token
            if self.local_dir is not None:
                dataset_local = importlib.import_module("alpamayo1_5_distill.data.dataset_local")
                interface = dataset_local.PhysicalAIAVDatasetInterfaceLocal
            else:
                physical_ai_av = importlib.import_module("physical_ai_av")
                interface = physical_ai_av.PhysicalAIAVDatasetInterface
            self._avdi = interface(**kwargs)
            self._avdi_owner = owner
        return self._avdi

    def _load_basic(self, index: int, visit_id: int | None = None) -> dict[str, Any]:
        clip_index, sample_slot = divmod(index, self.samples_per_clip)
        last_error: Exception | None = None
        for attempt in range(self.max_clip_retries + 1):
            candidate = (clip_index + attempt) % len(self.clip_ids)
            try:
                if self.random_t0:
                    t0_us = None
                    t0_random_bits = self._random_bits(
                        candidate,
                        sample_slot + attempt,
                        visit_id,
                    )
                else:
                    t0_us = self.t0_us_list[(sample_slot + attempt) % len(self.t0_us_list)]
                    t0_random_bits = None
                sample = load_physical_aiavdataset(
                    clip_id=self.clip_ids[candidate],
                    t0_us=t0_us,
                    t0_margin_us=self.margin_us,
                    t0_random_bits=t0_random_bits,
                    avdi=self._interface(),
                    **self.loader_kwargs,
                )
                if self.mixture_enabled:
                    return self.nav_store.attach_basic(sample)
                return self.nav_store.attach(sample)
            except InfeasibleClipError as error:
                last_error = error
                failed_clip_id = self.clip_ids[candidate]
                if failed_clip_id not in self._logged_infeasible_clip_ids:
                    logger.warning(
                        "Skipping infeasible clip %s while loading dataset "
                        "index %d (attempt %d/%d): %s",
                        failed_clip_id,
                        index,
                        attempt + 1,
                        self.max_clip_retries + 1,
                        error,
                    )
                    self._logged_infeasible_clip_ids.add(failed_clip_id)
        raise RuntimeError(
            f"failed to load index {index} after {self.max_clip_retries + 1} attempts"
        ) from last_error

    @staticmethod
    def _with_mixture_metadata(
        sample: dict[str, Any],
        *,
        requested_nav: bool,
        nav_t0_retries: int = 0,
        nav_clip_retries: int = 0,
        nav_fallback: bool = False,
    ) -> dict[str, Any]:
        sample["mixture_requested_nav"] = requested_nav
        sample["mixture_actual_nav"] = bool(sample.get("nav_enabled", False))
        sample["mixture_nav_t0_retries"] = int(nav_t0_retries)
        sample["mixture_nav_clip_retries"] = int(nav_clip_retries)
        sample["mixture_nav_fallback"] = bool(nav_fallback)
        return sample

    def _load_navigation(self, index: int, visit_id: int | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        failed_t0_attempts = 0
        clip_attempts = min(self.nav_sample_retries, len(self.nav_clip_ids))
        if clip_attempts:
            start = self._stable_bits("nav-clip", index, visit_id=visit_id) % len(self.nav_clip_ids)
            for clip_attempt in range(clip_attempts):
                clip_id = self.nav_clip_ids[(start + clip_attempt) % len(self.nav_clip_ids)]
                candidates = tuple(dict.fromkeys(self.nav_store.valid_t0_us(clip_id)))
                if not candidates:
                    continue
                candidate_start = self._stable_bits(
                    "nav-t0", index, clip_id, visit_id=visit_id
                ) % len(candidates)
                candidate_step = 1
                if len(candidates) > 1:
                    candidate_step = (
                        self._stable_bits("nav-t0-step", index, clip_id, visit_id=visit_id)
                        % (len(candidates) - 1)
                        + 1
                    )
                    while math.gcd(candidate_step, len(candidates)) != 1:
                        candidate_step = candidate_step % (len(candidates) - 1) + 1
                t0_attempts = min(self.nav_t0_retries, len(candidates))
                for t0_attempt in range(t0_attempts):
                    t0_us = candidates[
                        (candidate_start + t0_attempt * candidate_step) % len(candidates)
                    ]
                    try:
                        sample = load_physical_aiavdataset(
                            clip_id=clip_id,
                            t0_us=t0_us,
                            t0_margin_us=self.margin_us,
                            t0_random_bits=None,
                            avdi=self._interface(),
                            **self.loader_kwargs,
                        )
                        sample = self.nav_store.attach(sample)
                        if not bool(sample["nav_enabled"]):
                            failed_t0_attempts += 1
                            continue
                        return self._with_mixture_metadata(
                            sample,
                            requested_nav=True,
                            nav_t0_retries=failed_t0_attempts,
                            nav_clip_retries=clip_attempt,
                        )
                    except InfeasibleClipError as error:
                        last_error = error
                        failed_t0_attempts += 1

        if self.fallback_nav_to_basic:
            sample = self._load_basic(index, visit_id)
            return self._with_mixture_metadata(
                sample,
                requested_nav=True,
                nav_t0_retries=failed_t0_attempts,
                nav_clip_retries=clip_attempts,
                nav_fallback=True,
            )
        raise RuntimeError(
            f"failed to load navigation index {index} after trying {clip_attempts} clips"
        ) from last_error

    def __getitem__(self, index: SampleIndex) -> dict[str, Any]:
        index, visit_id = unpack_sample_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if not self.mixture_enabled:
            return self._load_basic(index, visit_id)
        if self._requests_navigation(index):
            return self._load_navigation(index, visit_id)
        return self._with_mixture_metadata(
            self._load_basic(index, visit_id),
            requested_nav=False,
        )

    def __getstate__(self) -> dict[str, Any]:
        """Avoid serializing a process-local dataset interface."""

        state = self.__dict__.copy()
        state["_avdi"] = None
        state["_avdi_owner"] = None
        return state


# Alternate word order retained as a discoverable convenience.
PhysicalAIAVOnlineDataset = OnlinePhysicalAIAVDataset
