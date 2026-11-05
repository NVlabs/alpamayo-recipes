# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Navigation-label sidecar loading for PhysicalAI-AV trajectory samples."""

from __future__ import annotations

import functools
import json
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


@functools.lru_cache(maxsize=256)
def _read_clip_rows(path: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return tuple(rows)


def _read_index(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"navigation clip index does not exist: {path}")
    clip_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.strip()
            if not value:
                continue
            if value.startswith("{"):
                row = json.loads(value)
                value = str(row.get("clip_id", row.get("name", "")))
                if not value:
                    raise ValueError(f"{path}:{line_number}: missing clip_id")
            clip_ids.add(value.removeprefix("clipgt-"))
    return clip_ids


class NavigationLabelStore:
    """Attach deterministic navigation conditions to trajectory samples."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.root = Path(str(config.get("root", ""))).expanduser()
        self.cohort = str(config.get("cohort", "train"))
        allowed_row_cohorts = config.get("allowed_row_cohorts")
        if allowed_row_cohorts is None:
            # Preserve the original strict contract for training and gold644:
            # a physical cohort directory contains rows bearing the same name.
            parsed_allowed_cohorts = [self.cohort]
        elif isinstance(allowed_row_cohorts, str):
            parsed_allowed_cohorts = allowed_row_cohorts.split(",")
        else:
            parsed_allowed_cohorts = list(allowed_row_cohorts)
        self.allowed_row_cohorts = frozenset(
            str(value).strip() for value in parsed_allowed_cohorts if str(value).strip()
        )
        if not self.allowed_row_cohorts:
            raise ValueError("navigation allowed_row_cohorts must not be empty")
        self.max_distance_m = float(config.get("max_distance_m", -1))
        self.exclude_unknown = bool(config.get("exclude_unknown", True))
        self._valid_t0_cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        index_value = config.get("clip_index_path")
        self.clip_index_path = (
            None if index_value in {None, ""} else Path(str(index_value)).expanduser()
        )

        if self.max_distance_m < 0.0 and self.max_distance_m != -1.0:
            raise ValueError("navigation max_distance_m must be -1 or non-negative")
        if self.enabled:
            if not self.root.is_dir():
                raise NotADirectoryError(f"navigation root is not a directory: {self.root}")
            if not self.cohort_dir.is_dir():
                raise NotADirectoryError(
                    f"navigation cohort directory does not exist: {self.cohort_dir}"
                )

    @property
    def cohort_dir(self) -> Path:
        return self.root / self.cohort

    def _validate_row_cohort(self, path: Path, row_cohort: str) -> None:
        if row_cohort not in self.allowed_row_cohorts:
            expected = ", ".join(sorted(repr(value) for value in self.allowed_row_cohorts))
            raise ValueError(
                f"navigation cohort mismatch in {path}: expected one of "
                f"[{expected}], got {row_cohort!r}"
            )

    def available_clip_ids(self) -> set[str]:
        """Return clips with sidecars, using an optional prebuilt text/JSONL index."""
        if not self.enabled:
            return set()
        if self.clip_index_path is not None:
            return _read_index(self.clip_index_path)
        return {path.stem for path in self.cohort_dir.glob("*.jsonl")}

    def valid_t0_us(self, clip_id: str) -> tuple[int, ...]:
        """Return sidecar timestamps that satisfy the configured nav filters."""
        if not self.enabled:
            return ()
        normalized_clip_id = str(clip_id).removeprefix("clipgt-")
        cached = self._valid_t0_cache.get(normalized_clip_id)
        if cached is not None:
            self._valid_t0_cache.move_to_end(normalized_clip_id)
            return cached
        path = self.cohort_dir / f"{normalized_clip_id}.jsonl"
        if not path.is_file():
            self._valid_t0_cache[normalized_clip_id] = ()
            if len(self._valid_t0_cache) > 1024:
                self._valid_t0_cache.popitem(last=False)
            return ()

        timestamps: list[int] = []
        for frame_index, row in enumerate(_read_clip_rows(str(path))):
            row_clip_id = str(row.get("clip_id", "")).removeprefix("clipgt-")
            row_t0_us = int(row.get("t0_relative_us", -1))
            expected_t0_us = frame_index * 100_000
            if row_clip_id != normalized_clip_id or row_t0_us != expected_t0_us:
                raise ValueError(
                    f"navigation row mismatch in {path}: expected "
                    f"{normalized_clip_id}@{expected_t0_us}, got "
                    f"{row_clip_id}@{row_t0_us}"
                )
            row_cohort = str(row.get("cohort", ""))
            self._validate_row_cohort(path, row_cohort)
            if row_cohort == "train" and str(row.get("hf_split", "")) != "train":
                raise ValueError(
                    f"navigation split mismatch in {path}: expected hf_split='train'"
                )
            if not bool(row.get("lookup_in_route_timestamp_range", False)):
                continue
            if not bool(row.get("has_future_instruction", False)):
                continue
            command = str(row.get("nav_command", "UNKNOWN"))
            if self.exclude_unknown and command.upper() == "UNKNOWN":
                continue
            distance_m = float(row["nav_distance_m"])
            if self.max_distance_m >= 0.0 and distance_m > self.max_distance_m:
                continue
            timestamps.append(row_t0_us)
        result = tuple(timestamps)
        self._valid_t0_cache[normalized_clip_id] = result
        if len(self._valid_t0_cache) > 1024:
            self._valid_t0_cache.popitem(last=False)
        return result

    @staticmethod
    def attach_basic(sample: dict[str, Any]) -> dict[str, Any]:
        """Return an unconditional sample without consulting a sidecar."""
        result = dict(sample)
        result.update(
            nav_available=False,
            nav_enabled=False,
            nav_text=None,
            nav_command=None,
            nav_distance_m=None,
            nav_label_t0_us=None,
            nav_unavailable_reason="navigation_disabled",
        )
        return result

    def attach(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return ``sample`` enriched with navigation fields."""
        result = self.attach_basic(sample)
        if not self.enabled:
            return result

        clip_id = str(sample["clip_id"]).removeprefix("clipgt-")
        t0_us = int(sample["t0_us"])
        if t0_us % 100_000:
            raise ValueError(f"navigation requires a 100000-us timestamp grid, got t0_us={t0_us}")
        path = self.cohort_dir / f"{clip_id}.jsonl"
        if not path.is_file():
            result["nav_unavailable_reason"] = "missing_clip_sidecar"
            return result
        rows = _read_clip_rows(str(path))
        frame_index = t0_us // 100_000
        if not 0 <= frame_index < len(rows):
            result["nav_unavailable_reason"] = "timestamp_out_of_range"
            return result
        row = rows[frame_index]
        row_clip_id = str(row.get("clip_id", "")).removeprefix("clipgt-")
        row_t0_us = int(row.get("t0_relative_us", -1))
        if row_clip_id != clip_id or row_t0_us != t0_us:
            raise ValueError(
                f"navigation row mismatch in {path}: expected {clip_id}@{t0_us}, "
                f"got {row_clip_id}@{row_t0_us}"
            )
        row_cohort = str(row.get("cohort", ""))
        self._validate_row_cohort(path, row_cohort)
        if row_cohort == "train" and str(row.get("hf_split", "")) != "train":
            raise ValueError(
                f"navigation split mismatch in {path}: expected hf_split='train'"
            )
        if not bool(row.get("lookup_in_route_timestamp_range", False)):
            result["nav_unavailable_reason"] = "outside_route_timestamp_range"
            return result
        if not bool(row.get("has_future_instruction", False)):
            result["nav_unavailable_reason"] = "no_future_instruction"
            return result
        command = str(row.get("nav_command", "UNKNOWN"))
        if self.exclude_unknown and command.upper() == "UNKNOWN":
            result["nav_unavailable_reason"] = "unknown_command"
            return result
        distance_m = float(row["nav_distance_m"])
        if self.max_distance_m >= 0.0 and distance_m > self.max_distance_m:
            result["nav_unavailable_reason"] = "distance_limit"
            return result

        result.update(
            nav_available=True,
            nav_enabled=True,
            nav_text=str(row["nav_text"]),
            nav_command=command,
            nav_distance_m=distance_m,
            nav_label_t0_us=row_t0_us,
            nav_unavailable_reason=None,
        )
        return result
