# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from alpamayo1_5_distill.data.nav_labels import NavigationLabelStore


def _write_clip(root, clip_id="clip-a", *, distance_m=450.0):
    cohort = root / "train"
    cohort.mkdir(parents=True)
    path = cohort / f"{clip_id}.jsonl"
    rows = []
    for frame_index in range(3):
        rows.append(
            {
                "clip_id": clip_id,
                "cohort": "train",
                "hf_split": "train",
                "t0_relative_us": frame_index * 100_000,
                "lookup_in_route_timestamp_range": True,
                "has_future_instruction": True,
                "nav_command": "TURN_LEFT",
                "nav_distance_m": distance_m,
                "nav_text": f"Turn left in {distance_m:.0f}m",
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_cohort_clip(root, directory, row_cohort, clip_id):
    cohort = root / directory
    cohort.mkdir(parents=True, exist_ok=True)
    path = cohort / f"{clip_id}.jsonl"
    rows = [
        {
            "clip_id": clip_id,
            "cohort": row_cohort,
            "t0_relative_us": frame_index * 100_000,
            "lookup_in_route_timestamp_range": True,
            "has_future_instruction": True,
            "nav_command": "GO_STRAIGHT",
            "nav_distance_m": 100.0,
            "nav_text": "Continue straight",
        }
        for frame_index in range(3)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_navigation_label_join_and_unlimited_distance(tmp_path):
    _write_clip(tmp_path)
    store = NavigationLabelStore(
        {
            "enabled": True,
            "root": str(tmp_path),
            "cohort": "train",
            "max_distance_m": -1,
        },
    )

    result = store.attach({"clip_id": "clip-a", "t0_us": 100_000})

    assert result["nav_available"] is True
    assert result["nav_enabled"] is True
    assert result["nav_text"] == "Turn left in 450m"
    assert result["nav_label_t0_us"] == 100_000


def test_navigation_distance_limit(tmp_path):
    _write_clip(tmp_path)
    limited = NavigationLabelStore(
        {
            "enabled": True,
            "root": str(tmp_path),
            "cohort": "train",
            "max_distance_m": 200,
        },
    )

    limited_result = limited.attach({"clip_id": "clip-a", "t0_us": 100_000})

    assert limited_result["nav_available"] is False
    assert limited_result["nav_unavailable_reason"] == "distance_limit"


def test_navigation_requires_grid_aligned_timestamps(tmp_path):
    _write_clip(tmp_path)
    store = NavigationLabelStore({"enabled": True, "root": str(tmp_path), "cohort": "train"})

    with pytest.raises(ValueError, match="100000-us"):
        store.attach({"clip_id": "clip-a", "t0_us": 100_001})


def test_valid_timestamps_are_filtered_ordered_and_cached(tmp_path):
    cohort = tmp_path / "train"
    cohort.mkdir()
    rows = [
        {
            "clip_id": "clip-a",
            "cohort": "train",
            "hf_split": "train",
            "t0_relative_us": index * 100_000,
            "lookup_in_route_timestamp_range": index != 1,
            "has_future_instruction": index != 2,
            "nav_command": "UNKNOWN" if index == 3 else "TURN_LEFT",
            "nav_distance_m": 400.0 if index == 4 else 100.0,
            "nav_text": "Turn left",
        }
        for index in range(6)
    ]
    path = cohort / "clip-a.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    store = NavigationLabelStore(
        {
            "enabled": True,
            "root": str(tmp_path),
            "cohort": "train",
            "max_distance_m": 300,
        },
    )

    assert store.valid_t0_us("clip-a") == (0, 500_000)
    path.write_text("", encoding="utf-8")
    assert store.valid_t0_us("clip-a") == (0, 500_000)
    assert store.valid_t0_us("missing") == ()


def test_attach_basic_never_uses_navigation(tmp_path):
    _write_clip(tmp_path)
    store = NavigationLabelStore({"enabled": True, "root": str(tmp_path), "cohort": "train"})

    result = store.attach_basic({"clip_id": "clip-a", "t0_us": 100_000})

    assert result["nav_available"] is False
    assert result["nav_enabled"] is False
    assert result["nav_unavailable_reason"] == "navigation_disabled"


def test_navigation_allows_explicit_mixed_row_cohorts(tmp_path):
    _write_cohort_clip(tmp_path, "eval", "checkpoint_val_5k", "clip-a")
    _write_cohort_clip(tmp_path, "eval", "opd_unseen_val_70k", "clip-b")
    store = NavigationLabelStore(
        {
            "enabled": True,
            "root": str(tmp_path),
            "cohort": "eval",
            "allowed_row_cohorts": ["checkpoint_val_5k", "opd_unseen_val_70k"],
        }
    )

    assert store.available_clip_ids() == {"clip-a", "clip-b"}
    assert store.valid_t0_us("clip-a") == (0, 100_000, 200_000)
    assert store.valid_t0_us("clip-b") == (0, 100_000, 200_000)
    assert store.attach({"clip_id": "clip-a", "t0_us": 100_000})["nav_enabled"] is True
    assert store.attach({"clip_id": "clip-b", "t0_us": 100_000})["nav_enabled"] is True


def test_navigation_remains_strict_without_allowed_row_cohorts(tmp_path):
    _write_cohort_clip(tmp_path, "gold644", "unexpected", "clip-a")
    store = NavigationLabelStore(
        {"enabled": True, "root": str(tmp_path), "cohort": "gold644"}
    )

    with pytest.raises(ValueError, match="expected one of.*'gold644'.*got 'unexpected'"):
        store.valid_t0_us("clip-a")


@pytest.mark.parametrize("value", [-2, -0.5])
def test_navigation_rejects_invalid_distance_limit(tmp_path, value):
    _write_clip(tmp_path)

    with pytest.raises(ValueError, match="max_distance_m"):
        NavigationLabelStore(
            {
                "enabled": True,
                "root": str(tmp_path),
                "cohort": "train",
                "max_distance_m": value,
            },
        )
