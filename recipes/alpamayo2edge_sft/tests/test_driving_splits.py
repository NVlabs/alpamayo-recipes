# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-recording time/location contracts; no models or downloaded data."""

from copy import deepcopy
import hashlib
import json

import pytest

from alpamayo2edge_sft.driving_splits import audit_split, load_split_manifest, manifest_sha256
from alpamayo2edge_sft.prepare_driving_splits import main


def _record(start, end, points):
    return {"start_time_us": start, "end_time_us": end, "time_reference": "unix",
            "positions_xy_m": points, "spatial_frame": "EPSG:26917", "spatial_units": "metres",
            "spatial_frame_verified": True,
            "spatial_frame_evidence": "Shared map coordinates verified against source metadata."}


@pytest.fixture
def manifest():
    return {"schema_version": 1, "dataset": "tartan", "train": ["train-1"], "val": ["val-1"],
            "recordings": {"train-1": _record(0, 1_000_000, [[0, 0], [1, 0]]),
                           "val-1": _record(86_401_000_000, 86_402_000_000, [[0, 1000], [1, 1000]])}}


def _write(tmp_path, manifest):
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    return path


def test_load_returns_all_ids_and_hash_of_exact_loaded_bytes(tmp_path, manifest):
    path = _write(tmp_path, manifest)
    loaded, digest = load_split_manifest(path, "tartan", return_sha256=True)
    assert loaded == manifest
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest() == manifest_sha256(path)
    assert load_split_manifest(path, "tartan") == manifest


@pytest.mark.parametrize("bad_id", ["", ".", "..", "../other", "a/b", "a\\b", " a", "a ", 1, None])
def test_loader_rejects_unsafe_recording_ids(tmp_path, manifest, bad_id):
    manifest["train"] = [bad_id]
    with pytest.raises(ValueError, match="safe, nonempty"):
        load_split_manifest(_write(tmp_path, manifest), "tartan")


@pytest.mark.parametrize("damage", ["boolean_version", "wrong_dataset", "empty_train", "empty_val",
                                    "duplicate", "overlap", "missing_metadata", "wrong_time_reference",
                                    "boolean_timestamp", "float_timestamp", "negative_timestamp",
                                    "backward_interval", "unsafe_source_partition"])
def test_rejects_invalid_manifest_without_dropping_recordings(tmp_path, manifest, damage):
    if damage == "boolean_version":
        manifest["schema_version"] = True
    elif damage == "wrong_dataset":
        manifest["dataset"] = "once"
    elif damage in ("empty_train", "empty_val"):
        manifest[damage[6:]] = []
    elif damage == "duplicate":
        manifest["train"] *= 2
    elif damage == "overlap":
        manifest["val"] = manifest["train"][:]
    elif damage == "missing_metadata":
        manifest["train"].append("missing")
    else:
        record = manifest["recordings"]["train-1"]
        if damage == "wrong_time_reference":
            record["time_reference"] = "sequence-relative"
        elif damage == "unsafe_source_partition":
            record["source_partition"] = "../../outside"
        else:
            record["start_time_us"] = {"boolean_timestamp": True, "float_timestamp": 0.0,
                                       "negative_timestamp": -1, "backward_interval": 2_000_000}[damage]
    with pytest.raises(ValueError):
        load_split_manifest(_write(tmp_path, manifest), "tartan")


def test_json_duplicate_keys_and_nonfinite_numbers_are_rejected(tmp_path, manifest):
    path = _write(tmp_path, manifest)
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_split_manifest(path, "tartan")
    for value in (float("nan"), float("inf")):
        manifest["recordings"]["train-1"]["positions_xy_m"] = [[value, 0]]
        with pytest.raises(ValueError, match="Nonfinite JSON"):
            load_split_manifest(_write(tmp_path, manifest), "tartan")


def test_exact_temporal_guard_boundary_passes_and_one_microsecond_short_fails(manifest):
    original = deepcopy(manifest)
    result = audit_split(manifest)
    assert result["passed"] and result["full_validation_passed"]
    assert result["min_time_gap_seconds"] == 86400
    assert manifest == original  # auditer never reassigns or drops records
    manifest["recordings"]["val-1"]["start_time_us"] -= 1
    assert not audit_split(manifest)["temporal_passed"]


@pytest.mark.parametrize("start", [0, 999_999, 1_000_000])
def test_overlapping_or_touching_intervals_fail_with_minimum_time_guard(manifest, start):
    manifest["recordings"]["val-1"].update(start_time_us=start, end_time_us=2_000_000)
    result = audit_split(manifest, min_time_gap_seconds=8.4)
    assert not result["passed"]
    assert result["pairs"][0]["time_intervals_overlap"]


def test_order_of_recordings_in_time_does_not_change_separation(manifest):
    manifest["train"], manifest["val"] = manifest["val"], manifest["train"]
    assert audit_split(manifest)["passed"]


def test_crossing_segments_cannot_pass_based_only_on_distant_vertices(manifest):
    manifest["recordings"]["train-1"]["positions_xy_m"] = [[-1000, 0], [1000, 0]]
    manifest["recordings"]["val-1"]["positions_xy_m"] = [[0, -1000], [0, 1000]]
    result = audit_split(manifest)
    pair = result["pairs"][0]["spatial"]
    assert pair["nearest_vertex_distance_m"] > 1000
    assert pair["polyline_distance_lower_bound_m"] == 0
    assert not result["passed"]


def test_all_route_vertices_are_used_including_context_far_from_anchor(manifest):
    manifest["recordings"]["val-1"]["positions_xy_m"] = [[0, 1000], [0, 0], [1, 1000]]
    assert not audit_split(manifest)["spatial_passed"]


def test_sparse_parallel_routes_can_fail_to_certify_without_claiming_overlap(manifest):
    manifest["recordings"]["train-1"]["positions_xy_m"] = [[0, 0], [1000, 0]]
    manifest["recordings"]["val-1"]["positions_xy_m"] = [[0, 300], [1000, 300]]
    result = audit_split(manifest)
    assert result["pairs"][0]["spatial"]["status"] == "not_certified"
    assert not result["passed"]


def test_metric_distance_boundary_with_stationary_routes(manifest):
    manifest["recordings"]["train-1"]["positions_xy_m"] = [[0, 0]]
    manifest["recordings"]["val-1"]["positions_xy_m"] = [[250, 0]]
    assert audit_split(manifest)["passed"]
    manifest["recordings"]["val-1"]["positions_xy_m"] = [[249.99, 0]]
    assert not audit_split(manifest)["passed"]


@pytest.mark.parametrize("update", [{"spatial_frame": "EPSG:32617"},
                                   {"spatial_frame": "sequence-local"},
                                   {"spatial_frame": "EPSG:4326"},
                                   {"spatial_units": "degrees"},
                                   {"spatial_frame_verified": False},
                                   {"spatial_frame_evidence": ""}])
def test_mismatched_or_unverified_coordinate_frames_never_pass(manifest, update):
    manifest["recordings"]["val-1"].update(update)
    result = audit_split(manifest)
    assert not result["passed"]
    assert result["pairs"][0]["spatial"]["status"] == "unverified"


def test_missing_geography_requires_explicit_temporal_only_mode(manifest):
    del manifest["recordings"]["val-1"]["positions_xy_m"]
    assert not audit_split(manifest)["passed"]
    result = audit_split(manifest, temporal_only=True)
    assert result["passed"] and result["status"] == "passed_temporal_only"
    assert not result["full_validation_passed"] and not result["spatial_passed"]
    assert result["pairs"][0]["spatial"]["status"] == "unverified"


@pytest.mark.parametrize("points", [[], [[0]], [[0, 1, 2]], [["0", 1]], [[True, 0]], [[float("nan"), 0]]])
def test_malformed_route_is_not_silently_ignored_in_temporal_only_mode(manifest, points):
    manifest["recordings"]["val-1"]["positions_xy_m"] = points
    with pytest.raises(ValueError):
        audit_split(manifest, temporal_only=True)


def test_every_cross_split_pair_is_checked(manifest):
    manifest["val"].append("val-2")
    manifest["recordings"]["val-2"] = _record(2_000_000, 3_000_000, [[0, 1000]])
    result = audit_split(manifest)
    assert len(result["pairs"]) == 2
    assert not result["passed"]


@pytest.mark.parametrize("kwargs", [{"min_time_gap_seconds": -1}, {"min_time_gap_seconds": True},
                                   {"min_location_gap_m": float("inf")},
                                   {"min_location_gap_m": 0}, {"temporal_only": 1}])
def test_invalid_policy_is_rejected(manifest, kwargs):
    with pytest.raises(ValueError):
        audit_split(manifest, **kwargs)


def test_cli_writes_report_and_nonzero_exit_for_unverified_geography(tmp_path, manifest):
    del manifest["recordings"]["val-1"]["positions_xy_m"]
    path = _write(tmp_path, manifest)
    output = tmp_path / "audit.json"
    args = ["--manifest", str(path), "--dataset", "tartan", "--output", str(output)]
    assert main(args) == 1
    assert json.loads(output.read_text())["status"] == "failed"
    assert main(args + ["--temporal-only"]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed_temporal_only"
    assert report["manifest_sha256"] == manifest_sha256(path)
    assert not report["full_validation_passed"]


def test_cli_invalid_manifest_produces_failure_report_and_preserves_input(tmp_path, manifest):
    manifest["train"].append("absent")
    path = _write(tmp_path, manifest)
    original = path.read_bytes()
    output = tmp_path / "audit.json"
    assert main(["--manifest", str(path), "--dataset", "tartan", "--output", str(output)]) == 1
    assert json.loads(output.read_text())["status"] == "invalid"
    assert path.read_bytes() == original
    with pytest.raises(SystemExit):
        main(["--manifest", str(path), "--dataset", "tartan", "--output", str(path)])
    assert path.read_bytes() == original


@pytest.mark.parametrize("guard", [0, .1, 8., 8.399999])
def test_audit_rejects_time_guards_below_the_model_context_floor(manifest, guard):
    with pytest.raises(ValueError):
        audit_split(manifest, min_time_gap_seconds=guard, temporal_only=True)


def test_minimum_time_guard_keeps_microsecond_boundary_exact(manifest):
    record = manifest["recordings"]["val-1"]
    record.update(start_time_us=9_400_000, end_time_us=10_400_000)
    assert audit_split(manifest, min_time_gap_seconds=8.4)["passed"]
    record["start_time_us"] -= 1
    result = audit_split(manifest, min_time_gap_seconds=8.4)
    assert not result["temporal_passed"]
