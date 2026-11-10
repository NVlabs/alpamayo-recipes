# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit explicit driving-recording splits using timestamps and global routes.

The manifest describes complete recording intervals and routes, including all
image/history/future context. It must not substitute sampled anchors or reset
sequence-relative poses for those data. Evidence fields record how the author
verified the shared spatial frame; this audit cannot independently establish
the truth of those declarations or the accuracy of GPS measurements.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# Minimum buffer policy for the default 2 s camera context + 6.4 s future.
MIN_TIME_GAP_SECONDS = 8.4

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_GEOGRAPHIC_EPSG = {"EPSG:4326", "EPSG:4269", "EPSG:4267", "EPSG:4258", "EPSG:4979"}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return result


def _safe_id(value, context):
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{context} must be a safe, nonempty recording/path-component ID")


def _timestamp(value, context):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError(f"{context} must be a nonnegative integer Unix timestamp in microseconds")


def _points(value, context):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must contain a nonempty list of [x, y] positions")
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"{context} must contain [x, y] positions")
        if any(type(x) not in (int, float) for x in row):
            raise ValueError(f"{context} coordinates must be finite numbers")
    try:
        points = np.asarray(value, dtype=np.float64)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{context} coordinates must be finite numbers") from error
    if not np.isfinite(points).all():
        raise ValueError(f"{context} coordinates must be finite numbers")
    return points


def _validate_manifest(manifest, dataset):
    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must be a JSON object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Split manifest schema_version must be integer 1")
    if dataset not in ("once", "tartan") or manifest.get("dataset") != dataset:
        raise ValueError(f"Split manifest dataset must match {dataset!r} (once or tartan)")
    for split in ("train", "val"):
        ids = manifest.get(split)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Split {split} must be a nonempty list of recording IDs")
        for identity in ids:
            _safe_id(identity, f"{split} recording ID")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Split {split} contains duplicate recording IDs")
    if set(manifest["train"]) & set(manifest["val"]):
        raise ValueError("Training and validation recording IDs overlap")
    records = manifest.get("recordings")
    if not isinstance(records, dict):
        raise ValueError("recordings must provide metadata for every selected recording")
    for identity in manifest["train"] + manifest["val"]:
        record = records.get(identity)
        if not isinstance(record, dict):
            raise ValueError(f"Missing recording metadata: {identity}")
        for field in ("start_time_us", "end_time_us"):
            _timestamp(record.get(field), f"{identity}.{field}")
        if record["start_time_us"] > record["end_time_us"]:
            raise ValueError(f"{identity}: recording start is later than its end")
        if record.get("time_reference") != "unix":
            raise ValueError(f"{identity}: time_reference must be 'unix', not a relative clock")
        if "source_partition" in record:
            _safe_id(record["source_partition"], f"{identity}.source_partition")
        if "positions_xy_m" in record:
            _points(record["positions_xy_m"], f"{identity}.positions_xy_m")
    return manifest


def load_split_manifest(path, dataset, *, return_sha256=False):
    """Load all selected IDs and their metadata; never silently drop a recording.

    ``return_sha256=True`` returns ``(manifest, hash)`` from the same byte read,
    so a dataset can identify the exact manifest that it used.
    """
    data = Path(path).read_bytes()
    try:
        value = json.loads(data, object_pairs_hook=_object,
                           parse_constant=_invalid_constant, parse_float=_json_float)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid split manifest {path}: {error}") from error
    value = _validate_manifest(value, dataset)
    return (value, hashlib.sha256(data).hexdigest()) if return_sha256 else value


def manifest_sha256(path):
    """Hash a manifest file without implying that its schema or split is valid."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spatial_record(record):
    required = ("positions_xy_m", "spatial_frame", "spatial_units",
                "spatial_frame_verified", "spatial_frame_evidence")
    missing = [key for key in required if key not in record]
    if missing:
        return None, f"Missing spatial metadata: {', '.join(missing)}"
    frame = record["spatial_frame"]
    if not isinstance(frame, str) or not frame.strip():
        return None, "Spatial frame must identify a shared global projection"
    if any(word in frame.lower() for word in ("local", "relative", "first-frame", "first_frame")):
        return None, "Sequence-relative/local frames cannot establish cross-recording separation"
    if frame.upper() in _GEOGRAPHIC_EPSG:
        return None, "Geographic degree coordinates are not projected metric coordinates"
    if record["spatial_units"] != "metres" or record["spatial_frame_verified"] is not True:
        return None, "A verified shared metric frame with spatial_units='metres' is required"
    evidence = record["spatial_frame_evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        return None, "Spatial-frame verification requires a nonempty evidence description"
    points = _points(record["positions_xy_m"], "positions_xy_m")
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.isfinite(lengths).all():
        raise ValueError("Route segment lengths overflow; verify the projected metric coordinates")
    return (frame, points, float(lengths.max()) if len(lengths) else 0.0), None


def audit_split(manifest, min_time_gap_seconds=86400, min_location_gap_m=250,
                temporal_only=False):
    """Check every train/validation pair without altering either split.

    A conservative full-polyline distance bound is computed from all route
    vertices. Failure of that bound means separation is *not certified*, not
    necessarily that the underlying routes overlap. ``temporal_only=True``
    explicitly selects a weaker check and never claims full spatial validation.
    """
    _validate_manifest(manifest, manifest.get("dataset") if isinstance(manifest, dict) else None)
    for name, value in (("min_time_gap_seconds", min_time_gap_seconds),
                        ("min_location_gap_m", min_location_gap_m)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if min_time_gap_seconds < MIN_TIME_GAP_SECONDS:
        raise ValueError(f"min_time_gap_seconds must be at least {MIN_TIME_GAP_SECONDS} seconds")
    if min_location_gap_m == 0:
        raise ValueError("min_location_gap_m must be positive; use temporal_only explicitly")
    if type(temporal_only) is not bool:
        raise ValueError("temporal_only must be a boolean")
    records = manifest["recordings"]
    spatial = {name: _spatial_record(records[name]) for name in manifest["train"] + manifest["val"]}
    trees = {name: cKDTree(item[0][1]) for name, item in spatial.items() if item[0] is not None}
    pairs = []
    for train_id in manifest["train"]:
        for val_id in manifest["val"]:
            train, val = records[train_id], records[val_id]
            gap_us = max(val["start_time_us"] - train["end_time_us"],
                         train["start_time_us"] - val["end_time_us"])
            overlaps = gap_us <= 0  # intervals include their boundary timestamps
            pair = {"train": train_id, "val": val_id,
                    "time_gap_seconds": gap_us / 1e6, "time_intervals_overlap": overlaps,
                    "temporal_passed": not overlaps and gap_us / 1e6 >= min_time_gap_seconds}
            a, a_error = spatial[train_id]
            b, b_error = spatial[val_id]
            if a is None or b is None:
                pair["spatial"] = {"status": "unverified", "reason": a_error or b_error}
            elif a[0] != b[0]:
                pair["spatial"] = {"status": "unverified", "reason": "Spatial frames differ"}
            else:
                distances, _ = trees[val_id].query(a[1], workers=1)
                nearest = float(distances.min())
                if not math.isfinite(nearest):
                    raise ValueError("Route distances overflow; verify projected metric coordinates")
                # Any point on a segment is within half its length of an endpoint.
                # The triangle inequality therefore bounds the distance between
                # every pair of segment points, including segments that cross.
                lower = max(0.0, nearest - 0.5 * (a[2] + b[2]))
                pair["spatial"] = {
                    "status": "passed" if lower >= min_location_gap_m else "not_certified",
                    "spatial_frame": a[0], "nearest_vertex_distance_m": nearest,
                    "polyline_distance_lower_bound_m": lower,
                    "train_max_segment_m": a[2], "val_max_segment_m": b[2],
                }
            pairs.append(pair)
    temporal_passed = all(pair["temporal_passed"] for pair in pairs)
    spatial_passed = all(pair["spatial"]["status"] == "passed" for pair in pairs)
    passed = temporal_passed and (temporal_only or spatial_passed)
    return {
        "schema_version": 1, "dataset": manifest["dataset"],
        "status": ("passed_temporal_only" if temporal_only else "passed") if passed else "failed",
        "passed": passed, "full_validation_passed": passed and not temporal_only,
        "temporal_passed": temporal_passed, "spatial_passed": spatial_passed,
        "temporal_only": temporal_only,
        "policy": {"min_time_gap_seconds": min_time_gap_seconds,
                   "min_location_gap_m": min_location_gap_m},
        "recording_counts": {"train": len(manifest["train"]), "val": len(manifest["val"])},
        "min_time_gap_seconds": min(pair["time_gap_seconds"] for pair in pairs),
        "pairs": pairs,
        "limitations": [
            "Separation is conditional on truthful, complete recording metadata and accurate clocks/routes.",
            "Route separation does not prove disjoint camera views or statistical independence.",
            "This audit does not check base-model pretraining overlap or model quality.",
        ] + (["Temporal-only mode does not require geographic separation."] if temporal_only else []),
    }
