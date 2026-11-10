# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Download the Physical AI AV dataset from Hugging Face.

example: for downloading camera features and egomotion for fine-tuning
python scripts/download_pai.py --chunk-ids 0-2 \
         --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
         --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions --labels egomotion egomotion.offline obstacle.offline

example: only download a reproducible random subset of N reasoning clips (and their chunks)
python scripts/download_pai.py --only-reasoning-chunks --num-reasoning-clips 64 \
     --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
     --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions --labels egomotion egomotion.offline obstacle.offline \
    --reasoning ood_reasoning.parquet --output-dir ./PAI_reasoning
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
import shlex
from pathlib import Path
import sys
import tempfile

# Support running this source-checkout CLI before installing a recipe environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alpamayo.data.pai_metadata import (
    EDGE_MIN_T0_US, EDGE_MAX_T0_US, LEGACY_MIN_T0_US,
    parse_chunk_ids, events_have_reasoning,
)

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
MANDATORY_PATTERNS = [
    "features.csv",
    "clip_index.parquet",
    "metadata/**",
]
OPTIONAL_COMPONENTS = ("camera", "calibration", "labels", "lidar", "radar", "reasoning")

# Phase 1 for ``--only-reasoning-chunks``: files needed to map ood_reasoning clip_ids -> chunk IDs.
PHASE1_INFER_CHUNKS_PATTERNS = [
    "clip_index.parquet",
    "reasoning/ood_reasoning.parquet",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PAI dataset downloader."""
    parser = argparse.ArgumentParser(
        description="Download the Physical AI AV dataset from Hugging Face Hub."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nvidia/PhysicalAI-Autonomous-Vehicles"),
        help="Local directory to store downloaded files.",
    )
    parser.add_argument(
        "--chunk-ids",
        type=str,
        default=None,
        help="Chunk IDs to download. Supports: single '0', multi '0 1', or range '0-3' (exclusive end, downloads 0,1,2). Downloads all if not specified. Ignored when --only-reasoning-chunks is set.",
    )
    parser.add_argument(
        "--camera",
        nargs="+",
        default=None,
        help="Camera subparts, e.g. camera_front_wide_120fov camera_cross_left_120fov.",
    )
    parser.add_argument(
        "--calibration",
        nargs="+",
        default=None,
        help="Calibration subparts, e.g. camera_intrinsics sensor_extrinsics.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels subparts, e.g. egomotion.",
    )
    parser.add_argument(
        "--lidar",
        nargs="+",
        default=None,
        help="Lidar subparts, e.g. lidar_top_360fov.",
    )
    parser.add_argument(
        "--radar",
        nargs="+",
        default=None,
        help="Radar subparts, e.g. radar_front_center_mrr_2.",
    )
    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=None,
        help="Reasoning subparts, e.g. reasoning_cot, or a single parquet at reasoning/<name>.parquet "
        "such as ood_reasoning.parquet.",
    )
    parser.add_argument(
        "--only-reasoning-chunks",
        action="store_true",
        dest="only_reasoning_chunks",
        help=(
            "Ignore --chunk-ids. Download clip_index.parquet and reasoning/ood_reasoning.parquet first, "
            "take clip_ids from ood_reasoning, map them to chunk IDs via clip_index, then run the normal "
            "download with those chunk IDs (camera, calibration, labels, reasoning, etc.)."
        ),
    )
    parser.add_argument(
        "--num-reasoning-clips",
        type=int,
        default=None,
        dest="num_reasoning_clips",
        help=(
            "Only valid with --only-reasoning-chunks. Randomly sample N clips from "
            "reasoning/ood_reasoning.parquet (after filtering empty annotations and invalid event times), then download only the "
            "chunks containing those clips. Also writes the corresponding rows from clip_index.parquet "
            "to <output-dir>/clip_index_reasoning_mini.parquet."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=11,
        dest="random_seed",
        help="Random seed used by --num-reasoning-clips sampling for reproducibility. Default: 11.",
    )
    parser.add_argument("--revision", default=None, help="Hugging Face dataset revision; resolved to a commit SHA and recorded locally.")
    parser.add_argument("--profile", choices=("legacy", "edge"), default="legacy", help="Reasoning event window: legacy (1.6, 13.6] s; edge [2.0, 13.6] s.")
    args = parser.parse_args()
    if args.num_reasoning_clips is not None and not args.only_reasoning_chunks:
        parser.error(
            "--num-reasoning-clips can only be used together with --only-reasoning-chunks."
        )
    if args.num_reasoning_clips is not None and args.num_reasoning_clips <= 0:
        parser.error("--num-reasoning-clips must be a positive integer.")
    return args


def parse_component_subparts(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Collect ``(component, subpart)`` pairs from the parsed CLI args.

    Iterates over ``OPTIONAL_COMPONENTS`` (camera, calibration, labels, lidar, radar,
    reasoning) and returns a flat list of non-empty, trimmed ``(component, subpart)``
    tuples, preserving the user-provided order.
    """
    component_pairs: list[tuple[str, str]] = []
    for component in OPTIONAL_COMPONENTS:
        subparts = getattr(args, component) or []
        for subpart in subparts:
            cleaned = subpart.strip().strip("/")
            if not cleaned:
                continue
            component_pairs.append((component, cleaned))
    return component_pairs


def build_allow_patterns(
    component_pairs: list[tuple[str, str]],
    chunk_ids: list[int] | None,
) -> list[str]:
    """Build the ``allow_patterns`` list passed to ``snapshot_download``.

    Always includes ``MANDATORY_PATTERNS`` plus one pattern per ``(component, subpart)``
    pair, scoped to the given ``chunk_ids`` when provided. Single-file reasoning
    releases (``reasoning/<name>.parquet``) are added verbatim. Returns a list with
    duplicates removed while preserving order.
    """
    patterns: list[str] = list(MANDATORY_PATTERNS)

    normalized_chunks = [f"chunk_{int(chunk):04d}" for chunk in (chunk_ids or [])]

    for component, subpart in component_pairs:
        # Chunked assets live under e.g. reasoning/reasoning_cot/reasoning_cot.chunk_0000.parquet.
        # Single-table releases (e.g. OOD reasoning) are one file: reasoning/ood_reasoning.parquet.
        if component == "reasoning" and subpart.endswith(".parquet"):
            patterns.append(f"{component}/{subpart}")
            continue
        if normalized_chunks:
            for chunk in normalized_chunks:
                patterns.append(f"{component}/{subpart}/{subpart}.{chunk}.*")
        else:
            patterns.append(f"{component}/{subpart}/{subpart}.*")

    # De-duplicate while preserving order.
    return list(dict.fromkeys(patterns))


def _ood_reasoning_events_nonempty(events_cell: object, min_t0_us: int = LEGACY_MIN_T0_US) -> bool:
    """Keep a clip when any labeled event falls within the requested recipe window."""
    return events_have_reasoning(events_cell, min_t0_us, EDGE_MAX_T0_US)


def infer_chunk_ids_from_ood_and_clip_index(
    output_dir: Path,
    num_reasoning_clips: int | None = None,
    random_seed: int = 42,
    min_t0_us: int = LEGACY_MIN_T0_US,
) -> list[int]:
    """Return sorted unique chunk IDs for clip_ids present in both ood_reasoning and clip_index.

    When ``num_reasoning_clips`` is given, reproducibly downsample ood_reasoning (after
    filtering empty annotations and invalid event times) to that many clips using ``random_seed``, and write the matching
    rows of ``clip_index.parquet`` to ``<output_dir>/clip_index_reasoning_mini.parquet``.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required for --only-reasoning-chunks. Install with: pip install pandas"
        ) from exc

    clip_index_path = output_dir / "clip_index.parquet"
    ood_path = output_dir / "reasoning" / "ood_reasoning.parquet"
    if not clip_index_path.is_file():
        raise SystemExit(f"Expected file missing after download: {clip_index_path}")
    if not ood_path.is_file():
        raise SystemExit(f"Expected file missing after download: {ood_path}")

    clip_index = pd.read_parquet(clip_index_path)
    ood = pd.read_parquet(ood_path)

    if "chunk" not in clip_index.columns:
        raise SystemExit("clip_index.parquet has no 'chunk' column.")

    # Filter out clips with missing or empty ``events`` — no CoT labels; PAIDataset drops
    # them anyway, so skip downloading their chunks.
    total_ood = len(ood)
    if "events" in ood.columns:
        nonempty = ood["events"].apply(lambda events: _ood_reasoning_events_nonempty(events, min_t0_us))
        ood = ood.loc[nonempty]
        skipped = total_ood - len(ood)
        if skipped:
            print(
                f"[download_pai] Skipping {skipped}/{total_ood} clip(s) with missing or empty "
                f"events in ood_reasoning; {len(ood)} clip(s) remain."
            )

    ood = ood.loc[ood.index.astype(str).isin(clip_index.index.astype(str))]

    if num_reasoning_clips is not None:
        available = len(ood)
        if available == 0:
            raise SystemExit("No reasoning clips available after filtering empty annotations and invalid event times.")
        k = min(num_reasoning_clips, available)
        if num_reasoning_clips > available:
            print(
                f"[download_pai] Warning: requested --num-reasoning-clips={num_reasoning_clips} "
                f"exceeds available {available}; using all {available}."
            )
        ood = ood.sample(n=k, random_state=random_seed).sort_index()
        print(
            f"[download_pai] Randomly sampled {k}/{available} clip(s) from ood_reasoning "
            f"with seed={random_seed}."
        )

    ood_ids = set(ood.index.astype(str))
    clip_index_str = clip_index.copy()
    clip_index_str.index = clip_index.index.astype(str)

    common = ood_ids & set(clip_index_str.index.astype(str))
    if not common:
        raise SystemExit(
            "No clip_id overlap between reasoning/ood_reasoning.parquet and clip_index.parquet."
        )

    missing = ood_ids - set(clip_index_str.index.astype(str))
    if missing:
        print(
            f"[download_pai] Warning: {len(missing)} clip_id(s) in ood_reasoning are absent from "
            "clip_index and will be skipped."
        )

    in_both = clip_index_str.index.astype(str).isin(common)

    if num_reasoning_clips is not None:
        # ``in_both`` is a positional bool mask aligned with clip_index (same row order as
        # clip_index_str). Using the mask preserves clip_index's original index dtype.
        mini = clip_index.loc[in_both]
        mini_path = output_dir / "clip_index_reasoning_mini.parquet"
        mini.to_parquet(mini_path)
        print(f"[download_pai] Wrote mini clip_index ({len(mini)} row(s)) to: {mini_path}")

    chunks = clip_index_str.loc[in_both, "chunk"].unique()
    return sorted(int(c) for c in chunks)


def _parse_cli_chunk_ids(chunk_ids: str) -> list[int]:
    """Parse the ``--chunk-ids`` CLI string into a list of integer chunk IDs.

    Supports three forms: a single ID (``"0"``), space-separated IDs (``"0 1 2"``),
    and a half-open range (``"0-3"`` -> ``[0, 1, 2]``).
    """
    return parse_chunk_ids(chunk_ids)


def _write_manifest(path: Path, manifest: dict) -> None:
    """Publish request state atomically, including before an interruptible transfer."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=".download_manifest_", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _locked_output(output_dir: Path):
    """Prevent concurrent callers from accepting different revisions or losing history."""
    # Keep the inode in place after releasing: unlinking a lock permits another
    # process to acquire a different inode while an existing waiter still owns it.
    with (output_dir / ".download_manifest.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                "Another downloader is using this output directory; wait for it to finish."
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _load_download_manifest(output_dir: Path, revision: str) -> dict:
    path = output_dir / "download_manifest.json"
    if not path.exists():
        if any(item.name != ".download_manifest.lock" for item in output_dir.iterdir()):
            raise SystemExit(
                "Output directory is populated but has no download_manifest.json; "
                "its dataset revision cannot be verified. Use a new empty directory. "
                "Existing files have not been changed."
            )
        return {"repo_id": DEFAULT_REPO_ID, "resolved_revision": revision, "requests": []}
    manifest = json.loads(path.read_text())
    if manifest.get("repo_id") != DEFAULT_REPO_ID:
        raise SystemExit("Output directory records a different dataset repository; use a new directory.")
    if manifest.get("resolved_revision") != revision:
        raise SystemExit(
            "Output directory contains a different recorded dataset revision. "
            f"Recorded SHA: {manifest.get('resolved_revision')!r}; resolved requested SHA: {revision!r}. "
            "To extend this directory, rerun the same download command with "
            f"--revision {shlex.quote(str(manifest.get('resolved_revision')))}. "
            "Use a new directory for the requested revision to avoid mixing data."
        )
    if "requests" not in manifest:
        # The older format was written only after success and retained the last
        # selection. Preserve what is known without inventing earlier requests.
        manifest["requests"] = [{
            key: manifest[key]
            for key in ("requested_revision", "profile", "chunk_ids", "allow_patterns")
            if key in manifest
        }]
        manifest["requests"][0]["status"] = "complete"
    if not isinstance(manifest["requests"], list):
        raise SystemExit("Invalid download manifest request history; use a new directory.")
    return manifest


def main() -> None:
    """Download one pinned revision, retaining interrupted and completed requests."""
    args = parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        ) from exc

    raw_chunk_ids = args.chunk_ids
    if not args.only_reasoning_chunks:
        args.chunk_ids = [] if raw_chunk_ids is None else _parse_cli_chunk_ids(raw_chunk_ids)
    component_pairs = parse_component_subparts(args)
    revision = HfApi().dataset_info(DEFAULT_REPO_ID, revision=args.revision).sha
    if not revision:
        raise SystemExit("Could not resolve the dataset revision to a commit SHA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "download_manifest.json"
    with _locked_output(args.output_dir):
        manifest = _load_download_manifest(args.output_dir, revision)
        request = {
            "requested_revision": args.revision,
            "profile": args.profile,
            "requested_chunk_ids": raw_chunk_ids,
            "chunk_ids": None if args.only_reasoning_chunks else args.chunk_ids,
            "allow_patterns": [],
            "status": "in_progress",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest["requests"].append(request)

        def record_request():
            # Retain the latest selection at the top level for old consumers;
            # requests is the complete history for sequential chunk downloads.
            for key in ("error_type", "finished_at"):
                if key not in request:
                    manifest.pop(key, None)
            manifest.update(request)
            _write_manifest(manifest_path, manifest)

        # Reserve the revision before any snapshot can leave downloaded files.
        # If the process is killed, in_progress explicitly marks the unfinished request.
        record_request()
        print(f"Dataset revision: {revision}")
        try:
            if args.only_reasoning_chunks:
                if raw_chunk_ids is not None:
                    print(
                        "[download_pai] Ignoring --chunk-ids; inferring chunk IDs from "
                        "reasoning/ood_reasoning.parquet and clip_index.parquet."
                    )
                print("Phase 1 (infer chunks):", PHASE1_INFER_CHUNKS_PATTERNS)
                request["allow_patterns"] = list(PHASE1_INFER_CHUNKS_PATTERNS)
                record_request()
                snapshot_download(
                    repo_id=DEFAULT_REPO_ID,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=str(args.output_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=PHASE1_INFER_CHUNKS_PATTERNS,
                )
                args.chunk_ids = infer_chunk_ids_from_ood_and_clip_index(
                    args.output_dir,
                    num_reasoning_clips=args.num_reasoning_clips,
                    random_seed=args.random_seed,
                    min_t0_us=EDGE_MIN_T0_US if args.profile == "edge" else LEGACY_MIN_T0_US,
                )
                print(
                    f"[download_pai] Inferred {len(args.chunk_ids)} chunk(s) from ood_reasoning "
                    f"clip_ids (via clip_index): {args.chunk_ids}"
                )
            else:
                print("downloading chunks: ", args.chunk_ids if args.chunk_ids else "all")

            allow_patterns = build_allow_patterns(component_pairs, args.chunk_ids)
            request["chunk_ids"] = args.chunk_ids
            request["allow_patterns"] = list(dict.fromkeys(
                request["allow_patterns"] + allow_patterns
            ))
            record_request()
            print("download patterns", allow_patterns)
            downloaded_path = snapshot_download(
                repo_id=DEFAULT_REPO_ID,
                repo_type="dataset",
                revision=revision,
                local_dir=str(args.output_dir),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )
        except BaseException as exc:
            request["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            request["error_type"] = type(exc).__name__
            request["finished_at"] = datetime.now(timezone.utc).isoformat()
            record_request()
            raise
        request["status"] = "complete"
        request["finished_at"] = datetime.now(timezone.utc).isoformat()
        record_request()
        print(f"Downloaded dataset snapshot to: {downloaded_path}")
        print("Included mandatory patterns: " + ", ".join(MANDATORY_PATTERNS))


if __name__ == "__main__":
    main()
