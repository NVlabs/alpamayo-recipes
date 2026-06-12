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

example: for downloading 4camera + egomotion for AR1 finetuning
python scripts/download_pai.py --chunk-ids 0-2 \
         --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
         --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions --labels egomotion egomotion.offline obstacle.offline

example: only download a reproducible random subset of N reasoning clips (and their chunks)
python scripts/download_pai.py --only-reasoning-chunks --num-reasoning-clips 64 \
     --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
     --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions --labels egomotion egomotion.offline obstacle.offline \
    --reasoning ood_reasoning.parquet --output-dir ./PAI_reasoning_obstacle
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
        default=16,
        dest="num_reasoning_clips",
        help=(
            "Only valid with --only-reasoning-chunks. Randomly sample N clips from "
            "reasoning/ood_reasoning.parquet (after filtering empty events), then download only the "
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


def _ood_reasoning_events_nonempty(events_cell: object) -> bool:
    """Return True if ``ood_reasoning.events`` has usable non-empty content.

    Aligns with ``pai_utils._read_reasoning_data``: None, NaN, blank/``[]`` JSON,
    and empty lists are treated as no CoT labels (skip downloading that clip's chunk).
    """
    import json

    import pandas as pd

    if events_cell is None:
        return False
    if isinstance(events_cell, str):
        stripped = events_cell.strip()
        if not stripped:
            return False
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        if isinstance(parsed, dict):
            return len(parsed) > 0
        if isinstance(parsed, (list, tuple)):
            return len(parsed) > 0
        return True
    if pd.api.types.is_scalar(events_cell) and pd.isna(events_cell):
        return False
    if isinstance(events_cell, (list, tuple)):
        return len(events_cell) > 0
    if isinstance(events_cell, dict):
        return len(events_cell) > 0
    if hasattr(events_cell, "__len__") and not isinstance(events_cell, (str, bytes)):
        try:
            return len(events_cell) > 0
        except TypeError:
            return True
    return bool(events_cell)


def infer_chunk_ids_from_ood_and_clip_index(
    output_dir: Path,
    num_reasoning_clips: int | None = None,
    random_seed: int = 42,
) -> list[int]:
    """Return sorted unique chunk IDs for clip_ids present in both ood_reasoning and clip_index.

    When ``num_reasoning_clips`` is given, reproducibly downsample ood_reasoning (after
    filtering empty events) to that many clips using ``random_seed``, and write the matching
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
        nonempty = ood["events"].apply(_ood_reasoning_events_nonempty)
        ood = ood.loc[nonempty]
        skipped = total_ood - len(ood)
        if skipped:
            print(
                f"[download_pai] Skipping {skipped}/{total_ood} clip(s) with missing or empty "
                f"events in ood_reasoning; {len(ood)} clip(s) remain."
            )

    if num_reasoning_clips is not None:
        available = len(ood)
        if available == 0:
            raise SystemExit("No reasoning clips available after filtering empty events.")
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
    if " " in chunk_ids:
        return [int(x) for x in chunk_ids.split()]
    if "-" in chunk_ids:
        start = int(chunk_ids.split("-")[0])
        end = int(chunk_ids.split("-")[1])
        return list(range(start, end))
    return [int(chunk_ids)]


def main() -> None:
    """Run the PAI dataset download.

    In ``--only-reasoning-chunks`` mode, first downloads ``clip_index.parquet`` and
    ``reasoning/ood_reasoning.parquet``, infers the minimal set of chunk IDs to fetch
    (optionally downsampled via ``--num-reasoning-clips``), then performs the main
    ``snapshot_download`` with the resolved ``allow_patterns``.
    """
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install with: pip install huggingface_hub"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_chunk_ids = args.chunk_ids

    if args.only_reasoning_chunks:
        if raw_chunk_ids is not None:
            print(
                "[download_pai] Ignoring --chunk-ids; inferring chunk IDs from "
                "reasoning/ood_reasoning.parquet and clip_index.parquet."
            )
        print("Phase 1 (infer chunks):", PHASE1_INFER_CHUNKS_PATTERNS)
        snapshot_download(
            repo_id=DEFAULT_REPO_ID,
            repo_type="dataset",
            local_dir=str(args.output_dir),
            local_dir_use_symlinks=False,
            allow_patterns=PHASE1_INFER_CHUNKS_PATTERNS,
        )
        inferred = infer_chunk_ids_from_ood_and_clip_index(
            args.output_dir,
            num_reasoning_clips=args.num_reasoning_clips,
            random_seed=args.random_seed,
        )
        args.chunk_ids = inferred
        print(
            f"[download_pai] Inferred {len(inferred)} chunk(s) from ood_reasoning "
            f"clip_ids (via clip_index): {inferred}"
        )
    else:
        if args.chunk_ids is None:
            args.chunk_ids = []
        else:
            args.chunk_ids = _parse_cli_chunk_ids(args.chunk_ids)

        print("downloading chunks: ", args.chunk_ids if args.chunk_ids else "all")

    try:
        component_pairs = parse_component_subparts(args)
        allow_patterns = build_allow_patterns(component_pairs, args.chunk_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("download patterns", allow_patterns)

    downloaded_path = snapshot_download(
        repo_id=DEFAULT_REPO_ID,
        repo_type="dataset",
        local_dir=str(args.output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    print(f"Downloaded dataset snapshot to: {downloaded_path}")
    print("Included mandatory patterns: " + ", ".join(MANDATORY_PATTERNS))


if __name__ == "__main__":
    main()
