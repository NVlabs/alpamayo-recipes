# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Curate target number of samples from Physical AI dataset so that you don't need to download the entire dataset.

example:
python scripts/curate_pai_samples.py \
  --clip-index-path /path/to/PAI_datset/clip_index.parquet \
  --chunk 3116-3119 --num-samples 16 \
  --output-path /path/to/PAI_datset/clip_index_3116_mini.parquet

example: only curate clips that include the reasoning labels in the ood_reasoning.parquet (optional --chunk to restrict by chunk id)
         typically you will run download_pai.py first to download the ood_reasoning.parquet and clip_index.parquet and if you already
         downloaded reasoning data with --num-reasoning-clips, you can skip this curate step and directly use the clip_index_reasoning_mini.parquet
python scripts/curate_pai_samples.py \
  --clip-index-path /path/to/PAI_dataset/clip_index.parquet \
  --only-reasoning-chunks --num-samples 16 \
  --output-path /path/to/clip_index_reasoning_mini_16.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CLIP_INDEX_PATH = "dataset/alpamayo/PAI_mini/clip_index.parquet"


def default_ood_reasoning_path(clip_index_path: str) -> Path:
    """``<dataset_root>/reasoning/ood_reasoning.parquet`` next to ``clip_index.parquet``."""
    return Path(clip_index_path).resolve().parent / "reasoning" / "ood_reasoning.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate PAI samples")
    parser.add_argument(
        "--clip-index-path",
        "-p",
        type=str,
        default=DEFAULT_CLIP_INDEX_PATH,
        help="Path to clip index parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk",
        "-c",
        type=str,
        default=None,
        help=(
            "chunk_id(s): single (e.g. 3116), space-separated, or range (e.g. 3116-3119). "
            "Optional when --only-reasoning-chunks is set (then all reasoning clips are eligible)."
        ),
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, required=True, help="Number of samples to curate"
    )
    parser.add_argument(
        "--output-path", "-o", type=str, required=True, help="Output path for the curated parquet"
    )
    parser.add_argument(
        "--only-reasoning-chunks",
        action="store_true",
        dest="only_reasoning_chunks",
        help=(
            "Restrict to clip_ids that appear in the OOD reasoning table (ood_reasoning.parquet). "
            "Use --ood-reasoning-path to override the default path beside clip_index."
        ),
    )
    parser.add_argument(
        "--ood-reasoning-path",
        type=str,
        default=None,
        help="Path to reasoning/ood_reasoning.parquet (default: <parent of clip_index>/reasoning/ood_reasoning.parquet)",
    )
    args = parser.parse_args()
    if args.chunk is None and not args.only_reasoning_chunks:
        parser.error("--chunk is required unless --only-reasoning-chunks is set.")
    return args


def _parse_chunk_ids(chunk_arg: str) -> list[str]:
    """Parse --chunk like download_pai: single, space-separated, or start-end range."""
    chunk_arg = chunk_arg.strip()
    if " " in chunk_arg:
        return [s.strip() for s in chunk_arg.split() if s.strip()]
    if "-" in chunk_arg:
        start_s, end_s = chunk_arg.split("-", 1)
        start, end = int(start_s.strip()), int(end_s.strip())
        return [str(i) for i in range(start, end)]
    return [chunk_arg]


def _ood_reasoning_events_nonempty(events_cell: object) -> bool:
    """Return True if ``ood_reasoning.events`` has usable non-empty content.

    Aligns with ``pai_utils._read_reasoning_data`` and ``download_pai``: None, NaN,
    blank/``[]`` JSON, and empty lists are excluded (no CoT labels).
    """
    import json

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


def _filter_to_reasoning_clip_ids(clip_index: pd.DataFrame, ood_path: Path) -> pd.DataFrame:
    """Keep rows whose index (clip_id) appears in ``ood_reasoning`` with non-empty ``events``."""
    if not ood_path.is_file():
        raise SystemExit(f"OOD reasoning table not found: {ood_path}")
    ood = pd.read_parquet(ood_path)
    events_in_ood = "events" in ood.columns
    total_ood = len(ood)
    if events_in_ood:
        nonempty = ood["events"].apply(_ood_reasoning_events_nonempty)
        ood = ood.loc[nonempty]
        skipped = total_ood - len(ood)
        if skipped:
            print(
                f"[curate_pai_samples] Skipping {skipped}/{total_ood} clip(s) with missing or empty "
                f"events in ood_reasoning; {len(ood)} clip(s) remain."
            )
    if len(ood) == 0:
        if events_in_ood:
            raise SystemExit(
                f"No ood_reasoning rows remain after excluding missing/empty events: {ood_path}"
            )
        raise SystemExit(f"ood_reasoning table is empty: {ood_path}")
    reasoning_ids = set(ood.index.astype(str))
    mask = clip_index.index.astype(str).isin(reasoning_ids)
    out = clip_index.loc[mask]
    if len(out) == 0:
        raise SystemExit(
            f"No clip_index rows overlap ood_reasoning ({ood_path}). "
            f"ood_reasoning rows: {len(ood)}; clip_index rows: {len(clip_index)}."
        )
    ood_ids = set(ood.index.astype(str))
    ci_ids = set(clip_index.index.astype(str))
    not_in_index = ood_ids - ci_ids
    if not_in_index:
        print(
            f"[curate_pai_samples] {len(not_in_index)} ood_reasoning clip_id(s) not in clip_index (skipped)."
        )
    print(
        f"[curate_pai_samples] Restricted to {len(out)} clip(s) present in both clip_index and ood_reasoning."
    )
    return out


def main() -> None:
    args = parse_args()

    clip_index = pd.read_parquet(args.clip_index_path)
    col = "chunk_id" if "chunk_id" in clip_index.columns else "chunk"

    if args.only_reasoning_chunks:
        ood_path = (
            Path(args.ood_reasoning_path).resolve()
            if args.ood_reasoning_path
            else default_ood_reasoning_path(args.clip_index_path)
        )
        print(f"[curate_pai_samples] Using OOD reasoning table: {ood_path}")
        chunk_subset = _filter_to_reasoning_clip_ids(clip_index, ood_path)
    else:
        chunk_subset = clip_index

    if args.chunk is not None:
        chunk_ids = _parse_chunk_ids(args.chunk)
        print(f"[curate_pai_samples] Restricting to chunk ID(s): {chunk_ids}")
        chunk_set = set(chunk_ids)
        chunk_subset = chunk_subset[chunk_subset[col].astype(str).isin(chunk_set)]

    if len(chunk_subset) == 0:
        raise SystemExit(
            f"No rows left after filters ({col}). "
            f"Available chunks (sample): {clip_index[col].dropna().unique()[:10].tolist()}"
        )

    n = min(args.num_samples, len(chunk_subset))
    if n < args.num_samples:
        print(f"Warning: only {len(chunk_subset)} rows available, sampling {n}.")
    curated_clip_index = chunk_subset.sample(n=n, random_state=11)
    curated_clip_index.to_parquet(args.output_path)


if __name__ == "__main__":
    main()
