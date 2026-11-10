# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare public PAI CoC labels or split existing paired Nav+CoC JSON.

Navigation is joined only at the exact clip-relative microsecond timestamp. This
script neither generates navigation instructions nor verifies their semantics.
It reads local files only; downloading PAI sensor data is a separate step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
import sys
import tempfile
from typing import TYPE_CHECKING

# Support running this source-checkout CLI before installing a recipe environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alpamayo.data.pai_metadata import (
    EDGE_MIN_T0_US, EDGE_MAX_T0_US, reasoning_text, validate_navigation,
)

if TYPE_CHECKING:
    import pandas as pd


SPLITS = ("train", "val")
MIN_T0_US = EDGE_MIN_T0_US
MAX_T0_US = EDGE_MAX_T0_US
OUTPUT_NAMES = (
    "cot_train.json", "cot_val.json", "nav_template.json",
    "nav_cot_train.json", "nav_cot_val.json", "preparation_report.json",
)


def _clip_id(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context}: clip_id must be a nonempty string without surrounding whitespace")
    return value


def _timestamp(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context}: timestamp must be integer clip-relative microseconds")
    return value


def _read_events(value: object, context: str) -> list:
    import pandas as pd

    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return []
    if not isinstance(value, str):
        raise ValueError(f"{context}: events must be a JSON-encoded list")
    if not value.strip():
        return []
    try:
        events = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context}: invalid events JSON: {exc}") from exc
    if not isinstance(events, list):
        raise ValueError(f"{context}: events must decode to a list")
    return events


def prepare_coc(clip_index: pd.DataFrame, reasoning: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Return split annotations, clip-to-chunk mapping, and preparation counts."""
    import pandas as pd

    chunk_column = "chunk" if "chunk" in clip_index.columns else "chunk_id"
    if chunk_column not in clip_index.columns:
        raise ValueError("clip index must have a 'chunk' or 'chunk_id' column")
    if not {"split", "events"}.issubset(reasoning.columns):
        raise ValueError("ood_reasoning.parquet must have 'split' and 'events' columns")
    chunks = {}
    for clip, chunk in clip_index[chunk_column].items():
        clip = _clip_id(clip, "clip index")
        if clip in chunks:
            raise ValueError(f"clip index: duplicate clip_id {clip!r}")
        if pd.isna(chunk) or not str(chunk).strip():
            raise ValueError(f"clip index: missing chunk for {clip!r}")
        chunks[clip] = str(chunk)
    if not chunks:
        raise ValueError("clip index is empty")

    dropped = {
        "reasoning_rows_outside_clip_index": 0,
        "empty_event_rows": 0,
        "empty_coc_events": 0,
        "out_of_window_events": 0,
        "identical_duplicate_events": 0,
    }
    clip_splits = {}
    events_by_key = {}
    for clip, row in reasoning.iterrows():
        clip = _clip_id(clip, "reasoning index")
        if clip not in chunks:
            dropped["reasoning_rows_outside_clip_index"] += 1
            continue
        context = f"reasoning clip {clip!r}"
        split = row["split"]
        if not isinstance(split, str) or split not in SPLITS:
            raise ValueError(f"{context}: split must be 'train' or 'val', got {split!r}")
        if clip in clip_splits and clip_splits[clip] != split:
            raise ValueError(f"{context}: conflicting official splits for the same clip")
        clip_splits[clip] = split
        events = _read_events(row["events"], context)
        if not events:
            dropped["empty_event_rows"] += 1
        for index, event in enumerate(events, 1):
            event_context = f"{context}, event {index}"
            if not isinstance(event, dict):
                raise ValueError(f"{event_context}: event must be an object")
            timestamp = _timestamp(event.get("event_start_timestamp"), event_context)
            cot = reasoning_text(event)
            if cot is None or (isinstance(cot, str) and not cot.strip()):
                dropped["empty_coc_events"] += 1
                continue
            if not isinstance(cot, str):
                raise ValueError(f"{event_context}: coc must be a string")
            if not MIN_T0_US <= timestamp <= MAX_T0_US:
                dropped["out_of_window_events"] += 1
                continue
            key = (clip, timestamp)
            annotation = {"clip_id": clip, "t0_relative": timestamp, "cot": cot.strip()}
            if key in events_by_key:
                if events_by_key[key][1] != annotation:
                    raise ValueError(f"{event_context}: conflicting CoC labels for duplicate event {key!r}")
                dropped["identical_duplicate_events"] += 1
                continue
            events_by_key[key] = (split, annotation)
    if not events_by_key:
        raise ValueError(
            "No usable CoC events in the selected clip index and Edge window "
            f"[{MIN_T0_US}, {MAX_T0_US}] us. Dropped: {json.dumps(dropped)}"
        )
    annotations = {split: [] for split in SPLITS}
    for key in sorted(events_by_key):
        split, annotation = events_by_key[key]
        annotations[split].append(annotation)
    counts = {
        "selected_index_clips": len(chunks),
        "selected_index_chunks": len(set(chunks.values())),
        "selected_clips_without_reasoning_rows": len(set(chunks) - set(clip_splits)),
        "selected_clips_without_usable_events": len(set(chunks) - {key[0] for key in events_by_key}),
        "dropped": dropped,
        "label_statistics": {split: label_statistics(rows) for split, rows in annotations.items()},
        "warnings": [
            "Public CoC annotations describe selected rare events; their distribution differs from the checkpoint training mixture.",
            "A lack of <NOMINAL> labels or route-aware reasoning can shift behavior during fine-tuning; assess retention separately.",
            "Official PAI split disjointness does not establish that clips were unseen by the released checkpoint.",
        ],
    }
    return annotations, chunks, counts


def join_navigation(path: Path, annotations: dict, *, validate_navigation_format: bool = True) -> tuple[dict, dict]:
    """Validate user-supplied navigation and join exact (clip_id, t0) keys."""
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if isinstance(document, list):
        rows, timestamp_field = document, "t0_relative"
    elif isinstance(document, dict) and isinstance(document.get("samples"), list):
        rows, timestamp_field = document["samples"], "t0_us"
    else:
        raise ValueError("navigation JSON must be a recipe-format list or an Edge manifest with a samples list")
    known = {
        (row["clip_id"], row["t0_relative"]): (split, row)
        for split in SPLITS for row in annotations[split]
    }
    navigation = {}
    seen = set()
    blank_count = 0
    unmatched = []
    for index, row in enumerate(rows, 1):
        context = f"navigation row {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{context}: annotation must be an object")
        clip = _clip_id(row.get("clip_id"), context)
        timestamp = _timestamp(row.get(timestamp_field), context)
        other_timestamp = "t0_us" if timestamp_field == "t0_relative" else "t0_relative"
        if other_timestamp in row:
            if _timestamp(row[other_timestamp], context) != timestamp:
                raise ValueError(f"{context}: conflicting timestamp fields")
        key = (clip, timestamp)
        if key in seen:
            raise ValueError(f"{context}: duplicate navigation key {key!r}")
        seen.add(key)
        nav = row.get("nav_text")
        if isinstance(nav, list) and len(nav) == 1:
            nav = nav[0]
        if not isinstance(nav, str):
            raise ValueError(f"{context}: nav_text must be a string or a list containing one string")
        if not nav.strip():
            blank_count += 1
            continue
        if key not in known:
            unmatched.append(key)
            continue
        split, annotation = known[key]
        if "split" in row and row["split"] != split:
            raise ValueError(f"{context}: split conflicts with the official CoC split")
        if "cot" in row and row["cot"] != annotation["cot"]:
            raise ValueError(f"{context}: cot conflicts with the public CoC event")
        navigation[key] = validate_navigation(nav, context) if validate_navigation_format else nav
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} nonempty navigation annotation(s) have no exact CoC event match; "
            f"first keys: {unmatched[:3]!r}. Match clip_id and timestamp in microseconds; "
            "the one-event Edge navigation demo cannot supply paired train/val labels."
        )
    joined = {split: [] for split in SPLITS}
    for split in SPLITS:
        for row in annotations[split]:
            key = (row["clip_id"], row["t0_relative"])
            if key in navigation:
                joined[split].append({**row, "nav_text": navigation[key]})
    if any(not joined[split] for split in SPLITS):
        sizes = ", ".join(f"{split}={len(joined[split])}" for split in SPLITS)
        raise ValueError(
            f"Nav+CoC requires nonempty official train and val splits after exact joining ({sizes}); "
            f"skipped {blank_count} blank navigation row(s). Supply reviewed navigation for both splits. "
            "A single demo is insufficient. No output files were written."
        )
    return joined, {"input_rows": len(rows), "blank_rows_skipped": blank_count}


def prepare_paired_json(
    path: Path, val_fraction: float, seed: int, clip_index=None, *,
    validate_navigation_format: bool = True,
) -> tuple[dict, dict]:
    """Validate existing labels and partition complete clips for a custom split."""
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be finite and strictly between 0 and 1")
    if type(seed) is not int:
        raise ValueError("--seed must be an integer")
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError("annotations JSON must be a recipe-format list")
    seen = set()
    for index, row in enumerate(rows, 1):
        context = f"annotation row {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{context}: annotation must be an object")
        clip = _clip_id(row.get("clip_id"), context)
        timestamp = _timestamp(row.get("t0_relative"), context)
        if not MIN_T0_US <= timestamp <= MAX_T0_US:
            raise ValueError(f"{context}: t0_relative must be in [{MIN_T0_US}, {MAX_T0_US}] us")
        key = (clip, timestamp)
        if key in seen:
            raise ValueError(f"{context}: duplicate event key {key!r}")
        seen.add(key)
        nav = row.get("nav_text")
        if isinstance(nav, list) and len(nav) == 1:
            nav = nav[0]
        if not isinstance(nav, str) or not nav.strip():
            raise ValueError(f"{context}: nav_text must be a nonblank string or a one-string list")
        if validate_navigation_format:
            validate_navigation(nav, context)
        if not isinstance(row.get("cot"), str) or not row["cot"].strip():
            raise ValueError(f"{context}: cot must be a nonblank string")
    clip_ids = sorted({row["clip_id"] for row in rows})
    val_count = math.ceil(len(clip_ids) * val_fraction)
    if not 0 < val_count < len(clip_ids):
        raise ValueError("Need at least two clips and a --val-fraction that leaves nonempty train and val splits")
    shuffled = clip_ids.copy()
    random.Random(seed).shuffle(shuffled)
    val_clips = set(shuffled[:val_count])
    annotations = {split: [] for split in SPLITS}
    for row in rows:
        annotations["val" if row["clip_id"] in val_clips else "train"].append(row)
    report = {
        "annotations_source": str(path.resolve()),
        "validate_navigation_format": validate_navigation_format,
        "split_method": "Randomized demo/custom split by clip; not the official PAI split.",
        "seed": seed,
        "val_fraction": val_fraction,
        "events": len(rows),
        "clips": len(clip_ids),
        "clip_ids": clip_ids,
        "splits": {
            split: {"events": len(events), "clips": len({row["clip_id"] for row in events}),
                    "clip_ids": sorted({row["clip_id"] for row in events})}
            for split, events in annotations.items()
        },
    }
    report["warnings"] = [
        "Demonstration only: this custom clip split is not an official PAI split or evidence of generalization.",
        "Different clips can share a drive; clip-disjoint does not imply drive-disjoint or checkpoint-unseen.",
        "Training demo clips can contaminate subsequent official-validation/test evaluations.",
        ("Navigation syntax is checked, but label/trajectory alignment still needs human review."
         if validate_navigation_format else
         "Navigation syntax validation was disabled; review the custom format and label/trajectory alignment."),
    ]
    for split, events in annotations.items():
        report["splits"][split]["label_statistics"] = label_statistics(events)
    if clip_index is not None:
        add_split_metadata(report, annotations, clip_index)
    return annotations, report


def label_statistics(rows):
    """Report descriptive counts; route-word counts are a heuristic, not semantic labels."""
    from collections import Counter
    import re

    text = [row.get("cot", "").strip() for row in rows]
    return {
        "events": len(rows),
        "nominal_targets": sum(value == "<NOMINAL>" for value in text),
        "reasoning_word_count": sum(len(value.split()) for value in text),
        "route_word_mentions": sum(bool(re.search(r"\b(route|navigation|destination)\b", value, re.I)) for value in text),
        "route_word_mentions_note": "Lexical heuristic only; not a route-awareness judgment.",
        "navigation_maneuvers": dict(Counter(row.get("nav_maneuver") or row.get("nav_text", "missing").split(" onto ")[0].split(" in ")[0] for row in rows if isinstance(row.get("nav_text", "missing"), str))),
    }


def add_split_metadata(report, annotations, clip_index):
    """Audit official split membership and drive overlap when metadata are supplied."""
    from collections import Counter

    session_column = next((key for key in ("session_id", "session", "drive_id") if key in clip_index.columns), None)
    sessions = {}
    for split, rows in annotations.items():
        ids = sorted({row["clip_id"] for row in rows})
        missing = sorted(set(ids) - set(clip_index.index))
        selected = clip_index.loc[clip_index.index.intersection(ids)]
        report["splits"][split]["missing_metadata_clips"] = missing
        if "split" in selected.columns:
            report["splits"][split]["official_clip_splits"] = dict(Counter(selected["split"].astype(str)))
        if session_column is not None:
            sessions[split] = set(selected[session_column].dropna().astype(str))
    if session_column is not None:
        report["shared_train_val_sessions"] = sorted(sessions.get("train", set()) & sessions.get("val", set()))
        report["session_column"] = session_column
    else:
        report["warnings"].append("No session/drive column available: drive disjointness was not verified.")


def _split_counts(rows: list, chunks: dict) -> dict:
    clip_ids = {row["clip_id"] for row in rows}
    chunk_ids = sorted({chunks[clip] for clip in clip_ids})
    return {"events": len(rows), "clips": len(clip_ids), "chunks": len(chunk_ids), "chunk_ids": chunk_ids}


def write_outputs(output_dir: Path, outputs: dict, overwrite: bool, input_paths: list[Path]) -> None:
    """Check all destinations and serialize before replacing any output files."""
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            if name not in outputs:
                raise ValueError(f"Stale output {path}; choose a new --output-dir to avoid mixing preparations")
            if not overwrite:
                raise ValueError(f"Output already exists: {path}; use --overwrite to replace generated files")
            if not path.is_file():
                raise ValueError(f"Output is not a file: {path}")
        if path.resolve() in input_paths:
            raise ValueError(f"Output would overwrite an input file: {path}; use a separate --output-dir")
    serialized = {name: json.dumps(value, indent=2, ensure_ascii=False) + "\n" for name, value in outputs.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare_pai_", dir=output_dir) as temporary:
        for name, content in serialized.items():
            (Path(temporary) / name).write_text(content, encoding="utf-8")
        for name in serialized:
            source, destination = Path(temporary) / name, output_dir / name
            if overwrite:
                os.replace(source, destination)
            else:
                # An exclusive link also protects against destinations created after preflight.
                os.link(source, destination)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pai-dir", type=Path, help="Local PAI dataset root for public CoC conversion")
    source.add_argument("--annotations-json", type=Path, help="Existing paired Nav+CoC recipe list; creates a custom clip-disjoint split without parquet")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation clip fraction for --annotations-json (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --annotations-json (default: 42)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for annotation JSON files")
    parser.add_argument("--clip-index", type=Path, help="Selected clip index; relative paths use the current directory (default: PAI_DIR/clip_index.parquet)")
    parser.add_argument("--navigation-json", type=Path, help="User-reviewed navigation, as a recipe list or Edge samples manifest")
    parser.add_argument(
        "--validate-navigation-format", action=argparse.BooleanOptionalAction, default=True,
        help="Check the release navigation syntax (default: enabled). Disable only for reviewed custom formats; event and split checks remain enabled.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated files after validation")
    args = parser.parse_args(argv)
    if args.annotations_json is not None:
        if args.navigation_json is not None:
            parser.error("--annotations-json cannot be combined with --navigation-json")
        try:
            clip_index = None
            if args.clip_index is not None:
                import pandas as pd
                clip_index = pd.read_parquet(args.clip_index)
            annotations, report = prepare_paired_json(
                args.annotations_json, args.val_fraction, args.seed, clip_index,
                validate_navigation_format=args.validate_navigation_format,
            )
            outputs = {f"nav_cot_{split}.json": annotations[split] for split in SPLITS}
            outputs["preparation_report.json"] = report
            write_outputs(args.output_dir, outputs, args.overwrite, [args.annotations_json.resolve()])
        except (ValueError, OSError, ImportError) as exc:
            parser.exit(2, f"error: {exc}\n")
        print(json.dumps(report, indent=2))
        print(f"Wrote annotations to {args.output_dir.resolve()}")
        print(f"Set TRAIN_ANNOTATIONS={args.output_dir.resolve() / 'nav_cot_train.json'}")
        print(f"Set VAL_ANNOTATIONS={args.output_dir.resolve() / 'nav_cot_val.json'}")
        print("Use the nav_cot task config with data.train_dataset.chunk_ids=null and data.val_dataset.chunk_ids=null.")
        return
    clip_path = args.clip_index or args.pai_dir / "clip_index.parquet"
    reasoning_path = args.pai_dir / "reasoning" / "ood_reasoning.parquet"
    input_paths = [clip_path.resolve(), reasoning_path.resolve()]
    try:
        import pandas as pd

        annotations, chunks, report = prepare_coc(pd.read_parquet(clip_path), pd.read_parquet(reasoning_path))
        joined = None
        navigation_counts = {"input_rows": 0, "blank_rows_skipped": 0}
        if args.navigation_json is not None:
            input_paths.append(args.navigation_json.resolve())
            joined, navigation_counts = join_navigation(
                args.navigation_json, annotations,
                validate_navigation_format=args.validate_navigation_format,
            )
        report.update({
            "clip_index": str(clip_path.resolve()),
            "reasoning_source": str(reasoning_path.resolve()),
            "navigation_source": str(args.navigation_json.resolve()) if args.navigation_json else None,
            "navigation_semantics": "User-supplied navigation; semantic alignment is not verified by this script.",
            "validate_navigation_format": args.validate_navigation_format,
            "timestamp_window_us": [MIN_T0_US, MAX_T0_US],
            "navigation": navigation_counts,
            "splits": {},
        })
        if not args.validate_navigation_format:
            report["warnings"].append("Navigation syntax validation was disabled; review the custom format and label/trajectory alignment.")
        outputs = {f"cot_{split}.json": annotations[split] for split in SPLITS}
        outputs["nav_template.json"] = [
            {**row, "split": split, "nav_text": ""}
            for split in SPLITS for row in annotations[split]
        ]
        for split in SPLITS:
            paired = joined[split] if joined is not None else []
            report["splits"][split] = {
                "cot": _split_counts(annotations[split], chunks),
                "nav_cot": _split_counts(paired, chunks),
                "missing_navigation_events": len(annotations[split]) - len(paired),
            }
            if joined is not None:
                outputs[f"nav_cot_{split}.json"] = paired
        outputs["preparation_report.json"] = report
        write_outputs(args.output_dir, outputs, args.overwrite, input_paths)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote annotations to {args.output_dir.resolve()}")
    variant = "nav_cot" if joined is not None else "cot"
    if joined is None:
        print("Copy nav_template.json to a separate reviewed navigation file, fill nav_text, then rerun with --navigation-json.")
        print("Blank nav_text rows remain unannotated; navigation is never generated from CoC text.")
    else:
        print("Navigation was joined by exact event keys; review of its meaning remains the annotator's responsibility.")
    if all(report["splits"][split][variant]["events"] for split in SPLITS):
        print(f"Set TRAIN_ANNOTATIONS={args.output_dir.resolve() / (variant + '_train.json')}")
        print(f"Set VAL_ANNOTATIONS={args.output_dir.resolve() / (variant + '_val.json')}")
        print(f"Use the {variant} task config with data.train_dataset.chunk_ids=null and data.val_dataset.chunk_ids=null.")
    else:
        print("One official CoC split is empty; select clips from both splits before training with held-out validation.")


if __name__ == "__main__":
    main()
