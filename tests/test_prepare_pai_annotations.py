# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise public annotation preparation with real Parquet files and CLI calls."""

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_pai_annotations.py"


def _event(t0, text="Yield to the crossing pedestrian."):
    return {"event_start_timestamp": t0, "coc": text}


@pytest.fixture
def corpus(tmp_path):
    pai = tmp_path / "pai"
    (pai / "reasoning").mkdir(parents=True)
    pd.DataFrame({"chunk": [7, 8]}, index=["train-clip", "val-clip"]).to_parquet(
        pai / "clip_index.parquet"
    )
    reasoning = pd.DataFrame([
        {"split": "train", "events": json.dumps([
            _event(2_000_000), _event(13_600_000, "Wait for the vehicle ahead."),
            _event(1_999_999), _event(13_600_001), _event(3_000_000, ""),
        ])},
        {"split": "val", "events": json.dumps([_event(4_000_000)])},
        {"split": "train", "events": json.dumps([_event(5_000_000)])},
    ], index=["train-clip", "val-clip", "unselected-clip"])
    reasoning.to_parquet(pai / "reasoning/ood_reasoning.parquet")
    return pai


def _run(pai, out, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--pai-dir", str(pai), "--output-dir", str(out), *args],
        text=True, capture_output=True, timeout=30,
    )


def _read(out, name):
    return json.loads((out / name).read_text())


def _nav_rows():
    # Synthetic fixtures exercise joining; these are not real navigation annotations.
    return [
        {"clip_id": "train-clip", "t0_relative": 2_000_000, "nav_text": "Continue straight"},
        {"clip_id": "val-clip", "t0_relative": 4_000_000, "nav_text": ["Turn right"]},
        {"clip_id": "train-clip", "t0_relative": 13_600_000, "nav_text": ""},
    ]


def _write_nav(tmp_path, rows):
    path = tmp_path / "navigation.json"
    path.write_text(json.dumps(rows))
    return str(path)


def test_public_coc_preserves_events_and_splits_without_inventing_navigation(corpus, tmp_path):
    out = tmp_path / "prepared"
    result = _run(corpus, out)
    assert result.returncode == 0, result.stderr
    train, val = _read(out, "cot_train.json"), _read(out, "cot_val.json")
    assert [row["t0_relative"] for row in train] == [2_000_000, 13_600_000]
    assert train[1]["cot"] == "Wait for the vehicle ahead."
    assert {row["clip_id"] for row in train} == {"train-clip"}
    assert {row["clip_id"] for row in val} == {"val-clip"}
    template = _read(out, "nav_template.json")
    assert len(template) == 3 and all(row["nav_text"] == "" for row in template)
    assert not (out / "nav_cot_train.json").exists()
    report = _read(out, "preparation_report.json")
    assert report["dropped"]["out_of_window_events"] == 2
    assert report["dropped"]["empty_coc_events"] == 1
    assert report["dropped"]["reasoning_rows_outside_clip_index"] == 1
    assert report["splits"]["train"]["cot"]["events"] == 2
    assert report["splits"]["val"]["cot"]["clips"] == 1


@pytest.mark.parametrize("edge_format", [False, True])
def test_navigation_joins_exact_events_and_leaves_unannotated_events_out(
    corpus, tmp_path, edge_format
):
    rows = _nav_rows()
    if edge_format:
        rows = {"samples": [
            {"clip_id": row["clip_id"], "t0_us": row["t0_relative"], "nav_text": row["nav_text"]}
            for row in rows
        ]}
    nav = _write_nav(tmp_path, rows)
    out = tmp_path / "paired"
    result = _run(corpus, out, "--navigation-json", nav)
    assert result.returncode == 0, result.stderr
    assert _read(out, "nav_cot_train.json") == [{
        "clip_id": "train-clip", "t0_relative": 2_000_000,
        "cot": "Yield to the crossing pedestrian.", "nav_text": "Continue straight",
    }]
    val = _read(out, "nav_cot_val.json")
    assert val[0]["clip_id"] == "val-clip" and val[0]["nav_text"] == "Turn right"
    report = _read(out, "preparation_report.json")
    assert report["navigation"]["blank_rows_skipped"] == 1
    assert report["splits"]["train"]["missing_navigation_events"] == 1


@pytest.mark.parametrize("problem", [
    "wrong-time", "wrong-clip", "duplicate", "float-time", "bool-time", "missing-nav",
    "conflicting-times", "conflicting-coc", "conflicting-split", "missing-val",
])
def test_invalid_or_incomplete_navigation_cannot_produce_training_files(corpus, tmp_path, problem):
    rows = _nav_rows()
    if problem == "wrong-time":
        rows[0]["t0_relative"] += 1  # A nearest-event join must not silently accept this.
    elif problem == "wrong-clip":
        rows[0]["clip_id"] = "unselected-clip"
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "float-time":
        rows[0]["t0_relative"] = 2_000_000.0
    elif problem == "bool-time":
        rows[0]["t0_relative"] = True
    elif problem == "missing-nav":
        del rows[0]["nav_text"]
    elif problem == "conflicting-times":
        rows[0]["t0_us"] = 4_000_000
    elif problem == "conflicting-coc":
        rows[0]["cot"] = "An unrelated event."
    elif problem == "conflicting-split":
        rows[0]["split"] = "val"
    elif problem == "missing-val":
        rows = [rows[0]]
    out = tmp_path / "invalid"
    result = _run(corpus, out, "--navigation-json", _write_nav(tmp_path, rows))
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("problem", ["clip-split", "event-target"])
def test_conflicting_public_annotations_are_rejected(corpus, tmp_path, problem):
    path = corpus / "reasoning/ood_reasoning.parquet"
    frame = pd.read_parquet(path)
    row = frame.loc[["train-clip"]].copy()
    if problem == "clip-split":
        row["split"] = "val"
    else:
        row["events"] = json.dumps([_event(2_000_000, "Conflicting target.")])
    pd.concat([frame, row]).to_parquet(path)
    out = tmp_path / "invalid"
    result = _run(corpus, out)
    assert result.returncode != 0
    assert "conflicting" in result.stderr
    assert not out.exists()


def test_explicit_clip_selection_and_overwrite_protection(corpus, tmp_path):
    selected = tmp_path / "selection.parquet"
    pd.read_parquet(corpus / "clip_index.parquet").loc[["val-clip"]].to_parquet(selected)
    out = tmp_path / "selected"
    result = _run(corpus, out, "--clip-index", str(selected))
    assert result.returncode == 0, result.stderr
    assert _read(out, "cot_train.json") == []
    assert "split is empty" in result.stdout
    original = (out / "cot_val.json").read_bytes()
    result = _run(corpus, out)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert (out / "cot_val.json").read_bytes() == original
    result = _run(corpus, out, "--overwrite")
    assert result.returncode == 0, result.stderr
    assert len(_read(out, "cot_train.json")) == 2


DEMO = SCRIPT.parents[1] / "recipes/alpamayo2edge_sft/assets/nav_demo_samples.json"


def _run_paired(source, out, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--annotations-json", str(source),
         "--output-dir", str(out), *args],
        text=True, capture_output=True, timeout=30,
    )


def test_bundled_json_splits_by_clip_without_pai_files_or_label_changes(tmp_path):
    source_bytes = DEMO.read_bytes()
    source_rows = json.loads(source_bytes)
    out = tmp_path / "splits"
    result = _run_paired(DEMO, out, "--val-fraction", "0.2", "--seed", "5")
    assert result.returncode == 0, result.stderr
    train, val = _read(out, "nav_cot_train.json"), _read(out, "nav_cot_val.json")
    train_clips, val_clips = ({r["clip_id"] for r in rows} for rows in (train, val))
    assert len(train) == 15 and len(val) == 4
    assert len(train_clips) == 14 and len(val_clips) == 4
    assert not train_clips.intersection(val_clips)
    # Pin the complete partition checked against driving-session metadata.
    # Changed clips require a new audit; clip-disjointness alone is insufficient.
    assert train_clips == {
        "1ae17e2a-bc07-4e57-a04b-e9d5a372cd1e", "681c26a7-168b-400d-bcf4-c3a807ca5f8b",
        "6fb1a5f2-fbbd-40fb-9bfa-bc3ada06e503", "709fe679-4f3e-4938-8767-23886741c6bc",
        "8455d8a6-3c18-473b-90fd-a89038684c6f", "99baefc2-e9c3-4e84-bea9-71608e48d580",
        "b7dd3c46-ca94-4ffa-a8cf-3a13350c8aba", "bd65ae5a-7c50-4d33-a953-bd382c108d04",
        "c9c045a3-ebe9-4569-9ce3-a44068cf2e3b", "e7471b86-88f5-40a4-9d99-a005ff13db15",
        "e8221498-a35e-42a2-9e5e-1a9a0185089b", "ea7bbd31-b7a5-4972-8dbd-7089e6b53de4",
        "f2205e0b-b16f-491d-a211-e65db71d4f1c", "f4fc523c-4dd8-4728-b35d-e8aed5399028",
    }
    assert val_clips == {
        "70d71413-d377-4836-ae6a-72f3b3bb5c49", "7ec7451c-a58a-49bc-a3b3-70330a314434",
        "82f9f689-6efb-4eb3-af10-070b52eeca22", "b4a2c25d-7f1c-4446-856a-a1c862af9c16",
    }
    assert Counter(row["nav_text"].split(" in ")[0] for row in val) == {
        "Turn left": 3, "Turn right": 1,
    }
    key = lambda row: (row["clip_id"], row["t0_relative"])
    assert sorted(train + val, key=key) == sorted(source_rows, key=key)
    assert DEMO.read_bytes() == source_bytes
    assert {p.name for p in out.iterdir()} == {
        "nav_cot_train.json", "nav_cot_val.json", "preparation_report.json",
    }
    report = _read(out, "preparation_report.json")
    assert report["seed"] == 5 and report["val_fraction"] == 0.2
    assert report["splits"]["train"]["clips"] == 14
    # Input row order must not change which clips belong to validation.
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(list(reversed(source_rows))))
    second = tmp_path / "second"
    result = _run_paired(reordered, second, "--seed", "5")
    assert result.returncode == 0, result.stderr
    assert {r["clip_id"] for r in _read(second, "nav_cot_val.json")} == val_clips


@pytest.mark.parametrize("problem", ["one-clip", "duplicate", "missing-cot", "blank-nav", "invalid-time"])
def test_paired_json_validation_precedes_output_writes(tmp_path, problem):
    rows = json.loads(DEMO.read_text())
    if problem == "one-clip":
        rows = [rows[0]]
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    elif problem == "missing-cot":
        del rows[0]["cot"]
    elif problem == "blank-nav":
        rows[0]["nav_text"] = ""
    elif problem == "invalid-time":
        rows[0]["t0_relative"] = 13_600_001
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(rows))
    out = tmp_path / "invalid-output"
    result = _run_paired(source, out)
    assert result.returncode != 0
    assert not out.exists()


@pytest.mark.parametrize("fraction", ["0", "1", "nan", "0.999"])
def test_paired_split_cannot_leave_empty_training_or_validation(tmp_path, fraction):
    out = tmp_path / "invalid-output"
    result = _run_paired(DEMO, out, "--val-fraction", fraction)
    assert result.returncode != 0
    assert not out.exists()


def test_paired_mode_rejects_conflicting_inputs_and_protects_existing_outputs(tmp_path):
    for extra in (["--pai-dir", str(tmp_path)], ["--navigation-json", str(DEMO)]):
        out = tmp_path / "invalid-output"
        result = _run_paired(DEMO, out, *extra)
        assert result.returncode != 0
        assert not out.exists()
    out = tmp_path / "splits"
    first = _run_paired(DEMO, out)
    assert first.returncode == 0, first.stderr
    original = (out / "nav_cot_train.json").read_bytes()
    rerun = _run_paired(DEMO, out, "--seed", "1")
    assert rerun.returncode != 0 and "already exists" in rerun.stderr
    assert (out / "nav_cot_train.json").read_bytes() == original


def test_legacy_cot_and_null_coc_fallback_keep_valid_events(corpus, tmp_path):
    path = corpus / "reasoning/ood_reasoning.parquet"
    frame = pd.read_parquet(path)
    frame.loc["train-clip", "events"] = json.dumps([
        {"event_start_timestamp": 2_000_000, "cot": "Legacy reasoning."},
        {"event_start_timestamp": 3_000_000, "coc": None, "cot": "Fallback reasoning."},
        {"event_start_timestamp": 4_000_000, "coc": None},
    ])
    frame.to_parquet(path)
    out = tmp_path / "legacy"
    result = _run(corpus, out)
    assert result.returncode == 0, result.stderr
    assert [row["cot"] for row in _read(out, "cot_train.json")] == ["Legacy reasoning.", "Fallback reasoning."]
    report = _read(out, "preparation_report.json")
    assert report["label_statistics"]["train"]["nominal_targets"] == 0
    assert any("rare events" in warning for warning in report["warnings"])


def test_demo_labels_remove_conflicting_row_and_absolute_capture_times():
    rows = json.loads(DEMO.read_text())
    assert len(rows) == 19 and len({row["clip_id"] for row in rows}) == 18
    assert not any(row["clip_id"] == "80a01f53-83eb-47b0-aa6c-901e9d7abf56" for row in rows)
    assert all("t0" not in row and "clip_start_timestamp" not in row for row in rows)


def test_demo_report_exposes_official_split_and_session_overlap(tmp_path):
    rows = json.loads(DEMO.read_text())
    ids = sorted({row["clip_id"] for row in rows})
    metadata = tmp_path / "clips.parquet"
    pd.DataFrame({"split": ["test"] * len(ids), "session_id": ["shared-drive"] * len(ids)}, index=ids).to_parquet(metadata)
    out = tmp_path / "audit"
    result = _run_paired(DEMO, out, "--clip-index", str(metadata))
    assert result.returncode == 0, result.stderr
    report = _read(out, "preparation_report.json")
    assert report["shared_train_val_sessions"] == ["shared-drive"]
    assert report["splits"]["train"]["official_clip_splits"] == {"test": 14}
    assert any("contaminate" in warning for warning in report["warnings"])
    assert report["splits"]["val"]["label_statistics"]["navigation_maneuvers"]


@pytest.mark.parametrize("nav", [
    "Turn left in 594m", "Turn left in 3.5m", "Take a left whenever",
    "Turn straight in 25m",
])
@pytest.mark.parametrize("mode", ["navigation", "paired"])
def test_navigation_schema_failure_precedes_output(corpus, tmp_path, nav, mode):
    rows = _nav_rows() if mode == "navigation" else json.loads(DEMO.read_text())
    rows[0]["nav_text"] = nav
    out = tmp_path / "bad-nav"
    source = _write_nav(tmp_path, rows)
    if mode == "navigation":
        result = _run(corpus, out, "--navigation-json", source)
    else:
        result = _run_paired(source, out)
    assert result.returncode != 0 and not out.exists()
    assert "unsupported navigation format" in result.stderr


@pytest.mark.parametrize("nav", ["Exit roundabout in 25m", "Exit rotary", "Take the ramp straight in 25m", "Merge straight in 25m"])
@pytest.mark.parametrize("mode", ["navigation", "paired"])
def test_release_navigation_formats_pass_preparation(corpus, tmp_path, nav, mode):
    rows = _nav_rows() if mode == "navigation" else json.loads(DEMO.read_text())
    rows[0]["nav_text"] = nav
    source = _write_nav(tmp_path, rows)
    out = tmp_path / "valid-nav"
    result = (_run(corpus, out, "--navigation-json", source) if mode == "navigation"
              else _run_paired(source, out))
    assert result.returncode == 0, result.stderr
    prepared = _read(out, "nav_cot_train.json") + _read(out, "nav_cot_val.json")
    assert any(row["nav_text"] == nav for row in prepared)
    assert _read(out, "preparation_report.json")["validate_navigation_format"] is True


@pytest.mark.parametrize("mode", ["navigation", "paired"])
def test_custom_navigation_opt_out_preserves_event_and_split_checks(corpus, tmp_path, mode):
    rows = _nav_rows() if mode == "navigation" else json.loads(DEMO.read_text())
    rows[0]["nav_text"] = "Follow the marked construction detour."
    source = _write_nav(tmp_path, rows)
    out = tmp_path / "custom-nav"
    args = ("--no-validate-navigation-format",)
    result = (_run(corpus, out, "--navigation-json", source, *args) if mode == "navigation"
              else _run_paired(source, out, *args))
    assert result.returncode == 0, result.stderr
    report = _read(out, "preparation_report.json")
    assert report["validate_navigation_format"] is False
    assert any("validation was disabled" in warning for warning in report["warnings"])
    train, val = _read(out, "nav_cot_train.json"), _read(out, "nav_cot_val.json")
    assert not {row["clip_id"] for row in train} & {row["clip_id"] for row in val}
    assert any(row["nav_text"] == rows[0]["nav_text"] for row in train + val)
    # Disabling a format validator must not bypass duplicate event rejection.
    rows.append(rows[0].copy())
    source = _write_nav(tmp_path, rows)
    bad_out = tmp_path / "duplicate-custom-nav"
    result = (_run(corpus, bad_out, "--navigation-json", source, *args) if mode == "navigation"
              else _run_paired(source, bad_out, *args))
    assert result.returncode != 0 and "duplicate" in result.stderr
    assert not bad_out.exists()
