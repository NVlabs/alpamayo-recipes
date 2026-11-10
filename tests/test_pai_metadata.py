# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata edge cases without downloading data or loading a model."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alpamayo.data.pai_metadata import (
    EDGE_MIN_T0_US, events_have_reasoning, parse_chunk_ids, validate_navigation,
)


@pytest.mark.parametrize("value,expected", [("99", [99]), ("0 2 99", [0, 2, 99]), ("0-3", [0, 1, 2]), (99, [99]), ([0, 2], [0, 2])])
def test_chunk_selection_is_integer_and_not_characters(value, expected):
    assert parse_chunk_ids(value) == expected


@pytest.mark.parametrize("value", ["", "9-9", "9-1", "-1", True, [False], [1.5], []])
def test_invalid_chunk_selection_is_not_silently_empty(value):
    with pytest.raises(ValueError):
        parse_chunk_ids(value)


def test_any_usable_event_survives_empty_first_event_and_window_filter():
    events = [{"event_start_timestamp": 3_000_000, "coc": None},
              {"event_start_timestamp": 2_000_000, "coc": "", "cot": "Legacy valid."}]
    assert events_have_reasoning(json.dumps(events), EDGE_MIN_T0_US)
    assert not events_have_reasoning([{"event_start_timestamp": 1_900_000, "coc": "Wait."}], EDGE_MIN_T0_US)
    assert not events_have_reasoning([{"event_start_timestamp": 2_000_000.1, "coc": "Wait."}], EDGE_MIN_T0_US)


@pytest.mark.parametrize("text", [
    "Continue straight", "Continue straight in 25m", "Keep straight", "Turn left in 250m",
    "Turn sharp right onto Main Street in 25m", "At roundabout, take exit 2 in 0m",
    "Make a U-turn", "Keep slight right", "Continue around the roundabout and take the next exit",
    "Merge", "Merge left in 25m", "Merge slight right onto Main Street in 25m",
    "Merge straight", "Merge straight onto Main Street in 25m",
    "Take the ramp", "Take the ramp right in 25m", "Take the ramp slight left",
    "Take the ramp straight", "Take the ramp straight in 25m",
    "Take the ramp straight onto Main Street in 25m",
    "Exit roundabout", "Exit roundabout in 100m", "Exit rotary onto Main Street in 0m",
])
def test_navigation_supported_formats(text):
    assert validate_navigation(text) == text


@pytest.mark.parametrize("text", ["Turn right in 251m", "Turn right onto Main in 251m", "Turn right in -1m", "Turn left in 1.5m", "Turn left.", "Proceed when green"])
def test_navigation_rejects_unsupported_syntax_and_horizon(text):
    with pytest.raises(ValueError):
        validate_navigation(text)


@pytest.mark.parametrize("suffix", ["", " in 25m", " onto Main Street in 25m"])
def test_navigation_rejects_turn_straight(suffix):
    with pytest.raises(ValueError, match="unsupported navigation format"):
        validate_navigation(f"Turn straight{suffix}")


def test_reader_official_split_filter_has_no_feature_presence_dependency(tmp_path, monkeypatch):
    import types
    import pandas as pd
    physical = types.ModuleType("physical_ai_av")
    physical.egomotion = types.SimpleNamespace()
    physical.video = types.SimpleNamespace()
    features = types.ModuleType("physical_ai_av.dataset")
    features.Features = lambda frame: types.SimpleNamespace(features_df=frame)
    monkeypatch.setitem(sys.modules, "physical_ai_av", physical)
    monkeypatch.setitem(sys.modules, "physical_ai_av.dataset", features)
    spec = importlib.util.spec_from_file_location("_pai_split_reader", ROOT / "src/alpamayo/data/pai_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pd.DataFrame({"feature": ["egomotion"], "clip_files_in_zip": ['{}']}).to_csv(tmp_path / "features.csv", index=False)
    pd.DataFrame({"chunk": [99, 1134], "split": ["train", "val"]}, index=["train", "val"]).to_parquet(tmp_path / "clip_index.parquet")
    reader = module.PhysicalAIAVDatasetLocalInterface(tmp_path, chunk_ids="1134", split="val")
    assert reader.get_all_clip_ids() == ["val"]
    with pytest.raises(ValueError, match="No clips selected"):
        module.PhysicalAIAVDatasetLocalInterface(tmp_path, chunk_ids="99", split="val")


def test_downloader_resolves_one_revision_for_both_phases_and_records_it(tmp_path, monkeypatch):
    import types
    import argparse
    spec = importlib.util.spec_from_file_location("_pai_download", ROOT / "scripts/download_pai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(output_dir=tmp_path, revision="release-tag", profile="edge", chunk_ids=None,
                             only_reasoning_chunks=True, num_reasoning_clips=2, random_seed=7,
                             camera=["camera_front_wide_120fov"], calibration=None, labels=["egomotion"],
                             lidar=None, radar=None, reasoning=["ood_reasoning.parquet"])
    monkeypatch.setattr(module, "parse_args", lambda: args)
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = lambda: types.SimpleNamespace(dataset_info=lambda repo, revision: types.SimpleNamespace(sha="a" * 40))
    hub.snapshot_download = lambda **kwargs: calls.append(kwargs) or str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    def infer(path, **kwargs):
        assert kwargs["min_t0_us"] == EDGE_MIN_T0_US
        return [7, 99]
    monkeypatch.setattr(module, "infer_chunk_ids_from_ood_and_clip_index", infer)
    module.main()
    assert len(calls) == 2 and all(call["revision"] == "a" * 40 for call in calls)
    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert manifest["resolved_revision"] == "a" * 40 and manifest["chunk_ids"] == [7, 99]
    hub.HfApi = lambda: types.SimpleNamespace(dataset_info=lambda repo, revision: types.SimpleNamespace(sha="b" * 40))
    with pytest.raises(SystemExit, match="different recorded dataset revision"):
        module.main()
    assert len(calls) == 2


def test_downloader_and_curator_agree_on_later_valid_event():
    events = json.dumps([{"event_start_timestamp": 2_000_000, "coc": ""},
                         {"event_start_timestamp": 3_000_000, "cot": "Proceed."}])
    for name in ("download_pai", "curate_pai_samples"):
        spec = importlib.util.spec_from_file_location("_" + name, ROOT / f"scripts/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._ood_reasoning_events_nonempty(events, EDGE_MIN_T0_US)
        assert not module._ood_reasoning_events_nonempty(json.dumps([{"event_start_timestamp": 1_900_000, "coc": "Wait."}]), EDGE_MIN_T0_US)


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    """An offline downloader with real filesystem manifests and locks."""
    import argparse
    import types
    spec = importlib.util.spec_from_file_location("_pai_download_guard", ROOT / "scripts/download_pai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(
        output_dir=tmp_path, revision="release-tag", profile="edge", chunk_ids="729-744",
        only_reasoning_chunks=False, num_reasoning_clips=None, random_seed=42,
        camera=["camera_front_wide_120fov"], calibration=None, labels=["egomotion"],
        lidar=None, radar=None, reasoning=None,
    )
    monkeypatch.setattr(module, "parse_args", lambda: argparse.Namespace(**vars(args)))
    hub = types.ModuleType("huggingface_hub")
    state = types.SimpleNamespace(revision="a" * 40, calls=[])
    hub.HfApi = lambda: types.SimpleNamespace(
        dataset_info=lambda *a, **kw: types.SimpleNamespace(sha=state.revision)
    )
    hub.snapshot_download = lambda **kwargs: state.calls.append(kwargs) or str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return module, args, hub, state


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_interrupted_download_reserves_revision_and_can_resume_same_revision(
    downloader, tmp_path, monkeypatch, reasoning, error_type,
):
    module, args, hub, state = downloader
    args.only_reasoning_chunks = reasoning
    monkeypatch.setattr(module, "infer_chunk_ids_from_ood_and_clip_index", lambda *a, **kw: [7, 99])

    def interrupted(**kwargs):
        state.calls.append(kwargs)
        manifest = json.loads((tmp_path / "download_manifest.json").read_text())
        assert manifest["resolved_revision"] == state.revision
        assert manifest["status"] == "in_progress"
        assert manifest["requests"][-1]["allow_patterns"]
        (tmp_path / "partial.bin").write_bytes(b"existing partial data")
        raise error_type("simulated interrupted transfer")

    hub.snapshot_download = interrupted
    with pytest.raises(error_type):
        module.main()
    saved = json.loads((tmp_path / "download_manifest.json").read_text())
    assert saved["status"] == ("interrupted" if error_type is KeyboardInterrupt else "failed")
    state.revision = "b" * 40
    with pytest.raises(SystemExit, match="different recorded dataset revision"):
        module.main()
    assert len(state.calls) == 1
    assert (tmp_path / "partial.bin").read_bytes() == b"existing partial data"

    state.revision = "a" * 40
    hub.snapshot_download = lambda **kwargs: state.calls.append(kwargs) or str(tmp_path)
    module.main()
    resumed = json.loads((tmp_path / "download_manifest.json").read_text())
    assert resumed["status"] == "complete" and "error_type" not in resumed
    assert [row["status"] for row in resumed["requests"]] == [saved["status"], "complete"]
    assert (tmp_path / "partial.bin").read_bytes() == b"existing partial data"
    assert all(call["revision"] == "a" * 40 for call in state.calls)


def test_sequential_default_downloads_preserve_both_chunk_selections(downloader, tmp_path):
    module, args, hub, state = downloader
    module.main()
    args.chunk_ids = "1134-1145"
    module.main()
    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert [row["chunk_ids"] for row in manifest["requests"]] == [list(range(729, 744)), list(range(1134, 1145))]
    assert all(row["status"] == "complete" for row in manifest["requests"])
    assert manifest["chunk_ids"] == list(range(1134, 1145))
    assert len(state.calls) == 2


def test_populated_directory_without_manifest_is_not_relabelled(downloader, tmp_path):
    module, args, hub, state = downloader
    original = tmp_path / "clip_index.parquet"
    original.write_bytes(b"unknown existing revision")
    with pytest.raises(SystemExit, match="Use a new empty directory"):
        module.main()
    assert original.read_bytes() == b"unknown existing revision"
    assert not (tmp_path / "download_manifest.json").exists()
    assert not state.calls


def test_concurrent_downloader_cannot_change_manifest_or_download(downloader, tmp_path):
    import subprocess
    module, args, hub, state = downloader
    holder = subprocess.Popen(
        [sys.executable, "-c", (
            "import fcntl,sys; "
            "stream=open(sys.argv[1],'a'); "
            "fcntl.flock(stream,fcntl.LOCK_EX); "
            "print('locked',flush=True); sys.stdin.readline()"
        ), str(tmp_path / ".download_manifest.lock")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(SystemExit, match="Another downloader is using"):
            module.main()
        assert not (tmp_path / "download_manifest.json").exists()
        assert not state.calls
    finally:
        holder.communicate("release\n", timeout=10)
    # The lock is reusable once the owner exits; no stale lock blocks resuming.
    module.main()
    assert len(state.calls) == 1


def test_previous_success_manifest_migrates_without_inventing_history(downloader, tmp_path):
    module, args, hub, state = downloader
    (tmp_path / "download_manifest.json").write_text(json.dumps({
        "repo_id": module.DEFAULT_REPO_ID, "resolved_revision": state.revision,
        "requested_revision": "old-tag", "profile": "edge", "chunk_ids": [7],
        "allow_patterns": ["features.csv"],
    }))
    module.main()
    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert len(manifest["requests"]) == 2
    assert manifest["requests"][0] == {
        "requested_revision": "old-tag", "profile": "edge", "chunk_ids": [7],
        "allow_patterns": ["features.csv"], "status": "complete",
    }


def test_revision_mismatch_shows_both_shas_and_preserves_extension_route(downloader, tmp_path):
    module, args, hub, state = downloader
    module.main()
    recorded = state.revision
    state.revision = "b" * 40
    original = (tmp_path / "download_manifest.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        module.main()
    message = str(error.value)
    assert recorded in message and state.revision in message
    assert f"--revision {recorded}" in message
    assert "extend this directory" in message
    assert (tmp_path / "download_manifest.json").read_bytes() == original
    assert len(state.calls) == 1


@pytest.mark.parametrize("name", ["download_pai", "curate_pai_samples"])
@pytest.mark.parametrize("profile", [None, "legacy", "edge"])
def test_reasoning_cli_profiles_preserve_legacy_and_offer_edge_window(name, profile, tmp_path, monkeypatch):
    """Shared CLIs retain older recipes' window; Edge explicitly selects its history window."""
    spec = importlib.util.spec_from_file_location("_profile_" + name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [name, "--only-reasoning-chunks"]
    if name == "curate_pai_samples":
        argv += ["--num-samples", "2", "--output-path", str(tmp_path / "clips.parquet")]
    if profile is not None:
        argv += ["--profile", profile]
    monkeypatch.setattr(sys, "argv", argv)
    args = module.parse_args()
    assert args.profile == (profile or "legacy")
    minimum = module.EDGE_MIN_T0_US if args.profile == "edge" else module.LEGACY_MIN_T0_US
    assert module._ood_reasoning_events_nonempty(json.dumps([{
        "event_start_timestamp": 1_900_000, "coc": "Wait."
    }]), minimum) is (args.profile == "legacy")
    assert module._ood_reasoning_events_nonempty(json.dumps([{
        "event_start_timestamp": 2_000_000, "coc": "Wait."
    }]), minimum)
