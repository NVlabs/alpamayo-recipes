# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data identity must follow contents across edits and relocations."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from alpamayo2edge_sft.evaluation_identity import comparable_protocol, validation_data_identity


def _protocol(root, *, pinned=True):
    root.mkdir()
    (root / "clip_index.parquet").write_bytes(b"default clip selection\n")
    if pinned:
        (root / "download_manifest.json").write_text(json.dumps({
            "repo_id": "example/driving", "resolved_revision": "a" * 40,
        }))
    annotations = root / "annotations.json"
    annotations.write_text('[{"clip_id": "clip-a", "cot": "Yield."}]')
    dataset = {"local_dir": str(root), "annotations_path": str(annotations),
               "chunk_ids": "1-3", "cache_dir": str(root / "cache")}
    return {"validation_dataset": dataset, "annotations_path": str(annotations),
            "validation_data_identity": validation_data_identity(dataset)}


def test_identical_pinned_inputs_can_be_relocated(tmp_path):
    first = _protocol(tmp_path / "one")
    second = _protocol(tmp_path / "two")
    original = deepcopy(first)
    assert comparable_protocol(first) == comparable_protocol(second)
    assert first == original  # normalization must not overwrite provenance


def test_annotation_edits_in_place_change_identity(tmp_path):
    first = _protocol(tmp_path / "one")
    second = deepcopy(first)
    (tmp_path / "one/annotations.json").write_text('[{"clip_id": "clip-a", "cot": "Proceed."}]')
    second["validation_data_identity"] = validation_data_identity(second["validation_dataset"])
    assert comparable_protocol(first) != comparable_protocol(second)


def test_pinned_revision_changes_are_not_path_changes(tmp_path):
    first = _protocol(tmp_path / "one")
    second = deepcopy(first)
    second["validation_data_identity"]["dataset"]["resolved_revision"] = "b" * 40
    assert comparable_protocol(first) != comparable_protocol(second)


def test_unverified_local_inputs_cannot_be_declared_identical_after_relocation(tmp_path):
    first = _protocol(tmp_path / "one", pinned=False)
    second = _protocol(tmp_path / "two", pinned=False)
    assert comparable_protocol(first) != comparable_protocol(second)


def test_legacy_protocols_retain_path_checks():
    first = {"validation_dataset": {"local_dir": "/data/one"}}
    second = {"validation_dataset": {"local_dir": "/data/two"}}
    assert comparable_protocol(first) != comparable_protocol(second)


@pytest.mark.parametrize("field,key", [
    ("annotations_path", "annotations_sha256"),
    ("clip_index_metadata", "clip_index_metadata_sha256"),
    ("split_manifest", "split_manifest_sha256"),
])
def test_selection_files_are_hashed_from_actual_bytes(tmp_path, field, key):
    path = tmp_path / "selection"
    path.write_bytes(b"actual file bytes\n")
    identity = validation_data_identity({field: str(path)})
    assert identity[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"edited bytes\n")
    assert validation_data_identity({field: str(path)}) != identity


@pytest.mark.parametrize("revision", [None, "main", "a" * 39, "g" * 40])
def test_manifest_requires_immutable_revision(tmp_path, revision):
    (tmp_path / "download_manifest.json").write_text(json.dumps({
        "repo_id": "example/driving", "resolved_revision": revision,
    }))
    with pytest.raises(ValueError, match="resolved commit SHA"):
        validation_data_identity({"local_dir": str(tmp_path)})


@pytest.mark.parametrize("digest", [None, "", "a" * 63, 42])
def test_configured_file_requires_content_hash(tmp_path, digest):
    protocol = _protocol(tmp_path / "one")
    protocol["validation_data_identity"]["annotations_sha256"] = digest
    with pytest.raises(ValueError, match="identity|SHA256"):
        comparable_protocol(protocol)


def test_missing_configured_file_is_not_silently_unidentified(tmp_path):
    with pytest.raises(FileNotFoundError):
        validation_data_identity({"annotations_path": str(tmp_path / "absent.json")})


@pytest.mark.parametrize("metadata_path", ["clip_index.parquet", "selection/clips.parquet"])
def test_pai_metadata_paths_resolve_under_dataset_root(tmp_path, monkeypatch, metadata_path):
    protocol = _protocol(tmp_path / "data")
    root = tmp_path / "data"
    selected = root / metadata_path
    selected.parent.mkdir(parents=True, exist_ok=True)
    selected.write_bytes(b"selected dataset clips\n")
    # A same-named CWD file must not be used to identify a different selection.
    cwd = tmp_path / "cwd"
    misleading = cwd / metadata_path
    misleading.parent.mkdir(parents=True)
    misleading.write_bytes(b"unrelated working directory clips\n")
    monkeypatch.chdir(cwd)
    relative = deepcopy(protocol)
    relative["validation_dataset"]["clip_index_metadata"] = metadata_path
    relative["validation_data_identity"] = validation_data_identity(relative["validation_dataset"])
    absolute = deepcopy(relative)
    absolute["validation_dataset"]["clip_index_metadata"] = str(selected)
    absolute["validation_data_identity"] = validation_data_identity(absolute["validation_dataset"])
    expected = hashlib.sha256(selected.read_bytes()).hexdigest()
    assert relative["validation_data_identity"]["clip_index_metadata_sha256"] == expected
    assert comparable_protocol(relative) == comparable_protocol(absolute)


def test_default_pai_clip_index_matches_explicit_path_and_detects_edits(tmp_path):
    implicit = _protocol(tmp_path / "data")
    selected = tmp_path / "data/clip_index.parquet"
    explicit = deepcopy(implicit)
    explicit["validation_dataset"]["clip_index_metadata"] = str(selected)
    explicit["validation_data_identity"] = validation_data_identity(explicit["validation_dataset"])
    assert implicit["validation_data_identity"]["clip_index_metadata_sha256"] == hashlib.sha256(
        selected.read_bytes()).hexdigest()
    assert comparable_protocol(implicit) == comparable_protocol(explicit)
    selected.write_bytes(b"different selected clips\n")
    changed = deepcopy(implicit)
    changed["validation_data_identity"] = validation_data_identity(changed["validation_dataset"])
    assert comparable_protocol(implicit) != comparable_protocol(changed)


@pytest.mark.parametrize("metadata_path", [None, "clip_index.parquet", "selection/clips.parquet"])
def test_tilde_dataset_root_matches_expanded_root(tmp_path, monkeypatch, metadata_path):
    protocol = _protocol(tmp_path / "data")
    config = protocol["validation_dataset"]
    if metadata_path is not None:
        selected = tmp_path / "data" / metadata_path
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(b"selected clips under expanded home\n")
        config["clip_index_metadata"] = metadata_path
    expected = validation_data_identity(config)
    # Simulate a home-relative dataset without changing the process HOME.
    expanduser = Path.expanduser
    monkeypatch.setattr(
        Path, "expanduser",
        lambda path: tmp_path / "data" if str(path) == "~/data" else expanduser(path),
    )
    config["local_dir"] = "~/data"
    actual = validation_data_identity(config)
    assert actual == expected
    assert actual["dataset"]["kind"] == "huggingface"
    assert actual["clip_index_metadata_sha256"] is not None


def test_absolute_pai_metadata_can_be_outside_dataset_root(tmp_path):
    protocol = _protocol(tmp_path / "data")
    selected = tmp_path / "external.parquet"
    selected.write_bytes(b"external clip selection\n")
    protocol["validation_dataset"]["clip_index_metadata"] = str(selected)
    identity = validation_data_identity(protocol["validation_dataset"])
    assert identity["clip_index_metadata_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()


@pytest.mark.parametrize("field,key", [
    ("annotations_path", "annotations_sha256"),
    ("split_manifest", "split_manifest_sha256"),
])
def test_nonmetadata_paths_remain_relative_to_working_directory(tmp_path, monkeypatch, field, key):
    protocol = _protocol(tmp_path / "data")
    (tmp_path / "data/selection.json").write_bytes(b"wrong dataset-relative file")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    selected = cwd / "selection.json"
    selected.write_bytes(b"working directory selection")
    monkeypatch.chdir(cwd)
    protocol["validation_dataset"][field] = "selection.json"
    identity = validation_data_identity(protocol["validation_dataset"])
    assert identity[key] == hashlib.sha256(selected.read_bytes()).hexdigest()


def test_missing_default_pai_clip_index_is_not_silently_unidentified(tmp_path):
    with pytest.raises(FileNotFoundError):
        validation_data_identity({"local_dir": str(tmp_path)})


def test_legacy_identity_without_implicit_clip_digest_remains_readable(tmp_path):
    protocol = _protocol(tmp_path / "data")
    protocol["validation_data_identity"]["clip_index_metadata_sha256"] = None
    assert comparable_protocol(protocol)["validation_data_identity"]["clip_index_metadata_sha256"] is None
