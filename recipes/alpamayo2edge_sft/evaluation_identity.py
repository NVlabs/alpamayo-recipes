# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content identities for evaluation data; local paths remain provenance only when identified."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-fA-F]{40}\Z")
_FILE_FIELDS = {
    "annotations_path": "annotations_sha256",
    "clip_index_metadata": "clip_index_metadata_sha256",
    "split_manifest": "split_manifest_sha256",
}


def validation_data_identity(dataset_config):
    """Identify pinned PAI data and exact annotation/selection-file bytes.

    A download manifest identifies the claimed immutable upstream revision; it
    does not rehash camera archives or prove local files have not been modified.
    Without a pinned manifest the directory remains an unverified local source,
    and comparisons retain its location rather than declaring different roots
    equivalent. PAI evaluation requires an existing clip-index file, defaulting
    to local_dir/clip_index.parquet, including with a custom dataset interface.
    Missing metadata must not silently produce an unidentified evaluation.
    """
    identity = {"schema_version": 1, "dataset": {"kind": "unverified_local"}}
    root = dataset_config.get("local_dir")
    if root is not None:
        root = Path(root).expanduser()
        manifest_path = root / "download_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            repository = manifest.get("repo_id") if isinstance(manifest, dict) else None
            revision = manifest.get("resolved_revision") if isinstance(manifest, dict) else None
            if (not isinstance(repository, str) or not repository.strip()
                    or not isinstance(revision, str) or not _REVISION.fullmatch(revision)):
                raise ValueError("download_manifest.json requires repo_id and a resolved commit SHA")
            identity["dataset"] = {"kind": "huggingface", "repo_id": repository,
                                   "resolved_revision": revision.lower()}
    for field, key in _FILE_FIELDS.items():
        default = "clip_index.parquet" if field == "clip_index_metadata" and root is not None else None
        path = dataset_config.get(field, default)
        if path:
            if field == "clip_index_metadata" and root is not None:
                # The PAI loader joins metadata paths to local_dir; absolute
                # paths override that root, including the implicit default.
                path = root / path
            else:
                path = Path(path).expanduser()
        identity[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path else None
    return identity


def comparable_protocol(protocol, excluded_fields=()):
    """Remove locations only when their recorded identities can be checked.

    Legacy protocols retain their original path comparison. Identity differences
    remain in the returned object and therefore reject pairing, including files
    edited in place between evaluations.
    """
    result = deepcopy({key: value for key, value in protocol.items() if key not in excluded_fields})
    identity = result.get("validation_data_identity")
    if identity is None:
        return result
    if not isinstance(identity, dict) or type(identity.get("schema_version")) is not int or identity["schema_version"] != 1:
        raise ValueError("Unsupported validation data identity")
    data = identity.get("dataset")
    if not isinstance(data, dict) or data.get("kind") not in {"huggingface", "unverified_local"}:
        raise ValueError("Invalid validation dataset identity")
    dataset_config = result.get("validation_dataset")
    if not isinstance(dataset_config, dict):
        raise ValueError("Data identity requires a recorded validation_dataset")
    if data["kind"] == "huggingface":
        if (not isinstance(data.get("repo_id"), str) or not data["repo_id"].strip()
                or not isinstance(data.get("resolved_revision"), str)
                or not _REVISION.fullmatch(data["resolved_revision"])):
            raise ValueError("Invalid pinned dataset identity")
        dataset_config.pop("local_dir", None)
    for field, key in _FILE_FIELDS.items():
        if key not in identity:
            raise ValueError(f"Missing validation identity field {key}")
        digest = identity[key]
        has_path = bool(dataset_config.get(field)) or (field == "annotations_path" and bool(result.get(field)))
        if digest is None:
            if has_path:
                raise ValueError(f"Missing content identity for {field}")
            continue
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"Invalid SHA256 for {field}")
        dataset_config.pop(field, None)
        if field == "annotations_path":
            result.pop(field, None)
    # Cache placement is not a model input; retain all sampling/selection fields.
    dataset_config.pop("cache_dir", None)
    return result
