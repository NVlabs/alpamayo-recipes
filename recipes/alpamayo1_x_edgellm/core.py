# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared recipe pieces: paths, JSON output, the model bundle, the checkpoint identity, the request
prompt, and the public clip list reader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECIPE_DIR = Path(__file__).resolve().parent

# Model bundle: the private weights and the files derived from them, one directory (README).
BUNDLE_FILES = {"config": "config.json", "drafter": "domino-drafter.safetensors", "scales": "fp8-scales.json", "vocab": "domino-vocab.json", "noise": "action-noise.bin"}
MANIFEST = "MANIFEST.json"
HASH_LIMIT = 64 << 20  # files above this are checked by size only when the bundle is opened


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def target_safetensors(directory: Path) -> Path:
    """The target checkpoint weights: exactly one ``model*.safetensors`` in the bundle."""
    files = sorted(directory.glob("model*.safetensors"))
    if len(files) != 1:
        raise ValueError(f"{directory}: expected one model*.safetensors file, found {[f.name for f in files]}")
    return files[0]


def bundle_files(directory: Path) -> list[Path]:
    """Every file the manifest covers, in manifest order."""
    paths = [directory / name for name in BUNDLE_FILES.values()] + [target_safetensors(directory)]
    paths += sorted(directory.glob("model.safetensors.index.json"))
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: bundle is missing {missing}")
    return paths


def write_manifest(directory: Path) -> Path:
    """Write ``MANIFEST.json`` with the size and SHA-256 of every bundle file."""
    directory = Path(directory)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in bundle_files(directory)}
    path = directory / MANIFEST
    write_json(path, {"schema": 1, "files": files}, sort_keys=True)
    return path


class Bundle:
    """An opened model bundle: the manifest is checked (size for every file, SHA-256 up to 64 MiB)."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path}: write it with `python -m alpamayo1_x_edgellm.bundle {directory}`")
        self.manifest_sha256 = sha256_file(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        expected = self.manifest["files"]
        files = {p.name: p for p in bundle_files(self.directory)}
        if set(files) != set(expected):
            raise ValueError(f"{self.directory}: manifest lists {sorted(expected)}, bundle holds {sorted(files)}")
        for name, item in expected.items():
            path = files[name]
            if path.stat().st_size != item["bytes"]:
                raise ValueError(f"{path}: size differs from the manifest")
            if item["bytes"] <= HASH_LIMIT and sha256_file(path) != item["sha256"]:
                raise ValueError(f"{path}: SHA-256 differs from the manifest")
        self.target = files[target_safetensors(self.directory).name]

    def path(self, key: str) -> Path:
        return self.directory / BUNDLE_FILES[key]

    def scales(self, section: str) -> dict:
        return json.loads(self.path("scales").read_text())[section]

    def vocab(self) -> list[int]:
        return json.loads(self.path("vocab").read_text())["compact_to_global"]

    def noise(self) -> bytes:
        return self.path("noise").read_bytes()


SOURCE_MODEL_TYPE = "alpamayo_reasoning_vla_expert_distill"
SOURCE_ARCHITECTURE = "ExpertModelDistill"
INFERENCE_MODEL_TYPE = "alpamayo_r1"
INFERENCE_ARCHITECTURE = "AlpamayoR1"


def write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    """Write one JSON document: parent created, trailing newline, atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=sort_keys) + "\n")
    temporary.replace(path)


def inference_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the 2B checkpoint config with its inference model identity (``alpamayo_r1``)."""
    model_type = config.get("model_type")
    architectures = config.get("architectures")
    identity = (model_type, tuple(architectures) if isinstance(architectures, list) else ())
    if identity not in {(SOURCE_MODEL_TYPE, (SOURCE_ARCHITECTURE,)), (INFERENCE_MODEL_TYPE, (INFERENCE_ARCHITECTURE,))}:
        raise ValueError("Expected the Alpamayo 2B training or inference identity; "
                         f"found model_type={model_type!r}, architectures={architectures!r}")
    return dict(config, model_type=INFERENCE_MODEL_TYPE, architectures=[INFERENCE_ARCHITECTURE])


# Request prompt and generation fields written into each request by prepare.py (the runtime decodes
# greedily and samples six action candidates regardless of these fields).
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
USER_PROMPT = (
    "output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory."
)


@dataclass(frozen=True)
class InferencePlan:
    action_batch: int = 6
    temperature: float = 0.6
    top_p: float = 0.98
    top_k: int = 1
    max_new_tokens: int = 128


PLAN = InferencePlan()


def read_clip_ids_from_parquet(parquet_path: str) -> list[str]:
    """Return unique clip IDs in first-occurrence order."""
    import pandas as pd

    df = pd.read_parquet(str(parquet_path))
    cols_lower = {c.lower(): c for c in df.columns}
    clip_ids = df[cols_lower["key"]].astype(str).tolist()
    seen: set[str] = set()
    unique: list[str] = []
    for cid in clip_ids:
        if cid not in seen:
            seen.add(cid)
            unique.append(cid)
    return unique
