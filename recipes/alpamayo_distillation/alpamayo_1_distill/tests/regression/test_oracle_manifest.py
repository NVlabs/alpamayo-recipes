from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


MANIFEST_PATH = Path(__file__).with_name("oracle_manifest.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _manifest() -> dict:
    with MANIFEST_PATH.open() as manifest_file:
        return json.load(manifest_file)


def test_oracle_manifest_is_internally_consistent() -> None:
    manifest = _manifest()
    training = manifest["training"]
    assert training["global_batch_size"] == (
        training["world_size"]
        * training["batch_size_per_gpu"]
        * training["gradient_accumulation_steps"]
    )

    for result in manifest["evaluation"].values():
        if not isinstance(result, dict) or "expected_samples" not in result:
            continue
        assert result["successful_samples"] + result["failed_samples"] == result[
            "expected_samples"
        ]


def test_alpamayo1_5_subtree_remains_frozen() -> None:
    """Fail if the frozen Alpamayo 1.5 package changes during this refactor."""

    if not (REPOSITORY_ROOT / "alpamayo1_5_distill").is_dir():
        pytest.skip(
            "Frozen-subtree gate applies only to the upstream refactor repository "
            "with its original alpamayo1_5_distill sibling directory."
        )

    expected_tree = _manifest()["source"]["frozen_alpamayo1_5_tree"]
    actual_tree = subprocess.run(
        ["git", "rev-parse", "HEAD:alpamayo1_5_distill"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert actual_tree == expected_tree

    worktree_diff = subprocess.run(
        ["git", "status", "--porcelain", "--", "alpamayo1_5_distill"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert worktree_diff == ""
