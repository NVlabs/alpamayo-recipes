#!/usr/bin/env python3
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

"""Shared helpers for checkpoint conversion scripts."""

from __future__ import annotations

import os
import shutil
from collections.abc import Collection
from pathlib import Path

DEFAULT_SYMLINK_EXTENSIONS = frozenset({".safetensors"})
DEFAULT_SYMLINK_NAMES = frozenset({"model.safetensors.index.json"})


def remap_target(target: str, table: dict[str, str]) -> str:
    """Remap a Hydra ``_target_`` string using the first matching prefix."""
    for old_prefix, new_prefix in table.items():
        if target.startswith(old_prefix):
            return new_prefix + target[len(old_prefix) :]
    return target


def remap_targets(obj: object, table: dict[str, str]) -> None:
    """Recursively remap all ``_target_`` values in a nested dict/list."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "_target_" and isinstance(value, str):
                obj[key] = remap_target(value, table)
            else:
                remap_targets(value, table)
    elif isinstance(obj, list):
        for item in obj:
            remap_targets(item, table)


def collect_targets(obj: object, prefix: str = "") -> dict[str, str]:
    """Recursively collect all ``_target_`` values with their JSON paths."""
    targets: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == "_target_" and isinstance(value, str):
                targets[prefix] = value
            else:
                targets.update(collect_targets(value, path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            targets.update(collect_targets(item, f"{prefix}[{i}]"))
    return targets


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Ensure *output_dir* is a usable, empty destination.

    Policy:
    * If *output_dir* does not exist, create it.
    * If it exists and is empty, reuse it.
    * If it exists and is non-empty:
        - ``overwrite=False`` (default): raise ``FileExistsError``.
        - ``overwrite=True``: remove the existing tree and recreate it.

    A non-directory at *output_dir* is always rejected.
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")

    if output_dir.exists():
        has_entries = any(output_dir.iterdir())
        if has_entries and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Pass --overwrite to replace its contents, or choose a different --output."
            )
        if has_entries:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def setup_checkpoint_output(
    input_dir: Path,
    output_dir: Path,
    *,
    symlink_extensions: Collection[str] = DEFAULT_SYMLINK_EXTENSIONS,
    symlink_names: Collection[str] = DEFAULT_SYMLINK_NAMES,
    copy_names: Collection[str] = frozenset(),
) -> list[str]:
    """Symlink checkpoint weights and copy selected auxiliary files.

    Assumes *output_dir* already exists. Returns human-readable actions for
    callers that print conversion summaries.
    """
    actions: list[str] = []
    for src in sorted(input_dir.iterdir()):
        if src.suffix in symlink_extensions or src.name in symlink_names:
            real_src = src.resolve()
            dst = output_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(real_src, dst)
            actions.append(f"symlink: {src.name} -> {real_src}")
        elif src.name in copy_names and src.is_file():
            shutil.copy2(src, output_dir / src.name)
            actions.append(f"copy: {src.name}")
    return actions