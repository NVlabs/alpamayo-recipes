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

"""Small shared helpers for distillation command-line tools."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf


def import_callable(path: str) -> Callable[..., Any]:
    """Import ``package.module:attribute`` and verify it is callable."""

    if ":" not in path:
        raise ValueError(f"expected module:callable, got {path!r}")
    module_name, attribute = path.split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"{path!r} is not callable")
    return value


def tokenizer_sha256(tokenizer: Any) -> str:
    """Hash the complete token-to-ID mapping used for direct logit KD."""
    vocabulary = tokenizer.get_vocab()
    encoded = json.dumps(vocabulary, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    result = OmegaConf.to_container(config, resolve=True)
    if not isinstance(result, dict):
        raise ValueError("top-level configuration must be a mapping")
    return result


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [to_device(item, device) for item in value]
    return value


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)
