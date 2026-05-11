# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Recipe-level configuration read from pyproject.toml.

Each recipe under ``recipes/`` declares its model-version-specific
settings in a ``[tool.alpamayo]`` section of its own ``pyproject.toml``:

    [tool.alpamayo]
    chat_template_version = "r1_5"

Recipe code uses :func:`get_chat_template_version` to read that value,
e.g. when constructing a :class:`QwenProcessor`::

    from alpamayo.config import get_chat_template_version
    proc = QwenProcessor(..., chat_template_version=get_chat_template_version())
"""

from __future__ import annotations

import inspect
import tomllib
from pathlib import Path


def get_chat_template_version(start: str | Path | None = None) -> str:
    """Return ``[tool.alpamayo].chat_template_version`` from the nearest pyproject.toml.

    Walks up from ``start`` (default: the caller's ``__file__``) until it
    finds a ``pyproject.toml`` declaring the field, and returns its value.

    Args:
        start: Path to start the walk from. Defaults to the caller's source file.

    Raises:
        RuntimeError: If no ``pyproject.toml`` with the field is found.
    """
    if start is None:
        start = inspect.stack()[1].filename
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent
    for d in [p, *p.parents]:
        pyproject = d / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as f:
                data = tomllib.load(f)
            version = data.get("tool", {}).get("alpamayo", {}).get("chat_template_version")
            if version is not None:
                return version
    raise RuntimeError(
        f"No pyproject.toml with [tool.alpamayo].chat_template_version found "
        f"walking up from {start}"
    )
