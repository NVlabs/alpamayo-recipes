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

"""Chat templates for the Alpamayo R1 model family.

Use :func:`get_template` to obtain the chat template for a specific
model version. Add a new version by creating a module here, defining
its template class, and registering it in ``_TEMPLATES``.

    template = get_template("r1_5")
    messages = template.build_conversation(data=..., ...)
"""

from .r1 import R1ChatTemplate
from .r1_5 import R1_5ChatTemplate

__all__ = ["R1ChatTemplate", "R1_5ChatTemplate", "get_template"]


_TEMPLATES: dict[str, type[R1ChatTemplate]] = {
    "r1": R1ChatTemplate,
    "r1_5": R1_5ChatTemplate,
    "r1.5": R1_5ChatTemplate,
}


def get_template(version: str) -> R1ChatTemplate:
    """Return a chat template instance for the given model version.

    Args:
        version: One of ``"r1"``, ``"r1_5"``, ``"r1.5"``.

    Raises:
        ValueError: If ``version`` is not registered.
    """
    try:
        cls = _TEMPLATES[version]
    except KeyError:
        raise ValueError(
            f"Unknown chat template version {version!r}. Known: {sorted(_TEMPLATES)}"
        ) from None
    return cls()
