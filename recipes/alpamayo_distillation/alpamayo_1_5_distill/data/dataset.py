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

"""Collation utilities for online teacher targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _collate_tensors(values: list[torch.Tensor]) -> tuple[Any, torch.Tensor | None]:
    if not values:
        return values, None
    if any(value.ndim != values[0].ndim for value in values):
        return values, None
    if all(value.shape == values[0].shape for value in values):
        return torch.stack(values), None
    if values[0].ndim == 0:
        return torch.stack(values), None

    max_shape = tuple(max(value.shape[dim] for value in values) for dim in range(values[0].ndim))
    padded = values[0].new_zeros((len(values), *max_shape))
    mask = torch.zeros((len(values), *max_shape), dtype=torch.bool)
    for batch_index, value in enumerate(values):
        slices = (batch_index, *(slice(0, size) for size in value.shape))
        padded[slices] = value
        mask[slices] = True
    return padded, mask


def recursive_collate(batch: Sequence[Any]) -> Any:
    """Recursively collate nested records and pad variable-shaped tensors.

    A variable tensor mapping value named ``foo`` produces a padded ``foo`` and
    boolean ``foo_mask`` with the same batched shape. Equal tensors are simply
    stacked. Non-tensor variable sequences and missing mapping values remain
    Python lists so no information is discarded.
    """

    values = list(batch)
    if not values:
        return values
    first = values[0]
    if all(isinstance(value, torch.Tensor) for value in values):
        return _collate_tensors(values)[0]
    if all(isinstance(value, Mapping) for value in values):
        keys = set().union(*(value.keys() for value in values))
        result: dict[str, Any] = {}
        for key in sorted(keys):
            if not all(key in value for value in values):
                result[key] = [value.get(key) for value in values]
                continue
            children = [value[key] for value in values]
            if all(isinstance(child, torch.Tensor) for child in children):
                collated, mask = _collate_tensors(children)
                result[key] = collated
                if mask is not None:
                    mask_key = f"{key}_mask"
                    if mask_key in keys:
                        raise ValueError(f"cannot synthesize {mask_key}: key already exists")
                    result[mask_key] = mask
            else:
                result[key] = recursive_collate(children)
        return result
    if all(isinstance(value, (int, float, bool)) for value in values):
        return torch.as_tensor(values)
    if all(isinstance(value, tuple) and len(value) == len(first) for value in values):
        return tuple(recursive_collate(items) for items in zip(*values, strict=True))
    return values
