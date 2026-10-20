# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serializable sample keys for stateless, per-visit training randomness."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SampleVisit:
    """Identify a dataset index and its position in the planned epoch schedule.

    Repeated indices get distinct visit IDs. Replaying a planned visit after
    resume keeps its ID, so workers need no mutable RNG or visit counters.
    """

    index: int
    visit_id: int


SampleIndex = int | SampleVisit


def unpack_sample_index(key: SampleIndex) -> tuple[int, int | None]:
    """Keep ordinary integer access compatible with fixed evaluation samples."""

    if isinstance(key, SampleVisit):
        return key.index, key.visit_id
    return key, None
