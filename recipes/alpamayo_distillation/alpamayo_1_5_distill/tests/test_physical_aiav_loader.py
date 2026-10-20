# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from alpamayo1_5_distill.data.physical_aiav_loader import (
    InfeasibleClipError,
    _sample_t0_from_range,
)


def test_random_t0_sampling_can_use_an_exact_grid():
    values = {_sample_t0_from_range(1_650_001, 2_049_999, bits, 100_000) for bits in range(20)}

    assert values <= {1_700_000, 1_800_000, 1_900_000, 2_000_000}
    assert all(value % 100_000 == 0 for value in values)


def test_random_t0_sampling_rejects_range_without_grid_point():
    with pytest.raises(InfeasibleClipError, match="no feasible t0"):
        _sample_t0_from_range(1_650_001, 1_699_999, 0, 100_000)
