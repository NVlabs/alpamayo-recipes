# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic regressions for stable pose-target interpolation."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alpamayo2edge_sft.dataset import _interpolate_pose_samples, sample_linear_poses


def test_interpolation_preserves_float32_rounding_at_a_halfway_value():
    # The exact result is halfway between two float32 values. A weighted
    # endpoint expression rounds slightly below that tie in float64.
    upper = 3.0 + 10 * 2.0**-22
    result = _interpolate_pose_samples(
        np.array([0.0, 20.0]),
        np.array([[3.0], [upper]]),
        np.array([19.0], dtype=np.float32),
    )
    assert result.dtype == np.float64
    assert result[0, 0] == 3.0 + 19 * 2.0**-23
    assert result.astype(np.float32)[0, 0] == np.float32(upper)


def test_interpolation_keeps_sorted_samples_endpoints_and_query_order():
    result = _interpolate_pose_samples(
        np.array([2.0, 0.0, 1.0], dtype=np.float32),
        np.array([[4.0, 8.0], [0.0, 0.0], [2.0, 4.0]], dtype=np.float32),
        np.array([2.0, 0.5, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(result, [[4.0, 8.0], [1.0, 2.0], [0.0, 0.0], [2.0, 4.0]])
    assert result.dtype == np.float64


@pytest.mark.parametrize("query", [-0.01, 2.01])
def test_interpolation_rejects_both_extrapolation_directions(query):
    with pytest.raises(ValueError, match="interpolation range"):
        _interpolate_pose_samples(
            np.array([0.0, 2.0]), np.array([[0.0], [2.0]]), np.array([query])
        )


@pytest.mark.parametrize(
    "timestamps",
    [np.array([0, 1, 1]), np.array([0, 2, 1]), np.array([0.0, 1.0, 2.0])],
)
def test_pose_sampling_still_requires_increasing_integer_timestamps(timestamps):
    source = SimpleNamespace(
        timestamps=timestamps,
        values=SimpleNamespace(
            pose=SimpleNamespace(
                translation=np.zeros((3, 3)),
                rotation=Rotation.from_quat(np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))),
            )
        ),
    )
    with pytest.raises(ValueError, match="increasing integer timestamps"):
        sample_linear_poses(source, 0, np.array([0]))
