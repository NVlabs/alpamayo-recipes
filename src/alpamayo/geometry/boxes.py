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

"""2D bounding box geometry utilities for collision detection."""

import torch

from alpamayo_r1.geometry.rotation import rotation_matrix_torch


def get_corners_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    heading: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    length_offset: float | torch.Tensor = 0.5,
    width_offset: float | torch.Tensor = 0.5,
) -> torch.Tensor:
    """Compute 2D bounding box corners from center, heading, and dimensions.

    Args:
        x: [...] x coordinate of bbox center.
        y: [...] y coordinate of bbox center.
        heading: [...] yaw angle in radians.
        length: [...] bbox length along heading direction.
        width: [...] bbox width perpendicular to heading.
        length_offset: Relative offset of center along length, default 0.5.
        width_offset: Relative offset of center along width, default 0.5.

    Returns:
        corners: [..., 4, 2] xy coordinates of bbox corners, clockwise from
            front_left: front_left, front_right, rear_right, rear_left.
    """
    xy = torch.stack([x, y], dim=-1)  # [..., 2]
    rot_mat = rotation_matrix_torch(heading)

    rel_front_right = torch.stack(
        [(1 - length_offset) * length, -width_offset * width],
        dim=-1,
    )
    rel_front_left = torch.stack(
        [(1 - length_offset) * length, (1 - width_offset) * width],
        dim=-1,
    )
    rel_rear_right = torch.stack(
        [-length_offset * length, -width_offset * width],
        dim=-1,
    )
    rel_rear_left = torch.stack(
        [-length_offset * length, (1 - width_offset) * width],
        dim=-1,
    )

    rel_corner_coords = torch.stack(
        [
            rel_front_left,
            rel_front_right,
            rel_rear_right,
            rel_rear_left,
        ],
        dim=-2,
    )  # [..., 4, 2]

    # project according to rotation matrix
    return xy[..., None, :] + torch.einsum("...jk,...ck->...cj", rot_mat, rel_corner_coords)


def point_in_box_2d(
    point_xy: torch.Tensor,
    box_xy: torch.Tensor,
    box_heading: torch.Tensor,
    box_length: torch.Tensor,
    box_width: torch.Tensor,
    box_length_offset: float | torch.Tensor = 0.5,
    box_width_offset: float | torch.Tensor = 0.5,
) -> torch.BoolTensor:
    """Check if a batch of points is in a batch of 2D bounding boxes.

    Args:
        point_xy: [..., 2] point coordinates.
        box_xy: [..., 2] box center coordinates.
        box_heading: [...] box heading in radians.
        box_length: [...] box length.
        box_width: [...] box width.
        box_length_offset: Fractional position of bbox origin along length.
        box_width_offset: Fractional position of bbox origin along width.

    Returns:
        [...] True if point is inside the bounding box.
    """
    # translate points to bbox origin
    point_xy = point_xy - box_xy

    # rotate points to bbox frame (by negative of box angle)
    rot_mat = rotation_matrix_torch(-box_heading)  # [..., 2, 2]
    point_xy = torch.einsum("...jk,...k->...j", rot_mat, point_xy)

    # get limits from box dimensions
    x_min = -box_length * box_length_offset
    x_max = box_length * (1 - box_length_offset)
    y_min = -box_width * box_width_offset
    y_max = box_width * (1 - box_width_offset)

    # check if transformed points are within limits
    return torch.logical_and(
        torch.logical_and(point_xy[..., 0] > x_min, point_xy[..., 0] < x_max),
        torch.logical_and(point_xy[..., 1] > y_min, point_xy[..., 1] < y_max),
    )
