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

"""OBB collision detection via edge-point sampling."""

import torch

from alpamayo.geometry import boxes


class BoxObstacle:
    """Represents a 2D box obstacle as a list of corners and interpolated points along each edge."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        heading: torch.Tensor,
        length: torch.Tensor,
        width: torch.Tensor,
        length_offset: float | torch.Tensor = 0.5,
        width_offset: float | torch.Tensor = 0.5,
        num_pts_per_edge: int = 50,
    ) -> None:
        """Initialize a box obstacle from center, heading, and dimensions.

        Args:
            x: [...] x coordinate.
            y: [...] y coordinate.
            heading: [...] yaw angle in radians.
            length: [...] bbox length.
            width: [...] bbox width.
            length_offset: Relative center offset along length, default 0.5.
            width_offset: Relative center offset along width, default 0.5.
            num_pts_per_edge: Points to sample per edge. More points give tighter
                collision bounds but cost more memory.
        """
        self.xy = torch.stack([x, y], dim=-1)
        self.heading = heading
        self.length = length
        self.width = width
        self.length_offset = length_offset
        self.width_offset = width_offset
        self.corners = boxes.get_corners_2d(
            x, y, heading, length, width, length_offset, width_offset
        )
        self.edge_points, self.delta = self._interpolate(self.corners, num_pts_per_edge)

    @staticmethod
    def _with_corner_dim(value: float | torch.Tensor) -> float | torch.Tensor:
        """Add the corner dimension to tensor parameters for point-in-box broadcasting."""
        return value[..., None] if torch.is_tensor(value) else value

    def _interpolate(
        self, pts: torch.Tensor, num_pts_per_edge: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate points along box edges.

        Args:
            pts: [..., K, 2] corner points.
            num_pts_per_edge: Number of points per edge.

        Returns:
            Tuple of (edge_points [..., K*N, 2], max_delta [...]).
        """
        interp_pts = []
        max_dist_between_pts = torch.zeros_like(pts[..., 0, 0])
        for start_idx in range(pts.shape[-2] - 1):
            start = pts[..., start_idx, :]  # [..., 2]
            end = pts[..., start_idx + 1, :]  # [..., 2]
            vector = end - start  # [..., 2]
            distance = vector.norm(p=2, dim=-1)  # [...]
            interp_pts.append(
                start[..., None, :]
                + vector[..., None, :]
                * torch.linspace(0, 1, num_pts_per_edge, device=pts.device)[:, None]
            )
            max_dist_between_pts = torch.maximum(max_dist_between_pts, distance / num_pts_per_edge)

        return (
            torch.cat(interp_pts, dim=-2),  # [..., K*N, 2]
            max_dist_between_pts,
        )

    def closest_distance(self, other: "BoxObstacle") -> torch.Tensor:
        """Return closest edge-point-to-edge-point distance between self and other.

        Args:
            other: BoxObstacle with matching batch dims [...].

        Returns:
            [...] minimum distance between any pair of edge points.
        """
        displacements = self.edge_points[..., None, :, :] - other.edge_points[..., :, None, :]
        distances = displacements.norm(p=2, dim=-1)
        min_dists = distances.min(-1).values.min(-1).values
        return min_dists

    def in_collision(self, other: "BoxObstacle") -> torch.Tensor:
        """Conservative collision check based on closest point-to-point distance.

        Threshold is based on max(self.delta, other.delta) / 2 to account for
        sampling discretization error. Corner containment is also checked to
        catch full-overlap cases where one box lies inside another and edge
        samples are not close.

        Args:
            other: BoxObstacle with matching batch dims [...].

        Returns:
            [...] boolean tensor, True if in collision.
        """
        threshold = torch.maximum(self.delta, other.delta) / 2
        edge_collision = self.closest_distance(other) < threshold

        self_corners_in_other = boxes.point_in_box_2d(
            self.corners,
            other.xy[..., None, :],
            other.heading[..., None],
            other.length[..., None],
            other.width[..., None],
            self._with_corner_dim(other.length_offset),
            self._with_corner_dim(other.width_offset),
        ).any(dim=-1)
        other_corners_in_self = boxes.point_in_box_2d(
            other.corners,
            self.xy[..., None, :],
            self.heading[..., None],
            self.length[..., None],
            self.width[..., None],
            self._with_corner_dim(self.length_offset),
            self._with_corner_dim(self.width_offset),
        ).any(dim=-1)

        return edge_collision | self_corners_in_other | other_corners_in_self


def collision_at_ts(
    ego_xyh: torch.Tensor,
    ego_length: torch.Tensor,
    ego_width: torch.Tensor,
    obstacle_xyh: torch.Tensor,
    obstacle_length: torch.Tensor,
    obstacle_width: torch.Tensor,
    timesteps: list[int],
) -> dict[str, torch.Tensor]:
    """Compute collision metrics at multiple future timesteps.

    Args:
        ego_xyh: [B, T, 3] ego x, y, heading.
        ego_length: [B, T] ego length.
        ego_width: [B, T] ego width.
        obstacle_xyh: [B, N, T, 3] obstacle positions and headings.
        obstacle_length: [B, N, T] obstacle lengths.
        obstacle_width: [B, N, T] obstacle widths.
        timesteps: List of timestep indices to evaluate collision up until.

    Returns:
        Dict mapping ``collision/by_t={t/10:.1f}`` to [B] boolean tensors.
    """
    ego_box = BoxObstacle(
        ego_xyh[:, None, ..., 0],
        ego_xyh[:, None, ..., 1],
        ego_xyh[:, None, ..., 2],
        ego_length[:, None, ...],
        ego_width[:, None, ...],
    )

    obstacle_boxes = BoxObstacle(
        obstacle_xyh[..., 0],
        obstacle_xyh[..., 1],
        obstacle_xyh[..., 2],
        obstacle_length,
        obstacle_width,
    )

    in_coll = ego_box.in_collision(obstacle_boxes)  # [B, N, T]
    in_collision_any_obstacle = in_coll.any(dim=1)

    output: dict[str, torch.Tensor] = {}
    for t in timesteps:
        output[f"collision/by_t={t / 10:0.1f}"] = in_collision_any_obstacle[:, :t].any(dim=-1)

    return output
