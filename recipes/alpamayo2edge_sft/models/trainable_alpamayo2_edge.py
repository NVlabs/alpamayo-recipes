# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training extension for the public Alpamayo 2 Edge inference model."""

from typing import Any

from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge

from alpamayo2edge_sft.models.sft import Alpamayo2EdgeSFTMixin


class TrainableAlpamayo2Edge(Alpamayo2EdgeSFTMixin, Alpamayo2Edge):
    """Train the public model with joint token and trajectory supervision."""

    trajectory_conditioning = "trajectory-only prompt through future-start; no generated CoT"

    def sample_trajectories_from_data(
        self, data: dict[str, Any], *, generate_cot: bool = False, **kwargs: Any
    ):
        """Sample from an anchored prompt or generate CoT with the native inference API."""
        if generate_cot:
            ids = data["tokenized_data"]["input_ids"]
            if (ids == self.config.traj_ids["future_start"]).any():
                raise ValueError("CoT generation requires a fresh assistant prompt, without future-start")
            return super().sample_trajectories_from_data(data=data, **kwargs)

        from alpamayo2edge_sft.models.public_inference import sample_anchored_trajectories

        return sample_anchored_trajectories(self, data, **kwargs)
