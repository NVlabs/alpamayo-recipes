# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training extension of the released flow-matching implementation."""

from typing import Literal

import torch
from alpamayo1_5.diffusion.flow_matching import FlowMatching as UpstreamFlowMatching


class FlowMatching(UpstreamFlowMatching):
    """Reuse upstream inference while adding distillation training pairs."""

    def __init__(
        self,
        *args,
        train_timestep_sampler: Literal["uniform", "beta"] = "beta",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.train_timestep_sampler = train_timestep_sampler
        if train_timestep_sampler == "beta":
            self.beta_dist = torch.distributions.Beta(1.5, 1.0)
            self.beta_scale_constant = 0.999
        elif train_timestep_sampler != "uniform":
            raise ValueError(f"Invalid time sampler: {train_timestep_sampler}")

    def sample_training_pair(
        self,
        target: torch.Tensor,
        noise: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct a rectified-flow point and its ``data - noise`` target."""
        if noise is None:
            noise = torch.randn_like(target)
        if noise.shape != target.shape:
            raise ValueError(f"noise shape {noise.shape} does not match target {target.shape}")

        batch_size = target.shape[0]
        if timesteps is None:
            if self.train_timestep_sampler == "uniform":
                timesteps = torch.rand(batch_size, device=target.device, dtype=target.dtype)
            else:
                # Beta's Dirichlet backend does not support bfloat16.
                beta = torch.distributions.Beta(
                    torch.as_tensor(self.beta_dist.concentration1, dtype=torch.float32),
                    torch.as_tensor(self.beta_dist.concentration0, dtype=torch.float32),
                )
                timesteps = self.beta_scale_constant * (1.0 - beta.sample((batch_size,)))
        timesteps = timesteps.to(device=target.device, dtype=target.dtype)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        if timesteps.shape != (batch_size,):
            raise ValueError(f"timesteps must have shape ({batch_size},), got {timesteps.shape}")

        broadcast_t = timesteps.view(batch_size, *([1] * (target.ndim - 1)))
        noisy_action = (1.0 - broadcast_t) * noise + broadcast_t * target
        return noisy_action, timesteps, target - noise
