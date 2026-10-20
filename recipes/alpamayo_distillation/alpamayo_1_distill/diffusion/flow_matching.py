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

from typing import Literal

import torch
from alpamayo1_distill.diffusion.base import BaseDiffusion, StepFn


import math
from typing import Any, Literal, Optional

import torch
from diffusers.utils.torch_utils import randn_tensor

from alpamayo1_distill.diffusion.base import BaseDiffusion, StepFn


class FlowMatching(BaseDiffusion):
    """Flow Matching model.

    References:
    Flow Matching for Generative Modeling
        https://arxiv.org/pdf/2210.02747
    Guided Flows for Generative Modeling and Decision Making
        https://arxiv.org/pdf/2311.13443
    """

    def __init__(
        self,
        int_method: Literal["euler", "midpoint", "sde"] = "euler",
        train_timestep_sampler: Literal["uniform", "beta"] = "beta",
        num_inference_steps: int = 10,
        train_ignore_guidance_rate: float = 0.1,
        inference_guidance_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        """Initialize the FlowMatching model.

        Args:
            int_method: The integration method used in inference.
            train_timestep_sampler: How we sample timesteps during training.
                "uniform": Sample timesteps uniformly from [0, 1].
                "beta": Sample timesteps from a beta distribution.
                    (ref: pi-zero https://www.physicalintelligence.company/download/pi0.pdf)
            num_inference_steps: The number of inference steps.
            train_ignore_guidance_rate: The rate at which we ignore guidance during training.
            inference_guidance_weight: The weight of the guidance during inference.
        """
        super().__init__(*args, **kwargs)
        self.int_method = int_method
        self.train_timestep_sampler = train_timestep_sampler
        self.num_inference_steps = num_inference_steps
        self.train_ignore_guidance_rate = train_ignore_guidance_rate
        self.inference_guidance_weight = inference_guidance_weight
        if self.train_timestep_sampler == "beta":
            self.beta_dist = torch.distributions.beta.Beta(1.5, 1)
            self.beta_scale_constant = 0.999

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        guidance: Any | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        timesteps: list[float] | None = None,
        int_method: Literal["euler", "midpoint", "sde"] | None = None,
        inference_guidance_weight: float | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sample data from the model.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function.
            guidance: The guidance input.
            device: The device to use.
            return_all_steps: Whether to return all steps.
            inference_step: The number of inference steps. (override self.num_inference_steps)
            int_method: The integration method used in inference. (override self.int_method)

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        int_method = int_method or self.int_method
        inference_guidance_weight = inference_guidance_weight or self.inference_guidance_weight
        inference_step = inference_step if inference_step is not None else self.num_inference_steps
        if int_method == "euler":
            return self._euler(
                batch_size=batch_size,
                step_fn=step_fn,
                guidance=guidance,
                device=device,
                return_all_steps=return_all_steps,
                inference_step=inference_step,
                inference_guidance_weight=inference_guidance_weight,
            )
        elif int_method == "midpoint":
            return self._midpoint(
                batch_size=batch_size,
                step_fn=step_fn,
                guidance=guidance,
                device=device,
                return_all_steps=return_all_steps,
                inference_step=inference_step,
                inference_guidance_weight=inference_guidance_weight,
            )
        elif int_method == "sde":
            x, log_prob, all_steps, timesteps = self._sde(
                batch_size,
                step_fn,
                guidance,
                device,
                inference_step,
                timesteps,
                inference_guidance_weight,
                noise_level=kwargs.get("noise_level", 0.7),
                samples_list=kwargs.get("samples_list", None),
                generator=kwargs.get("generator", None),
                sde_type=kwargs.get("sde_type", "sde"),
            )
            if kwargs.get("return_info", False):
                return dict(x=x, log_prob=log_prob, all_steps=all_steps, timesteps=timesteps)
            elif return_all_steps:
                return all_steps, timesteps
            else:
                return x
        else:
            raise ValueError(f"Invalid integration method: {self.int_method}")

    def _guided_v(
        self,
        step_fn: StepFn,
        x: torch.Tensor,
        t: torch.Tensor,
        guidance: Any,
        inference_guidance_weight: float,
    ) -> torch.Tensor:
        """Guided v for flow matching.

        eq 6 in https://arxiv.org/pdf/2311.13443
        Guided Flows for Generative Modeling and Decision Making

        Args:
            step_fn: The denoising step function.
            x: The input tensor.
            t: The timestep.
            guidance: The guidance input.
            inference_guidance_weight: The weight of the guidance during inference.
        """
        guided_v = step_fn(x=x, t=t, guidance=guidance)
        unguided_v = step_fn(x=x, t=t, guidance=None)
        return (1 - inference_guidance_weight) * unguided_v + inference_guidance_weight * guided_v

    def _euler(
        self,
        batch_size: int,
        step_fn: StepFn,
        guidance: Any | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        inference_guidance_weight: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Euler integration for flow matching.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function.
            guidance: The guidance input.
            device: The device to use.
            return_all_steps: Whether to return all steps.
            inference_step: The inference step.
            inference_guidance_weight: The weight of the guidance during inference.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        x = torch.randn(batch_size, *self.x_dims, device=device)
        inference_step = inference_step or self.num_inference_steps
        timesteps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.x_dims)
        if return_all_steps:
            all_steps = [x]

        for i in range(inference_step):
            dt = timesteps[i + 1] - timesteps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = timesteps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            if self.use_classifier_free_guidance:
                v = self._guided_v(
                    step_fn=step_fn,
                    x=x,
                    t=t_start,
                    guidance=guidance,
                    inference_guidance_weight=inference_guidance_weight,
                )
            else:
                if guidance:
                    v = step_fn(x=x, t=t_start, guidance=guidance)
                else:
                    v = step_fn(x=x, t=t_start)
            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)
        if return_all_steps:
            return torch.stack(all_steps, dim=1), timesteps
        return x

    def _midpoint(
        self,
        batch_size: int,
        step_fn: StepFn,
        guidance: Any | None = None,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        inference_step: int | None = None,
        inference_guidance_weight: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Midpoint integration for flow matching.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function.
            guidance: The guidance input.
            device: The device to use.
            return_all_steps: Whether to return all steps.
            inference_step: The inference step.
            inference_guidance_weight: The weight of the guidance during inference.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                The final sampled tensor [B, *x_dims] if return_all_steps is False,
                otherwise a tuple of all sampled tensors [B, T, *x_dims] and the time steps [T].
        """
        x = torch.randn(batch_size, *self.x_dims, device=device)
        inference_step = inference_step or self.num_inference_steps
        timesteps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(self.x_dims)
        if return_all_steps:
            all_steps = [x]

        for i in range(inference_step):
            dt = (
                (timesteps[i + 1] - timesteps[i])
                .view(1, *[1] * n_dim)
                .expand(batch_size, *[1] * n_dim)
            )
            t_start = timesteps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_end = timesteps[i + 1].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            if self.use_classifier_free_guidance:
                v_t = self._guided_v(
                    step_fn=step_fn,
                    x=x,
                    t=t_start,
                    guidance=guidance,
                    inference_guidance_weight=inference_guidance_weight,
                )
            else:
                v_t = step_fn(x=x, t=t_start, guidance=guidance)

            x_mid = x + v_t * (t_end - t_start) / 2
            t_mid = t_start + (t_end - t_start) / 2
            if self.use_classifier_free_guidance:
                v_mid = self._guided_v(
                    step_fn=step_fn,
                    x=x_mid,
                    t=t_mid,
                    guidance=guidance,
                    inference_guidance_weight=inference_guidance_weight,
                )
            else:
                v_mid = step_fn(x=x_mid, t=t_mid, guidance=guidance)

            x = x + dt * v_mid
            if return_all_steps:
                all_steps.append(x)
        if return_all_steps:
            return torch.stack(all_steps, dim=1), timesteps
        return x

    def construct_training_data(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Construct the training data for the flow matching model."""
        batch_size = x.shape[0]
        if self.train_timestep_sampler == "uniform":
            t = torch.rand((batch_size,), device=x.device)
        elif self.train_timestep_sampler == "beta":
            # Dirichlet (used internally by Beta) doesn't support BFloat16;
            # rebuild with explicit float32 concentrations each time.
            t = torch.distributions.Beta(
                torch.as_tensor(self.beta_dist.concentration1, dtype=torch.float32),
                torch.as_tensor(self.beta_dist.concentration0, dtype=torch.float32),
            ).sample((batch_size,)).to(x.device)
            t = self.beta_scale_constant - t * self.beta_scale_constant
        else:
            raise ValueError(f"Invalid time sampler: {self.train_timestep_sampler}")
        while len(t.shape) < len(x.shape):
            t = t.unsqueeze(-1)

        noise = torch.randn_like(x)
        noisy_x = t * x + (1 - t) * noise
        training_data = {
            "x": x,
            "noisy_x": noisy_x,
            "timesteps": t,
            "noise": noise,
            "is_drop_guidance": None,
        }
        if self.use_classifier_free_guidance:
            is_drop_guidance = (
                torch.rand(batch_size, device=x.device) < self.train_ignore_guidance_rate
            )
            training_data["is_drop_guidance"] = is_drop_guidance
        return training_data

    def compute_loss_from_pred(
        self, training_data: dict[str, torch.Tensor], pred: torch.Tensor
    ) -> torch.Tensor:
        """Training step for the flow matching model."""
        x = training_data["x"]
        noise = training_data["noise"]
        return torch.nn.functional.mse_loss(x - noise, pred)

    def compute_loss_from_step_fn(
        self, x: torch.Tensor, step_fn: StepFn, guidance: Any | None = None
    ) -> torch.Tensor:
        """Handy function to compute the loss for the simple diffusion model
        (step_fn is simple and differentiable).

        Args:
            x: The input data.
            step_fn: The denoising step function.
            guidance: The guidance input.

        Returns:
            torch.Tensor: The training loss.
        """
        batch_size = x.shape[0]
        training_data = self.construct_training_data(x)
        noisy_x = training_data["noisy_x"]
        t = training_data["timesteps"]
        if self.use_classifier_free_guidance:
            pred_v = step_fn(
                x=noisy_x,
                t=t,
                guidance=guidance,
                is_drop_guidance=torch.rand(batch_size, device=x.device)
                < self.train_ignore_guidance_rate,
            )
        else:
            pred_v = step_fn(x=noisy_x, t=t)
        return self.compute_loss_from_pred(training_data, pred_v)

    def _sde(
        self,
        batch_size: int,
        step_fn: StepFn,
        guidance: Any | None = None,
        device: torch.device = torch.device("cpu"),
        inference_step: int | None = None,
        timesteps: list[float] | None = None,
        inference_guidance_weight: float | None = None,
        noise_level: float = 0.7,
        samples_list: list[torch.Tensor] | None = None,
        generator: Optional[torch.Generator] = None,
        sde_type: Optional[str] = "sde",
        use_classifier_free_guidance: bool = False,
    ) -> dict[str, torch.Tensor]:
        """SDE integration for flow matching.

        Args:
            batch_size: The batch size.
            step_fn: The denoising step function.
            guidance: The guidance input.
            device: The device to use.
        """

        def _guided_v(x: torch.Tensor, t: torch.Tensor, guidance: Any):
            # eq 6 in https://arxiv.org/pdf/2311.13443
            # Guided Flows for Generative Modeling and Decision Making
            guided_v = step_fn(x=x, t=t, guidance=guidance)
            unguided_v = step_fn(x=x, t=t, guidance=None)
            return (
                1 - inference_guidance_weight
            ) * unguided_v + inference_guidance_weight * guided_v

        if samples_list is not None:
            assert len(samples_list) == inference_step + 1, (
                "samples_list must have length inference_step + 1"
            )
            x = samples_list[0]
        else:
            x = torch.randn(batch_size, *self.x_dims, device=device)
        all_steps = [x]
        if timesteps is not None:
            assert len(timesteps) == inference_step + 1, (
                "timesteps must have length inference_step + 1"
            )
        else:
            timesteps = torch.linspace(0.0, 1.0, inference_step + 1).tolist()
        log_prob_list = []
        for i in range(inference_step):
            t = timesteps[i] * torch.ones([batch_size, *[1] * len(self.x_dims)], device=device)
            next_sample = samples_list[i + 1] if samples_list is not None else None
            if use_classifier_free_guidance:
                v = _guided_v(x=x, t=t, guidance=guidance)
            else:
                v = step_fn(x=x, t=t)
            xn, log_prob, next_sample_mean, std_dev_t = self._sde_step_with_logprob(
                v, timesteps[i], timesteps, x, noise_level, next_sample, generator, sde_type
            )
            if samples_list is not None:
                x = samples_list[i + 1]
            else:
                x = xn
            all_steps.append(x)
            log_prob_list.append(log_prob)
        log_prob = torch.stack(log_prob_list, dim=1).mean(dim=1)
        return (
            x,
            log_prob,
            torch.stack(all_steps, dim=1),
            torch.tensor(timesteps, device=device, dtype=torch.float32),
        )

    def _batched_sde_logprob(
        self,
        model_output: torch.FloatTensor,
        timesteps: torch.FloatTensor,
        sample: torch.FloatTensor,
        ref_model_output: torch.FloatTensor | None = None,
        noise_level: float = 0.7,
    ) -> dict[str, torch.Tensor]:
        """Compute the log probability of the sample.

        Args:
            model_output: (B, T, ...) The model output.
            ref_model_output: (B, T, ...) The reference model output.
            timesteps: (B, T+1) The timesteps.
            sample: (B, T+1, ...) The sample.
            noise_level: The noise level.
            sde_type: The SDE type.
        """
        t = timesteps[:, :-1]
        t_next = timesteps[:, 1:]
        sigma = 1 - t
        sigma_max = 1 - timesteps[0, 1].item()
        dt = t_next - t
        sqdt = torch.sqrt(dt)
        x = sample[:, :-1]
        x_next = sample[:, 1:]
        expand_shape = [x.shape[0], -1, *[1] * len(x.shape[2:])]
        std_dev_t = torch.sqrt(sigma / (1 - sigma.clip(max=sigma_max))) * noise_level
        x_next_mean = x * (1 + std_dev_t**2 / (2 * sigma) * dt).reshape(
            expand_shape
        ) + model_output * ((1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt).reshape(
            expand_shape
        )
        log_prob = (
            -((x_next.detach() - x_next_mean) ** 2)
            / (2 * ((std_dev_t * sqdt) ** 2)).reshape(expand_shape)
            - torch.log(std_dev_t * sqdt).reshape(expand_shape)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        if ref_model_output is not None:
            kl_div = (dt / 2 * (std_dev_t * t / 2 / (1 - t) + 1 / std_dev_t) ** 2).reshape(
                expand_shape
            ) * (ref_model_output - model_output) ** 2
            kl_div = kl_div.mean(dim=tuple(range(1, kl_div.ndim)))
        else:
            kl_div = None

        return log_prob, kl_div

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _sde_step_with_logprob(
        self,
        model_output: torch.FloatTensor,
        t: float,
        timesteps: list[float],
        sample: torch.FloatTensor,
        noise_level: float = 0.7,
        next_sample: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        sde_type: Optional[str] = "sde",
    ):
        """Predict the sample from the previous timestep by reversing the SDE. This function
        propagates the flow matching process from the learned model outputs (most often the
        predicted velocity).
        Adopted from https://github.com/yifan123/flow_grpo
        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned flow model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
        # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
        model_output = model_output.float()
        sample = sample.float()
        if next_sample is not None:
            prev_sample = next_sample.float()
        assert t in timesteps, f"timestep {t} not in {timesteps}"
        assert t != 1.0, "timestep 1.0 is not supported"

        step_index = timesteps.index(t)
        next_step_index = step_index + 1
        next_t = timesteps[next_step_index]
        sigma = torch.tensor(1 - t, device=model_output.device, dtype=model_output.dtype)
        sigma_next = torch.tensor(1 - next_t, device=model_output.device, dtype=model_output.dtype)
        sigma_max = 1 - timesteps[1]
        dt = next_t - t
        sqdt = math.sqrt(dt)

        if sde_type == "sde":
            std_dev_t = torch.sqrt(sigma / (1 - sigma.clip(max=sigma_max))) * noise_level

            # our sde
            next_sample_mean = (
                sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            if next_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                next_sample = next_sample_mean + std_dev_t * sqdt * variance_noise

            log_prob = (
                -((next_sample.detach() - next_sample_mean) ** 2) / (2 * ((std_dev_t * sqdt) ** 2))
                - torch.log(std_dev_t * sqdt)
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif sde_type == "cps":
            raise NotImplementedError("CPS is not implemented")
            std_dev_t = sigma_next * math.sin(noise_level * math.pi / 2)  # sigma_t in paper
            pred_original_sample = sample - sigma * model_output  # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (
                1 - sigma_next
            ) + noise_estimate * torch.sqrt(sigma_next**2 - std_dev_t**2)

            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            # remove all constants
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        return next_sample, log_prob, next_sample_mean, std_dev_t
