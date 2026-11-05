# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed validation losses and trajectory minADE metrics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from alpamayo1_5_distill.training.factory import (
    build_online_teacher_batch,
    build_student_inference_batch,
)


VALIDATION_SEED = 42


class RankShardSampler(Sampler[int]):
    """Shard indices by rank without padding or duplicating validation samples."""

    def __init__(self, dataset_size: int, rank: int, world_size: int) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.dataset_size - self.rank + self.world_size - 1) // self.world_size)


def minade_values(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> torch.Tensor:
    """Compute one best-of-K average displacement error per batch sample."""
    pred_xy = pred_xyz[..., :2]
    gt_xy = gt_xyz[..., :2]
    if gt_xy.ndim >= 4 and gt_xy.shape[1] == 1:
        gt_xy = gt_xy[:, 0]
    batch_size, num_steps = pred_xy.shape[0], pred_xy.shape[-2]
    pred_xy = pred_xy.reshape(batch_size, -1, num_steps, 2)
    if gt_xy.shape != (batch_size, num_steps, 2):
        raise ValueError(
            f"ground truth must have shape {(batch_size, num_steps, 2)}, got {gt_xy.shape}"
        )
    distances = torch.linalg.vector_norm(pred_xy - gt_xy[:, None], dim=-1)
    return distances.mean(dim=-1).min(dim=1).values


def _snapshot_modes(*modules: torch.nn.Module) -> dict[torch.nn.Module, bool]:
    return {child: child.training for module in modules for child in module.modules()}


def _restore_modes(modes: dict[torch.nn.Module, bool]) -> None:
    for module, training in modes.items():
        module.training = training


def _sample_minade(
    model: torch.nn.Module,
    inference_batch: dict[str, Any],
    *,
    num_samples: int,
    greedy: bool,
    max_generation_length: int,
    top_p: float,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    method = getattr(model, "sample_trajectories_from_data_with_vlm_rollout", None)
    if method is None:
        raise AttributeError(
            "student model must implement sample_trajectories_from_data_with_vlm_rollout"
        )
    kwargs: dict[str, Any] = {
        "data": inference_batch,
        "num_traj_samples": num_samples,
        "max_generation_length": max_generation_length,
        # Keep fixed-length validation robust to a rare rollout that exhausts
        # its token budget before emitting <traj_future_start>.
        "allow_missing_future_start": True,
    }
    if greedy:
        kwargs.update(top_p=1.0, temperature=1.0, do_sample=False)
    else:
        kwargs.update(top_p=top_p, temperature=temperature)
    # Reset immediately before every rollout so stochastic minADE is stable
    # across validation runs and matches standalone Alpamayo 1.5 evaluation.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pred_xyz = method(**kwargs)[0]
    return minade_values(pred_xyz, inference_batch["ego_future_xyz"])


@torch.no_grad()
def run_validation(
    *,
    module: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Any,
    accelerator: Any,
    teacher_processor: Any,
    student_processor: Any,
    online_teacher_config: dict[str, Any],
    loss_temperature: float,
    compute_minade: bool = True,
    stochastic_samples: int = 6,
    top_p: float = 0.98,
    sampling_temperature: float = 0.6,
    max_generation_length: int = 256,
    inference_seed: int = VALIDATION_SEED,
) -> dict[str, float]:
    """Evaluate losses plus paired basic and navigation-conditioned trajectories."""
    if stochastic_samples <= 0:
        raise ValueError("stochastic_samples must be positive")
    student = module.model
    objective_config = getattr(getattr(module, "objective", None), "config", None)
    compute_teacher_vector = float(getattr(objective_config, "teacher_vector_weight", 0.0)) != 0.0
    modes = _snapshot_modes(module, teacher)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    module.eval()
    teacher.eval()
    loss_sums: dict[str, float] = {}
    stochastic_sum = 0.0
    greedy_sum = 0.0
    nav_subset_basic_stochastic_sum = 0.0
    nav_subset_basic_greedy_sum = 0.0
    nav_stochastic_sum = 0.0
    nav_greedy_sum = 0.0
    sample_count = 0
    nav_sample_count = 0
    try:
        for raw_samples in loader:
            batch_size = len(raw_samples)
            teacher_batch = build_online_teacher_batch(
                teacher,
                raw_samples,
                student_model=student,
                teacher_processor=teacher_processor,
                student_processor=student_processor,
                top_k=int(online_teacher_config.get("top_k", 64)),
                temperature=loss_temperature,
                max_new_tokens=int(online_teacher_config.get("max_new_tokens", 256)),
                compute_teacher_vector=compute_teacher_vector,
            )
            with accelerator.autocast():
                losses = module(teacher_batch)
            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value) * batch_size

            if compute_minade:
                basic_samples = [{**sample, "nav_enabled": False} for sample in raw_samples]
                basic_batch = build_student_inference_batch(
                    student,
                    basic_samples,
                    processor=student_processor,
                )
                with accelerator.autocast():
                    stochastic = _sample_minade(
                        student,
                        basic_batch,
                        num_samples=stochastic_samples,
                        greedy=False,
                        max_generation_length=max_generation_length,
                        top_p=top_p,
                        temperature=sampling_temperature,
                        seed=inference_seed,
                    )
                    greedy = _sample_minade(
                        student,
                        basic_batch,
                        num_samples=1,
                        greedy=True,
                        max_generation_length=max_generation_length,
                        top_p=1.0,
                        temperature=1.0,
                        seed=inference_seed,
                    )
                stochastic_sum += stochastic.double().sum().item()
                greedy_sum += greedy.double().sum().item()
                nav_indices = [
                    index
                    for index, sample in enumerate(raw_samples)
                    if bool(sample.get("nav_enabled", False))
                ]
                nav_samples = [raw_samples[index] for index in nav_indices]
                if nav_samples:
                    nav_index_tensor = torch.tensor(
                        nav_indices,
                        dtype=torch.long,
                        device=stochastic.device,
                    )
                    nav_subset_basic_stochastic_sum += (
                        stochastic[nav_index_tensor].double().sum().item()
                    )
                    nav_subset_basic_greedy_sum += greedy[nav_index_tensor].double().sum().item()
                    nav_batch = build_student_inference_batch(
                        student,
                        nav_samples,
                        processor=student_processor,
                    )
                    with accelerator.autocast():
                        nav_stochastic = _sample_minade(
                            student,
                            nav_batch,
                            num_samples=stochastic_samples,
                            greedy=False,
                            max_generation_length=max_generation_length,
                            top_p=top_p,
                            temperature=sampling_temperature,
                            seed=inference_seed,
                        )
                        nav_greedy = _sample_minade(
                            student,
                            nav_batch,
                            num_samples=1,
                            greedy=True,
                            max_generation_length=max_generation_length,
                            top_p=1.0,
                            temperature=1.0,
                            seed=inference_seed,
                        )
                    nav_stochastic_sum += nav_stochastic.double().sum().item()
                    nav_greedy_sum += nav_greedy.double().sum().item()
                    nav_sample_count += len(nav_samples)
            sample_count += batch_size

        loss_names = sorted(loss_sums)
        if dist.is_available() and dist.is_initialized():
            gathered_names: list[list[str] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_names, loss_names)
            loss_names = sorted(
                {name for rank_names in gathered_names for name in (rank_names or [])}
            )
        packed = torch.tensor(
            [loss_sums.get(name, 0.0) for name in loss_names]
            + [
                stochastic_sum,
                greedy_sum,
                nav_subset_basic_stochastic_sum,
                nav_subset_basic_greedy_sum,
                nav_stochastic_sum,
                nav_greedy_sum,
                float(sample_count),
                float(nav_sample_count),
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )
        packed = accelerator.reduce(packed, reduction="sum")
        total_count = float(packed[-2].item())
        total_nav_count = float(packed[-1].item())
        if total_count <= 0:
            raise ValueError("validation dataset contains no samples")
        metrics = {
            f"val/{name}": float(packed[index].item() / total_count)
            for index, name in enumerate(loss_names)
        }
        if compute_minade:
            metrics[f"val/minADE@{stochastic_samples}"] = float(packed[-8].item() / total_count)
            metrics["val/greedy_minADE@1"] = float(packed[-7].item() / total_count)
            metrics[f"val/basic_minADE@{stochastic_samples}"] = metrics[
                f"val/minADE@{stochastic_samples}"
            ]
            metrics["val/basic_greedy_minADE@1"] = metrics["val/greedy_minADE@1"]
            if total_nav_count > 0:
                metrics[f"val/nav_subset_basic_minADE@{stochastic_samples}"] = float(
                    packed[-6].item() / total_nav_count
                )
                metrics["val/nav_subset_basic_greedy_minADE@1"] = float(
                    packed[-5].item() / total_nav_count
                )
                metrics[f"val/nav_minADE@{stochastic_samples}"] = float(
                    packed[-4].item() / total_nav_count
                )
                metrics["val/nav_greedy_minADE@1"] = float(packed[-3].item() / total_nav_count)
                metrics["val/nav_samples"] = total_nav_count
        metrics["val/samples"] = total_count
        return metrics
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        _restore_modes(modes)
