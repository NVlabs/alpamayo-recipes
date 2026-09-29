# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SFT evaluation metrics."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from alpamayo2_sft.evaluate_hf import compute_min_ade, evaluate_trajectory


def test_compute_min_ade_selects_best_trajectory_sample() -> None:
    target = torch.zeros(1, 2, 3)
    predictions = torch.zeros(1, 1, 2, 2, 3)
    predictions[:, :, 0, :, 0] = 2.0
    predictions[:, :, 1, :, 0] = 0.5

    min_ade = compute_min_ade(predictions, target)

    torch.testing.assert_close(min_ade, torch.tensor([0.5]))


def test_trajectory_evaluation_runs_sampler_and_logs_min_ade(monkeypatch) -> None:
    sample_chunks = []

    class Model:
        def eval(self) -> None:
            pass

        def sample_trajectories_from_data(self, data, **kwargs):
            assert data["tokenized_data"]["input_ids"].shape == (1, 2)
            num_samples = kwargs["num_traj_samples"]
            sample_chunks.append(num_samples)
            predictions = torch.zeros(1, 1, num_samples, 2, 3)
            predictions[..., 0] = 0.5 if num_samples == 1 else 2.0
            return predictions, torch.empty(0), torch.empty(0)

    class Accelerator:
        num_processes = 1
        is_main_process = True

        def prepare_model(self, model, evaluation_mode):
            assert evaluation_mode
            return model

        def unwrap_model(self, model):
            return model

        def gather_for_metrics(self, value):
            return value

    target = torch.zeros(1, 1, 2, 3)
    batch = {
        "tokenized_data": {"input_ids": torch.ones(1, 2, dtype=torch.long)},
        "traj_data": {
            "ego_history_xyz": torch.zeros(1, 1, 1, 3),
            "ego_future_xyz": target,
        },
    }

    class Trainer:
        accelerator = Accelerator()
        model = Model()

        def get_eval_dataloader(self):
            return [batch]

        def log(self, metrics):
            self.logged_metrics = metrics

    monkeypatch.setattr(
        "alpamayo2_sft.evaluate_hf.torch.autocast",
        lambda *args, **kwargs: nullcontext(),
    )
    config = SimpleNamespace(
        evaluate=SimpleNamespace(
            max_eval_steps=-1,
            autocast_dtype="bfloat16",
            num_traj_samples=3,
            num_samples_per_forward=2,
            num_traj_sets=1,
            top_p=0.98,
            temperature=0.6,
            max_generation_length=256,
        )
    )
    trainer = Trainer()

    metrics = evaluate_trajectory(trainer, config)

    assert metrics == {"eval_min_ade": 0.5}
    assert trainer.logged_metrics == metrics
    assert sample_chunks == [2, 1]
