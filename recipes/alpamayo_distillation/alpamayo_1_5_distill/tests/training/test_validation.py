# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path

import pytest
import torch
from alpamayo1_5_distill.data.physical_aiav_dataset import _read_clip_ids
from alpamayo1_5_distill.training import validation
from alpamayo1_5_distill.training.common import load_config
from alpamayo1_5_distill.training.validation import RankShardSampler, minade_values
from torch import nn


def test_minade_values_selects_best_trajectory_per_sample():
    ground_truth = torch.zeros(2, 1, 3, 3)
    predictions = torch.zeros(2, 1, 2, 3, 3)
    predictions[0, 0, 0, :, 0] = 2.0
    predictions[0, 0, 1, :, 0] = 1.0
    predictions[1, 0, 0, :, 1] = 3.0
    predictions[1, 0, 1, :, 1] = 4.0

    assert minade_values(predictions, ground_truth).tolist() == pytest.approx([1.0, 3.0])


def test_rank_shards_cover_each_index_once_without_padding():
    shards = [list(RankShardSampler(7, rank, 3)) for rank in range(3)]

    assert shards == [[0, 3, 6], [1, 4], [2, 5]]
    assert sorted(index for shard in shards for index in shard) == list(range(7))


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/baseline.yaml",
        "configs/on_policy_nav_lingoqa.yaml",
    ],
)
def test_recipes_enable_fixed_seed_gold_644_validation(config_path):
    root = Path(__file__).parents[2]
    config = load_config(root / config_path)
    validation_config = config["validation"]
    package, relative = (
        validation_config["clip_allowlist_path"].removeprefix("pkg://").split("/", 1)
    )
    clip_ids = files(package).joinpath(relative).read_text(encoding="utf-8").splitlines()

    assert validation_config["enabled"] is True
    assert validation_config["seed"] == 42
    assert validation_config["every_steps"] == 2500
    assert validation_config["t0_us"] == 5_100_000
    assert validation_config["compute_minade"] is True
    assert validation_config["stochastic_samples"] == 6
    assert validation_config["top_p"] == pytest.approx(0.98)
    assert validation_config["temperature"] == pytest.approx(0.6)
    assert len(clip_ids) == len(set(clip_ids)) == 644


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/baseline.yaml",
        "configs/on_policy_nav_lingoqa.yaml",
        "../alpamayo1_x_distillation_expert_sft/configs/expert_sft_am15.yaml",
        "../alpamayo1_x_distillation_expert_sft/configs/expert_sft_am1.yaml",
    ],
)
def test_training_excludes_entire_gold_validation_set(config_path):
    root = Path(__file__).parents[2]
    config = load_config(root / config_path)
    gold_clip_ids = _read_clip_ids(config["validation"]["clip_allowlist_path"])
    excluded_clip_ids = _read_clip_ids(config["data"]["exclude_clip_ids_path"])

    assert len(gold_clip_ids) == len(set(gold_clip_ids)) == 644
    assert excluded_clip_ids == gold_clip_ids


def test_run_validation_restores_exact_modes_and_weights_samples(monkeypatch):
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.rollout_kwargs = []

        def sample_trajectories_from_data_with_vlm_rollout(
            self, *, data, num_traj_samples, do_sample=True, **kwargs
        ):
            self.rollout_kwargs.append(kwargs)
            batch_size = data["ego_future_xyz"].shape[0]
            prediction = torch.full(
                (batch_size, 1, num_traj_samples, 2, 3),
                2.0 if not do_sample else 1.0,
            )
            if do_sample and num_traj_samples > 1:
                prediction[:, :, -1] = 0.0
            return prediction, torch.empty(0)

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Student()

        def forward(self, batch):
            del batch
            return {"loss": self.model.anchor * 0 + 3.0}

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def reduce(tensor, reduction):
            assert reduction == "sum"
            return tensor

    module = Module()
    teacher = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
    module.train()
    module.model.eval()
    teacher.eval()
    teacher[0].train()
    initial_rng_state = torch.get_rng_state().clone()
    expected_modes = {
        child: child.training for root in (module, teacher) for child in root.modules()
    }

    monkeypatch.setattr(
        validation,
        "build_online_teacher_batch",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        validation,
        "build_student_inference_batch",
        lambda model, samples, processor: {"ego_future_xyz": torch.zeros(len(samples), 1, 2, 3)},
    )
    cpu_seeds = []
    cuda_seeds = []
    original_cuda_manual_seed_all = torch.cuda.manual_seed_all

    def record_cpu_seed(seed):
        cpu_seeds.append(seed)
        return torch.random.default_generator.manual_seed(seed)

    def record_cuda_seed(seed):
        cuda_seeds.append(seed)
        return original_cuda_manual_seed_all(seed)

    monkeypatch.setattr(torch, "manual_seed", record_cpu_seed)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", record_cuda_seed)

    metrics = validation.run_validation(
        module=module,
        teacher=teacher,
        loader=[[{"nav_enabled": True}, {}], [{}]],
        accelerator=Accelerator(),
        teacher_processor=object(),
        student_processor=object(),
        online_teacher_config={},
        loss_temperature=0.5,
    )

    assert metrics["val/loss"] == pytest.approx(3.0)
    assert metrics["val/minADE@6"] == pytest.approx(0.0)
    assert metrics["val/greedy_minADE@1"] == pytest.approx(2**0.5 * 2.0)
    assert metrics["val/basic_minADE@6"] == pytest.approx(0.0)
    assert metrics["val/basic_greedy_minADE@1"] == pytest.approx(2**0.5 * 2.0)
    assert metrics["val/nav_subset_basic_minADE@6"] == pytest.approx(0.0)
    assert metrics["val/nav_subset_basic_greedy_minADE@1"] == pytest.approx(2**0.5 * 2.0)
    assert metrics["val/nav_minADE@6"] == pytest.approx(0.0)
    assert metrics["val/nav_greedy_minADE@1"] == pytest.approx(2**0.5 * 2.0)
    assert metrics["val/nav_samples"] == 1
    assert metrics["val/samples"] == 3
    assert module.model.rollout_kwargs
    assert all(
        kwargs["allow_missing_future_start"] is True for kwargs in module.model.rollout_kwargs
    )
    assert cpu_seeds == [42] * len(module.model.rollout_kwargs)
    assert cuda_seeds == [42] * len(module.model.rollout_kwargs)
    assert torch.equal(torch.get_rng_state(), initial_rng_state)
    assert all(child.training == mode for child, mode in expected_modes.items())


def test_run_validation_can_skip_minade(monkeypatch):
    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1)

        def forward(self, batch):
            del batch
            return {"loss": self.model.weight.sum() * 0 + 2.0}

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def reduce(tensor, reduction):
            assert reduction == "sum"
            return tensor

    monkeypatch.setattr(validation, "build_online_teacher_batch", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        validation,
        "build_student_inference_batch",
        lambda *args, **kwargs: pytest.fail("minADE inference should be skipped"),
    )

    metrics = validation.run_validation(
        module=Module(),
        teacher=nn.Linear(1, 1),
        loader=[[{}]],
        accelerator=Accelerator(),
        teacher_processor=object(),
        student_processor=object(),
        online_teacher_config={},
        loss_temperature=0.5,
        compute_minade=False,
    )

    assert metrics == {"val/loss": pytest.approx(2.0), "val/samples": 1.0}


def test_run_validation_reduces_rank_with_no_local_samples(monkeypatch):
    class Student(nn.Module):
        pass

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Student()

    class Accelerator:
        device = torch.device("cpu")

        @staticmethod
        def autocast():
            return nullcontext()

        @staticmethod
        def reduce(tensor, reduction):
            assert reduction == "sum"
            # Simulate the other rank contributing one named loss and two samples.
            return tensor + torch.tensor(
                [6.0, 2.0, 4.0, 1.5, 2.5, 1.0, 2.0, 2.0, 1.0],
                dtype=tensor.dtype,
            )

    monkeypatch.setattr(validation.dist, "is_available", lambda: True)
    monkeypatch.setattr(validation.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(validation.dist, "get_world_size", lambda: 2)

    def all_gather_object(gathered, local):
        assert local == []
        gathered[:] = [[], ["loss"]]

    monkeypatch.setattr(validation.dist, "all_gather_object", all_gather_object)

    metrics = validation.run_validation(
        module=Module(),
        teacher=nn.Linear(1, 1),
        loader=[],
        accelerator=Accelerator(),
        teacher_processor=object(),
        student_processor=object(),
        online_teacher_config={},
        loss_temperature=0.5,
    )

    assert metrics["val/loss"] == pytest.approx(3.0)
    assert metrics["val/minADE@6"] == pytest.approx(1.0)
    assert metrics["val/greedy_minADE@1"] == pytest.approx(2.0)
    assert metrics["val/samples"] == 2
    assert metrics["val/nav_subset_basic_minADE@6"] == pytest.approx(1.5)
    assert metrics["val/nav_subset_basic_greedy_minADE@1"] == pytest.approx(2.5)
    assert metrics["val/nav_minADE@6"] == pytest.approx(1.0)
    assert metrics["val/nav_greedy_minADE@1"] == pytest.approx(2.0)
    assert metrics["val/nav_samples"] == 1
