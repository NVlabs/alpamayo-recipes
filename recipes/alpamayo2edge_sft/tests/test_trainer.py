# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for optimizer precision outside DeepSpeed."""

import pytest
import torch
from transformers import Trainer

from alpamayo2edge_sft.trainer import Alpamayo2EdgeTrainer, TrainingArguments


@pytest.mark.parametrize("multipliers", [None, {"weight": 0.1}])
def test_plain_adamw_rejects_bfloat16_master_parameters(tmp_path, multipliers):
    model = torch.nn.Linear(2, 1).bfloat16()
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[], lr_multiplier=multipliers
    )
    trainer = Alpamayo2EdgeTrainer(model=model, args=args)
    with pytest.raises(ValueError, match="model.dtype=float32"):
        trainer.create_optimizer()


@pytest.mark.parametrize("multipliers", [None, {"weight": 0.1}])
def test_plain_adamw_keeps_small_updates_and_moments_in_float32(tmp_path, multipliers):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        learning_rate=1e-4,
        weight_decay=0,
        lr_multiplier=multipliers,
        optim="adamw_torch",
    )
    trainer = Alpamayo2EdgeTrainer(model=model, args=args)
    optimizer = trainer.create_optimizer()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert torch.all(model.weight < 1)
    state = optimizer.state[model.weight]
    assert model.weight.dtype == torch.float32
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    assert trainer.create_optimizer() is optimizer


def test_evaluation_reuses_noise_and_preserves_training_rng(tmp_path, monkeypatch):
    trainer = Alpamayo2EdgeTrainer(
        model=torch.nn.Linear(2, 1),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], seed=42),
    )
    monkeypatch.setattr(Trainer, "evaluate", lambda *args, **kwargs: torch.randn(10))
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    first = trainer.evaluate()
    assert torch.equal(torch.get_rng_state(), before)
    torch.randn(7)
    second = trainer.evaluate()
    assert torch.equal(first, second)


@pytest.mark.parametrize("multipliers", [None, {"0": 0.1}])
def test_native_weight_decay_includes_bias_and_normalization(tmp_path, multipliers):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.LayerNorm(2))
    trainer = Alpamayo2EdgeTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            weight_decay=0.1,
            lr_multiplier=multipliers,
        ),
    )
    optimizer = trainer.create_optimizer()
    decayed = {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.1
        for parameter in group["params"]
    }
    assert decayed == {id(parameter) for parameter in model.parameters()}


def test_native_cosine_warmup_reaches_peak_on_index_nineteen(tmp_path):
    trainer = Alpamayo2EdgeTrainer(
        model=torch.nn.Linear(1, 1),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            learning_rate=1.0,
            warmup_steps=20,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.01},
        ),
    )
    optimizer = trainer.create_optimizer()
    scheduler = trainer.create_scheduler(200)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(200):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == 0
    assert rates[1] == pytest.approx(1 / 19)
    assert rates[19] == rates[20] == 1
    assert all(a > b for a, b in zip(rates[20:], rates[21:]))
    assert rates[-1] == pytest.approx(0.01)


class _LossComponents(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, value):
        ce = self.weight * value
        fm = self.weight.square() * value * 2
        return {"loss": ce + fm, "vlm_ce": ce.detach(), "expert_diffusion": fm.detach()}


def test_joint_losses_are_logged_separately_without_mixing_train_and_eval(tmp_path):
    model = _LossComponents()
    trainer = Alpamayo2EdgeTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    trainer.compute_loss(model, {"value": torch.tensor(2.0)}).backward()
    trainer.compute_loss(model, {"value": torch.tensor(4.0)})
    model.eval()
    trainer.compute_loss(model, {"value": torch.tensor(9.0)})
    trainer.log({"eval_loss": 27.0})
    assert trainer.state.log_history[-1]["eval_vlm_ce"] == pytest.approx(9.0)
    assert trainer.state.log_history[-1]["eval_expert_diffusion"] == pytest.approx(18.0)
    trainer.log({"loss": 9.0})
    assert trainer.state.log_history[-1]["vlm_ce"] == pytest.approx(3.0)
    assert trainer.state.log_history[-1]["expert_diffusion"] == pytest.approx(6.0)
    assert trainer._component_losses == {}
    assert model.weight.grad.item() == pytest.approx(10.0)


def test_joint_optimizer_assigns_each_parameter_to_one_module_learning_rate(tmp_path):
    model = torch.nn.Module()
    model.vlm = torch.nn.Module()
    model.vlm.model = torch.nn.Module()
    model.vlm.model.visual = torch.nn.Linear(2, 2)
    model.vlm.model.projector = torch.nn.Linear(2, 2)
    model.vlm.model.language_model = torch.nn.Embedding(2, 2)
    model.vlm.lm_head = torch.nn.Linear(2, 2, bias=False)
    model.vlm.lm_head.weight = model.vlm.model.language_model.weight
    model.expert = torch.nn.Linear(2, 2)
    trainer = Alpamayo2EdgeTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            learning_rate=1e-4, weight_decay=0.1,
            lr_multiplier={"vlm.model.visual": 0.1,
                           "vlm.model.projector": 0.1, "expert": 4.0},
        ),
    )
    groups = trainer.create_optimizer().param_groups
    assert groups[0]["lr"] == pytest.approx(trainer.args.learning_rate)
    assert any(parameter is model.vlm.model.language_model.weight for parameter in groups[0]["params"])
    assigned = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(assigned) == len(set(assigned)) == len(list(model.parameters()))
    rates = {id(parameter): group["lr"] for group in groups for parameter in group["params"]}
    for part, expected in (
        (model.vlm.model.visual, 1e-5), (model.vlm.model.projector, 1e-5),
        (model.vlm.model.language_model, 1e-4), (model.expert, 4e-4),
    ):
        for parameter in part.parameters():
            assert rates[id(parameter)] == pytest.approx(expected)


def test_eval_component_means_follow_gathered_samples_not_padded_rank_counts(tmp_path, monkeypatch):
    model = _LossComponents().eval()
    trainer = Alpamayo2EdgeTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    # Simulate two rank batches, with the final duplicated padding row removed.
    gathered = iter([torch.tensor([[2.0, 4.0], [6.0, 12.0]]),
                     torch.tensor([[10.0, 20.0]])])
    monkeypatch.setattr(trainer.accelerator, "gather_for_metrics", lambda values: next(gathered))
    trainer.compute_loss(model, {"value": torch.tensor(2.0)})
    trainer.compute_loss(model, {"value": torch.tensor(10.0)})
    trainer.log({"eval_loss": 18.0})
    logged = trainer.state.log_history[-1]
    assert logged["eval_vlm_ce"] == pytest.approx(6.0)
    assert logged["eval_expert_diffusion"] == pytest.approx(12.0)
    assert logged["eval_loss"] == pytest.approx(logged["eval_vlm_ce"] + logged["eval_expert_diffusion"])


@pytest.mark.parametrize("deepspeed_enabled", [False, True])
def test_preparation_preserves_geometric_values_without_private_deepspeed_dtype(
    tmp_path, deepspeed_enabled
):
    trainer = Alpamayo2EdgeTrainer(
        model=torch.nn.Linear(2, 1),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    trainer.is_deepspeed_enabled = deepspeed_enabled
    # No DeepSpeed plugin is installed. Preparing a batch must not access its
    # private hf_ds_config._dtype, even if Trainer considers the engine enabled.
    trajectory = torch.tensor([1.000123, -0.00023456], dtype=torch.float32)
    assert not torch.equal(trajectory.bfloat16().float(), trajectory)
    batch = {
        "tokenized_data": {"input_ids": torch.tensor([[1, 2]]),
                           "pixel_values": torch.tensor([1.234567])},
        "traj_data": {"ego_history_xyz": trajectory,
                      "ego_future_rot": torch.eye(3)},
        "labels_mask": torch.tensor([[False, True]]),
        "other": ([torch.tensor(0.123456, dtype=torch.float64)], "metadata"),
    }
    prepared = trainer._prepare_input(batch)
    torch.testing.assert_close(prepared["traj_data"]["ego_history_xyz"], trajectory, rtol=0, atol=0)
    assert prepared["traj_data"]["ego_history_xyz"].dtype == torch.float32
    assert prepared["tokenized_data"]["pixel_values"].dtype == torch.float32
    assert prepared["tokenized_data"]["input_ids"].dtype == torch.long
    assert prepared["labels_mask"].dtype == torch.bool
    assert isinstance(prepared["other"], tuple)
    assert isinstance(prepared["other"][0], list)
    assert prepared["other"][0][0].dtype == torch.float32
    assert prepared["other"][1] == "metadata"


@pytest.mark.parametrize("multipliers", [None, {"weight": 0.1}])
def test_optimizer_honors_explicit_model_argument(tmp_path, multipliers):
    original = torch.nn.Linear(2, 1)
    wrapped = torch.nn.Linear(3, 2)
    trainer = Alpamayo2EdgeTrainer(
        model=original,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[], lr_multiplier=multipliers
        ),
    )
    optimizer = trainer.create_optimizer(model=wrapped)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in wrapped.parameters()
    }


def test_fractional_warmup_scales_with_training_budget(tmp_path):
    trainer = Alpamayo2EdgeTrainer(
        model=torch.nn.Linear(1, 1),
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            learning_rate=1.0, warmup_steps=0.04,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr_rate": 0.01},
        ),
    )
    assert trainer.args.get_warmup_steps(100) == 4
    assert trainer.args.get_warmup_steps(1000) == 40
    optimizer = trainer.create_optimizer()
    scheduler = trainer.create_scheduler(100)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == 0
    assert rates[3] == rates[4] == 1
    assert rates[5] < 1
    assert rates[-1] == pytest.approx(0.01)


def test_epoch_budget_reaches_dataset_even_with_a_seedable_sampler(tmp_path):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.epoch = 0
            self.visited_epochs = []
        def __len__(self):
            return 8
        def set_epoch(self, epoch):
            self.epoch = epoch
        def __getitem__(self, index):
            self.visited_epochs.append(self.epoch)
            return {"value": torch.tensor(float(index + 1))}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def forward(self, value):
            return {"loss": (self.weight * value).square().mean()}

    dataset = Dataset()
    trainer = Alpamayo2EdgeTrainer(
        model=Model(), train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            num_train_epochs=3, max_steps=-1, per_device_train_batch_size=2,
            save_strategy="no", logging_strategy="no", disable_tqdm=True,
        ),
    )
    result = trainer.train()
    assert result.global_step == 12
    assert dataset.visited_epochs == [0] * 8 + [1] * 8 + [2] * 8
