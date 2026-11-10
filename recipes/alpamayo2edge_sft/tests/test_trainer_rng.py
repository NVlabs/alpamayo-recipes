# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check rank flow streams and real Trainer RNG restoration on CPU."""

from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import SeedableRandomSampler
from transformers import TrainerCallback, TrainerControl, TrainerState

from alpamayo2edge_sft.trainer import (
    Alpamayo2EdgeTrainer,
    IndependentDPRNGCallback,
    TrainingArguments,
)


def flow_draw():
    """Exercise the CPU timestep and noise distributions without an Edge model."""
    time = torch.distributions.Beta(1.0, 1.5).sample((4,))
    return torch.cat((time, torch.randn(8)))


def invoke_callback(*, rank=0, enabled=True, stage="expert", step=0):
    args = SimpleNamespace(independent_dp_rng=enabled, seed=42, process_index=rank)
    state = TrainerState(global_step=step)
    control = TrainerControl()
    result = IndependentDPRNGCallback().on_train_begin(
        args, state, control, model=SimpleNamespace(training_stage=stage)
    )
    assert result is control


def test_fresh_expert_streams_match_rank_seed_and_are_distinct():
    draws = []
    for rank in (0, 1, 7):
        torch.manual_seed(42 + rank)
        expected = flow_draw()
        torch.manual_seed(999)
        torch.randn(13)  # Simulate RNG consumed during model/optimizer initialization.
        invoke_callback(rank=rank)
        actual = flow_draw()
        assert torch.equal(actual, expected)
        draws.append(actual)
    assert all(not torch.equal(draws[0], other) for other in draws[1:])


@pytest.mark.parametrize(
    "overrides",
    [{"enabled": False}, {"stage": "vlm"}, {"stage": None}, {"step": 100}],
)
def test_inactive_or_resumed_callback_preserves_existing_rng(overrides):
    torch.manual_seed(123)
    torch.randn(11)
    before = torch.get_rng_state().clone()
    invoke_callback(**overrides)
    assert torch.equal(torch.get_rng_state(), before)


def test_rank_flow_seeding_does_not_change_explicit_data_seed_order():
    orders = []
    for rank in (0, 1, 7):
        invoke_callback(rank=rank)
        orders.append(list(SeedableRandomSampler(range(32), data_seed=42)))
    assert orders[0] == orders[1] == orders[2]


class StochasticExpert(torch.nn.Module):
    """Tiny stochastic loss used only to test the actual Trainer lifecycle."""

    training_stage = "expert"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.draws = []

    def forward(self, x):
        draw = flow_draw()
        self.draws.append(draw.detach().clone())
        loss = (self.weight * x.float().mean() - draw.mean()).square()
        return {"loss": loss}


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control


def make_trainer(output, *, callbacks=None):
    args = TrainingArguments(
        output_dir=str(output),
        use_cpu=True,
        report_to=[],
        seed=42,
        data_seed=42,
        independent_dp_rng=True,
        max_steps=4,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=2,
        logging_strategy="no",
        disable_tqdm=True,
        dataloader_pin_memory=False,
    )
    trainer = Alpamayo2EdgeTrainer(
        model=StochasticExpert(),
        args=args,
        train_dataset=[{"x": torch.tensor([float(i + 1)])} for i in range(8)],
        callbacks=callbacks,
    )
    assert any(isinstance(c, IndependentDPRNGCallback) for c in trainer.callback_handler.callbacks)
    return trainer


def test_real_trainer_resume_keeps_noise_and_parameter_updates(tmp_path):
    uninterrupted = make_trainer(tmp_path / "uninterrupted")
    uninterrupted.train()

    interrupted = make_trainer(tmp_path / "resumed", callbacks=[StopAtTwo()])
    interrupted.train()
    assert interrupted.state.global_step == 2
    resumed = make_trainer(tmp_path / "resumed")
    resumed.train(resume_from_checkpoint=str(tmp_path / "resumed" / "checkpoint-2"))

    actual = interrupted.model.draws + resumed.model.draws
    assert len(actual) == len(uninterrupted.model.draws) == 4
    assert all(torch.equal(a, b) for a, b in zip(actual, uninterrupted.model.draws))
    assert resumed.state.global_step == 4
    assert torch.equal(resumed.model.weight, uninterrupted.model.weight)
