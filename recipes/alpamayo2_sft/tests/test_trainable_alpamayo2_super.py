# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Alpamayo 2 Super training extension."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
from alpamayo2_sft.models import TrainableAlpamayo2Super
import alpamayo2_sft.models.trainable_alpamayo2_super as trainable_module


def _model_shell(enable_expert: bool = True) -> TrainableAlpamayo2Super:
    model = object.__new__(TrainableAlpamayo2Super)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        enable_expert=enable_expert,
        traj_ids={"future_start": 7},
    )
    model.vlm = torch.nn.Linear(2, 2)
    if enable_expert:
        model.expert = torch.nn.Linear(2, 2)
    return model


def test_training_model_inherits_release_model() -> None:
    assert issubclass(TrainableAlpamayo2Super, Alpamayo2Super)


def test_training_stage_selects_one_trainable_model_part() -> None:
    model = _model_shell()

    model.set_training_stage("vlm")
    assert all(parameter.requires_grad for parameter in model.vlm.parameters())
    assert not any(parameter.requires_grad for parameter in model.expert.parameters())

    model.set_training_stage("expert")
    assert not any(parameter.requires_grad for parameter in model.vlm.parameters())
    assert all(parameter.requires_grad for parameter in model.expert.parameters())


def test_stage1_vlm_checkpoint_overlays_release_vlm(tmp_path: Path) -> None:
    model = _model_shell()
    replacement_weight = torch.full_like(model.vlm.weight, 2.0)
    replacement_bias = torch.full_like(model.vlm.bias, 3.0)
    save_file(
        {
            "vlm.weight": replacement_weight,
            "vlm.bias": replacement_bias,
        },
        tmp_path / "model.safetensors",
    )

    model.load_stage1_vlm(str(tmp_path))

    torch.testing.assert_close(model.vlm.weight, replacement_weight)
    torch.testing.assert_close(model.vlm.bias, replacement_bias)


def test_stage1_vlm_checkpoint_uses_zero3_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = _model_shell()
    replacement_weight = torch.full_like(model.vlm.weight, 2.0)
    replacement_bias = torch.full_like(model.vlm.bias, 3.0)
    save_file(
        {
            "vlm.weight": replacement_weight,
            "vlm.bias": replacement_bias,
        },
        tmp_path / "model.safetensors",
    )

    def fake_zero3_load(vlm, state_dict):
        vlm.load_state_dict(state_dict, strict=True)
        return []

    monkeypatch.setattr(trainable_module, "is_deepspeed_zero3_enabled", lambda: True)
    monkeypatch.setattr(trainable_module, "_load_state_dict_into_zero3_model", fake_zero3_load)

    model.load_stage1_vlm(str(tmp_path))

    torch.testing.assert_close(model.vlm.weight, replacement_weight)
    torch.testing.assert_close(model.vlm.bias, replacement_bias)


def test_vlm_training_forward_does_not_dispatch_through_release_forward(monkeypatch) -> None:
    model = _model_shell(enable_expert=False)
    model.training_stage = "vlm"
    model.history_traj_tokenizer = None
    model.future_traj_tokenizer = None
    model.config.traj_ids = {
        "future_id0": 8,
        "future_start": 6,
        "future_end": 7,
    }
    model.config.future_vocab_size = 2
    model.config.loss_weights = {"future_traj": 1.0, "others": 1.0}

    class DummyVLM(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            del kwargs
            logits = torch.zeros((*input_ids.shape, 16), requires_grad=True)
            return SimpleNamespace(logits=logits)

    model.vlm = DummyVLM()
    monkeypatch.setattr(trainable_module, "fuse_traj_tokens", lambda *args, **kwargs: args[2])
    monkeypatch.setattr(
        Alpamayo2Super,
        "forward",
        lambda *args, **kwargs: pytest.fail("release forward should not be called"),
    )

    output = model(
        tokenized_data={"input_ids": torch.tensor([[1, 6, 8, 7]])},
        traj_data={},
        labels_mask=torch.ones((1, 4), dtype=torch.bool),
    )

    assert torch.isfinite(output.loss)
    assert output.future_traj_ce is not None
    assert output.others_ce is not None


def test_expert_crop_ends_after_aligned_future_start_tokens() -> None:
    model = _model_shell()
    input_ids = torch.tensor([[0, 0, 1, 7, 2], [0, 0, 3, 7, 4]])

    assert model._future_start_crop_length(input_ids) == 4


@pytest.mark.parametrize(
    "input_ids",
    [
        torch.tensor([[0, 1, 7], [0, 1, 2]]),
        torch.tensor([[0, 7, 2], [7, 2, 3]]),
        torch.tensor([[7, 1, 7], [0, 1, 7]]),
    ],
)
def test_expert_crop_rejects_invalid_future_start_layout(input_ids: torch.Tensor) -> None:
    model = _model_shell()

    with pytest.raises(ValueError):
        model._future_start_crop_length(input_ids)
