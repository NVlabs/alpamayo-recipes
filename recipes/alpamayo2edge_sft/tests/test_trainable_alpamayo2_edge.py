# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Alpamayo 2 Edge training extension."""

from pathlib import Path
from types import SimpleNamespace

import alpamayo2edge_sft.models.sft as trainable_module
import pytest
import torch
from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge
from alpamayo2edge_sft.models import TrainableAlpamayo2Edge
from safetensors.torch import save_file


def _model_shell(enable_expert: bool = True) -> TrainableAlpamayo2Edge:
    model = object.__new__(TrainableAlpamayo2Edge)
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
    assert issubclass(TrainableAlpamayo2Edge, Alpamayo2Edge)


def test_checkpoint_save_keeps_native_keys_and_inference_architecture(monkeypatch, tmp_path):
    model = _model_shell()
    received = {}
    saved_assets = []
    model.config.save_pretrained = lambda path: saved_assets.append(("config", path))
    model.tokenizer = SimpleNamespace(
        save_pretrained=lambda path: saved_assets.append(("tokenizer", path))
    )
    monkeypatch.setattr(
        Alpamayo2Edge,
        "save_pretrained",
        lambda self, path, **kwargs: received.update(kwargs),
        raising=False,
    )
    model.save_pretrained(tmp_path)
    assert received["save_original_format"] is False
    assert model.config.architectures == ["Alpamayo2Edge"]
    assert saved_assets == [("config", tmp_path), ("tokenizer", tmp_path)]


def test_gradient_checkpointing_accepts_transformers5_trainer_arguments(monkeypatch) -> None:
    model = _model_shell()
    model.set_training_stage("vlm")
    received = {}
    monkeypatch.setattr(
        model.vlm,
        "gradient_checkpointing_enable",
        lambda **kwargs: received.update(kwargs),
        raising=False,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}, every_n_layers=2, offload=True
    )
    assert received == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "every_n_layers": 2,
        "offload": True,
    }


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
        return [], set()

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
    model.config.vlm_config = SimpleNamespace(text_config=SimpleNamespace(eos_token_id=2))
    model.config.future_vocab_size = 2
    model.config.loss_weights = {"future_traj": 1.0, "others": 1.0}

    class DummyVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBody()
            self.lm_head = torch.nn.Linear(4, 16, bias=False)

    class DummyBody(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            del kwargs
            return SimpleNamespace(last_hidden_state=torch.ones((*input_ids.shape, 4)))

    model.vlm = DummyVLM()
    monkeypatch.setattr(trainable_module, "fuse_traj_tokens", lambda *args, **kwargs: args[2])
    monkeypatch.setattr(
        Alpamayo2Edge,
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
    assert output.supervised_logits.shape == (3, 16)


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


def test_stage1_vlm_checkpoint_rejects_unloaded_zero3_parameters(monkeypatch, tmp_path):
    model = _model_shell()
    save_file(
        {"vlm.weight": model.vlm.weight.detach(), "vlm.bias": model.vlm.bias.detach()},
        tmp_path / "model.safetensors",
    )
    monkeypatch.setattr(trainable_module, "is_deepspeed_zero3_enabled", lambda: True)
    monkeypatch.setattr(
        trainable_module,
        "_load_state_dict_into_zero3_model",
        lambda *args: ([], {"weight"}),
    )
    with pytest.raises(RuntimeError, match="unloaded=.*weight"):
        model.load_stage1_vlm(str(tmp_path))


def test_complete_checkpoint_loading_returns_trainable_model(monkeypatch):
    model = _model_shell()
    model.training_stage = "vlm"
    received = {}

    def load(cls, path, **kwargs):
        received.update(kwargs)
        return model, {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }

    monkeypatch.setattr(Alpamayo2Edge, "from_pretrained", classmethod(load), raising=False)
    loaded = TrainableAlpamayo2Edge.from_pretrained("checkpoint", require_complete_weights=True)
    assert loaded is model
    assert received == {"output_loading_info": True}
    assert all(parameter.requires_grad for parameter in model.vlm.parameters())
    assert not any(parameter.requires_grad for parameter in model.expert.parameters())


@pytest.mark.parametrize(
    "problem",
    [
        {"missing_keys": ["vlm.model.language_model.layers.0.self_attn.q_proj.weight"]},
        {"unexpected_keys": ["unknown.weight"]},
        {"mismatched_keys": [("vlm.lm_head.weight", (2, 2), (3, 2))]},
        {"error_msgs": ["A checkpoint tensor could not be loaded"]},
    ],
)
def test_incomplete_checkpoint_cannot_reach_overlay_or_training(monkeypatch, problem):
    model = _model_shell()
    monkeypatch.setattr(
        model,
        "set_training_stage",
        lambda *args: pytest.fail("An incomplete model must not be handed off to training"),
    )
    monkeypatch.setattr(
        model,
        "load_stage1_vlm",
        lambda *args: pytest.fail("An overlay must not hide an incomplete release checkpoint"),
    )
    monkeypatch.setattr(
        Alpamayo2Edge,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: (model, problem)),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="Checkpoint did not load completely"):
        TrainableAlpamayo2Edge.from_pretrained(
            "checkpoint",
            require_complete_weights=True,
            stage1_vlm_checkpoint_path="stage1",
        )


def test_public_loading_keeps_default_model_return_contract(monkeypatch):
    model = _model_shell()
    model.training_stage = "vlm"
    received = {}

    def load(cls, path, **kwargs):
        received.update(kwargs)
        return model

    monkeypatch.setattr(Alpamayo2Edge, "from_pretrained", classmethod(load), raising=False)
    assert TrainableAlpamayo2Edge.from_pretrained("checkpoint") is model
    assert "output_loading_info" not in received


@pytest.mark.parametrize(
    "weights,expected",
    [({"text": 0.1}, {"future_traj": 1.0, "text": 0.1}),
     ({"future_traj": 2}, {"future_traj": 2.0, "text": 0.2}),
     ({}, {"future_traj": 1.0, "text": 0.2}),
     (None, None)],
)
def test_checkpoint_loading_resolves_partial_token_weights_and_legacy_null(
    monkeypatch, weights, expected
):
    model = _model_shell()
    model.training_stage = "joint"
    monkeypatch.setattr(
        Alpamayo2Edge, "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model), raising=False,
    )
    loaded = TrainableAlpamayo2Edge.from_pretrained(
        "checkpoint", lm_token_loss_weights=weights
    )
    assert loaded.config.lm_token_loss_weights == expected
    assert loaded.config.loss_weights == {"future_traj": 1.0, "others": 1.0}


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "0.2", True])
@pytest.mark.parametrize("field,key", [("lm_token_loss_weights", "text"), ("loss_weights", "others")])
def test_objective_weights_reject_invalid_values(monkeypatch, value, field, key):
    model = _model_shell()
    model.training_stage = "joint"
    monkeypatch.setattr(
        Alpamayo2Edge, "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model), raising=False,
    )
    with pytest.raises(ValueError, match=f"{field}.{key}.*nonnegative finite"):
        TrainableAlpamayo2Edge.from_pretrained("checkpoint", **{field: {key: value}})


def test_resolved_objective_weights_survive_config_save_and_reload(tmp_path):
    from transformers import PretrainedConfig

    model = _model_shell()
    model.config = PretrainedConfig(loss_weights={"future_traj": 2.0})
    model._validate_objective_weights()
    model.config.save_pretrained(tmp_path)
    reloaded = PretrainedConfig.from_pretrained(tmp_path)
    assert reloaded.loss_weights == {"future_traj": 2.0, "others": 1.0}
    assert reloaded.lm_token_loss_weights == {"future_traj": 1.0, "text": 0.2}


def test_loaded_objective_weights_are_preserved_without_explicit_override(monkeypatch):
    model = _model_shell()
    model.training_stage = "joint"
    model.config.loss_weights = {"future_traj": 0.3, "others": 2.0}
    model.config.lm_token_loss_weights = {"future_traj": 0.7, "text": 0.1}
    monkeypatch.setattr(
        Alpamayo2Edge, "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model), raising=False,
    )
    loaded = TrainableAlpamayo2Edge.from_pretrained("checkpoint")
    assert loaded.config.loss_weights == {"future_traj": 0.3, "others": 2.0}
    assert loaded.config.lm_token_loss_weights == {"future_traj": 0.7, "text": 0.1}


@pytest.mark.parametrize("enabled", [False, True])
def test_qk_gradient_layout_override_is_training_config_not_inference_keyword(monkeypatch, enabled):
    model = _model_shell()
    model.training_stage = "joint"
    received = {}
    def load(cls, checkpoint, **kwargs):
        received.update(kwargs)
        return model
    monkeypatch.setattr(Alpamayo2Edge, "from_pretrained", classmethod(load), raising=False)
    monkeypatch.setattr(model, "_restore_training_buffers", lambda: None)
    loaded = TrainableAlpamayo2Edge.from_pretrained("checkpoint", qk_gradient_layout=enabled)
    assert "qk_gradient_layout" not in received
    assert loaded.config.qk_gradient_layout is enabled
    assert hasattr(loaded, "_joint_sft_qk_layout") is enabled
