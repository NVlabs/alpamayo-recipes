# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise frozen-VLM expert training with small differentiable CPU modules."""

from types import SimpleNamespace

import pytest
import torch
from alpamayo2edge_sft.models import TrainableAlpamayo2Edge
from safetensors.torch import save_file
from transformers.modeling_outputs import BaseModelOutputWithPast


class _Cache:
    def __init__(self) -> None:
        self.layers = [
            SimpleNamespace(
                keys=torch.ones((2, 1, 5, 2), requires_grad=True),
                values=torch.ones((2, 1, 5, 2), requires_grad=True),
            )
        ]

    def crop(self, length: int) -> None:
        for layer in self.layers:
            layer.keys = layer.keys[:, :, :length]
            layer.values = layer.values[:, :, :length]

    def get_seq_length(self) -> int:
        return self.layers[0].keys.shape[-2]


class _VLMBody(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.rope_deltas = torch.tensor([[2], [4]])
        self.cache = _Cache()

    def forward(self, input_ids, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["use_cache"]
        return BaseModelOutputWithPast(
            last_hidden_state=torch.ones((*input_ids.shape, 2)), past_key_values=self.cache
        )


class _VLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _VLMBody()

    def forward(self, **kwargs):
        raise AssertionError("Frozen expert training must bypass the VLM language-model head")


class _ActionSpace:
    def get_action_space_dims(self):
        return (2, 2)

    def traj_to_action(self, **trajectories):
        assert all(value.dtype == torch.float32 for value in trajectories.values())
        assert not torch.is_autocast_enabled("cpu")
        return trajectories["traj_future_xyz"][..., :2]


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.ones(1))


class _ActionIn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)
        self.sinus = torch.nn.ModuleList([_Encoder()])
        self.timestep_fourier_encoder = _Encoder()

    def forward(self, noisy_x, timesteps):
        return self.projection(noisy_x + timesteps)


class _Decoder(torch.nn.Module):
    def __init__(self, implementation: str) -> None:
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.projection = torch.nn.Linear(2, 2)
        self.inputs = None

    def forward(self, inputs_embeds, **kwargs):
        self.inputs = kwargs
        assert self.config.is_causal is False
        for layer in kwargs["past_key_values"].layers:
            assert not layer.keys.requires_grad
            assert not layer.values.requires_grad
        return BaseModelOutputWithPast(last_hidden_state=self.projection(inputs_embeds))


def _expert_model(implementation: str = "sdpa") -> TrainableAlpamayo2Edge:
    model = object.__new__(TrainableAlpamayo2Edge)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        enable_expert=True,
        cotrain_expert_vlm=True,
        loss_weights={"future_traj": 1.0},
        traj_ids={"future_start": 7, "history_pad": 98, "future_pad": 99},
        expert_config=SimpleNamespace(diffusion_cfg={"train_timestep_sampler": "beta"}),
    )
    model.vlm = _VLM()
    model.history_traj_tokenizer = None
    model.future_traj_tokenizer = None
    model.expert = torch.nn.Module()
    model.expert.config = SimpleNamespace(expert_non_causal_attention=True)
    model.expert.action_space = _ActionSpace()
    model.expert.action_in_proj = _ActionIn()
    model.expert.action_out_proj = torch.nn.Linear(2, 2)
    model.expert.expert = _Decoder(implementation)
    model.set_training_stage("expert")
    return model


def _batch():
    return {
        "tokenized_data": {
            "input_ids": torch.tensor([[0, 1, 2, 7, 8], [1, 2, 3, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        },
        "traj_data": {
            "ego_history_xyz": torch.zeros((2, 1, 3, 3), dtype=torch.bfloat16),
            "ego_history_rot": torch.eye(3, dtype=torch.bfloat16).expand(2, 1, 3, 3, 3),
            "ego_future_xyz": torch.ones((2, 1, 2, 3), dtype=torch.bfloat16),
            "ego_future_rot": torch.eye(3, dtype=torch.bfloat16).expand(2, 1, 2, 3, 3),
        },
    }


@pytest.mark.parametrize("implementation", ["sdpa", "flash_attention_2"])
def test_expert_gradients_use_frozen_cropped_context(implementation: str) -> None:
    torch.manual_seed(19)
    model = _expert_model(implementation)
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(**_batch())
    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (2, 2, 2)
    output.loss.backward()
    assert all(parameter.grad is None for parameter in model.vlm.parameters())
    assert all(parameter.grad is not None for parameter in model.expert.parameters())
    assert sum(parameter.grad.abs().sum() for parameter in model.expert.parameters()) > 0
    assert model.vlm.model.cache.get_seq_length() == 4
    assert not model.vlm.training
    assert model.expert.training

    inputs = model.expert.expert.inputs
    # HF5 mRoPE deltas need the cached left padding removed.
    expected_positions = torch.tensor([[5, 6], [8, 9]]).expand(3, -1, -1)
    torch.testing.assert_close(inputs["position_ids"], expected_positions)
    assert inputs["is_causal"] is False
    mask = inputs["attention_mask"]
    assert mask.tolist() == [[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]


@pytest.mark.parametrize("sampler", ["beta", "uniform"])
def test_flow_matching_interpolation_and_velocity_target(sampler: str) -> None:
    model = _expert_model()
    model.config.expert_config.diffusion_cfg["train_timestep_sampler"] = sampler
    action = torch.tensor([[[1.0, -2.0], [3.0, -4.0]]])
    training = model._flow_matching_targets(action)
    t, noise = training["timesteps"], training["noise"]
    assert t.dtype == torch.float32
    assert ((0 <= t) & (t <= 1)).all()
    torch.testing.assert_close(training["noisy_x"], t * action + (1 - t) * noise)
    velocity = action - noise
    torch.testing.assert_close(training["noisy_x"] + (1 - t) * velocity, action)


def test_beta_sampler_preserves_native_fp32_rng_and_operation_order() -> None:
    model = _expert_model()
    action = torch.arange(4000, dtype=torch.float32).reshape(1000, 2, 2) / 4000
    torch.manual_seed(37)
    beta = torch.distributions.Beta(
        torch.tensor(1.5, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)
    )
    sampled = beta.sample((1000,))
    expected_t = (0.999 - sampled * 0.999).reshape(-1, 1, 1)
    expected_noise = torch.randn_like(action)
    torch.manual_seed(37)
    result = model._flow_matching_targets(action)
    torch.testing.assert_close(result["timesteps"], expected_t, rtol=0, atol=0)
    torch.testing.assert_close(result["noise"], expected_noise, rtol=0, atol=0)


def test_stage_choice_overrides_joint_release_and_is_saved_in_config() -> None:
    model = _expert_model()
    assert model.config.training_stage == "expert"
    assert model.config.cotrain_expert_vlm is False
    assert not any(parameter.requires_grad for parameter in model.vlm.parameters())

    model.set_training_stage("vlm")
    model.train()
    assert model.config.training_stage == "vlm"
    assert model.config.cotrain_expert_vlm is False
    assert model.vlm.training
    assert not model.expert.training
    assert all(parameter.requires_grad for parameter in model.vlm.parameters())
    assert not any(parameter.requires_grad for parameter in model.expert.parameters())


def test_expert_stage_requires_trajectory_only_batches() -> None:
    model = _expert_model()
    with pytest.raises(ValueError, match="trajectory for every sample"):
        model(tokenized_data=_batch()["tokenized_data"], traj_data=[None, None])


def test_stage1_overlay_restores_a_safetensors_deduplicated_tied_weight(tmp_path) -> None:
    model = _expert_model()
    model.vlm = torch.nn.Module()
    model.vlm.embedding = torch.nn.Embedding(4, 2)
    model.vlm.lm_head = torch.nn.Linear(2, 4, bias=False)
    model.vlm.lm_head.weight = model.vlm.embedding.weight
    replacement = torch.full((4, 2), 7.0)
    save_file({"vlm.embedding.weight": replacement}, tmp_path / "model.safetensors")

    model.load_stage1_vlm(str(tmp_path))

    torch.testing.assert_close(model.vlm.embedding.weight, replacement)
    torch.testing.assert_close(model.vlm.lm_head.weight, replacement)
    assert model.vlm.embedding.weight is model.vlm.lm_head.weight
