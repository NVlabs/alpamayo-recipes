# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint CE/flow training with real differentiable Cosmos3-Edge KV caches on CPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from alpamayo2edge_sft.models import TrainableAlpamayo2Edge
from alpamayo2edge_sft.models.utils import future_traj_loss_mask
from transformers import Cosmos3EdgeConfig, Cosmos3EdgeForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPast


class _ActionSpace:
    def get_action_space_dims(self):
        return (2, 2)

    def traj_to_action(self, **trajectories):
        assert all(value.dtype == torch.float32 for value in trajectories.values())
        assert not torch.is_autocast_enabled("cpu")
        return trajectories["traj_future_xyz"][..., :2]


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.ones(1))


class _ActionIn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 8)
        self.sinus = torch.nn.ModuleList([_Encoder()])
        self.timestep_fourier_encoder = _Encoder()

    def forward(self, noisy_x, timesteps):
        return self.projection(noisy_x + timesteps)


class _CacheConsumingDecoder(torch.nn.Module):
    """An expert whose loss would reach the VLM if either K or V were attached."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.projection = torch.nn.Linear(8, 8)
        self.calls = []

    def forward(self, inputs_embeds, past_key_values, **kwargs):
        layers = [(layer.keys, layer.values) for layer in past_key_values.layers]
        self.calls.append({"cache": past_key_values, "layers": layers, **kwargs})
        prefix_length = layers[0][0].shape[-2]
        assert self.config.is_causal is False
        valid = kwargs["attention_mask"][:, :prefix_length].bool()[:, None, :, None]
        context = sum(
            ((keys.square() + 0.7 * values.square()) * valid).sum(dim=(1, 2, 3))
            / (valid.sum(dim=(1, 2, 3)) * keys.shape[1] * keys.shape[-1])
            for keys, values in layers
        )[:, None, None]
        hidden = self.projection(torch.tanh(inputs_embeds + context))
        # Real HF decoder forwards append action tokens. Exercise that mutation
        # so merely detaching the original cache in place cannot pass the test.
        for layer in past_key_values.layers:
            extra_shape = (*layer.keys.shape[:2], inputs_embeds.shape[1], layer.keys.shape[-1])
            layer.keys = torch.cat([layer.keys, layer.keys.new_zeros(extra_shape)], dim=-2)
            layer.values = torch.cat([layer.values, layer.values.new_zeros(extra_shape)], dim=-2)
        return BaseModelOutputWithPast(last_hidden_state=hidden)


def _joint_model(loss_weights=None, token_weights=None):
    torch.manual_seed(81)
    vlm_config = Cosmos3EdgeConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 5000000.0,
                "mrope_section": [2, 3, 3],
            },
        },
        vision_config={
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_patches": 4,
        },
        projector_hidden_size=48,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    vlm_config._attn_implementation = "sdpa"
    model = object.__new__(TrainableAlpamayo2Edge)
    torch.nn.Module.__init__(model)
    model.vlm = Cosmos3EdgeForConditionalGeneration(vlm_config)
    model.vlm.model.rope_deltas = torch.zeros((2, 1), dtype=torch.long)
    model.config = SimpleNamespace(
        enable_expert=True,
        vlm_config=vlm_config,
        traj_ids={
            "future_start": 6,
            "future_end": 7,
            "future_id0": 8,
            "history_pad": 25,
            "future_pad": 26,
        },
        future_vocab_size=2,
        loss_weights=dict(loss_weights or {}),
        expert_config=SimpleNamespace(diffusion_cfg={"train_timestep_sampler": "beta"}),
    )
    if token_weights is not None:
        model.config.lm_token_loss_weights = dict(token_weights)
    model.history_traj_tokenizer = model.future_traj_tokenizer = None
    model.expert = torch.nn.Module()
    model.expert.config = SimpleNamespace(expert_non_causal_attention=True)
    model.expert.action_space = _ActionSpace()
    model.expert.action_in_proj = _ActionIn()
    model.expert.expert = _CacheConsumingDecoder()
    model.expert.action_out_proj = torch.nn.Linear(8, 2)
    model.set_training_stage("joint")
    model.train()
    model.vlm_calls = []
    model.expert_predictions = []

    def capture_cache(module, args, kwargs, output):
        cache = output.past_key_values
        model.vlm_calls.append({
            "cache": cache,
            "use_cache": kwargs["use_cache"],
            "layers": [(layer.keys, layer.values) for layer in cache.layers],
        })

    model.vlm.model.register_forward_hook(capture_cache, with_kwargs=True)
    model.expert.action_out_proj.register_forward_hook(
        lambda module, args, output: model.expert_predictions.append(output)
    )
    # Fixed flow samples make branch-gradient comparisons exact and independent
    # of model construction, dropout, and the beta sampler's RNG consumption.
    def targets(action):
        noise = torch.full_like(action, -0.5)
        timesteps = action.new_full((action.shape[0], 1, 1), 0.25)
        return {
            "x": action,
            "noise": noise,
            "timesteps": timesteps,
            "noisy_x": timesteps * action + (1 - timesteps) * noise,
        }

    model._flow_matching_targets = targets
    return model


def _batch():
    ids = torch.tensor([[0, 3, 4, 6, 8, 7, 2], [3, 4, 5, 6, 9, 7, 2]])
    attention_mask = ids.ne(0).long()
    positions = attention_mask.cumsum(-1).sub(1).masked_fill(attention_mask == 0, 0)
    return {
        "tokenized_data": {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "position_ids": positions.unsqueeze(0).expand(3, -1, -1),
        },
        "labels_mask": torch.tensor([[False, False, True, True, True, True, True]] * 2),
        "traj_data": {
            "ego_history_xyz": torch.zeros((2, 1, 3, 3)),
            "ego_history_rot": torch.eye(3).expand(2, 1, 3, 3, 3),
            "ego_future_xyz": torch.tensor([
                [[[0.2, 0.4, 0.0], [0.6, 0.8, 0.0]]],
                [[[0.3, 0.5, 0.0], [0.7, 0.9, 0.0]]],
            ]),
            "ego_future_rot": torch.eye(3).expand(2, 1, 2, 3, 3),
        },
    }


def test_joint_stage_enables_both_modules_and_persists_detached_cache_contract():
    model = _joint_model()
    assert model.config.training_stage == "joint"
    assert model.config.cotrain_expert_vlm is True
    assert model.config.cotrain_detach_expert_kv is True
    assert all(parameter.requires_grad for parameter in model.vlm.parameters())
    assert all(parameter.requires_grad for parameter in model.expert.parameters())
    assert model.vlm.training and model.expert.training
    model.eval()
    assert not model.vlm.training and not model.expert.training
    model.train()
    assert model.vlm.training and model.expert.training


@pytest.mark.parametrize(
    "weights,token_weights",
    [({}, None), ({"future_traj": 2.5, "others": 1.7}, {"future_traj": 1.0, "text": 0.2})],
)
def test_joint_loss_is_global_token_mean_plus_separately_weighted_flow(weights, token_weights):
    model = _joint_model(weights, token_weights)
    batch = _batch()
    output = model(**batch)
    labels = batch["tokenized_data"]["input_ids"].masked_fill(~batch["labels_mask"], -100)
    selected = labels[:, 1:].ne(-100)
    trajectory = future_traj_loss_mask(labels, model.config.traj_ids, 2, 2)[:, 1:][selected]
    token_losses = F.cross_entropy(
        output.supervised_logits, labels[:, 1:][selected], reduction="none"
    )
    token_weights = token_weights or {}
    per_token_weights = torch.where(
        trajectory, token_weights.get("future_traj", 1.0), token_weights.get("text", 1.0)
    )
    expected_ce = (token_losses * per_token_weights).mean() * weights.get("others", 1.0)
    actions = batch["traj_data"]["ego_future_xyz"][..., :2].reshape(2, 2, 2)
    expected_flow = F.mse_loss(model.expert_predictions[0].float(), actions + 0.5)
    expected_flow *= weights.get("future_traj", 1.0)
    torch.testing.assert_close(output.vlm_ce, expected_ce)
    torch.testing.assert_close(output.expert_diffusion, expected_flow)
    torch.testing.assert_close(output.loss, expected_ce + expected_flow)
    torch.testing.assert_close(
        output.future_traj_ce, token_losses[trajectory].mean() * weights.get("others", 1.0)
    )
    torch.testing.assert_close(
        output.others_ce, token_losses[~trajectory].mean() * weights.get("others", 1.0)
    )
    assert output.vlm_ce.grad_fn is None
    assert output.expert_diffusion.grad_fn is None
    assert output.supervised_logits.shape == (10, 32)
    assert len(model.vlm_calls) == 1 and model.vlm_calls[0]["use_cache"] is True


def test_joint_flow_does_not_change_vlm_gradients_and_both_modules_update():
    joint = _joint_model({"future_traj": 2.3})
    ce_only = _joint_model({"future_traj": 0.0})
    joint(**_batch()).loss.backward()
    ce_only(**_batch()).loss.backward()
    reference = dict(ce_only.vlm.named_parameters())
    for name, parameter in joint.vlm.named_parameters():
        other = reference[name]
        if other.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-6, atol=1e-7)
    assert joint.vlm.lm_head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in joint.expert.parameters())
    assert all(parameter.grad.abs().sum() > 0 for parameter in joint.expert.parameters())
    old_vlm = joint.vlm.lm_head.weight.detach().clone()
    old_expert = joint.expert.action_out_proj.weight.detach().clone()
    torch.optim.SGD(joint.parameters(), lr=0.01).step()
    assert not torch.equal(joint.vlm.lm_head.weight, old_vlm)
    assert not torch.equal(joint.expert.action_out_proj.weight, old_expert)


def test_expert_only_joint_loss_cannot_backpropagate_through_either_cache_tensor():
    model = _joint_model({"others": 0.0})
    output = model(**_batch())
    output.loss.backward()
    assert output.vlm_ce.item() == 0
    assert output.expert_diffusion.item() > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.vlm.parameters()
    )
    assert all(parameter.grad.abs().sum() > 0 for parameter in model.expert.parameters())
    assert all(
        keys.requires_grad and values.requires_grad
        for keys, values in model.vlm_calls[0]["layers"]
    )


def test_joint_expert_receives_detached_prefix_copy_and_preserves_live_vlm_cache():
    model = _joint_model()
    model(**_batch()).loss.backward()
    original = model.vlm_calls[0]
    expert = model.expert.expert.calls[0]
    assert expert["cache"] is not original["cache"]
    assert original["cache"].get_seq_length() == 7
    for live, (keys, values), (expert_keys, expert_values) in zip(
        original["cache"].layers, original["layers"], expert["layers"], strict=True
    ):
        assert live.keys is keys and live.values is values
        assert keys.requires_grad and values.requires_grad
        assert not expert_keys.requires_grad and not expert_values.requires_grad
        assert expert_keys.shape[-2] == expert_values.shape[-2] == 4
        torch.testing.assert_close(expert_keys, keys[:, :, :4])
        torch.testing.assert_close(expert_values, values[:, :, :4])
    assert expert["is_causal"] is False
    mask = expert["attention_mask"]
    assert mask.tolist() == [[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    expected_positions = torch.tensor([[3, 4], [4, 5]]).expand(3, -1, -1)
    torch.testing.assert_close(expert["position_ids"], expected_positions)


def test_joint_expert_does_not_see_ground_truth_tokens_after_future_start():
    model = _joint_model()
    baseline = model(**_batch())
    changed = _batch()
    changed["tokenized_data"]["input_ids"][:, 4:] = torch.tensor([[9, 12, 13], [8, 14, 15]])
    actual = model(**changed)
    torch.testing.assert_close(actual.expert_diffusion, baseline.expert_diffusion, rtol=0, atol=0)
    for first, second in zip(
        model.expert.expert.calls[0]["layers"], model.expert.expert.calls[1]["layers"], strict=True
    ):
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


def test_joint_all_ignored_labels_keeps_finite_flow_loss_and_zero_vlm_gradients():
    model = _joint_model()
    batch = _batch()
    batch["labels_mask"].zero_()
    output = model(**batch)
    assert output.vlm_ce.item() == 0
    assert torch.isfinite(output.loss)
    torch.testing.assert_close(output.loss, output.expert_diffusion)
    output.loss.backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.vlm.parameters()
    )
    assert model.expert.action_out_proj.weight.grad.abs().sum() > 0


def test_joint_variable_future_start_columns_mask_each_rows_ground_truth_tokens():
    model = _joint_model()
    batch = _batch()
    ids = batch["tokenized_data"]["input_ids"]
    ids[1, 2], ids[1, 3] = 6, 5
    baseline = model(**batch)
    expert = model.expert.expert.calls[0]
    assert expert["layers"][0][0].shape[-2] == 4
    mask = expert["attention_mask"]
    assert mask.tolist() == [[0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 1]]
    expected_positions = torch.tensor([[3, 4], [3, 4]]).expand(3, -1, -1)
    torch.testing.assert_close(expert["position_ids"], expected_positions)
    ids[1, 3:] = torch.tensor([12, 13, 14, 15])
    changed = model(**batch)
    torch.testing.assert_close(changed.expert_diffusion, baseline.expert_diffusion, rtol=0, atol=0)


def test_joint_rejects_duplicate_future_start_prefixes():
    model = _joint_model()
    batch = _batch()
    ids = batch["tokenized_data"]["input_ids"]
    ids[0, 4] = 6
    with pytest.raises(ValueError, match="future-start|left padding"):
        model(**batch)
    assert not model.expert.expert.calls


def test_joint_gradient_checkpointing_is_rejected_with_cache_explanation():
    model = _joint_model()
    with pytest.raises(ValueError, match="(?i)checkpoint") as error:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert "cache" in str(error.value).lower()
