# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed driving/text joint objectives and graph-connected expert dummy passes."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Cosmos3EdgeTextConfig, Cosmos3EdgeTextModel

from alpamayo2edge_sft.models.sft import Alpamayo2EdgeSFTMixin
from test_joint_training import _batch, _joint_model


def _text_batch(trajectory=None):
    batch = _batch()
    batch["tokenized_data"]["input_ids"] = torch.tensor([
        [0, 3, 4, 5, 10, 11, 2], [3, 4, 5, 10, 11, 12, 2],
    ])
    batch["traj_data"] = trajectory
    return batch


def _text_model(**kwargs):
    model = _joint_model(**kwargs)
    config = Cosmos3EdgeTextConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=1, num_key_value_heads=1,
        head_dim=8, rope_parameters={
            "rope_type": "default", "rope_theta": 5000000.,
            "mrope_section": [1, 1, 2],
        },
    )
    config._attn_implementation = "sdpa"
    model.expert.expert = Cosmos3EdgeTextModel(config)
    del model.expert.expert.embed_tokens
    model.dummy_calls = []
    model.diffusion_calls = []

    def targets(action):
        result = Alpamayo2EdgeSFTMixin._flow_matching_targets(model, action)
        model.diffusion_calls.append({key: value.detach().clone() for key, value in result.items()})
        return result

    model._flow_matching_targets = targets
    model.expert.diffusion = SimpleNamespace(construct_training_data=targets)
    model.expert.expert.register_forward_pre_hook(
        lambda module, args, kwargs: model.dummy_calls.append(dict(kwargs)), with_kwargs=True,
    )
    return model


def _rows(batch):
    return [{key: value[row] for key, value in batch["traj_data"].items()} for row in range(2)]


@pytest.mark.parametrize("trajectory", [None, [None, None]])
def test_text_joint_supervises_ce_and_visits_every_expert_parameter_with_zero_gradients(trajectory):
    model = _text_model(loss_weights={"others": 1.7}, token_weights={"future_traj": 1., "text": .2})
    batch = _text_batch(trajectory)
    output = model(**batch)
    labels = batch["tokenized_data"]["input_ids"].masked_fill(~batch["labels_mask"], -100)
    targets = labels[:, 1:][labels[:, 1:].ne(-100)]
    expected = F.cross_entropy(output.supervised_logits.float(), targets) * .2 * 1.7
    torch.testing.assert_close(output.loss, expected)
    assert output.expert_diffusion.item() == 0 and output.future_traj_ce.item() == 0
    assert output.others_ce.item() > 0 and output.logits.shape == (1, 2, 2)
    output.loss.backward()
    assert model.vlm.lm_head.weight.grad.abs().sum() > 0
    for name, parameter in model.expert.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) == 0, name
    assert len(model.dummy_calls) == 1
    call = model.dummy_calls[0]
    assert call["past_key_values"] is None and call["attention_mask"] is None
    assert call["is_causal"] is False and call["use_cache"] is True
    assert call["position_ids"].shape == (3, 1, 2)
    assert torch.count_nonzero(call["position_ids"]) == 0
    assert model.diffusion_calls[0]["x"].shape == (1, 2, 2)
    assert torch.count_nonzero(model.diffusion_calls[0]["x"]) == 0
    assert model.diffusion_calls[0]["x"].dtype == torch.float32


def test_text_dummy_consumes_one_real_flow_draw_and_has_no_prompt_conditioning():
    model = _text_model()
    first = _text_batch()
    torch.manual_seed(1337)
    output = model(**first)
    rng_after = torch.get_rng_state()
    torch.manual_seed(1337)
    expected = Alpamayo2EdgeSFTMixin._flow_matching_targets(model, torch.zeros(1, 2, 2))
    assert torch.equal(rng_after, torch.get_rng_state())
    for key in expected:
        torch.testing.assert_close(model.diffusion_calls[0][key], expected[key], rtol=0, atol=0)
    second = _text_batch()
    second["tokenized_data"]["input_ids"][:, -3:] = torch.tensor([[14, 15, 2], [16, 17, 2]])
    torch.manual_seed(1337)
    changed = model(**second)
    torch.testing.assert_close(output.logits, changed.logits, rtol=0, atol=0)
    assert not torch.equal(output.supervised_logits, changed.supervised_logits)


def test_text_joint_all_ignored_labels_has_finite_zero_loss_and_connected_expert():
    model = _text_model()
    batch = _text_batch()
    batch["labels_mask"].zero_()
    output = model(**batch)
    assert output.loss.item() == 0 and torch.isfinite(output.loss)
    output.loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.expert.parameters())


def test_labelled_rows_without_future_anchor_use_zero_weight_expert():
    model = _text_model()
    output = model(**_text_batch(_batch()["traj_data"]))
    assert output.expert_diffusion.item() == 0
    assert len(model.diffusion_calls) == 1 and len(model.dummy_calls) == 1


def test_mixed_rows_use_only_anchored_trajectory_and_local_trajectory_mean():
    model = _joint_model({"future_traj": 2.3})
    batch = _batch()
    trajectory_rows = _rows(batch)
    batch["traj_data"] = [None, trajectory_rows[1]]
    batch["tokenized_data"]["input_ids"][0, 3:] = torch.tensor([10, 11, 12, 2])
    output = model(**batch)
    pred = model.expert_predictions[0]
    action = trajectory_rows[1]["ego_future_xyz"][..., :2].reshape(1, 2, 2)
    expected = F.mse_loss(pred.float(), action + .5) * 2.3
    torch.testing.assert_close(output.expert_diffusion, expected)
    call = model.expert.expert.calls[0]
    assert pred.shape[0] == 1 and call["layers"][0][0].shape[0] == 1
    for (actual_keys, actual_values), live in zip(
        call["layers"], model.vlm_calls[0]["layers"], strict=True,
    ):
        torch.testing.assert_close(actual_keys, live[0][1:2, :, :4].detach())
        torch.testing.assert_close(actual_values, live[1][1:2, :, :4].detach())
        assert not actual_keys.requires_grad and not actual_values.requires_grad
    output.loss.backward()
    assert all(p.grad is not None for p in model.expert.parameters())


def test_mixed_labelled_navigation_row_is_demoted_without_diluting_expert_loss():
    model = _joint_model()
    batch = _batch()
    batch["tokenized_data"]["input_ids"][0, 3:] = torch.tensor([10, 11, 12, 2])
    output = model(**batch)
    action = batch["traj_data"]["ego_future_xyz"][1:2, ..., :2].reshape(1, 2, 2)
    expected = F.mse_loss(model.expert_predictions[0].float(), action + .5)
    torch.testing.assert_close(output.expert_diffusion, expected)
    assert model.expert_predictions[0].shape[0] == 1


def test_per_row_all_driving_labels_match_stacked_driving_objective_and_gradients():
    stacked = _joint_model()
    listed = _joint_model()
    first = _batch()
    second = _batch()
    second["traj_data"] = _rows(second)
    a, b = stacked(**first), listed(**second)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
    a.loss.backward();b.loss.backward()
    for (name, p), (_, q) in zip(stacked.named_parameters(), listed.named_parameters(), strict=True):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("error", ["unlabelled_anchor", "duplicate", "bad_length", "padding_anchor"])
def test_mixed_joint_rejects_invalid_anchor_label_contract(error):
    model = _joint_model()
    batch = _batch()
    rows = _rows(batch)
    batch["traj_data"] = rows
    if error == "unlabelled_anchor":
        batch["traj_data"][0] = None
    elif error == "duplicate":
        batch["tokenized_data"]["input_ids"][0, 4] = 6
    elif error == "bad_length":
        batch["traj_data"] = rows[:1]
    else:
        batch["tokenized_data"]["attention_mask"][0, 3] = 0
    with pytest.raises(ValueError, match="future-start|batch size"):
        model(**batch)
    assert not model.expert.expert.calls


def test_nonfinite_dummy_prediction_keeps_zero_loss_and_zero_prediction_gradient():
    model = _text_model()
    prediction = torch.tensor([[[float('nan'), float('inf')], [-float('inf'), 3.]]], requires_grad=True)
    hook = model.expert.action_out_proj.register_forward_hook(lambda module, args, output: prediction)
    output = model(**_text_batch())
    assert torch.isfinite(output.loss) and output.expert_diffusion.item() == 0
    output.loss.backward()
    assert prediction.grad is not None and torch.count_nonzero(prediction.grad) == 0
    hook.remove()


@pytest.mark.parametrize("autocast", [False, True])
def test_text_dummy_ignores_bf16_cached_beta_and_keeps_zero_expert_gradients(autocast):
    """ZeRO-3 can cast the cached distribution despite explicit FP32 construction."""
    model = _text_model()
    cached_beta = torch.distributions.Beta(
        torch.tensor(1.5, dtype=torch.bfloat16),
        torch.tensor(1.0, dtype=torch.bfloat16),
    )
    assert cached_beta.concentration1.dtype == torch.bfloat16
    with pytest.raises(NotImplementedError, match="dirichlet.*BFloat16"):
        cached_beta.sample((1,))

    def cached_targets(action):
        # This is the failing inference-distribution path reached before the fix.
        return {"timesteps": cached_beta.sample((action.shape[0],))}

    model.expert.diffusion = SimpleNamespace(construct_training_data=cached_targets)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = model(**_text_batch())
    assert torch.isfinite(output.loss) and output.vlm_ce.item() > 0
    assert torch.isfinite(output.logits).all() and output.expert_diffusion.item() == 0
    output.loss.backward()
    assert model.vlm.lm_head.weight.grad.abs().sum() > 0
    for name, parameter in model.expert.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) == 0, name
    assert len(model.diffusion_calls) == 1
    assert all(value.dtype == torch.float32 for value in model.diffusion_calls[0].values())


def test_text_dummy_matches_released_flow_targets_and_rng_state(pytestconfig):
    if pytestconfig.getoption("--edge-release-stub"):
        pytest.skip("Requires the real inference diffusion sampler")
    from alpamayo2_edge.diffusion.flow_matching import FlowMatching

    model = _text_model()
    reference = FlowMatching(x_dims=(2, 2), train_timestep_sampler="beta")
    assert reference.beta_dist.concentration1.dtype == torch.float32
    torch.manual_seed(1337)
    model(**_text_batch())
    actual_rng_state = torch.get_rng_state()
    torch.manual_seed(1337)
    expected = reference.construct_training_data(torch.zeros(1, 2, 2, dtype=torch.float32))
    assert torch.equal(actual_rng_state, torch.get_rng_state())
    assert len(model.diffusion_calls) == 1
    for key in ("x", "timesteps", "noise", "noisy_x"):
        torch.testing.assert_close(model.diffusion_calls[0][key], expected[key], rtol=0, atol=0)
