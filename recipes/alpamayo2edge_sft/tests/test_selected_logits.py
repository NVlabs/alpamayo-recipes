# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real tiny Cosmos3-Edge checks; the public inference wrapper remains an explicit double."""

import math
from types import SimpleNamespace

import pytest
import torch
from alpamayo2edge_sft.models import TrainableAlpamayo2Edge
from alpamayo2edge_sft.models.utils import compute_next_token_loss, future_traj_loss_mask
from transformers import Cosmos3EdgeConfig, Cosmos3EdgeForConditionalGeneration
from transformers.processing_utils import ProcessorMixin


def _tiny_model(cap):
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
    model.config = SimpleNamespace(
        enable_expert=False,
        vlm_config=vlm_config,
        traj_ids={
            "future_start": 6,
            "future_end": 7,
            "future_id0": 8,
            "history_pad": 25,
            "future_pad": 26,
        },
        future_vocab_size=2,
        loss_weights={"future_traj": 0.7, "others": 1.3},
        logit_cap=cap,
    )
    model.history_traj_tokenizer = model.future_traj_tokenizer = None
    model.set_training_stage("vlm")
    return model


@pytest.mark.parametrize("cap", [0.0, 2.0])
def test_selected_projection_matches_dense_cosmos3_edge_loss_and_gradients(cap):
    torch.manual_seed(35)
    model = _tiny_model(cap)
    model.train()
    ids = torch.tensor([[0, 3, 4, 6, 8, 7, 2], [3, 4, 5, 6, 9, 7, 2]])
    data = {"input_ids": ids, "attention_mask": ids.ne(0).long()}
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, :2] = False
    labels = torch.where(mask, ids, -100)
    dense = model.vlm(**data, use_cache=False).logits
    if cap:
        dense = cap * torch.tanh(dense / cap)
    traj_mask = future_traj_loss_mask(labels, model.config.traj_ids, 2, 2)
    reference = 0.7 * compute_next_token_loss(dense, labels, traj_mask)
    reference += 1.3 * compute_next_token_loss(dense, labels, ~traj_mask)
    reference.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    projected_shapes = []
    handle = model.vlm.lm_head.register_forward_pre_hook(
        lambda module, args: projected_shapes.append(tuple(args[0].shape))
    )
    actual = model(tokenized_data=data, traj_data={}, labels_mask=mask)
    handle.remove()
    torch.testing.assert_close(actual.loss, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        actual.supervised_logits, dense[:, :-1][mask[:, 1:]], rtol=1e-6, atol=1e-6
    )
    assert projected_shapes == [(10, 32)]
    actual.loss.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=2e-5, atol=2e-6)
        else:
            assert parameter.grad is None


def test_supervised_target_cannot_be_predicted_from_padding():
    model = _tiny_model(0.0)
    ids = torch.tensor([[0, 3, 6, 8, 7, 2]])
    with pytest.raises(ValueError, match="non-padding"):
        model(
            tokenized_data={"input_ids": ids, "attention_mask": ids.ne(0).long()},
            traj_data={},
            labels_mask=ids.ne(0),
        )


def test_stage1_vqa_without_trajectory_trains_only_text_ce():
    model = _tiny_model(0.0)
    ids = torch.tensor([[3, 4, 5, 2]])
    mask = torch.tensor([[False, False, True, True]])
    output = model(tokenized_data={"input_ids": ids}, labels_mask=mask)
    assert output.future_traj_ce.item() == 0
    assert output.others_ce.item() > 0
    torch.testing.assert_close(output.loss, output.others_ce)
    output.loss.backward()
    assert model.vlm.lm_head.weight.grad.abs().sum() > 0


def test_legacy_modality_ids_match_processor_including_video_timestamps():
    model = _tiny_model(0.0)
    ids = torch.tensor([[0, 3, 27, 29, 28, 4, 27, 30, 30, 28, 2]])
    data = {"input_ids": ids, "image_grid_thw": torch.tensor([[1, 2, 2]])}
    processor = SimpleNamespace(image_token_ids=[29], video_token_ids=[30], audio_token_ids=[])
    expected = torch.tensor(ProcessorMixin.create_mm_token_type_ids(processor, ids.tolist()))
    prepared = model._prepare_vlm_inputs(data)
    assert "mm_token_type_ids" not in data
    assert prepared["mm_token_type_ids"].dtype == torch.int32
    torch.testing.assert_close(prepared["mm_token_type_ids"].long(), expected, rtol=0, atol=0)
    # Keep authoritative processor metadata, including custom token layouts.
    supplied = {**data, "mm_token_type_ids": expected}
    assert model._prepare_vlm_inputs(supplied)["mm_token_type_ids"] is expected


def test_legacy_multimodal_stage1_matches_explicit_modality_loss_and_gradients():
    torch.manual_seed(36)
    model = _tiny_model(0.0)
    model.train()
    ids = torch.tensor([[3, 27, 29, 28, 4, 6, 8, 7, 2]])
    mask = torch.tensor([[False, False, False, False, False, True, True, True, True]])
    data = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.randn(4, 768),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    explicit = {**data, "mm_token_type_ids": (ids == 29).to(torch.int32)}
    reference = model(tokenized_data=explicit, labels_mask=mask)
    reference.loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual = model(tokenized_data=data, labels_mask=mask)
    torch.testing.assert_close(actual.loss, reference.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.supervised_logits, reference.supervised_logits, rtol=0, atol=0)
    actual.loss.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
    assert "mm_token_type_ids" not in data


def test_loading_restores_nonpersistent_math_constants_from_config():
    model = _tiny_model(0.0)
    config = {
        "accel_mean": 0.02902694707164455,
        "accel_std": 0.6810426736454882,
        "curvature_mean": 0.0002692167976330542,
        "curvature_std": 0.026148280660833106,
    }

    def action_space():
        module = torch.nn.Module()
        for name, value in config.items():
            module.register_buffer(name, torch.tensor(value, dtype=torch.bfloat16), persistent=False)
        return module

    def encoder():
        module = torch.nn.Module()
        module.out_dim, module.max_freq = 20, 100.0
        module.register_buffer(
            "freqs", torch.logspace(0, 2, 10).to(torch.bfloat16)[None, :], persistent=False
        )
        return module

    model.expert = torch.nn.Module()
    model.expert.action_space = action_space()
    model.expert.action_in_proj = SimpleNamespace(
        sinus=[encoder(), encoder()], timestep_fourier_encoder=encoder()
    )
    model.config.expert_config = SimpleNamespace(action_space_cfg=config)
    model.config.future_traj_tokenizer_cfg = {"action_space_cfg": config}
    model.future_traj_tokenizer = SimpleNamespace(action_space=action_space())
    model._restore_training_buffers()
    for space in (model.expert.action_space, model.future_traj_tokenizer.action_space):
        for name, value in config.items():
            torch.testing.assert_close(getattr(space, name), torch.tensor(value), rtol=0, atol=0)
            assert getattr(space, name).dtype == torch.float32
        assert space.state_dict() == {}
    for item in [
        *model.expert.action_in_proj.sinus,
        model.expert.action_in_proj.timestep_fourier_encoder,
    ]:
        expected = torch.logspace(0, math.log10(item.max_freq), 10)[None, :]
        torch.testing.assert_close(item.freqs, expected, rtol=0, atol=0)
        assert item.freqs.dtype == torch.float32
        assert item.state_dict() == {}
