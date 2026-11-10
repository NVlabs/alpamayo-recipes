# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep configuration-derived constants exact after engine dtype/device casts."""

import math
from types import SimpleNamespace

import pytest
import torch

from alpamayo2edge_sft.models.sft import Alpamayo2EdgeSFTMixin


class FourierEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_freq = 100.0
        self.out_dim = 20
        self.register_buffer("freqs", torch.ones(1, 10), persistent=False)


class ActionSpace(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std"):
            self.register_buffer(name, torch.ones(()), persistent=False)


class BufferModel(Alpamayo2EdgeSFTMixin, torch.nn.Module):
    """Use real module casting with the checkpoint's nonpersistent-buffer layout."""

    def __init__(self, with_expert=True):
        torch.nn.Module.__init__(self)
        expert_cfg = {
            "accel_mean": 0.123456,
            "accel_std": 4.123456,
            "curvature_mean": -0.003456,
            "curvature_std": 0.013579,
        }
        tokenizer_cfg = {"accel_std": 5.123456, "curvature_std": 0.024681}
        self.config = SimpleNamespace(
            expert_config=SimpleNamespace(action_space_cfg=expert_cfg),
            future_traj_tokenizer_cfg={"action_space_cfg": tokenizer_cfg},
        )
        self.vlm = torch.nn.Linear(2, 2)
        if with_expert:
            self.expert = torch.nn.Module()
            self.expert.action_space = ActionSpace()
            self.expert.action_in_proj = torch.nn.Module()
            self.expert.action_in_proj.sinus = torch.nn.ModuleList(
                [FourierEncoder(), FourierEncoder()]
            )
            self.expert.action_in_proj.timestep_fourier_encoder = FourierEncoder()
            self.expert.action_out_proj = torch.nn.Linear(2, 2)
        self.future_traj_tokenizer = torch.nn.Module()
        self.future_traj_tokenizer.action_space = ActionSpace()
        self._restore_training_buffers()


def assert_exact_math_buffers(model):
    spaces = [
        (
            model.future_traj_tokenizer.action_space,
            model.config.future_traj_tokenizer_cfg["action_space_cfg"],
        )
    ]
    if hasattr(model, "expert"):
        spaces.append((model.expert.action_space, model.config.expert_config.action_space_cfg))
        projection = model.expert.action_in_proj
        for encoder in [*projection.sinus, projection.timestep_fourier_encoder]:
            expected = torch.logspace(
                0, math.log10(encoder.max_freq), encoder.out_dim // 2
            )[None]
            assert encoder.freqs.dtype == torch.float32
            torch.testing.assert_close(encoder.freqs, expected, rtol=0, atol=0)
    for space, config in spaces:
        for name, value in space.named_buffers():
            expected = torch.tensor(config.get(name, 1.0 if name.endswith("std") else 0.0))
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert all(name.endswith(("weight", "bias")) for name in model.state_dict())


@pytest.mark.parametrize("with_expert", [True, False])
def test_engine_bfloat16_cast_preserves_exact_derived_constants(with_expert):
    model = BufferModel(with_expert=with_expert)
    original = {name: value.clone() for name, value in model.named_buffers()}
    assert any(not torch.equal(value.bfloat16().float(), value) for value in original.values())

    assert model.bfloat16() is model

    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert_exact_math_buffers(model)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_to_dtype_and_meta_materialization_restore_constants_after_each_move():
    model = BufferModel()
    assert model.to(dtype=torch.float16, device="cpu") is model
    assert all(parameter.dtype == torch.float16 for parameter in model.parameters())
    assert_exact_math_buffers(model)

    model.to(device="meta", dtype=torch.bfloat16)
    assert all(value.is_meta for value in model.buffers())
    assert all(value.dtype == torch.float32 for value in model.buffers())
    assert model.to_empty(device="cpu") is model

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert_exact_math_buffers(model)


def test_cast_before_optional_modules_are_installed_is_safe():
    model = BufferModel.__new__(BufferModel)
    torch.nn.Module.__init__(model)
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))

    assert model.bfloat16() is model
    assert model.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("max_freq", [None, 37.0])
def test_release_fourier_buffers_restore_from_config(max_freq, request):
    if request.config.getoption("--edge-release-stub"):
        pytest.skip("Requires the actual inference package")
    from alpamayo2_edge.models.action_in_proj import FourierEncoderV2

    model = BufferModel()
    projection = model.expert.action_in_proj
    projection.sinus = torch.nn.ModuleList([FourierEncoderV2(20), FourierEncoderV2(20)])
    projection.timestep_fourier_encoder = FourierEncoderV2(20)
    model.config.expert_config.action_in_proj_cfg = (
        {} if max_freq is None else {"max_freq": max_freq}
    )
    model.bfloat16()
    expected = torch.logspace(0, math.log10(max_freq or 100.0), 10)[None]
    for encoder in [*projection.sinus, projection.timestep_fourier_encoder]:
        assert not hasattr(encoder, "max_freq")
        assert encoder.freqs.dtype == torch.float32
        torch.testing.assert_close(encoder.freqs, expected, rtol=0, atol=0)


@pytest.mark.parametrize("via_meta", [False, True])
def test_cosmos3_edge_text_and_expert_rope_are_rebuilt_in_exact_fp32(via_meta):
    from transformers import Cosmos3EdgeTextConfig
    from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeTextRotaryEmbedding

    model = BufferModel()
    text_config = Cosmos3EdgeTextConfig(
        hidden_size=128, num_attention_heads=1, head_dim=128,
        rope_parameters={
            "rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [24, 20, 20],
        },
    )
    model.vlm.rotary_emb = Cosmos3EdgeTextRotaryEmbedding(text_config)
    model.expert.rotary_emb = Cosmos3EdgeTextRotaryEmbedding(text_config)
    original = {
        name: value.clone() for name, value in model.named_buffers() if "inv_freq" in name
    }
    assert len(original) == 4
    assert any(not torch.equal(value.bfloat16().float(), value) for value in original.values())

    model.bfloat16()
    if via_meta:
        model.to("meta")
        model.to_empty(device="cpu")
    for name, value in model.named_buffers():
        if name in original:
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert all("inv_freq" not in name for name in model.state_dict())
