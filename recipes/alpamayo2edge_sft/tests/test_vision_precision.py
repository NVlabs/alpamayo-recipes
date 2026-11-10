# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint-only vision precision, including the actual Qwen attention operation."""

from types import SimpleNamespace

import pytest
import torch

from alpamayo2edge_sft.models.vision_precision import joint_sft_vision_precision


class InspectAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("inv_freq", torch.tensor([0.1234567, 0.9876543]))

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        return position_embeddings


def shell(blocks=12):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Module()
    model.model.visual.blocks = torch.nn.ModuleList()
    for _ in range(blocks):
        block = torch.nn.Module()
        block.attn = InspectAttention()
        model.model.visual.blocks.append(block)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keyword", [False, True])
def test_all_blocks_round_only_scoped_inputs_without_changing_buffers(dtype, keyword):
    model = shell()
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    hidden = torch.ones(3, 2, dtype=dtype)
    positions = (torch.tensor([[0.1234567, 0.9876543]]), torch.tensor([[0.456789, -0.234567]]))
    saved_positions = tuple(t.clone() for t in positions)
    sequence_lengths = torch.tensor([0, 3], dtype=torch.int32)

    def invoke(attention):
        if keyword:
            return attention(hidden_states=hidden, cu_seqlens=sequence_lengths,
                             position_embeddings=positions)
        return attention(hidden, sequence_lengths, positions)

    for _ in range(2):
        with joint_sft_vision_precision(model):
            for block in model.model.visual.blocks:
                output = invoke(block.attn)
                for actual, expected in zip(output, positions):
                    assert actual.dtype == dtype
                    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)
                    if dtype == torch.float32:
                        assert actual is expected
    for block in model.model.visual.blocks:
        assert len(block.attn._forward_pre_hooks) == 1
        assert invoke(block.attn) is positions
    assert set(model.state_dict()) == set(original)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    for value, saved in zip(positions, saved_positions):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)


def test_scope_is_instance_local_nested_and_exception_safe():
    model, other = shell(1), shell(1)
    hidden = torch.ones(1, 2, dtype=torch.bfloat16)
    positions = (torch.ones(1, 2), torch.zeros(1, 2))
    with pytest.raises(RuntimeError, match="test failure"):
        with joint_sft_vision_precision(model):
            with joint_sft_vision_precision(model):
                result = model.model.visual.blocks[0].attn(hidden, None, positions)
                assert result[0].dtype == torch.bfloat16
            result = model.model.visual.blocks[0].attn(hidden, None, positions)
            assert result[0].dtype == torch.bfloat16
            assert other.model.visual.blocks[0].attn(hidden, None, positions) is positions
            raise RuntimeError("test failure")
    assert model.model.visual.blocks[0].attn(hidden, None, positions) is positions
    assert not other.model.visual.blocks[0].attn._forward_pre_hooks
    with joint_sft_vision_precision(SimpleNamespace(model=SimpleNamespace())):
        pass


def test_autocast_uses_attention_compute_dtype():
    model = shell(1)
    hidden = torch.ones(1, 2)
    positions = (torch.ones(1, 2), torch.zeros(1, 2))
    with joint_sft_vision_precision(model), torch.autocast("cpu", dtype=torch.bfloat16):
        result = model.model.visual.blocks[0].attn(hidden, None, positions)
    assert all(value.dtype == torch.bfloat16 for value in result)
    assert all(value.dtype == torch.float32 for value in positions)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_qwen_attention_matches_explicit_boundary_rounding_and_gradients(dtype):
    from transformers import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention

    config = Qwen3VLVisionConfig(hidden_size=32, num_heads=2)
    config._attn_implementation = "sdpa"
    model = shell(1)
    attention = Qwen3VLVisionAttention(config).to(dtype)
    original_forward = Qwen3VLVisionAttention.forward
    model.model.visual.blocks[0].attn = attention
    generator = torch.Generator().manual_seed(183)
    hidden = torch.randn(4, 32, generator=generator).to(dtype).requires_grad_()
    angles = torch.randn(4, 16, generator=generator)
    positions = (angles.cos(), angles.sin())
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    expected = attention(hidden, cu_seqlens, tuple(value.to(dtype) for value in positions))
    expected.float().square().mean().backward()
    expected_grads = {name: p.grad.clone() for name, p in attention.named_parameters()}
    expected_input_grad = hidden.grad.clone()
    attention.zero_grad(set_to_none=True)
    hidden.grad = None
    with joint_sft_vision_precision(model):
        actual = attention(hidden, cu_seqlens, positions)
    actual.float().square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, expected_input_grad, rtol=0, atol=0)
    for name, p in attention.named_parameters():
        torch.testing.assert_close(p.grad, expected_grads[name], rtol=0, atol=0)
    assert Qwen3VLVisionAttention.forward is original_forward
