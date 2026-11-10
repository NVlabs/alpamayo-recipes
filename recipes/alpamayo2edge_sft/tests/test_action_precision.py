# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint action precision preserves forward values and fixes cast-gradient rounding."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from alpamayo2edge_sft.models import action_precision


class DoubleCastNorm(torch.nn.Module):
    """The public action RMSNorm's two-cast formula, without model dependencies."""

    def __init__(self, dim=512):
        super().__init__()
        self.eps = 1e-5
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.register_buffer("sentinel", torch.tensor([0.1234567]))

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


@pytest.fixture(autouse=True)
def public_type(monkeypatch):
    monkeypatch.setattr(action_precision, "_rms_norm_type", lambda: DoubleCastNorm)


def shell():
    projection = torch.nn.Module()
    projection.encoder = torch.nn.Module()
    projection.encoder.trunk = torch.nn.Sequential(DoubleCastNorm(), torch.nn.SiLU(), DoubleCastNorm())
    projection.other_norm = DoubleCastNorm()
    return projection


def reference(x, weight, eps):
    full_precision = x.float()
    normalized = full_precision * torch.rsqrt(full_precision.pow(2).mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_forward_and_gradients_match_shared_fp32_reference(input_dtype, weight_dtype):
    generator = torch.Generator().manual_seed(42)
    projection = shell()
    norm = projection.encoder.trunk[0].to(dtype=weight_dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(512, generator=generator).to(weight_dtype))
    value = torch.randn(64, 512, generator=generator).to(input_dtype).requires_grad_()
    upstream = torch.randn(64, 512, generator=generator).to(torch.promote_types(input_dtype, weight_dtype))
    baseline = norm(value)
    baseline_grads = torch.autograd.grad(baseline, (value, norm.weight), upstream)
    expected = reference(value, norm.weight, norm.eps)
    expected_grads = torch.autograd.grad(expected, (value, norm.weight), upstream)
    with action_precision.joint_sft_action_precision(projection):
        actual = norm(x=value)
    # Backward occurs after the context is restored, as in normal training.
    actual_grads = torch.autograd.grad(actual, (value, norm.weight), upstream)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads[1], baseline_grads[1], rtol=0, atol=0)
    if input_dtype == torch.bfloat16:
        assert not torch.equal(baseline_grads[0], expected_grads[0])
    else:
        torch.testing.assert_close(baseline_grads[0], expected_grads[0], rtol=0, atol=0)


def test_modules_parameters_buffers_state_and_existing_hooks_are_preserved():
    projection = shell().to(dtype=torch.bfloat16)
    identities = {name: id(value) for name, value in projection.named_parameters()}
    saved = {name: value.clone() for name, value in projection.state_dict().items()}
    hooks = []
    handles = []
    for index in (0, 2):
        norm = projection.encoder.trunk[index]
        handles.append(norm.register_forward_pre_hook(lambda module, args: hooks.append(("pre", id(module)))))
        handles.append(norm.register_forward_hook(lambda module, args, output: hooks.append(("post", id(module)))))
    original_class_forward = DoubleCastNorm.forward
    with action_precision.joint_sft_action_precision(projection):
        assert all("forward" in projection.encoder.trunk[index].__dict__ for index in (0, 2))
        assert "forward" not in projection.other_norm.__dict__
        projection.encoder.trunk(torch.ones(2, 512, dtype=torch.bfloat16))
    assert len(hooks) == 4
    assert DoubleCastNorm.forward is original_class_forward
    assert all("forward" not in projection.encoder.trunk[index].__dict__ for index in (0, 2))
    assert {name: id(value) for name, value in projection.named_parameters()} == identities
    assert set(projection.state_dict()) == set(saved)
    for name, value in projection.state_dict().items():
        assert value.dtype == saved[name].dtype
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    for handle in handles:
        handle.remove()


def test_nested_instances_and_exception_restore_original_override():
    projection, other = shell(), shell()
    norm = projection.encoder.trunk[0]
    original = MethodType(DoubleCastNorm.forward, norm)
    norm.forward = original
    with pytest.raises(RuntimeError, match="deliberate"):
        with action_precision.joint_sft_action_precision(projection):
            outer = norm.forward
            assert outer is not original
            with action_precision.joint_sft_action_precision(projection):
                assert norm.forward is not outer
            assert norm.forward is outer
            assert "forward" not in other.encoder.trunk[0].__dict__
            raise RuntimeError("deliberate")
    assert norm.forward is original
    assert "forward" not in projection.encoder.trunk[2].__dict__


def test_subclasses_and_absent_action_encoder_are_not_modified():
    class CustomNorm(DoubleCastNorm):
        pass

    projection = shell()
    custom = CustomNorm()
    projection.encoder.trunk[0] = custom
    with action_precision.joint_sft_action_precision(projection):
        assert "forward" not in custom.__dict__
    with action_precision.joint_sft_action_precision(SimpleNamespace()):
        pass
