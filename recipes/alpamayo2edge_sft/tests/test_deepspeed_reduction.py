# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual DS method fixtures with CPU tensors and deterministic collectives."""
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if importlib.util.find_spec("deepspeed") is None:
    pytest.skip("DeepSpeed is not installed", allow_module_level=True)
if importlib.metadata.version("deepspeed") != "0.19.1":
    pytest.skip("Fixtures require the validated DeepSpeed version", allow_module_level=True)
pytest.importorskip("deepspeed")

adapter = importlib.import_module("alpamayo2edge_sft.deepspeed_reduction")


@pytest.fixture
def environment(monkeypatch):
    import deepspeed.runtime.zero.stage3 as stage3
    import deepspeed.utils.nvtx as nvtx

    monkeypatch.setattr(nvtx, "enable_nvtx", False)
    monkeypatch.setattr(stage3, "get_accelerator",
                        lambda: SimpleNamespace(device_count=lambda: 8))
    monkeypatch.setattr(stage3, "dist", SimpleNamespace(get_world_size=lambda *args: 8))
    calls = []
    reduced = torch.tensor([1.001953125, -1.001953125], dtype=torch.float32)

    def collective(gradients, group):
        calls.append((gradients, group))
        assert all(gradient.dtype == torch.float32 for gradient in gradients)
        return [reduced.clone()]

    monkeypatch.setattr(stage3, "reduce_scatter_coalesced", collective)
    zero = object.__new__(stage3.DeepSpeedZeroOptimizer_Stage3)
    zero.dtype = torch.bfloat16
    zero.communication_data_type = torch.float32
    zero.gradient_accumulation_dtype = torch.float32
    zero.grad_partitions_flat_buffer = torch.zeros(2)
    zero.fp32_partitioned_groups_flat = [torch.zeros(2)]
    zero.reduce_scatter = True
    zero.postscale_gradients = True
    zero.gradient_predivide_factor = 1.0
    zero.sequence_parallel_size = 1
    zero.all2all_process_group = None
    zero.zeropp_loco_param = None
    zero.zenflow = None
    zero.use_muon = False
    zero.dp_process_group = object()
    zero.offload_optimizer = False
    parameter = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    parameter.grad = torch.ones_like(parameter)
    return zero, [parameter], calls, reduced


def test_original_negative_control_rounds_but_adapter_retains_fp32(environment):
    zero, parameters, calls, reduced = environment
    original = getattr(zero, adapter._METHOD)
    baseline = original(parameters, torch.float32)
    assert baseline[0].dtype == torch.bfloat16
    assert not torch.equal(baseline[0].float(), reduced)
    assert adapter._install_fp32_reduction(zero)
    actual = getattr(zero, adapter._METHOD)(parameters, torch.float32)
    torch.testing.assert_close(actual[0], reduced, rtol=0, atol=0)
    assert calls[0][1] is calls[1][1] is zero.dp_process_group
    torch.testing.assert_close(calls[0][0][0], calls[1][0][0], rtol=0, atol=0)
    assert zero.dtype == torch.bfloat16
    assert not adapter._install_fp32_reduction(zero)


@pytest.mark.parametrize("offload", [False, True])
def test_accumulation_and_offload_partition_owner_is_unchanged(environment, offload):
    zero, parameters, calls, reduced = environment
    zero.offload_optimizer = offload
    original_partition = zero.partition_grads
    accumulation = zero.grad_partitions_flat_buffer
    masters = zero.fp32_partitioned_groups_flat
    assert adapter._install_fp32_reduction(zero)
    for _ in range(8):
        actual = getattr(zero, adapter._METHOD)(parameters, torch.float32)
        # Represent the unchanged caller's FP32 accumulation across eight
        # microbatches. The adapter must not normalize by GAS a second time.
        accumulation.add_(actual[0])
    torch.testing.assert_close(accumulation, reduced * 8, rtol=0, atol=0)
    assert zero.partition_grads == original_partition
    assert zero.grad_partitions_flat_buffer is accumulation
    assert zero.fp32_partitioned_groups_flat is masters
    assert zero.offload_optimizer is offload and len(calls) == 8


@pytest.mark.parametrize("attribute,value", [
    ("dtype", torch.float32),
    ("communication_data_type", torch.bfloat16),
    ("gradient_accumulation_dtype", torch.bfloat16),
    ("reduce_scatter", False),
    ("postscale_gradients", False),
    ("gradient_predivide_factor", 2.0),
    ("sequence_parallel_size", 2),
    ("all2all_process_group", object()),
    ("zeropp_loco_param", {}),
    ("zenflow", True),
    ("use_muon", True),
])
def test_unsupported_configuration_is_rejected_without_installation(
    environment, attribute, value
):
    zero, *_ = environment
    original = getattr(zero, adapter._METHOD)
    setattr(zero, attribute, value)
    with pytest.raises(ValueError):
        adapter._install_fp32_reduction(zero)
    assert getattr(zero, adapter._METHOD) == original


def test_changed_version_is_rejected(environment, monkeypatch):
    zero, *_ = environment
    monkeypatch.setattr(adapter, "version", lambda name: "0.20.0")
    with pytest.raises(RuntimeError, match="0.19.1"):
        adapter._install_fp32_reduction(zero)


def test_replaced_method_is_rejected_before_and_after_installation(environment):
    zero, *_ = environment
    original = getattr(zero, adapter._METHOD)
    setattr(zero, adapter._METHOD, lambda *args: None)
    with pytest.raises(RuntimeError, match="already replaced"):
        adapter._install_fp32_reduction(zero)
    setattr(zero, adapter._METHOD, original)
    adapter._install_fp32_reduction(zero)
    setattr(zero, adapter._METHOD, original)
    with pytest.raises(RuntimeError, match="after installation"):
        adapter._install_fp32_reduction(zero)


def test_changed_upstream_source_is_rejected(environment, monkeypatch):
    zero, *_ = environment
    monkeypatch.setattr(adapter, "_ORIGINAL_SOURCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="implementation differs"):
        adapter._install_fp32_reduction(zero)


def test_changed_runtime_communication_dtype_is_rejected(environment):
    zero, parameters, *_ = environment
    adapter._install_fp32_reduction(zero)
    with pytest.raises(RuntimeError, match="communication precision"):
        getattr(zero, adapter._METHOD)(parameters, torch.bfloat16)


def test_non_deepspeed_model_is_untouched():
    assert not adapter.configure_joint_deepspeed_reduction(SimpleNamespace())
