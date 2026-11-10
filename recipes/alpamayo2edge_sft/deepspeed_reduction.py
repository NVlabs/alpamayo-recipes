# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""Preserve FP32 reduced gradients before ZeRO-3 accumulates microbatches.

The reduction body is adapted from DeepSpeed 0.19.1's Apache-2.0-licensed
DeepSpeedZeroOptimizer_Stage3.__avg_scatter_grads. Its only arithmetic change
is omitting the final cast of reduced partitions back to the BF16 model dtype.
Upstream source:
https://github.com/deepspeedai/DeepSpeed/blob/v0.19.1/deepspeed/runtime/zero/stage3.py
"""

import hashlib
import inspect
import sys
import textwrap
from importlib.metadata import version
from pathlib import Path
from types import MethodType

import torch


_SUPPORTED_DEEPSPEED_VERSION = "0.19.1"
_METHOD = "_DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads"
_MARKER = "_alpamayo2edge_fp32_reduction_method"
_ORIGINAL_SOURCE_SHA256 = "9bc52dce57cf52a8e6f2d8b5fc5fae39d06c7b90237b83463db5aa9d970b2c72"
_NVTX_SOURCE_SHA256 = "1ca82972b9f95b3571ec044e7334cac732ceb306e196fed8d49e256da29ae6d6"


def _validate_original_method(optimizer):
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

    if version("deepspeed") != _SUPPORTED_DEEPSPEED_VERSION:
        raise RuntimeError("Edge FP32 reduction requires DeepSpeed 0.19.1; validate newer versions first")
    if not isinstance(optimizer, DeepSpeedZeroOptimizer_Stage3):
        raise RuntimeError("Expected the actual DeepSpeed ZeRO-3 optimizer")
    original = getattr(optimizer, _METHOD)
    if (
        getattr(original, "__self__", None) is not optimizer
        or getattr(original, "__func__", None) is not getattr(DeepSpeedZeroOptimizer_Stage3, _METHOD)
    ):
        raise RuntimeError("DeepSpeed reduction method was already replaced")
    wrapper = original.__func__
    # DS's NVTX decorator has no __wrapped__. Verify the known wrapper before
    # reading its function closure; do not unwrap arbitrary third-party hooks.
    if (
        wrapper.__qualname__ != "instrument_w_nvtx.<locals>.wrapped_fn"
        or hashlib.sha256(Path(inspect.getfile(wrapper)).read_bytes()).hexdigest()
        != _NVTX_SOURCE_SHA256
    ):
        raise RuntimeError("Unsupported DeepSpeed reduction wrapper")
    closure = inspect.getclosurevars(wrapper).nonlocals
    if set(closure) != {"func"} or not inspect.isfunction(closure["func"]):
        raise RuntimeError("Unsupported DeepSpeed reduction closure")
    function = closure["func"]
    if (
        function.__module__ != "deepspeed.runtime.zero.stage3"
        or function.__qualname__ != "DeepSpeedZeroOptimizer_Stage3.__avg_scatter_grads"
        or hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()
        != _ORIGINAL_SOURCE_SHA256
    ):
        raise RuntimeError("DeepSpeed reduction implementation differs from the validated version")
    return original


def _validate_configuration(optimizer):
    if (
        optimizer.dtype != torch.bfloat16
        or optimizer.communication_data_type != torch.float32
        or optimizer.gradient_accumulation_dtype != torch.float32
        or optimizer.grad_partitions_flat_buffer.dtype != torch.float32
    ):
        raise ValueError("Edge reduction requires BF16 parameters and FP32 communication/accumulation")
    if not optimizer.fp32_partitioned_groups_flat or any(
        parameter.dtype != torch.float32 for parameter in optimizer.fp32_partitioned_groups_flat
    ):
        raise ValueError("Edge reduction requires FP32 optimizer masters")
    if (
        not optimizer.reduce_scatter
        or not optimizer.postscale_gradients
        or optimizer.gradient_predivide_factor != 1.0
        or getattr(optimizer, "sequence_parallel_size", 1) != 1
    ):
        raise ValueError("Edge reduction requires ordinary ZeRO-3 data-parallel reduce-scatter")
    if (
        optimizer.all2all_process_group is not None
        or getattr(optimizer, "zeropp_loco_param", None) is not None
        or getattr(optimizer, "zenflow", None)
        or getattr(optimizer, "use_muon", False)
    ):
        raise ValueError("Quantized gradients, ZenFlow and Muon require separate reduction validation")


def _avg_scatter_grads_fp32(self, params_to_reduce, communication_data_type):
    """Original reduction order and collectives, retaining their FP32 outputs."""
    from deepspeed.runtime.zero import stage3

    if communication_data_type != torch.float32:
        raise RuntimeError("Gradient communication precision changed after Edge reduction setup")
    full_grads_for_rank = [parameter.grad for parameter in params_to_reduce]
    if communication_data_type != self.dtype:
        full_grads_for_rank = [gradient.to(communication_data_type) for gradient in full_grads_for_rank]

    if self.postscale_gradients and self.gradient_predivide_factor != 1.0:
        full_grads_for_rank = [
            gradient.div(self.gradient_predivide_factor) for gradient in full_grads_for_rank
        ]

    local_world_size = stage3.get_accelerator().device_count()
    global_world_size = stage3.dist.get_world_size()
    num_nodes = global_world_size // local_world_size
    if self.all2all_process_group is not None and num_nodes > 1:
        # These branches retain upstream behavior but are rejected by the
        # installer until the quantized communication modes are validated.
        grad_partitions_for_rank = (
            stage3.all_to_all_loco_quant_reduce(
                params_to_reduce, self.all2all_process_group, self.zeropp_loco_param
            )
            if self.zeropp_loco_param is not None
            else stage3.all_to_all_quant_reduce(full_grads_for_rank, self.all2all_process_group)
        )
    else:
        grad_partitions_for_rank = stage3.reduce_scatter_coalesced(
            full_grads_for_rank, self.dp_process_group
        )

    if (
        self.postscale_gradients
        and self.gradient_predivide_factor != 1.0
        and self.gradient_predivide_factor != stage3.dist.get_world_size(self.dp_process_group)
    ):
        grad_partitions_for_rank = [
            gradient.mul(self.gradient_predivide_factor) for gradient in grad_partitions_for_rank
        ]

    # DeepSpeed's partition_grads still owns accumulation, CPU/NVMe offload,
    # stream management and partition lifetimes. Preserve its FP32 inputs.
    if any(gradient.dtype != torch.float32 for gradient in grad_partitions_for_rank):
        raise RuntimeError("Expected FP32 reduced gradient partitions")
    return grad_partitions_for_rank


def _install_fp32_reduction(optimizer):
    installed = getattr(optimizer, _MARKER, None)
    if installed is not None:
        if getattr(optimizer, _METHOD) is not installed:
            raise RuntimeError("The Edge reduction adapter was replaced after installation")
        return False
    _validate_original_method(optimizer)
    _validate_configuration(optimizer)
    from deepspeed.utils.nvtx import instrument_w_nvtx

    bound = MethodType(instrument_w_nvtx(_avg_scatter_grads_fp32), optimizer)
    setattr(optimizer, _METHOD, bound)
    setattr(optimizer, _MARKER, bound)
    return True


def configure_joint_deepspeed_reduction(model):
    """Install once for joint BF16 ZeRO-3; plain Torch and FSDP remain untouched."""
    engine_type = getattr(sys.modules.get("deepspeed.runtime.engine"), "DeepSpeedEngine", None)
    if engine_type is None or not isinstance(model, engine_type):
        return False
    if getattr(model.module, "training_stage", None) != "joint":
        return False
    if model.zero_optimization_stage() != 3:
        return False
    return _install_fp32_reduction(model.optimizer)
