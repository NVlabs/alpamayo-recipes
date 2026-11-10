# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-ordered FP32 FSDP-style global L2 norm for joint training.

This is an algorithm helper, not an installed adapter. It has no model names,
checkpoint shapes, capture files, environment switches, or diagnostic imports.
"""
import math

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard


def _stride(shape):
    result, value = [], 1
    for size in reversed(shape):
        result.append(value)
        value *= size
    return tuple(reversed(result))


def _row_range(shape, rank, world):
    rows = (shape[0] + world - 1) // world
    begin = min(rank * rows, shape[0])
    count = min(rows, shape[0] - begin)
    tail = math.prod(shape[1:])
    return begin * tail, count * tail, (count, *shape[1:])


@torch.no_grad()
def fp32_global_norm(gradients):
    """Explicit foreach path; DTensor materialization owns DP reduction."""
    values = list(gradients)
    if any(value.dtype != torch.float32 for value in values):
        raise ValueError("The FP32 norm policy requires FP32 accumulated gradients")
    norm = torch.nn.utils.get_total_norm(values, 2.0, False, True)
    return norm.full_tensor() if isinstance(norm, DTensor) else norm


@torch.no_grad()
def fp32_clip_coefficient(max_norm, norm):
    return torch.clamp(max_norm / (norm + 1e-6), max=1.0)


@torch.no_grad()
def clip_fp32_with_norm_(gradients, max_norm, norm):
    """Native coefficient expression, including configurable thresholds."""
    values = list(gradients)
    if max_norm <= 0:
        return
    coefficient = fp32_clip_coefficient(max_norm, norm)
    # Each ZeRO subgroup is one actual FP32 master gradient. This helper also
    # accepts FSDP DTensor gradients and plain FP32 gradients under that policy.
    torch._foreach_mul_(values, coefficient)


class ZeroFp32GlobalNorm:
    """Derive order/shapes from this model and live managed optimizer tensors.

    Mismatched flat/row partition widths require a full FP32 parameter gather on
    every call. Extra communication and temporary memory scale with that one
    parameter (approximately four bytes per element), including large embeddings.
    """
    def __init__(self, model, optimizer, *, device_type="cuda"):
        if optimizer.offload_optimizer or optimizer.swap_optimizer:
            raise ValueError("FP32 FSDP-style norm requires GPU-resident ZeRO gradients")
        if optimizer.model_parallel_group is not None:
            raise ValueError("FP32 FSDP-style norm currently supports data parallelism only")
        self.optimizer = optimizer
        self.group = optimizer.dp_process_group
        self.world, self.rank = dist.get_world_size(self.group), dist.get_rank(self.group)
        managed = {id(p): p for group in optimizer.fp16_groups for p in group}
        if not managed:
            raise ValueError("No optimizer-managed parameters are available")
        self.parameters = [p for p in model.parameters() if id(p) in managed]
        if len(self.parameters) != len(managed):
            raise ValueError("Optimizer-managed parameters are not covered by model traversal")
        self.shapes = [tuple(p.ds_shape) for p in self.parameters]
        if any(not shape or any(size <= 0 for size in shape) for shape in self.shapes):
            raise ValueError("FSDP-style norm requires positive, non-scalar parameter shapes")
        if any(math.prod(shape) != p.ds_numel for p, shape in zip(self.parameters,self.shapes)):
            raise ValueError("Inconsistent ZeRO global shape metadata")
        self.mesh = DeviceMesh.from_group(self.group, device_type, mesh_dim_names=("dp_shard",))
        self.device_type = device_type

    @torch.no_grad()
    def __call__(self):
        optimizer = self.optimizer
        live = {}
        for group_index, parameters in enumerate(optimizer.fp16_groups):
            for parameter, gradient in zip(parameters, optimizer.averaged_gradients[group_index], strict=True):
                if id(parameter) in live:
                    raise ValueError("Duplicate optimizer gradient")
                live[id(parameter)] = gradient.detach()
        if set(live) != {id(p) for p in self.parameters}:
            raise ValueError("Incomplete accumulated gradient coverage")
        distributed = []
        for parameter, shape in zip(self.parameters, self.shapes, strict=True):
            if tuple(parameter.ds_shape) != shape or parameter.ds_numel != math.prod(shape):
                raise RuntimeError("ZeRO global shape metadata changed after norm setup")
            gradient = live[id(parameter)]
            width = (parameter.ds_numel + self.world - 1) // self.world
            if (gradient.dtype != torch.float32 or gradient.device.type != self.device_type
                    or not gradient.is_contiguous() or gradient.numel() != width):
                raise ValueError("Unexpected actual ZeRO gradient partition")
            start, count, local_shape = _row_range(shape, self.rank, self.world)
            row_width = ((shape[0] + self.world - 1) // self.world) * math.prod(shape[1:])
            if row_width == width:
                local = gradient.view(-1)[:count].view(local_shape)
            else:
                full = gradient.new_empty(width * self.world)
                dist.all_gather_into_tensor(full, gradient.view(-1), group=self.group)
                # Release the full gathered parameter after retaining only this
                # rank's FSDP logical row slice. Bound temporary memory to one
                # irregular parameter, not all such parameters in the model.
                local = full[start:start+count].clone().view(local_shape)
                del full
            distributed.append(DTensor.from_local(local, self.mesh, [Shard(0)],
                run_check=False, shape=torch.Size(shape), stride=_stride(shape)))
        return fp32_global_norm(distributed)


def validate_norm_algorithm(value):
    if value not in {"auto", "fsdp_fp32", "engine"}:
        raise ValueError("gradient_norm_algorithm must be auto, fsdp_fp32 or engine")
    return value


def zero_norm_unsupported_reason(model, optimizer):
    """Validate supported ownership without initializing CUDA or collectives."""
    if optimizer.offload_optimizer or getattr(optimizer, "swap_optimizer", False):
        return "optimizer offload uses a different gradient storage path"
    if getattr(optimizer, "model_parallel_group", None) is not None:
        return "model-parallel gradient ownership needs separate validation"
    if getattr(optimizer, "model_parallel_rank", 0) != 0:
        return "model-parallel rank filtering needs separate validation"
    if getattr(optimizer, "sequence_parallel_size", 1) != 1:
        return "sequence parallelism needs separate validation"
    if torch.__version__.split("+")[0] != "2.8.0":
        return "the FP32 DTensor norm is validated with Torch 2.8.0"
    masters = optimizer.fp32_partitioned_groups_flat
    if not masters or any(value.device.type != "cuda" for value in masters):
        return "the matched ZeRO norm requires GPU-resident optimizer gradients"
    groups = getattr(optimizer, "fp16_groups", None)
    if not groups or not hasattr(optimizer, "averaged_gradients"):
        return "the validated ZeRO gradient partition interface is unavailable"
    managed = {id(p): p for group in groups for p in group}
    if not managed:
        return "no optimizer-managed parameters are available"
    if sum(len(group) for group in groups) != len(managed):
        return "duplicate optimizer-managed parameters need separate validation"
    selected = [p for p in model.parameters() if id(p) in managed]
    if len(selected) != len(managed):
        return "optimizer parameters are not covered by model traversal"
    for parameter in selected:
        if isinstance(parameter, DTensor):
            return "tensor-parallel model parameters need separate validation"
        shape = getattr(parameter, "ds_shape", None)
        if not shape or any(size <= 0 for size in shape):
            return "FSDP row sharding requires positive, non-scalar parameter shapes"
        if math.prod(shape) != getattr(parameter, "ds_numel", None):
            return "ZeRO global shape metadata is inconsistent"
    return None


def fsdp_norm_unsupported_reason(gradients):
    """Require one ordinary data-parallel Shard(0) mesh and FP32 gradients."""
    values = list(gradients)
    if torch.__version__.split("+")[0] != "2.8.0":
        return "the FP32 DTensor norm is validated with Torch 2.8.0"
    if not values:
        return "no materialized gradients are available"
    if any(value.device.type != "cuda" for value in values):
        return "the matched FSDP training path requires GPU-resident gradients"
    if any(not isinstance(value, DTensor) for value in values):
        return "mixed DTensor/plain gradient ownership needs separate validation"
    mesh = values[0].device_mesh
    for value in values:
        if value.dtype != torch.float32 or value.layout != torch.strided:
            return "the FSDP norm requires dense FP32 accumulated gradients"
        if value.device_mesh != mesh or mesh.ndim != 1:
            return "mixed or multidimensional meshes need separate validation"
        if len(value.placements) != 1 or value.placements[0] != Shard(0):
            return "only ordinary data-parallel Shard(0) gradients are supported"
        if not value.to_local().is_contiguous():
            return "non-contiguous FSDP gradient layout needs separate validation"
    return None
