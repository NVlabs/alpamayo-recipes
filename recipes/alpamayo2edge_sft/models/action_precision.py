# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action-encoder RMSNorm precision for joint supervised training."""

from contextlib import contextmanager
from types import MethodType

import torch


def _rms_norm_type():
    # Keep this helper importable by CPU-only tests without loading a model.
    from alpamayo2_edge.models.action_in_proj import RMSNorm

    return RMSNorm


def _shared_cast_forward(module, x):
    # Sharing the FP32 tensor accumulates both derivative branches before the
    # cast backward rounds to the input dtype. Two x.float() calls instead
    # round each branch separately when the input is BF16.
    x_fp32 = x.float()
    output = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + module.eps)
    return output.type_as(x) * module.weight


@contextmanager
def joint_sft_action_precision(action_in_proj):
    """Share the FP32 normalization input during a joint SFT forward only.

    Targets only the public action MLP's RMSNorm modules. Forward values,
    parameters, buffers and normal module hooks remain unchanged. Original
    instance methods are restored on exit, including nested/error paths.
    Joint SFT disallows gradient checkpointing, so normalization cannot be
    recomputed later outside this scope. Inference calls remain unchanged.
    """
    trunk = getattr(getattr(action_in_proj, "encoder", None), "trunk", None)
    if trunk is None:
        yield
        return
    rms_type = _rms_norm_type()
    missing = object()
    replaced = []
    try:
        for module in trunk.modules():
            if type(module) is rms_type:
                original = module.__dict__.get("forward", missing)
                module.forward = MethodType(_shared_cast_forward, module)
                replaced.append((module, original))
        yield
    finally:
        for module, original in reversed(replaced):
            if original is missing:
                del module.forward
            else:
                module.forward = original
