# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision-attention precision compatibility for joint supervised training."""

from contextlib import contextmanager

import torch


class _VisionRoPEPrecision:
    """Cast attention inputs locally; retain the original FP32 RoPE producer."""

    def __init__(self, visual):
        self.active = False
        self.handles = [
            block.attn.register_forward_pre_hook(self._prepare_attention, with_kwargs=True)
            for block in visual.blocks
        ]

    def _prepare_attention(self, module, args, kwargs):
        if not self.active:
            return None
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        positions = kwargs.get("position_embeddings", args[2] if len(args) > 2 else None)
        if (
            not isinstance(hidden, torch.Tensor)
            or not isinstance(positions, (tuple, list))
            or len(positions) != 2
        ):
            raise ValueError("Joint SFT vision attention requires hidden states and a cos/sin pair")
        dtype = hidden.dtype
        if torch.is_autocast_enabled(hidden.device.type):
            dtype = torch.get_autocast_dtype(hidden.device.type)
        if all(value.dtype == dtype for value in positions):
            return None
        # Native mixed-precision training rounds these two inputs at each vision
        # block boundary. Casting the small inputs here preserves that behavior
        # without rounding the persistent/model-wide FP32 mathematical constants.
        positions = tuple(value.to(dtype=dtype) for value in positions)
        if "position_embeddings" in kwargs:
            kwargs = {**kwargs, "position_embeddings": positions}
        else:
            args = (*args[:2], positions, *args[3:])
        return args, kwargs


@contextmanager
def joint_sft_vision_precision(vlm):
    """Use native vision input precision during a joint SFT forward only.

    Loss evaluation uses the same numerical path as training. Calls directly to
    the VLM, including inference/generation, are unchanged outside this scope.
    Joint SFT already disallows gradient checkpointing, so there is no deferred
    vision recomputation after this scope exits. Legacy component modes retain
    their existing behavior.
    """
    visual = getattr(getattr(vlm, "model", None), "visual", None)
    # Cosmos3-Edge's SigLIP2 encoder uses learned spatial embeddings, not
    # Qwen vision RoPE. Its native mixed-precision forward needs no adaptation.
    if visual is None or (
        getattr(getattr(visual, "config", None), "model_type", None) == "cosmos3_edge_vision"
    ):
        yield
        return
    precision = getattr(vlm, "_joint_sft_vision_precision", None)
    if precision is None:
        precision = _VisionRoPEPrecision(visual)
        vlm._joint_sft_vision_precision = precision
    previous = precision.active
    precision.active = True
    try:
        yield
    finally:
        precision.active = previous
