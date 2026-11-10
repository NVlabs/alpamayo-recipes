# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contiguous Q/K normalization backward gradients during joint SFT.

Forward values and dtypes are unchanged. Noncontiguous upstream gradients incur
one contiguous copy per hooked normalization; its reduction order may change.
This control is independent of the optimizer's gradient-norm algorithm.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _attention_types():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
    return Qwen3Attention, Qwen3VLTextAttention


def _contiguous_gradient(gradient):
    # Same logical values, shape and dtype; choose token-major storage before
    # normalization backward reduces over tokens/heads for its scale gradient.
    return gradient.contiguous(memory_format=torch.contiguous_format)


class _JointQKGradientLayout:
    def __init__(self, model):
        self.model = model
        self.handles = []
        seen = set()
        attention_types = _attention_types()
        for attention in model.modules():
            if not isinstance(attention, attention_types):
                continue
            for norm in (attention.q_norm, attention.k_norm):
                if id(norm) in seen:
                    continue
                seen.add(id(norm))
                self.handles.append(norm.register_forward_hook(self._after_norm))

    def _after_norm(self, module, args, output):
        if (not self.model.training or getattr(self.model, "training_stage", None) != "joint"
                or not torch.is_grad_enabled()):
            return None
        if not isinstance(output, torch.Tensor):
            raise TypeError("Q/K normalization must return one tensor")
        if output.requires_grad:
            output.register_hook(_contiguous_gradient)
        return None  # Preserve the exact output object and forward graph.


def configure_joint_qk_gradient_layout(model, enabled=True):
    """Install idempotent instance hooks that also cover checkpoint recomputation.

    Hooks are inert in eval, legacy component modes and no-grad forwards. The
    persistent forward hook attaches to the actual recomputation output during
    reentrant backward. No module, parameter, buffer or class method is replaced.
    """
    if not isinstance(enabled, bool):
        raise ValueError("qk_gradient_layout must be a boolean")
    existing = getattr(model, "_joint_sft_qk_layout", None)
    if existing is not None:
        if enabled:
            return False
        for handle in existing.handles:
            handle.remove()
        del model._joint_sft_qk_layout
    if enabled:
        model._joint_sft_qk_layout = _JointQKGradientLayout(model)
    count = len(model._joint_sft_qk_layout.handles) if enabled else 0
    logger.info("Joint Q/K gradient layout: enabled=%s, normalization_modules=%s", enabled, count)
    if enabled and count == 0:
        logger.warning("Q/K gradient layout found no supported Qwen attention modules")
    return True
