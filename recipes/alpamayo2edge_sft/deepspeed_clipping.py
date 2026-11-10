# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint ZeRO-3 clipping with an explicit FSDP-compatible FP32 norm policy."""

import inspect
import logging
import sys
from importlib.metadata import version
from types import MethodType

import torch

from .global_norm import (
    ZeroFp32GlobalNorm,
    fp32_clip_coefficient,
    validate_norm_algorithm,
    zero_norm_unsupported_reason,
)


_SUPPORTED_DEEPSPEED_VERSION = '0.19.1'
_MARKER = '_alpamayo2edge_fp32_clip_method'
_NORM_MARKER = '_alpamayo2edge_fp32_norm_method'
_POLICY_MARKER = '_alpamayo2edge_norm_policy'
logger = logging.getLogger(__name__)


def _install_fp32_clip_input(optimizer, model=None, norm_algorithm="engine"):
    """Adapt one already-created, validated ZeRO-3 optimizer instance."""
    validate_norm_algorithm(norm_algorithm)
    policy = getattr(optimizer, _POLICY_MARKER, None)
    installed = getattr(optimizer, _MARKER, None)
    if policy is not None:
        if installed is not None and optimizer.unscale_and_clip_grads is not installed:
            raise RuntimeError('The Edge clipping adapter was replaced after installation')
        if getattr(optimizer, _POLICY_MARKER)["requested"] != norm_algorithm:
            raise RuntimeError('The gradient norm policy changed after optimizer setup')
        norm_method = getattr(optimizer, _NORM_MARKER, None)
        if norm_method is not None and optimizer._get_norm_groups is not norm_method:
            raise RuntimeError('The Edge norm adapter was replaced after installation')
        return False
    computer = None
    reason = None
    if norm_algorithm != "engine":
        reason = ("the owning model is unavailable" if model is None
                  else zero_norm_unsupported_reason(model, optimizer))
        if reason is not None:
            if norm_algorithm == "fsdp_fp32":
                raise ValueError(f'fsdp_fp32 gradient norm is unsupported: {reason}')
            logger.warning('Joint gradient norm uses the engine algorithm: %s', reason)
        else:
            original_norm = optimizer._get_norm_groups
            try:
                inspect.signature(original_norm).bind()
            except (TypeError, ValueError) as error:
                raise RuntimeError('Unsupported DeepSpeed norm signature') from error
            computer = ZeroFp32GlobalNorm(model, optimizer)
    policy = dict(requested=norm_algorithm,
                  resolved="fsdp_fp32" if computer is not None else "engine",
                  fallback_reason=reason)
    if computer is None:
        # Engine and auto fallback must leave both backend methods untouched.
        setattr(optimizer, _POLICY_MARKER, policy)
        logger.info("Joint gradient norm policy: %s", policy)
        return True
    if optimizer.dtype != torch.bfloat16 or optimizer.gradient_accumulation_dtype != torch.float32:
        raise ValueError('Edge FP32 clipping requires BF16 computation and FP32 accumulation')
    if optimizer.loss_scale != 1.0:
        raise ValueError('Edge BF16 clipping requires the original unit loss scale')
    if any(p.dtype != torch.float32 for p in optimizer.fp32_partitioned_groups_flat):
        raise ValueError('Edge FP32 clipping requires FP32 optimizer masters')
    original = optimizer.unscale_and_clip_grads
    try:
        # DeepSpeed's NVTX decorator exposes (*args, **kwargs), without
        # __wrapped__. Validate the actual positional call, not parameter names.
        inspect.signature(original).bind(0, None)
    except (TypeError, ValueError) as error:
        raise RuntimeError('Unsupported DeepSpeed clipping signature') from error

    use_matched_clip = False

    if computer is not None:
        def norm_groups(self):
            nonlocal use_matched_clip
            use_matched_clip = False
            if self.clip_grad <= 0:
                return original_norm()
            if self.loss_scale != 1.0:
                raise RuntimeError('BF16 loss scaling changed after clipping setup')
            norm = computer()
            # Preserve DS's existing nonfinite group-sentinel behavior. Its
            # overflow check still runs before this method in the original step.
            if not bool(torch.isfinite(norm)):
                return original_norm()
            use_matched_clip = True
            # DS step only consumes the list as vector_norm(stack(norm_groups)).
            return [norm]
        norm_bound = MethodType(norm_groups, optimizer)

    def clip_with_fp32_norm(self, sub_group_id, total_norm):
        # Disabled clipping, unsupported backends and nonfinite norms retain
        # the existing engine path. DS still owns overflow, stepping and storage.
        if self.clip_grad <= 0 or not use_matched_clip:
            return original(sub_group_id, total_norm)
        if self.loss_scale != 1.0:
            raise RuntimeError('BF16 loss scaling changed after clipping setup')
        if not isinstance(total_norm, torch.Tensor) or total_norm.numel() != 1:
            raise RuntimeError('Expected the scalar DeepSpeed global gradient norm')
        if computer is not None and use_matched_clip and self.clip_grad != 1.0:
            gradient = self.fp32_partitioned_groups_flat[sub_group_id].grad
            if gradient is None or gradient.dtype != torch.float32:
                raise RuntimeError('Expected the actual FP32 optimizer subgroup gradient')
            # Direct division matches Torch for nonunit thresholds too; DS's
            # reciprocal-of-division expression can otherwise round differently.
            gradient.mul_(fp32_clip_coefficient(self.clip_grad, total_norm.float()))
            return None
        # At the recipe's unit threshold, the original DS expression is already
        # exactly the native expression. Preserve its actual multiplication path.
        return original(sub_group_id, total_norm.float())

    bound = MethodType(clip_with_fp32_norm, optimizer)
    optimizer.unscale_and_clip_grads = bound
    setattr(optimizer, _MARKER, bound)
    if computer is not None:
        optimizer._get_norm_groups = norm_bound
        setattr(optimizer, _NORM_MARKER, norm_bound)
    setattr(optimizer, _POLICY_MARKER, policy)
    logger.info("Joint gradient norm policy: %s", policy)
    return True


def configure_joint_deepspeed_clipping(model, norm_algorithm="auto"):
    """Install once on a joint-training DeepSpeed ZeRO-3 engine, otherwise no-op.

    The runtime imports are lazy so CPU/plain-Torch/FSDP use does not initialize
    DeepSpeed. This uses a version-guarded instance adapter because the pinned
    DeepSpeed API does not expose the model-ordered FSDP norm algorithm.
    """
    validate_norm_algorithm(norm_algorithm)
    engine_type = getattr(sys.modules.get('deepspeed.runtime.engine'), 'DeepSpeedEngine', None)
    if engine_type is None or not isinstance(model, engine_type):
        return False
    if getattr(model.module, 'training_stage', None) != 'joint':
        return False
    if model.zero_optimization_stage() != 3:
        reason = "the matched distributed norm requires ZeRO-3 or FSDP2"
        if norm_algorithm == "fsdp_fp32":
            raise ValueError(f"fsdp_fp32 gradient norm is unsupported: {reason}")
        policy = dict(requested=norm_algorithm, resolved="engine",
                      fallback_reason=reason if norm_algorithm == "auto" else None)
        previous = getattr(model.optimizer, _POLICY_MARKER, None)
        if previous is not None:
            if previous != policy:
                raise RuntimeError("The gradient norm policy changed after optimizer setup")
            return False
        setattr(model.optimizer, _POLICY_MARKER, policy)
        if policy["fallback_reason"] is not None:
            logger.warning("Joint gradient norm uses the engine algorithm: %s", reason)
        logger.info("Joint gradient norm policy: %s", policy)
        return True
    if getattr(model.optimizer, _POLICY_MARKER, None) is not None:
        return _install_fp32_clip_input(model.optimizer, model.module, norm_algorithm)
    if version('deepspeed') != _SUPPORTED_DEEPSPEED_VERSION:
        raise RuntimeError('Edge clipping adapter requires DeepSpeed 0.19.1; validate newer versions before enabling it')
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

    if not isinstance(model.optimizer, DeepSpeedZeroOptimizer_Stage3):
        raise RuntimeError('Expected the actual DeepSpeed ZeRO-3 optimizer')
    return _install_fp32_clip_input(model.optimizer, model.module, norm_algorithm)
