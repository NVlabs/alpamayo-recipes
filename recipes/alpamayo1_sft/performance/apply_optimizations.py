# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime training optimizations toggled via Hydra ``performance`` config."""

from __future__ import annotations

from typing import Any

import torch


def apply_runtime_optimizations(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply global PyTorch / CUDA knobs before model construction."""
    applied: dict[str, Any] = {}
    opt = cfg.get("optimizations") or {}

    if opt.get("tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        applied["tf32"] = True

    if opt.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        applied["cudnn_benchmark"] = True

    if opt.get("matmul_precision", None):
        precision = opt["matmul_precision"]
        torch.set_float32_matmul_precision(precision)
        applied["matmul_precision"] = precision

    return applied


def apply_model_optimizations(model: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply model-level optimizations after instantiation."""
    applied: dict[str, Any] = {}
    opt = cfg.get("optimizations") or {}

    if opt.get("channels_last", False):
        if hasattr(model, "vlm") and hasattr(model.vlm, "model"):
            visual = getattr(model.vlm.model, "visual", None)
            if visual is not None:
                converted = 0
                for module in visual.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                        module.to(memory_format=torch.channels_last)
                        converted += 1
                applied["channels_last_visual_conv2d"] = converted

    return applied


def enable_zip_cache() -> None:
    """Monkey-patch PAI chunk reader with a per-dataset-instance feature cache."""
    from alpamayo.data import pai_utils

    if getattr(pai_utils.PhysicalAIAVDatasetLocalInterface, "_zip_cache_enabled", False):
        return

    original = pai_utils.PhysicalAIAVDatasetLocalInterface.get_clip_feature

    def get_clip_feature_cached(self, clip_id: str, feature: str, maybe_stream: bool = False):
        cache = getattr(self, "_feature_cache", None)
        if cache is None:
            self._feature_cache = {}
            cache = self._feature_cache
        key = (clip_id, feature)
        if key not in cache:
            cache[key] = original(self, clip_id, feature, maybe_stream=maybe_stream)
        return cache[key]

    pai_utils.PhysicalAIAVDatasetLocalInterface.get_clip_feature = get_clip_feature_cached  # type: ignore[method-assign]
    pai_utils.PhysicalAIAVDatasetLocalInterface._zip_cache_enabled = True
