# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache VLA preprocess outputs (chat template + image_processor) per (clip_id, t0_us)."""

from __future__ import annotations

import time
from typing import Any

import torch

_CACHE_STATS = {"hits": 0, "misses": 0}


def cache_stats() -> dict[str, int]:
    return dict(_CACHE_STATS)


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return value


def _clone_mapping(data: dict[str, Any]) -> dict[str, Any]:
    return {key: _clone_value(value) for key, value in data.items()}


_SAMPLE_SIDE_EFFECT_KEYS = (
    "relative_timestamps",
    "image_frames",
    "camera_indices",
    "label_components",
    "generation_mode",
)


def _capture_sample_fields(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: _clone_value(sample[key]) for key in _SAMPLE_SIDE_EFFECT_KEYS if key in sample}


def _apply_sample_fields(sample: dict[str, Any], fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        sample[key] = _clone_value(value)


def _cache_key(sample: dict[str, Any]) -> tuple[str, int]:
    clip_id = sample["clip_id"]
    t0_us = int(sample["t0_us"])
    return clip_id, t0_us


def enable_preprocess_cache(dataset: Any) -> None:
    """Wrap ``vla_preprocess_func`` with a per-dataset dict cache."""
    if getattr(dataset, "_preprocess_cache_enabled", False):
        return
    if dataset.vla_preprocess_func is None:
        return

    dataset._preprocess_cache: dict[tuple[str, int], dict[str, Any]] = {}
    dataset._preprocess_cache_enabled = True
    original = dataset.vla_preprocess_func

    def cached_preprocess(*args: Any, **kwargs: Any):
        sample = kwargs.get("data")
        if sample is None and args:
            sample = args[0]
        if not isinstance(sample, dict):
            raise TypeError("vla_preprocess_func expects a sample dict")

        key = _cache_key(sample)
        cache = dataset._preprocess_cache
        if key in cache:
            _CACHE_STATS["hits"] += 1
            entry = cache[key]
            _apply_sample_fields(sample, entry["sample_fields"])
            return _clone_mapping(entry["tokenized_data"])

        _CACHE_STATS["misses"] += 1
        tokenized_data = original(*args, **kwargs)
        cache[key] = {
            "tokenized_data": _clone_mapping(tokenized_data),
            "sample_fields": _capture_sample_fields(sample),
        }
        return _clone_mapping(cache[key]["tokenized_data"])

    cached_preprocess._preprocess_cache_wrapped = True  # type: ignore[attr-defined]
    dataset.vla_preprocess_func = cached_preprocess


def preload_preprocess_cache(dataset: Any, verbose: bool = True) -> dict[str, Any]:
    """Warm the preprocess cache before DataLoader workers fork (copy-on-write)."""
    if not getattr(dataset, "_preprocess_cache_enabled", False):
        enable_preprocess_cache(dataset)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    n = len(dataset)
    before = len(dataset._preprocess_cache)
    t0 = time.perf_counter()
    for idx in range(n):
        dataset[idx]
    elapsed = time.perf_counter() - t0
    after = len(dataset._preprocess_cache)

    stats = {
        "clips": n,
        "cache_entries": after,
        "new_entries": after - before,
        "preload_sec": elapsed,
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
    }
    if verbose and rank == 0:
        print(
            f"[preprocess_cache] preloaded {stats['cache_entries']} entries "
            f"from {n} clips in {elapsed:.1f}s",
            flush=True,
        )
    return stats
