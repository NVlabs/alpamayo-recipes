# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-sample cache for PAIDataset (load + reasoning + preprocess) per (clip_id, t0_us)."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import Dataset

_CACHE_STATS = {"hits": 0, "misses": 0}


def cache_stats() -> dict[str, int]:
    return dict(_CACHE_STATS)


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return value


def _clone_sample(sample: dict[str, Any] | None) -> dict[str, Any] | None:
    if sample is None:
        return None
    return {
        key: _clone_value(value)
        for key, value in sample.items()
        if not str(key).startswith("_dl_timing")
    }


def _t0_for_clip(dataset: Any, clip_id: str) -> int:
    if dataset.use_default_keyframe:
        return int(dataset.DEFAULT_T0_US)
    return int(dataset.avdi.get_clip_key_frame(clip_id))


def _cache_key(dataset: Any, idx: int) -> tuple[str, int]:
    clip_id = dataset.clip_ids[idx]
    return clip_id, _t0_for_clip(dataset, clip_id)


class LoadCachedDataset(Dataset):
    """Wrap a map-style dataset and cache full ``__getitem__`` outputs."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._cache: dict[tuple[str, int], dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def __getitem__(self, idx: int):
        key = _cache_key(self._dataset, idx)
        if key in self._cache:
            _CACHE_STATS["hits"] += 1
            return _clone_sample(self._cache[key])

        _CACHE_STATS["misses"] += 1
        sample = self._dataset[idx]
        self._cache[key] = _clone_sample(sample)
        return _clone_sample(self._cache[key])


def enable_load_cache(dataset: Dataset) -> Dataset:
    """Return a dataset wrapper with full-sample caching (idempotent)."""
    if isinstance(dataset, LoadCachedDataset):
        return dataset
    return LoadCachedDataset(dataset)


def preload_load_cache(dataset: Dataset, verbose: bool = True) -> dict[str, Any]:
    """Warm full-sample cache before DataLoader workers fork (copy-on-write)."""
    if not isinstance(dataset, LoadCachedDataset):
        dataset = enable_load_cache(dataset)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0

    n = len(dataset)
    before = len(dataset._cache)
    t0 = time.perf_counter()
    for idx in range(n):
        dataset[idx]
    elapsed = time.perf_counter() - t0
    after = len(dataset._cache)

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
            f"[load_cache] preloaded {stats['cache_entries']} samples "
            f"from {n} clips in {elapsed:.1f}s",
            flush=True,
        )
    return stats
