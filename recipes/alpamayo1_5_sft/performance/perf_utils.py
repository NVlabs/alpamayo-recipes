# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for reading performance optimization flags."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def perf_plain(cfg: DictConfig) -> dict[str, Any]:
    perf = cfg.get("performance")
    if perf is None:
        return {}
    return OmegaConf.to_container(perf, resolve=True)  # type: ignore[return-value]
