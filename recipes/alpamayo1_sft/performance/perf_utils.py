# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for reading performance optimization flags."""

from __future__ import annotations

from functools import partial
from typing import Any

import hydra.utils as hyu
from omegaconf import DictConfig, OmegaConf

from alpamayo1_sft.performance.collate_cache import collate_fn_from_model_config_cached


def perf_plain(cfg: DictConfig) -> dict[str, Any]:
    perf = cfg.get("performance")
    if perf is None:
        return {}
    return OmegaConf.to_container(perf, resolve=True)  # type: ignore[return-value]


def build_collate_fn(cfg: DictConfig, model, perf: dict[str, Any]):
    chat_template_version = "r1"
    collate_cfg = cfg.data.get("collate_fn") or {}
    if "chat_template_version" in collate_cfg:
        chat_template_version = collate_cfg.chat_template_version

    if perf.get("collate_cache", False):
        return partial(
            collate_fn_from_model_config_cached,
            model_config=model.config,
            chat_template_version=chat_template_version,
        )
    return hyu.instantiate(cfg.data.collate_fn, _convert_="partial", model_config=model.config)
