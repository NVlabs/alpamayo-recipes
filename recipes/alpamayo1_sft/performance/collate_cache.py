# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse a single QwenProcessor for collate to avoid per-batch reconstruction."""

from __future__ import annotations

from typing import Any

from alpamayo.processor.qwen_processor import QwenProcessor

_COLLATE_PROCESSORS: dict[tuple, QwenProcessor] = {}


def collate_fn_from_model_config_cached(
    data: list[dict[str, Any]],
    model_config=None,
    padding_side: str = "left",
    include_camera_ids: bool = False,
    include_frame_nums: bool = False,
    chat_template_version: str = "r1",
) -> dict[str, Any]:
    key = (
        model_config.vlm_name_or_path,
        model_config.traj_vocab_size,
        getattr(model_config, "min_pixels", None),
        getattr(model_config, "max_pixels", None),
        include_camera_ids,
        include_frame_nums,
        chat_template_version,
        padding_side,
    )
    if key not in _COLLATE_PROCESSORS:
        _COLLATE_PROCESSORS[key] = QwenProcessor(
            vlm_name_or_path=model_config.vlm_name_or_path,
            traj_vocab_size=model_config.traj_vocab_size,
            min_pixels=getattr(model_config, "min_pixels", None),
            max_pixels=getattr(model_config, "max_pixels", None),
            include_camera_ids=include_camera_ids,
            include_frame_nums=include_frame_nums,
            chat_template_version=chat_template_version,
        )
    return _COLLATE_PROCESSORS[key].collate_fn(data, padding_side=padding_side)
