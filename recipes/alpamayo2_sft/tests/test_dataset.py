# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Alpamayo 2 Super SFT dataset adapter."""

from typing import Any

from alpamayo2_sft import dataset as dataset_module
from alpamayo2_sft.dataset import PhysicalAIDataset


class _DatasetInterface:
    reasoning_db = None

    def get_clip_key_frame(self, clip_id: str) -> int:
        assert clip_id == "clip"
        return 5_100_000


def test_dataset_forwards_explicit_camera_and_frame_configuration(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_load(clip_id: str, **kwargs: Any) -> dict[str, Any]:
        captured["clip_id"] = clip_id
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(dataset_module, "load_physical_aiavdataset", fake_load)
    dataset = object.__new__(PhysicalAIDataset)
    dataset.dataset = _DatasetInterface()
    dataset.clip_ids = ["clip"]
    dataset.use_default_keyframe = False
    dataset.num_history_steps = 16
    dataset.num_future_steps = 64
    dataset.time_step = 0.1
    dataset.camera_features = [f"camera/camera_{index}" for index in range(7)]
    dataset.num_frames = 4
    dataset.include_calibration = False
    dataset.preprocess = None

    dataset[0]

    assert captured["camera_features"] == dataset.camera_features
    assert captured["num_frames"] == 4
    assert captured["include_calibration"] is False
