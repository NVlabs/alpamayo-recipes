# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PhysicalAI-AV dataset adapter for Alpamayo 2 Super SFT."""

from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset


class PhysicalAIDataset(Dataset):
    """Load local PhysicalAI-AV clips and apply a configured SFT preprocessor."""

    DEFAULT_T0_US = 5_100_000

    def __init__(
        self,
        local_dir: str,
        chunk_ids: list[int] | str | None = None,
        model_config: Any | None = None,
        preprocess: dict[str, Any] | None = None,
        use_default_keyframe: bool = False,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        reasoning_metadata: str | None = None,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        time_step: float = 0.1,
        camera_features: list[str] | None = None,
        num_frames: int = 4,
        include_calibration: bool = False,
    ) -> None:
        self.dataset = PhysicalAIAVDatasetLocalInterface(
            local_dir=local_dir,
            chunk_ids=chunk_ids,
            features_metadata=features_metadata,
            clip_index_metadata=clip_index_metadata,
            reasoning_metadata=reasoning_metadata,
        )
        self.clip_ids = self.dataset.get_all_clip_ids()
        self.use_default_keyframe = use_default_keyframe
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.time_step = time_step
        self.camera_features = camera_features
        self.num_frames = num_frames
        self.include_calibration = include_calibration

        if isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        self.preprocess = (
            instantiate(preprocess, model_config=model_config)
            if preprocess is not None
            else None
        )

    def __len__(self) -> int:
        """Return the number of indexed clips."""
        return len(self.clip_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one clip at its configured keyframe."""
        clip_id = self.clip_ids[index]
        t0_us = (
            self.DEFAULT_T0_US
            if self.use_default_keyframe
            else self.dataset.get_clip_key_frame(clip_id)
        )
        sample = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            avdi=self.dataset,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
            camera_features=self.camera_features,
            num_frames=self.num_frames,
            include_calibration=self.include_calibration,
        )
        for key in tuple(sample):
            if key.startswith("ego_"):
                sample[key] = sample[key].squeeze(0)
        if self.dataset.reasoning_db is not None:
            sample.update(self.dataset.get_reasoning_data(clip_id, t0_us))
        if self.preprocess is not None:
            sample["tokenized_data"] = self.preprocess(data=sample)
        return sample
