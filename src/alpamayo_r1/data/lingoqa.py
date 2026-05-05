# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LingoQA VQA dataset for the OSS release.

Loads the publicly available LingoQA Scenery dataset
(https://github.com/wayveai/LingoQA) from its raw parquet + image format.

Internal counterpart: ``src/alpamayo/data/dataset/vqa_clip_cache_dataset.py``
driven by ``src/base_configs/data2/dataset/lingoqa_train.yaml`` with the
``add_vqa`` transform.  Internally the raw parquet is first converted into a
camera-keyed metadb; this OSS version reads the public format directly.

Raw parquet schema (``Scenery/train.parquet``)::

    question_id   str        unique QA-pair hash
    segment_id    str        video-segment hash (shared across ~42 QA pairs)
    images        ndarray    5 relative JPEG paths (front camera, 1 Hz)
    question      str        natural-language question
    answer        str        natural-language answer

Images live at ``{data_root}/images/train/{segment_id}/{0..4}.jpg``.

Example usage::

    dataset = LingoQADataset(
        data_root="/path/to/LingoQA/Scenery",
        model_config=model.config,
        vla_preprocess_args={...},
    )
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

import alpamayo_r1.common.constants as constants
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=False)

MAX_CHAR_LENGTH = 2048

FRONT_CAMERA_INDEX = constants.CAMERA_NAMES_TO_INDICES[constants.FRONT_WIDE_CAMERA_NAME]


class LingoQADataset(torch.utils.data.Dataset):
    """Image-VQA dataset backed by the public LingoQA Scenery parquet.

    Each row in the parquet is already a single (segment, QA-pair), so the
    dataset length equals the number of rows.
    """

    def __init__(
        self,
        data_root: str,
        parquet_name: str = "train.parquet",
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        n_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the dataset.

        Args:
            data_root: Root directory of the LingoQA Scenery download,
                e.g. ``/path/to/LingoQA/Scenery``.  Must contain the parquet
                file and an ``images/`` subdirectory.
            parquet_name: Filename of the parquet inside *data_root*.
            model_config: Model config forwarded to the VLA preprocessor.
            vla_preprocess_args: Hydra config dict to instantiate the VLA
                preprocessor.
            n_frames: Max frames to load per sample.  ``None`` loads all 5.
            **kwargs: Absorbs unused keys inherited from base config
                (e.g. ``local_dir``, ``chunk_ids``).
        """
        super().__init__()
        self.data_root = data_root
        self.n_frames = n_frames

        parquet_path = os.path.join(data_root, parquet_name)
        self.df = pd.read_parquet(parquet_path)
        logger.info(
            "Loaded LingoQA parquet from %s  (%d QA pairs, %d segments)",
            parquet_path,
            len(self.df),
            self.df["segment_id"].nunique(),
        )

        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        self.vla_preprocess_func: Callable[..., Any] | None = None
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        question: str = row["question"]
        answer: str = row["answer"]
        if len(question) > MAX_CHAR_LENGTH or len(answer) > MAX_CHAR_LENGTH:
            return self.__getitem__(random.randint(0, len(self) - 1))

        image_paths: np.ndarray = row["images"]
        if self.n_frames is not None and len(image_paths) > self.n_frames:
            image_paths = image_paths[-self.n_frames :]

        frames = []
        for rel_path in image_paths:
            full_path = os.path.join(self.data_root, rel_path)
            img = Image.open(full_path)
            arr = np.array(img)  # (H, W, 3)
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3, H, W)
        # (1_camera, N_frames, 3, H, W)
        image_frames = torch.stack(frames, dim=0).unsqueeze(0)

        n = image_frames.shape[1]
        relative_timestamps = torch.arange(n, dtype=torch.float32).unsqueeze(0)  # (1, N)
        camera_indices = torch.tensor([FRONT_CAMERA_INDEX], dtype=torch.int64)  # (1,)

        sample_data: dict[str, Any] = {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "relative_timestamps": relative_timestamps,
            "question": question,
            "answer": answer,
        }

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
