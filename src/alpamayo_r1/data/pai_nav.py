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

"""PAI dataset with navigation / text annotations loaded from a JSON file.

Example JSON (``nav_demo_samples.json``)::

    [
      {
        "clip_id": "1ae17e2a-...",
        "t0": 29708785000,
        "nav_text": "Turn left in 11m",
        "cot": "Yield to the cross-traffic vehicle ..."
      },
      ...
    ]
"""

import json
from typing import Any

from alpamayo_r1.data.pai import PAIDataset
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


class PAIDatasetWithNav(PAIDataset):
    """PAIDataset driven by a JSON annotation file.

    Each entry in the JSON list becomes a separate sample. The ``t0`` field
    (microseconds) is used as the sampling timestamp, and text fields
    (``nav_text``, ``cot``, etc.) are merged into the sample dict.
    """

    def __init__(self, annotations_path: str, **kwargs: Any):
        """Initialize the dataset.

        Args:
            annotations_path: Path to a JSON file containing a list of sample
                dicts.  Each dict must have ``clip_id`` and ``t0``.
            **kwargs: Forwarded to :class:`PAIDataset` (``local_dir``,
                ``model_config``, ``vla_preprocess_args``, etc.).
        """
        super().__init__(**kwargs)

        with open(annotations_path) as f:
            self._samples: list[dict[str, Any]] = json.load(f)
        self.clip_ids = [s["clip_id"] for s in self._samples]

    def __len__(self) -> int:
        """Return the number of annotated samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """Load a sample using the annotation's clip_id and t0."""
        entry = self._samples[idx]
        clip_id = entry["clip_id"]
        t0_us = int(entry["t0_relative"])

        sample_data = load_physical_aiavdataset(
            clip_id,
            t0_us=t0_us,
            avdi=self.avdi,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            time_step=self.time_step,
        )

        sample_data["nav_text"] = entry["nav_text"]

        for key in sample_data.keys():
            if key.startswith("ego_"):
                sample_data[key] = sample_data[key].squeeze(0)

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
