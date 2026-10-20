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

"""Direct PhysicalAI loading, with the frame/history convention used by SFT."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from physical_ai_av.dataset import Features, PhysicalAIAVDatasetInterface

from .contracts import image_paths


class LocalPhysicalAI(PhysicalAIAVDatasetInterface):
    """Use public feature decoding without HF lookups or implicit downloads.

    Like alpamayo.data.pai_utils.PhysicalAIAVDatasetLocalInterface in the SFT
    recipe, read the local features/clip index and native camera/egomotion ZIPs.
    The inherited get_clip_feature/get_clip_chunk need only these attributes
    and open_file; remote snapshot/download helpers are intentionally not used.
    """

    def __init__(self, root):
        self.local_dir = Path(root)
        features = pd.read_csv(self.local_dir / "features.csv", index_col="feature")
        features["clip_files_in_zip"] = features["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features)
        self.clip_index = pd.read_parquet(self.local_dir / "clip_index.parquet")
        if not self.clip_index.index.is_unique:
            raise ValueError("PhysicalAI clip index contains duplicate clip IDs")

    def open_file(self, filename, maybe_stream=False):
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected dataset-relative feature filename: {filename}")
        path = self.local_dir / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing PhysicalAI feature: {path}; download the selected chunks first"
            )
        return path.open("rb")

    def __repr__(self):
        return f"LocalPhysicalAI({str(self.local_dir)!r})"


class InputData:
    """Decode local camera-major inputs without exposing source answers to the target."""

    def __init__(self, args):
        self.root, self.source = args.data_root, args.data_source
        self.avdi = LocalPhysicalAI(self.root) if self.source == "physicalai" else None
        self.contract = {"source": self.source, "input_version": 1}

    def load(self, row: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return 16 camera-major RGB frames and ego history."""
        if self.avdi is not None:
            from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

            sample = load_physical_aiavdataset(
                row["name"],
                t0_us=row["t0_us"],
                avdi=self.avdi,
                maybe_stream=False,
                num_history_steps=16,
                num_future_steps=64,
                time_step=0.1,
                num_frames=4,
            )
            frames = sample["image_frames"]
            if frames.ndim != 5 or tuple(frames.shape[:3]) != (4, 4, 3):
                raise ValueError(f"Expected four frames per camera, got {tuple(frames.shape)}")
            frames = frames.flatten(0, 1)
            xyz, rot = sample["ego_history_xyz"], sample["ego_history_rot"]
            cameras = sample["camera_indices"].tolist()
            # No future trajectory, source CoC, or navigation is returned to the target.
        else:
            from PIL import Image
            from scipy.spatial.transform import Rotation

            images = []
            for relative in image_paths(row):
                with Image.open(self.root / relative) as image:
                    images.append(torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1))
            frames = torch.stack(images)
            xyz = torch.tensor([row[f"past_pos_{axis}"] for axis in "xyz"], dtype=torch.float32).T[
                None, None
            ]
            quat = np.array([row[f"past_q{axis}"] for axis in "xyzw"]).T
            rot = torch.tensor(Rotation.from_quat(quat).as_matrix(), dtype=torch.float32)[
                None, None
            ]
            cameras = [0, 1, 2, 6]

        if (
            cameras != [0, 1, 2, 6]
            or frames.dtype != torch.uint8
            or tuple(frames.shape[:2]) != (16, 3)
        ):
            raise ValueError(
                f"Invalid camera-major RGB input: {cameras}, {frames.shape}, {frames.dtype}"
            )
        if tuple(xyz.shape) != (1, 1, 16, 3) or tuple(rot.shape) != (1, 1, 16, 3, 3):
            raise ValueError("Expected 16 ego-history positions/rotations in the anchor frame")
        if not bool(torch.isfinite(xyz).all() and torch.isfinite(rot).all()):
            raise ValueError("Nonfinite ego history")
        return frames, {"ego_history_xyz": xyz, "ego_history_rot": rot}
