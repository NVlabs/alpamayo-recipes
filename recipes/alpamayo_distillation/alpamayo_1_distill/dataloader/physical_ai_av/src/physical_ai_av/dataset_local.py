# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import dataclasses
import json
import io
import os
import logging
import pathlib
import types
import zipfile
from typing import Any, Iterable

import huggingface_hub
import huggingface_hub.hf_api

import pandas as pd
from alpamayo1_distill.dataloader.physical_ai_av.src.physical_ai_av import egomotion, video
from alpamayo1_distill.dataloader.physical_ai_av.src.physical_ai_av.utils import hf_interface
from alpamayo1_distill.dataloader.physical_ai_av.src.physical_ai_av.dataset import PhysicalAIAVDatasetInterface, Features

logger = logging.getLogger(__name__)


class PhysicalAIAVDatasetInterfaceLocal(PhysicalAIAVDatasetInterface):
    """Interface for interacting with the PhysicalAI-Autonomous-Vehicles dataset on Hugging Face.

    See also the parent class `hf_interface.HfRepoInterface` for additional attributes.

    Attributes:
        revision (`str`): A Git revision id, which can be a branch name, a tag, or a commit hash
            (if not supplied at initialization, the latest commit hash on `main` will be used).
        token (`str | bool | None`): A valid user access token (string). Defaults to the locally
            saved token, which is the recommended method for authentication (see
            https://huggingface.co/docs/huggingface_hub/quick-start#authentication).
            To disable authentication, pass `False`.
        cache_dir (`str | pathlib.Path | None`): Path to the dir where cached files are stored.
        local_dir (`str | pathlib.Path | None`): If provided, downloaded files will be placed under
            this directory.
        confirm_download_threshold_gb (`float`): The threshold (in GB) of additional (uncached) file
            size beyond which the user is prompted for confirmation before downloading. Set to
            `float("inf")` to disable confirmation.
        features (`Features`): A representation of dataset features amenable to `.`-autocompletion.
        clip_index (`pd.DataFrame`): A clip index mapping `clip_id`s to `chunk` indices.
        sensor_presence (`pd.DataFrame`): A table mapping `clip_id`s to available sensors (notably,
            includes the radar config & radar sensor models for each clip).
        chunk_sensor_presence (`pd.DataFrame`): A table of sensor presence aggregated by chunk; used
            to determine which per-chunk packed files should exist in the dataset.
    """

    def __init__(
        self,
        revision: str | None = None,
        *,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        local_files_only: bool = False,
        confirm_download_threshold_gb: float = 10.0,
    ) -> None:
        self.token = token
        self.api = huggingface_hub.HfApi(token=self.token)
        self.fs = huggingface_hub.HfFileSystem(token=self.token)
        feature_dir = os.path.join(local_dir, 'features.csv')
        clip_index_dir = os.path.join(local_dir, 'clip_index.parquet')
        sensor_presence_dir = os.path.join(local_dir, "metadata/sensor_presence.parquet")
        features_df = pd.read_csv(feature_dir, index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(clip_index_dir)

        self.sensor_presence = pd.read_parquet(sensor_presence_dir)
        self.chunk_sensor_presence = (
            pd.concat(
                [self.clip_index[["chunk"]], self.sensor_presence.select_dtypes(include=bool)],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )
        self.repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"
        self.repo_type = "dataset"

        self.repo_snapshot_info = {
            "repo_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
            "repo_type": "dataset",
            "revision": None,
        }
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.local_files_only = local_files_only
        self.confirm_download_threshold_gb = confirm_download_threshold_gb

    def get_clip_feature(self, clip_id: str, feature: str, mode="rb", maybe_stream: bool=False) -> Any:
        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        chunk_filename = os.path.join(self.local_dir, chunk_filename)
        #with self.open_file(chunk_filename, maybe_stream=maybe_stream) as f:
        with open(chunk_filename, mode=mode) as f:
            if chunk_filename.endswith(".parquet"):
                return pd.read_parquet(f).loc[clip_id]
            elif chunk_filename.endswith(".zip"):
                clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(f, "r") as zf:
                    if feature == "egomotion":
                        egomotion_df = pd.read_parquet(
                            io.BytesIO(zf.read(clip_files_in_zip["egomotion"]))
                        )
                        return egomotion.EgomotionState.from_egomotion_df(
                            egomotion_df
                        ).create_interpolator(egomotion_df["timestamp"].to_numpy())
                    elif feature.startswith("camera"):
                        return video.SeekVideoReader(
                            video_data=io.BytesIO(zf.read(clip_files_in_zip["video"])),
                            timestamps=pd.read_parquet(
                                io.BytesIO(zf.read(clip_files_in_zip["frame_timestamps"]))
                            )["timestamp"].to_numpy(),
                        )
                    else:
                        logger.warning(
                            f"Feature-specific data reader for {feature=} not implemented yet."
                        )
                        return {
                            k: pd.read_parquet(io.BytesIO(zf.read(v)))
                            if v.endswith(".parquet")
                            else io.BytesIO(zf.read(v))
                            for k, v in self.features.get_clip_files_in_zip(
                                clip_id, feature
                            ).items()
                        }
            else:
                raise ValueError(f"Unexpected file extension: {chunk_filename=}.")
