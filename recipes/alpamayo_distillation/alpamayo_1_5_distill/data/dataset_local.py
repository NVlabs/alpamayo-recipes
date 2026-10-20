# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-disk PhysicalAI-AV interface for cluster training."""

from __future__ import annotations

import io
import json
import logging
import os
import pathlib
import zipfile
from typing import Any

import huggingface_hub
import pandas as pd
from physical_ai_av import egomotion, video
from physical_ai_av.dataset import Features, PhysicalAIAVDatasetInterface

logger = logging.getLogger(__name__)


class PhysicalAIAVDatasetInterfaceLocal(PhysicalAIAVDatasetInterface):
    """Read PhysicalAI-AV features from a fully materialized local mirror."""

    def __init__(
        self,
        revision: str | None = None,
        *,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        local_files_only: bool = True,
        confirm_download_threshold_gb: float = 10.0,
    ) -> None:
        if local_dir is None:
            raise ValueError("local_dir is required for PhysicalAIAVDatasetInterfaceLocal")

        self.token = token
        self.api = huggingface_hub.HfApi(token=self.token)
        self.fs = huggingface_hub.HfFileSystem(token=self.token)

        features_path = os.path.join(local_dir, "features.csv")
        clip_index_path = os.path.join(local_dir, "clip_index.parquet")
        sensor_presence_path = os.path.join(local_dir, "metadata/sensor_presence.parquet")
        features_df = pd.read_csv(features_path, index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(clip_index_path)
        self.sensor_presence = pd.read_parquet(sensor_presence_path)
        self.chunk_sensor_presence = (
            pd.concat(
                [
                    self.clip_index[["chunk"]],
                    self.sensor_presence.select_dtypes(include=bool),
                ],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )
        self.repo_id = "nvidia/PhysicalAI-Autonomous-Vehicles"
        self.repo_type = "dataset"
        self.repo_snapshot_info = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": revision,
        }
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.local_files_only = local_files_only
        self.confirm_download_threshold_gb = confirm_download_threshold_gb

    def get_clip_feature(
        self,
        clip_id: str,
        feature: str,
        mode: str = "rb",
        maybe_stream: bool = False,
    ) -> Any:
        del maybe_stream
        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        chunk_filename = os.path.join(self.local_dir, chunk_filename)
        with open(chunk_filename, mode=mode) as feature_file:
            if chunk_filename.endswith(".parquet"):
                return pd.read_parquet(feature_file).loc[clip_id]
            if chunk_filename.endswith(".zip"):
                clip_files = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(feature_file, "r") as zip_file:
                    if feature == "egomotion":
                        egomotion_df = pd.read_parquet(
                            io.BytesIO(zip_file.read(clip_files["egomotion"]))
                        )
                        return egomotion.EgomotionState.from_egomotion_df(
                            egomotion_df
                        ).create_interpolator(egomotion_df["timestamp"].to_numpy())
                    if feature.startswith("camera"):
                        return video.SeekVideoReader(
                            video_data=io.BytesIO(zip_file.read(clip_files["video"])),
                            timestamps=pd.read_parquet(
                                io.BytesIO(zip_file.read(clip_files["frame_timestamps"]))
                            )["timestamp"].to_numpy(),
                        )
                    logger.warning(
                        "Feature-specific data reader for feature=%s is not implemented.",
                        feature,
                    )
                    return {
                        key: (
                            pd.read_parquet(io.BytesIO(zip_file.read(filename)))
                            if filename.endswith(".parquet")
                            else io.BytesIO(zip_file.read(filename))
                        )
                        for key, filename in clip_files.items()
                    }
            raise ValueError(f"Unexpected file extension: {chunk_filename=}.")
