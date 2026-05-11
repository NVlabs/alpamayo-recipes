# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Local interface for loading PAI data from a local directory."""

import os
import json
import io
import pathlib
import zipfile
from typing import Any, Iterable
import pandas as pd
import numpy as np

from physical_ai_av import egomotion, video
from physical_ai_av.dataset import Features

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=False)
logger.setLevel("INFO")

# PAI clips are a fixed 20s relative timeline (µs); use this instead of per-row ``end_timestamp``.
CLIP_RELATIVE_DURATION_US = 20_000_000


class PhysicalAIAVDatasetLocalInterface:
    """Local filesystem interface for a PAI AV dataset.

    Loads features/clip-index/(optional) reasoning metadata from ``local_dir`` and
    exposes helpers to query clip ids, chunks, keyframes, reasoning, and features.
    """

    def __init__(
        self,
        local_dir: str | pathlib.Path,
        chunk_ids: list[int] | None = None,
        features_metadata: str = "features.csv",
        clip_index_metadata: str = "clip_index.parquet",
        start_safe_margin_seconds: float = 1.6,
        end_safe_margin_seconds: float = 6.4,
        reasoning_metadata: str | None = None,
    ) -> None:
        """Initialize the local PAI dataset interface.

        Args:
            local_dir: Path to the local directory containing the PAI dataset.
            chunk_ids: List of chunk IDs to load, or a range string (e.g. "0-9").
                      If None, all available chunks will be loaded.
        """
        self.local_dir = local_dir
        self.chunk_ids = None
        if chunk_ids is not None:
            if isinstance(chunk_ids, str) and "-" in chunk_ids:
                chunk_start = int(chunk_ids.split("-")[0])
                chunk_end = int(chunk_ids.split("-")[1])
                self.chunk_ids = list(range(chunk_start, chunk_end))
            elif isinstance(chunk_ids, (list, tuple, Iterable)):
                self.chunk_ids = list(chunk_ids)
            elif isinstance(chunk_ids, int):
                self.chunk_ids = [chunk_ids]
            else:
                logger.error(f"Invalid chunk_ids: {chunk_ids} {type(chunk_ids)}")
        else:
            logger.info("Loading all chunks")

        logger.info(f"Loading from {local_dir} with chunk_ids: {self.chunk_ids}")

        self.start_safe_margin_seconds = start_safe_margin_seconds
        self.end_safe_margin_seconds = end_safe_margin_seconds

        features_df = pd.read_csv(
            os.path.join(self.local_dir, features_metadata), index_col="feature"
        )
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(os.path.join(self.local_dir, clip_index_metadata))
        self.reasoning_db = None
        if reasoning_metadata is not None:
            reasoning_metadata_path = (
                reasoning_metadata
                if os.path.isabs(reasoning_metadata)
                else os.path.join(local_dir, reasoning_metadata)
            )
            if not os.path.exists(reasoning_metadata_path):
                raise ValueError(
                    f"[PAIDataset] Reasoning metadata file {reasoning_metadata_path} does not exist"
                )
            reasoning_metadata_df = pd.read_parquet(reasoning_metadata_path)
            self.reasoning_db = self._read_reasoning_data(reasoning_metadata_df)

        self.filter_clips_by_event_t0s()

        self.sensor_presence = pd.read_parquet(
            os.path.join(self.local_dir, "metadata/feature_presence.parquet")
        )
        self.chunk_sensor_presence = (
            pd.concat(
                [self.clip_index[["chunk"]], self.sensor_presence.select_dtypes(include=bool)],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )

    @staticmethod
    def _read_reasoning_data(reasoning_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Parse reasoning parquet: clip_id -> ``event_t0s`` (us) and ``cot`` from ``events``."""
        out: dict[str, dict[str, Any]] = {}
        if "events" not in reasoning_df.columns:
            return out
        for clip_id, events_cell in reasoning_df["events"].items():
            cid = str(clip_id)
            ts_list: list[int] = []
            cot_list: list[str] = []
            if events_cell is None or (np.isscalar(events_cell) and pd.isna(events_cell)):
                out[cid] = {"event_t0s": np.array([], dtype=np.int64), "cot": []}
                continue
            if isinstance(events_cell, str):
                if not events_cell.strip():
                    parsed: Any = []
                else:
                    parsed = json.loads(events_cell)
            else:
                parsed = events_cell
            if parsed is None or not hasattr(parsed, "__iter__") or len(parsed) == 0:
                out[cid] = {"event_t0s": np.array([], dtype=np.int64), "cot": []}
                continue
            for ev in parsed:
                if isinstance(ev, dict) and "event_start_timestamp" in ev:
                    ts_list.append(int(ev["event_start_timestamp"]))
                    cot_list.append(str(ev.get("cot", "")))
            out[cid] = {
                "event_t0s": np.asarray(ts_list, dtype=np.int64),
                "cot": cot_list,
            }
        return out

    def filter_clips_by_event_t0s(self) -> None:
        """Filter clip_index using ``self.reasoning_db`` per-clip ``event_t0s`` / ``cot``.

        Each reasoning entry comes from parsed ``events`` (``event_start_timestamp``,
        ``cot``). Keeps only events where ``event_t0 >= start_safe_margin_seconds``
        (in µs) and         ``event_t0 + end_safe_margin_seconds`` (in µs) <= ``CLIP_RELATIVE_DURATION_US``
        (20 s, fixed relative clip length). Drops rows
        with no reasoning entry or empty ``event_t0s`` after filtering. Writes aligned
        ``event_t0s`` and ``cot`` columns on ``clip_index``.
        """
        if self.reasoning_db is None or len(self.reasoning_db) == 0:
            return
        start_margin_us = int(self.start_safe_margin_seconds * 1_000_000)
        end_margin_us = int(self.end_safe_margin_seconds * 1_000_000)

        def filter_events(row: pd.Series) -> pd.Series:
            cid = str(row.name)
            entry = self.reasoning_db.get(cid)
            if entry is None:
                return pd.Series(
                    {"event_t0s": np.array([], dtype=np.int64), "cot": []},
                )
            arr = np.asarray(entry["event_t0s"], dtype=np.int64)
            cots = list(entry["cot"])
            if len(cots) < len(arr):
                cots.extend([""] * (len(arr) - len(cots)))
            elif len(cots) > len(arr):
                cots = cots[: len(arr)]
            mask = arr >= start_margin_us
            mask &= (arr + end_margin_us) <= CLIP_RELATIVE_DURATION_US
            idx = np.flatnonzero(mask)
            return pd.Series(
                {
                    "event_t0s": arr[mask],
                    "cot": [cots[i] for i in idx],
                },
            )

        filtered = self.clip_index.apply(filter_events, axis=1)
        self.clip_index["event_t0s"] = filtered["event_t0s"]
        self.clip_index["cot"] = filtered["cot"]
        non_empty = self.clip_index["event_t0s"].apply(lambda x: x is not None and len(x) > 0)
        removed = (~non_empty).sum()
        self.clip_index = self.clip_index.loc[non_empty]

        if removed > 0:
            logger.info(
                "[PAIDataset] filter_clip_index_by_event_t0s: removed %d rows with empty event_t0s after filtering, remaining %d rows",
                removed,
                len(self.clip_index),
            )

    def get_all_clip_ids(self):
        """Return all clip ids, filtered by ``self.chunk_ids`` if set."""
        if self.chunk_ids is not None:
            return self.clip_index.loc[self.clip_index["chunk"].isin(self.chunk_ids)].index.tolist()
        else:
            return self.clip_index.index.tolist()

    def get_clip_chunk(self, clip_id: str) -> int:
        """Returns the chunk index for `clip_id`."""
        return self.clip_index.at[clip_id, "chunk"]

    def get_clip_key_frame(self, clip_id: str, sample_index_in_clip: int = 0) -> np.int64:
        """Keyframe time (us) from filtered ``event_t0s`` on ``clip_index``."""
        if "event_t0s" in self.clip_index.columns:
            et0s = self.clip_index.at[clip_id, "event_t0s"]
            t0 = et0s[sample_index_in_clip]
            return np.asarray(t0, dtype=np.int64)

        if self.reasoning_db is not None and clip_id in self.reasoning_db:
            arr = self.reasoning_db[clip_id]["event_t0s"]
            t0 = arr[sample_index_in_clip]
            return np.asarray(t0, dtype=np.int64)
        raise KeyError(f"No event_t0s for {clip_id} (missing clip_index column and reasoning row)")

    def get_reasoning_data(self, clip_id: str, keyframe_timestamp: int) -> dict[str, Any]:
        """Get the reasoning data for a given clip_id and keyframe_timestamp."""
        if self.reasoning_db is not None and clip_id in self.reasoning_db:
            arr = self.reasoning_db[clip_id]["event_t0s"]
            matches = np.nonzero(arr == keyframe_timestamp)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"Event timestamp {keyframe_timestamp} not found for {clip_id} in reasoning_db."
                )
            return {"cot": self.reasoning_db[clip_id]["cot"][matches[0]]}
        else:
            return None

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        """Load a feature for ``clip_id`` from the on-disk parquet/zip chunk file.

        Returns ``None`` if the feature is not present in ``features_df``.
        """
        if feature not in self.features.features_df.index:
            logger.warning(
                "Feature %r is not in features_df (available: %s). Returning None.",
                feature,
                list(self.features.features_df.index),
            )
            return None
        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        chunk_filename = os.path.join(self.local_dir, chunk_filename)
        with open(chunk_filename, "rb") as f:
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