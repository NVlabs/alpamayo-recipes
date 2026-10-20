# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic distributed sampling and trainer-cursor serialization."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any

from torch.utils.data import DistributedSampler

_COMPATIBILITY_KEYS = (
    "seed",
    "world_size",
    "dataset_length",
    "per_rank_batch_size",
    "gradient_accumulation_steps",
    "sampler_epoch",
)


class ResumeDistributedSampler(DistributedSampler):
    """Deterministically shard an epoch and start at a rank-local offset."""

    def __init__(
        self,
        dataset: Sized,
        *,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if not drop_last:
            raise ValueError("ResumeDistributedSampler currently requires drop_last=True")
        super().__init__(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(shuffle),
            seed=int(seed),
            drop_last=bool(drop_last),
        )
        self.sample_offset = 0

    def __iter__(self) -> Iterator[int]:
        rank_indices = list(super().__iter__())
        return iter(rank_indices[self.sample_offset :])

    def __len__(self) -> int:
        return self.num_samples - self.sample_offset

    def set_epoch(self, epoch: int) -> None:
        """Select an epoch and reset its rank-local sample cursor."""

        super().set_epoch(int(epoch))
        self.sample_offset = 0

    def set_sample_offset(self, sample_offset: int) -> None:
        """Start iteration at a rank-local sample offset in this epoch."""

        sample_offset = int(sample_offset)
        if not 0 <= sample_offset <= self.num_samples:
            raise ValueError(
                f"sample_offset must be in [0, {self.num_samples}], got {sample_offset}"
            )
        self.sample_offset = sample_offset


def make_trainer_state(
    *,
    global_step: int,
    epoch: int,
    sample_offset: int,
    seed: int,
    world_size: int,
    dataset_length: int,
    per_rank_batch_size: int,
    gradient_accumulation_steps: int,
    sampler_epoch: int,
) -> dict[str, int]:
    """Build the JSON state needed to validate and restore a training cursor."""

    return {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "sample_offset": int(sample_offset),
        "seed": int(seed),
        "world_size": int(world_size),
        "dataset_length": int(dataset_length),
        "per_rank_batch_size": int(per_rank_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "sampler_epoch": int(sampler_epoch),
    }


def load_trainer_state(
    path: Path,
    *,
    expected: dict[str, int],
) -> dict[str, int]:
    """Load a cursor, accepting legacy batch cursors and checking compatibility."""

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    try:
        state = {
            "global_step": int(raw["global_step"]),
            "epoch": int(raw["epoch"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid trainer state in {path}") from error

    has_new_metadata = "sample_offset" in raw or any(key in raw for key in _COMPATIBILITY_KEYS)
    if has_new_metadata:
        missing = [key for key in ("sample_offset", *_COMPATIBILITY_KEYS) if key not in raw]
        if missing:
            raise ValueError(f"trainer state is missing compatibility metadata: {missing}")
        state["sample_offset"] = int(raw["sample_offset"])
        incompatible = {
            key: (int(raw[key]), int(expected[key]))
            for key in _COMPATIBILITY_KEYS[:-1]
            if int(raw[key]) != int(expected[key])
        }
        if incompatible:
            details = ", ".join(
                f"{key}=saved:{saved}/current:{current}"
                for key, (saved, current) in incompatible.items()
            )
            raise ValueError(f"incompatible resumed configuration: {details}")
        state["sampler_epoch"] = int(raw["sampler_epoch"])
        if state["sampler_epoch"] != state["epoch"]:
            raise ValueError("incompatible trainer state: sampler_epoch does not match epoch")
    elif "batch_in_epoch" in raw:
        state["sample_offset"] = int(raw["batch_in_epoch"]) * int(expected["per_rank_batch_size"])
        state["sampler_epoch"] = state["epoch"]
    else:
        raise ValueError("trainer state has neither sample_offset nor batch_in_epoch")

    if state["sample_offset"] < 0:
        raise ValueError("trainer state sample_offset must be non-negative")
    return state
