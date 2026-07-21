# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer callback for lightweight per-step wall-clock benchmarking."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class StepTimeCallback(TrainerCallback):
    """Record step-end intervals so data loading and collation are included."""

    def __init__(
        self,
        output_path: str,
        stable_start_step: int = 5,
        stable_end_step: int = 20,
    ) -> None:
        self.output_path = Path(output_path)
        self.stable_start_step = stable_start_step
        self.stable_end_step = stable_end_step
        self._last_step_end: float | None = None
        self._step_begin: float | None = None
        self.records: list[dict[str, float | int]] = []

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._sync_cuda()
        self._step_begin = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._sync_cuda()
        now = time.perf_counter()
        record: dict[str, float | int] = {"step": int(state.global_step)}

        if self._step_begin is not None:
            record["compute_seconds"] = now - self._step_begin
        if self._last_step_end is not None:
            record["step_seconds"] = now - self._last_step_end

        self.records.append(record)
        self._last_step_end = now

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not state.is_world_process_zero:
            return

        stable_records = [
            record
            for record in self.records
            if self.stable_start_step <= record["step"] <= self.stable_end_step
            and "step_seconds" in record
        ]
        stable_step_seconds = [float(record["step_seconds"]) for record in stable_records]
        summary = {
            "stable_start_step": self.stable_start_step,
            "stable_end_step": self.stable_end_step,
            "stable_step_count": len(stable_step_seconds),
            "stable_avg_step_seconds": mean(stable_step_seconds) if stable_step_seconds else None,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": self.records}, f, indent=2)
