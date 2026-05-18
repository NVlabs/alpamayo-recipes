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

"""Reasoning graders for Alpamayo reasoning grading.

Provides a base class and concrete implementations:
- **LingoJudgeGrader**: wayveai/Lingo-Judge sequence classifier

All graders expose a unified ``score(predictions, references) -> Tensor[N]``
interface so callers are agnostic to the underlying model.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Sequence, Union

import torch

TextLike = Union[str, Sequence[str]]


def _to_string_list(texts: TextLike) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _resolve_device(device: str) -> torch.device:
    """Resolve device string, supporting ``"auto"`` for the local GPU.

    ``"auto"`` picks ``cuda:{LOCAL_RANK}`` when CUDA is available,
    falling back to ``cpu``.  This works correctly under ``torchrun``
    where each process has ``LOCAL_RANK`` set.
    """
    if device == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseReasoningGrader(ABC):
    """Abstract base for all reasoning graders.

    Subclasses must implement :meth:`score` which computes quality / similarity
    scores between predicted and reference reasoning texts.
    """

    def __init__(self, model_dir: str, *, device: str = "cpu"):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Reasoning grading model directory does not exist: {model_dir!r}. "
            )
        self.model_dir = model_dir
        self.device = _resolve_device(device)
        self._forward_lock = threading.Lock()

    @abstractmethod
    def score(self, predictions: TextLike, references: TextLike) -> torch.Tensor:
        """Score *predictions* against *references*.

        Args:
            predictions: Predicted reasoning text(s).
            references: Reference / ground-truth reasoning text(s).
                Must be broadcastable to the same length as *predictions*
                (i.e. same length, or a single string).

        Returns:
            Float32 tensor of shape ``[N]`` with scores (higher = better).
        """
        ...


# ---------------------------------------------------------------------------
# Lingo-Judge classification grader
# Clarification: this grader is just for example purpose to show how to use a
# reasoning grading model to compute the score of the reasoning text. Developers
# can implement their own reasoning grading model by subclassing the BaseReasoningGrader class.
# ---------------------------------------------------------------------------


class LingoJudgeGrader(BaseReasoningGrader):
    """wayveai/Lingo-Judge sequence-classification grader.

    Scores prediction–reference pairs using a fine-tuned text classifier.
    Output is a sigmoid probability indicating truthfulness / quality.
    """

    MAX_LEN = 256
    DEFAULT_QUESTION = "What is the reasoning process of driving?"

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cpu",
        question: str | None = None,
    ):
        super().__init__(model_dir, device=device)
        self.question = question or self.DEFAULT_QUESTION

        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(device=self.device)

    @torch.inference_mode()
    def score(self, predictions: TextLike, references: TextLike) -> torch.Tensor:
        pred_list = _to_string_list(predictions)
        ref_list = _to_string_list(references)

        if len(ref_list) == 1 and len(pred_list) > 1:
            ref_list = ref_list * len(pred_list)
        if len(pred_list) != len(ref_list):
            raise ValueError(
                f"predictions ({len(pred_list)}) and references ({len(ref_list)}) "
                "must have the same length (or references length 1 for broadcast)"
            )
        if len(pred_list) == 0:
            return torch.empty((0,), dtype=torch.float32)

        texts = [
            f"{self.tokenizer.cls_token}\nQuestion: {self.question}\nAnswer: {ref}\nStudent: {pred}"
            for pred, ref in zip(pred_list, ref_list)
        ]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_LEN,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self._forward_lock:
            output = self.model(**inputs)
            logits = output.logits.squeeze(-1)

        return torch.sigmoid(logits.float())


# Backward-compatibility alias so existing imports keep working.
ReasoningGrader = LingoJudgeGrader

GRADER_REGISTRY: dict[str, type[BaseReasoningGrader]] = {
    "lingo_judge": LingoJudgeGrader,
}


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_SINGLETON: BaseReasoningGrader | None = None
_SINGLETON_LOCK = threading.Lock()


def get_reasoning_grader(
    model_dir: str | None = None,
    *,
    device: str | None = None,
    grader_type: str | None = None,
) -> BaseReasoningGrader:
    """Get per-process singleton reasoning grader.

    Resolution order for each parameter: explicit arg > env var > default.
    """
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                resolved_dir = model_dir or os.getenv("ALPAMAYO_REASONING_GRADING_MODEL_PATH")
                if not resolved_dir:
                    raise ValueError(
                        "Missing env var `ALPAMAYO_REASONING_GRADING_MODEL_PATH` "
                        "(local model directory)."
                    )
                resolved_device = device or os.getenv(
                    "ALPAMAYO_REASONING_GRADING_DEVICE",
                    "cpu",
                )
                resolved_type = grader_type or os.getenv(
                    "ALPAMAYO_REASONING_GRADER_TYPE",
                    "lingo_judge",
                )

                grader_cls = GRADER_REGISTRY.get(resolved_type)
                if grader_cls is None:
                    raise ValueError(
                        f"Unknown grader_type={resolved_type!r}. "
                        f"Available: {sorted(GRADER_REGISTRY)}"
                    )
                _SINGLETON = grader_cls(resolved_dir, device=resolved_device)
    else:
        if model_dir and getattr(_SINGLETON, "model_dir", None) != model_dir:
            print(
                "[ReasoningGrader] Warning: get_reasoning_grader() called with a different "
                f"model_dir after initialization. Using existing singleton: {_SINGLETON.model_dir}"
            )
    return _SINGLETON


def get_reasoning_grader_from_config(config: object | None) -> BaseReasoningGrader:
    """Resolve grader settings from Cosmos config and return singleton."""
    custom = getattr(config, "custom", {}) if config is not None else {}
    alp_custom = custom.get("alpamayo", {}) if isinstance(custom, dict) else {}
    if isinstance(alp_custom, dict):
        model_dir = alp_custom.get("reasoning_grading_model_path")
        device = alp_custom.get("reasoning_grading_device")
        grader_type = alp_custom.get("reasoning_grader_type")
    else:
        model_dir = None
        device = None
        grader_type = None
    return get_reasoning_grader(model_dir=model_dir, device=device, grader_type=grader_type)
