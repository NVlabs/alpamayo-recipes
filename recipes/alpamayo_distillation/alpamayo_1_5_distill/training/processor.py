# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processor adapter for model-specific teacher and student backbones."""

from typing import Any

from alpamayo1_5 import helper as upstream
from transformers import AutoProcessor, AutoTokenizer

BASE_PROCESSOR_NAME = upstream.BASE_PROCESSOR_NAME
MAX_PIXELS = upstream.MAX_PIXELS
MIN_PIXELS = upstream.MIN_PIXELS
create_message = upstream.create_message
create_vqa_message = upstream.create_vqa_message
to_device = upstream.to_device

_QWEN_VISION_CONTROL_TOKEN_IDS = (151652, 151653, 151654, 151655)


def vqa_answer_stop_kwargs(model: Any) -> dict[str, Any]:
    """Build generation kwargs that terminate a structured VQA answer."""
    from alpamayo1_5.models.token_utils import StopAfterEOS, to_special_token
    from transformers import StoppingCriteriaList

    answer_end_id = model.tokenizer.convert_tokens_to_ids(to_special_token("answer_end"))
    banned_names = (
        "traj_history_start",
        "traj_history_end",
        "traj_future_start",
        "traj_future_end",
        "question_start",
        "question_end",
        "answer_start",
    )
    banned_ids = [
        model.tokenizer.convert_tokens_to_ids(to_special_token(name))
        for name in banned_names
    ]
    return {
        "stopping_criteria": StoppingCriteriaList(
            [StopAfterEOS(eos_token_id=answer_end_id)]
        ),
        "bad_words_ids": [
            [int(token_id)]
            for token_id in (*_QWEN_VISION_CONTROL_TOKEN_IDS, *banned_ids)
        ],
    }


def get_processor(
    tokenizer: AutoTokenizer,
    processor_name_or_path: str | None = None,
) -> AutoProcessor:
    """Load the matching image processor and install the model tokenizer."""
    processor: Any = AutoProcessor.from_pretrained(
        processor_name_or_path or BASE_PROCESSOR_NAME,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    processor.tokenizer = tokenizer
    return processor
