from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from alpamayo1_5_distill.data.sample_index import SampleIndex, SampleVisit, unpack_sample_index
from alpamayo1_5_distill.training.lingoqa import (
    InvalidVQAGenerationError,
    LingoQADataset,
    PAIVQADataset,
    TaskConcatDataset,
    WeightedTaskDistributedBatchSampler,
    _answer_end_offset,
    _messages,
    build_lingoqa_gt_batch,
    build_lingoqa_on_policy_batch,
    build_pai_vqa_teacher_batch,
)
from alpamayo1_5_distill.training.losses import DistillationLoss, DistillationLossConfig
from alpamayo1_5_distill.training.pai_vqa_questions import PAI_VQA_QUESTIONS
from PIL import Image
from torch.utils.data import Dataset


def test_lingoqa_default_preserves_all_five_benchmark_frames(tmp_path) -> None:
    relative_paths = []
    for frame_index in range(5):
        relative_path = f"images/train/segment/{frame_index}.jpg"
        image_path = tmp_path / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color=(frame_index, 0, 0)).save(image_path)
        relative_paths.append(relative_path)
    pd.DataFrame(
        [{
            "question_id": "q1",
            "images": relative_paths,
            "question": "What changed?",
            "answer": "The traffic light.",
        }]
    ).to_parquet(tmp_path / "train.parquet", index=False)

    sample = LingoQADataset(tmp_path)[0]
    messages = _messages(sample)
    user_content = messages[1]["content"]

    assert sample["image_frames"].shape == (1, 5, 3, 6, 8)
    assert sum(item["type"] == "image" for item in user_content) == 5
    assert any(item.get("text") == "frame 4 " for item in user_content)


class _RangeDataset(Dataset[dict[str, Any]]):
    def __init__(self, length: int, source: str) -> None:
        self.length = length
        self.source = source

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: SampleIndex) -> dict[str, Any]:
        index, visit_id = unpack_sample_index(index)
        return {"index": index, "visit_id": visit_id, "source": self.source}


def _sampler(rank: int) -> WeightedTaskDistributedBatchSampler:
    dataset = TaskConcatDataset(
        _RangeDataset(24, "pai"),
        _RangeDataset(12, "lingoqa"),
    )
    return WeightedTaskDistributedBatchSampler(
        dataset,
        batch_size=2,
        world_size=2,
        rank=rank,
        seed=42,
        pai_weight=1,
        lingoqa_weight=2,
    )


def test_weighted_sampler_keeps_batches_homogeneous_and_ranks_aligned() -> None:
    samplers = [_sampler(rank) for rank in range(2)]
    rank_batches = [list(sampler) for sampler in samplers]
    assert len(rank_batches[0]) == len(rank_batches[1]) == 12
    assert samplers[0].global_source_samples == (24, 24)

    source_counts = {"pai": 0, "lingoqa": 0}
    for batches_at_step in zip(*rank_batches, strict=True):
        sources = {
            "pai" if unpack_sample_index(index)[0] < 24 else "lingoqa"
            for batch in batches_at_step
            for index in batch
        }
        assert len(sources) == 1
        source_counts[sources.pop()] += 1
        rank_indices = [
            {unpack_sample_index(index)[0] for index in batch} for batch in batches_at_step
        ]
        assert rank_indices[0].isdisjoint(rank_indices[1])

    # 24 PAI examples at weight 1 and 12 LingoQA examples at weight 2.
    assert source_counts == {"pai": 6, "lingoqa": 6}


def test_weighted_sampler_resume_is_exact_suffix() -> None:
    sampler = _sampler(rank=0)
    full = list(sampler)
    sampler.set_epoch(0)
    sampler.set_sample_offset(6)
    assert list(sampler) == full[3:]


def test_pai_vqa_questions_are_deterministic_per_epoch_and_include_reasoning() -> None:
    dataset = PAIVQADataset(_RangeDataset(8, "pai"), seed=42)
    first = [dataset[index]["question"] for index in range(len(dataset))]
    assert first == [dataset[index]["question"] for index in range(len(dataset))]
    assert all(dataset[index]["vqa_source"] == "pai" for index in range(len(dataset)))
    assert "What is the reasoning process of driving?" in PAI_VQA_QUESTIONS

    dataset.set_epoch(1)
    second = [dataset[index]["question"] for index in range(len(dataset))]
    assert first != second


def test_three_stream_sampler_randomly_splits_fixed_vqa_budget() -> None:
    pai = _RangeDataset(24, "trajectory")
    lingoqa = _RangeDataset(120, "lingoqa")
    pai_vqa = _RangeDataset(24, "pai_vqa")
    dataset = TaskConcatDataset(pai, lingoqa, pai_vqa)
    samplers = [
        WeightedTaskDistributedBatchSampler(
            dataset,
            batch_size=2,
            world_size=2,
            rank=rank,
            seed=42,
            pai_weight=1,
            lingoqa_weight=1,
            pai_vqa_probability=0.5,
        )
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]

    assert len(rank_batches[0]) == len(rank_batches[1]) == 36
    trajectory, lingoqa_vqa, pai_vqa_samples = samplers[0].global_source_samples
    assert trajectory == 24
    assert lingoqa_vqa + pai_vqa_samples == 120
    assert lingoqa_vqa > 0
    assert pai_vqa_samples > 0

    for batches_at_step in zip(*rank_batches, strict=True):
        source_ids = set()
        for batch in batches_at_step:
            for index in batch:
                index, _ = unpack_sample_index(index)
                if index < 24:
                    source_ids.add("trajectory")
                elif index < 144:
                    source_ids.add("lingoqa")
                else:
                    source_ids.add("pai_vqa")
        assert len(source_ids) == 1
        rank_indices = [
            {unpack_sample_index(index)[0] for index in batch} for batch in batches_at_step
        ]
        assert rank_indices[0].isdisjoint(rank_indices[1])

    full = rank_batches[0]
    samplers[0].set_epoch(0)
    samplers[0].set_sample_offset(6)
    assert list(samplers[0]) == full[3:]


@pytest.mark.parametrize("epoch", [0, 3])
def test_weighted_repeats_get_distinct_visits_without_changing_exposure(epoch) -> None:
    dataset = TaskConcatDataset(_RangeDataset(24, "pai"), _RangeDataset(12, "lingoqa"))
    samplers = [
        WeightedTaskDistributedBatchSampler(
            dataset,
            batch_size=2,
            world_size=2,
            rank=rank,
            seed=42,
            pai_weight=4,
            lingoqa_weight=2,
        )
        for rank in range(2)
    ]
    for sampler in samplers:
        sampler.set_epoch(epoch)
    rank_batches = [list(sampler) for sampler in samplers]
    visits_by_index = {index: set() for index in range(24)}
    for step, batches_at_step in enumerate(zip(*rank_batches, strict=True)):
        for rank, batch in enumerate(batches_at_step):
            for position, key in enumerate(batch):
                if isinstance(key, SampleVisit):
                    assert key.visit_id == step * 4 + position * 2 + rank
                    visits_by_index[key.index].add(key.visit_id)
                else:
                    assert 24 <= key < 36
    assert all(len(visits) == 4 for visits in visits_by_index.values())
    assert samplers[0].global_source_samples == (96, 24)
    samplers[0].set_sample_offset(10)
    assert list(samplers[0]) == rank_batches[0][5:]


def test_task_concat_preserves_visits_for_both_physicalai_streams() -> None:
    pai = _RangeDataset(4, "pai")
    dataset = TaskConcatDataset(pai, _RangeDataset(2, "lingoqa"), PAIVQADataset(pai))
    trajectory = dataset[SampleVisit(1, 7)]
    lingoqa = dataset[SampleVisit(5, 7)]
    pai_vqa = dataset[SampleVisit(7, 8)]

    assert (trajectory["index"], trajectory["visit_id"]) == (1, 7)
    assert trajectory["task_type"] == "trajectory"
    assert (lingoqa["index"], lingoqa["visit_id"]) == (1, None)
    assert (pai_vqa["index"], pai_vqa["visit_id"]) == (1, 8)
    assert pai_vqa["vqa_source"] == "pai"
    assert dataset[SampleVisit(-3, 8)] == pai_vqa
    with pytest.raises(IndexError):
        dataset[SampleVisit(len(dataset), 8)]


def test_pai_vqa_uses_inference_camera_and_frame_layout() -> None:
    messages = _messages({
        "image_frames": torch.zeros(4, 4, 3, 6, 8),
        "camera_indices": torch.tensor([0, 1, 2, 3]),
        "question": "Describe the scene.",
    })
    user_content = messages[1]["content"]
    assert sum(item["type"] == "image" for item in user_content) == 16
    assert sum(item.get("text") == "frame 3 " for item in user_content) == 4


def test_answer_end_boundary_requires_exactly_one_marker() -> None:
    sequence = torch.tensor([[1, 2, 3, 9, 4]])
    assert _answer_end_offset(sequence, answer_end_id=9, prompt_length=2) == 3
    with pytest.raises(InvalidVQAGenerationError, match="exactly one"):
        _answer_end_offset(torch.tensor([[1, 2, 3]]), 9, 2)
    with pytest.raises(InvalidVQAGenerationError, match="exactly one"):
        _answer_end_offset(torch.tensor([[1, 2, 9, 9]]), 9, 2)


def test_off_policy_lingoqa_batch_contains_gt_labels_but_no_teacher_targets() -> None:
    class _Model:
        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(()))

    class _Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            text = messages[-1]["content"][0]["text"]
            input_ids = [1, 2, 3, 4, 5] if "<|answer_end|>" in text else [1, 2, 3]
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }

    batch = build_lingoqa_gt_batch(
        _Model(),
        [{
            "image_frames": torch.zeros(1, 1, 3, 4, 4),
            "camera_indices": torch.tensor([1]),
            "question": "What is ahead?",
            "answer": "A car.",
            "question_id": "q1",
        }],
        processor=_Processor(),
    )

    assert batch["on_policy"] is False
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    assert batch["loss_mask"].tolist() == [[False, False, True, True, False]]
    assert not any(key.startswith("teacher_") for key in batch)


def test_pai_vqa_teacher_answer_provides_ce_and_kd_targets() -> None:
    class _Model:
        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(()))

    class _Tokenizer:
        pad_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return 9 if "answer_end" in token else 10

        @staticmethod
        def decode(token_ids, skip_special_tokens=True) -> str:
            del token_ids, skip_special_tokens
            return "The ego vehicle should slow down."

    class _Teacher(_Model):
        tokenizer = _Tokenizer()

        class _VLM:
            calls: list[dict[str, Any]] = []
            generate_calls: list[dict[str, Any]] = []
            generation_config = type("GenerationConfig", (), {})()

            @classmethod
            def __call__(cls, **kwargs):
                cls.calls.append(kwargs)
                input_ids = kwargs["input_ids"]
                outputs = type("Outputs", (), {})()
                outputs.logits = torch.arange(
                    input_ids.numel() * 7,
                    dtype=torch.float32,
                ).reshape(*input_ids.shape, 7)
                return outputs

            @classmethod
            def generate(cls, **kwargs):
                cls.generate_calls.append(kwargs)
                generated = type("Generated", (), {})()
                generated.sequences = torch.tensor([[1, 2, 3, 4, 5, 9]])
                return generated

        vlm = _VLM()

    class _Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            text = messages[-1]["content"][0]["text"]
            input_ids = [1, 2, 3, 4, 5] if "<|answer_end|>" in text else [1, 2, 3]
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }

    batch = build_pai_vqa_teacher_batch(
        _Teacher(),
        [{
            "vqa_source": "pai",
            "image_frames": torch.zeros(4, 4, 3, 4, 4),
            "camera_indices": torch.tensor([0, 1, 2, 3]),
            "question": "What should the ego vehicle do next?",
            "question_id": "pai:0:1",
        }],
        student=_Model(),
        teacher_processor=_Processor(),
        student_processor=_Processor(),
        top_p=0.98,
        temperature=0.6,
        max_new_tokens=256,
    )

    assert batch["on_policy"] is False
    assert batch["teacher_answers"] == ["The ego vehicle should slow down."]
    assert batch["labels"].tolist() == [[-100, -100, -100, 4, 5]]
    assert batch["prompt_mask"].tolist() == [[True, True, True, False, False]]
    assert batch["teacher_logits"].shape == (1, 5, 7)
    assert batch["teacher_logits"].dtype == torch.bfloat16
    gen_call = _Teacher._VLM.generate_calls[-1]
    assert gen_call["generation_config"].top_p == 0.98
    assert gen_call["generation_config"].temperature == 0.6
    assert gen_call["generation_config"].max_new_tokens == 256
    assert gen_call["stopping_criteria"] is not None
    assert gen_call["bad_words_ids"] is not None
    assert _Teacher._VLM.calls[-1]["input_ids"].tolist() == [[1, 2, 3, 4, 5]]

    language = DistillationLoss(
        DistillationLossConfig(
            ce_weight=1.0,
            kd_weight=1.0,
            temperature=1.0,
            kd_include_prompt=True,
        )
    ).language_terms(
        {"student_logits": torch.zeros_like(batch["teacher_logits"])},
        batch,
    )
    assert set(language) == {"ce_loss", "kd_loss"}
    assert all(loss.isfinite() for loss in language.values())


def test_pai_vqa_teacher_batch_uses_truncated_answer_instead_of_raising() -> None:
    """If the teacher never emits <|answer_end|> within the token budget, the
    truncated generation is used as the pseudo-label rather than raising
    InvalidVQAGenerationError and dropping the whole batch."""

    class _Model:
        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(()))

    class _Tokenizer:
        pad_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return 9 if "answer_end" in token else 10

        @staticmethod
        def decode(token_ids, skip_special_tokens=True) -> str:
            del token_ids, skip_special_tokens
            return "The ego vehicle kept describing the scene without stopping"

    class _Teacher(_Model):
        tokenizer = _Tokenizer()

        class _VLM:
            calls: list[dict[str, Any]] = []
            generate_calls: list[dict[str, Any]] = []
            generation_config = type("GenerationConfig", (), {})()

            @classmethod
            def __call__(cls, **kwargs):
                cls.calls.append(kwargs)
                input_ids = kwargs["input_ids"]
                outputs = type("Outputs", (), {})()
                outputs.logits = torch.arange(
                    input_ids.numel() * 7,
                    dtype=torch.float32,
                ).reshape(*input_ids.shape, 7)
                return outputs

            @classmethod
            def generate(cls, **kwargs):
                cls.generate_calls.append(kwargs)
                generated = type("Generated", (), {})()
                # No answer_end (id 9) anywhere in the generated tokens: the
                # budget ran out before the teacher closed the answer.
                generated.sequences = torch.tensor([[1, 2, 3, 4, 5]])
                return generated

        vlm = _VLM()

    class _Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            text = messages[-1]["content"][0]["text"]
            input_ids = [1, 2, 3, 4, 5] if "<|answer_end|>" in text else [1, 2, 3]
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }

    batch = build_pai_vqa_teacher_batch(
        _Teacher(),
        [{
            "vqa_source": "pai",
            "image_frames": torch.zeros(4, 4, 3, 4, 4),
            "camera_indices": torch.tensor([0, 1, 2, 3]),
            "question": "What should the ego vehicle do next?",
            "question_id": "pai:0:1",
        }],
        student=_Model(),
        teacher_processor=_Processor(),
        student_processor=_Processor(),
        top_p=0.98,
        temperature=0.6,
        max_new_tokens=256,
    )

    assert batch["teacher_answers"] == [
        "The ego vehicle kept describing the scene without stopping"
    ]


def test_pai_vqa_on_policy_uses_student_rollout_and_teacher_scores() -> None:
    class _Tokenizer:
        pad_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return 9 if "answer_end" in token else 10

    class _StudentVLM:
        training = True
        generation_config = type("GenerationConfig", (), {})()
        calls: list[dict[str, Any]] = []

        @classmethod
        def eval(cls):
            cls.training = False

        @classmethod
        def train(cls):
            cls.training = True

        @classmethod
        def generate(cls, **kwargs):
            cls.calls.append(kwargs)
            generated = type("Generated", (), {})()
            generated.sequences = torch.tensor([[1, 2, 3, 4, 9]])
            return generated

    class _Student:
        tokenizer = _Tokenizer()
        vlm = _StudentVLM()
        config = type(
            "Config",
            (),
            {"traj_token_start_idx": 100, "traj_vocab_size": 8},
        )()

        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(()))

    class _TeacherVLM:
        @staticmethod
        def __call__(**kwargs):
            input_ids = kwargs["input_ids"]
            outputs = type("Outputs", (), {})()
            outputs.logits = torch.randn(*input_ids.shape, 11)
            return outputs

    class _Teacher:
        vlm = _TeacherVLM()

        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(()))

    class _Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del messages, kwargs
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
                "pixel_values": torch.ones(2, 3),
                "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
            }

    batch = build_lingoqa_on_policy_batch(
        _Teacher(),
        [{
            "vqa_source": "pai",
            "image_frames": torch.zeros(4, 4, 3, 4, 4),
            "camera_indices": torch.tensor([0, 1, 2, 3]),
            "question": "What should the ego vehicle do next?",
            "question_id": "pai:0:1",
        }],
        student=_Student(),
        teacher_processor=_Processor(),
        student_processor=_Processor(),
        sample_temperature=0.6,
        sample_top_p=0.98,
        max_new_tokens=16,
    )

    assert batch["on_policy"] is True
    assert batch["labels"].eq(-100).all()
    assert batch["on_policy_kl_mask"].tolist() == [
        [False, False, False, True, True]
    ]
    assert batch["teacher_logits"].shape == (1, 5, 11)
    assert _StudentVLM.calls[-1]["generation_config"].max_new_tokens == 17

    language = DistillationLoss(
        DistillationLossConfig(ce_weight=1.0, kd_weight=1.0)
    ).language_terms(
        {"student_logits": torch.zeros_like(batch["teacher_logits"])},
        batch,
    )
    assert set(language) == {"ce_loss", "kd_loss", "on_policy_kl_loss"}
    assert language["ce_loss"].item() == 0.0
    assert language["kd_loss"].item() == 0.0
    assert language["on_policy_kl_loss"].isfinite()
