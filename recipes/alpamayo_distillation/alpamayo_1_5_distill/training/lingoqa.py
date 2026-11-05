# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public LingoQA loading, mixture sampling, and VQA distillation batches."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from alpamayo1_5.models.token_utils import StopAfterEOS, to_special_token
from PIL import Image
from torch.utils.data import Dataset, Sampler
from transformers import LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5_distill.data.dataset import recursive_collate
from alpamayo1_5_distill.data.sample_index import SampleIndex, SampleVisit, unpack_sample_index
from alpamayo1_5_distill.models.modeling_distill import (
    DistillAlpamayo1_5,
    ExpertLogitsProcessor,
)
from alpamayo1_5_distill.training import processor as helper
from alpamayo1_5_distill.training.pai_vqa_questions import PAI_VQA_QUESTIONS

FRONT_CAMERA_INDEX = 1
MAX_CHAR_LENGTH = 2048
_FLAT_MULTIMODAL_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
}
_QWEN_VISION_CONTROL_TOKEN_IDS = (151652, 151653, 151654, 151655)


class InvalidVQAGenerationError(RuntimeError):
    """Raised when a student VQA rollout does not close its answer."""


class LingoQADataset(Dataset[dict[str, Any]]):
    """Read the public LingoQA Scenery parquet and JPEG layout directly."""

    def __init__(
        self,
        data_root: str | Path,
        parquet_name: str = "train.parquet",
        n_frames: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        parquet_path = self.data_root / parquet_name
        if not parquet_path.is_file():
            raise FileNotFoundError(f"LingoQA parquet does not exist: {parquet_path}")
        if n_frames is not None and n_frames <= 0:
            raise ValueError("LingoQA n_frames must be positive or null")
        self.n_frames = n_frames
        frame = pd.read_parquet(parquet_path)
        required = {"images", "question", "answer"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"LingoQA parquet is missing columns: {sorted(missing)}")
        valid = frame["question"].str.len().le(MAX_CHAR_LENGTH)
        valid &= frame["answer"].str.len().le(MAX_CHAR_LENGTH)
        self.frame = frame.loc[valid].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError("LingoQA parquet contains no usable question-answer rows")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_paths = list(row["images"])
        if self.n_frames is not None:
            # An explicit limit retains the most recent part of the sequence;
            # the default preserves all five public LingoQA benchmark frames.
            image_paths = image_paths[-self.n_frames :]
        if not image_paths:
            raise ValueError(f"LingoQA row {index} has no images")

        frames = []
        for relative_path in image_paths:
            image_path = self.data_root / str(relative_path)
            with Image.open(image_path) as image:
                array = np.array(image.convert("RGB"), copy=True)
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return {
            "task_type": "vqa",
            "vqa_source": "lingoqa",
            "image_frames": torch.stack(frames).unsqueeze(0),
            "camera_indices": torch.tensor([FRONT_CAMERA_INDEX], dtype=torch.long),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "question_id": str(row.get("question_id", index)),
        }


class PAIVQADataset(Dataset[dict[str, Any]]):
    """Expose PhysicalAI-AV samples as VQA prompts with no fabricated GT."""

    def __init__(
        self,
        pai: Dataset,
        *,
        questions: list[str] | tuple[str, ...] = PAI_VQA_QUESTIONS,
        seed: int = 42,
    ) -> None:
        cleaned = tuple(
            dict.fromkeys(question.strip() for question in questions if question.strip())
        )
        if not cleaned:
            raise ValueError("PAI VQA needs at least one non-empty question")
        self.pai = pai
        self.questions = cleaned
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.pai)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.pai, "set_epoch"):
            self.pai.set_epoch(epoch)

    def _question_index(self, index: int) -> int:
        payload = f"{self.seed}:{self.epoch}:{index}".encode()
        random_bits = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return random_bits % len(self.questions)

    def __getitem__(self, index: SampleIndex) -> dict[str, Any]:
        sample = dict(self.pai[index])
        index, _ = unpack_sample_index(index)
        sample.update(
            task_type="vqa",
            vqa_source="pai",
            question=self.questions[self._question_index(index)],
            question_id=f"pai:{self.epoch}:{index}",
        )
        # PAI has no VQA labels. The frozen Alpamayo teacher generates the
        # supervision online, so carrying trajectory targets into this branch
        # is unnecessary and could make accidental trajectory training easier.
        sample.pop("answer", None)
        return sample


class TaskConcatDataset(Dataset[dict[str, Any]]):
    """Concatenate trajectory, LingoQA VQA, and optional PAI VQA streams."""

    def __init__(
        self,
        pai: Dataset,
        lingoqa: Dataset,
        pai_vqa: Dataset | None = None,
    ) -> None:
        self.pai = pai
        self.lingoqa = lingoqa
        self.pai_vqa = pai_vqa
        self.pai_length = len(pai)
        self.source_datasets = (pai, lingoqa) if pai_vqa is None else (pai, lingoqa, pai_vqa)
        self.source_lengths = tuple(len(source) for source in self.source_datasets)
        offsets = []
        total = 0
        for length in self.source_lengths:
            offsets.append(total)
            total += length
        self.source_offsets = tuple(offsets)
        self.total_length = total

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: SampleIndex) -> dict[str, Any]:
        index, visit_id = unpack_sample_index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        for source_id, (source, offset, length) in enumerate(
            zip(
                self.source_datasets,
                self.source_offsets,
                self.source_lengths,
                strict=True,
            )
        ):
            if index < offset + length:
                local_index = index - offset
                source_index = (
                    SampleVisit(local_index, visit_id)
                    if visit_id is not None and source_id != 1
                    else local_index
                )
                sample = dict(source[source_index])
                if source_id == 0:
                    sample["task_type"] = "trajectory"
                return sample
        raise AssertionError("dataset index resolution failed")

    def set_epoch(self, epoch: int) -> None:
        for source in self.source_datasets:
            if hasattr(source, "set_epoch"):
                source.set_epoch(epoch)


class WeightedTaskDistributedBatchSampler(Sampler[list[SampleIndex]]):
    """Interleave homogeneous trajectory and VQA batches across all ranks.

    Without PAI VQA, each source is traversed ``weight`` times per epoch, as in
    the original two-stream sampler. With PAI VQA, the existing LingoQA-sized
    VQA budget is retained and each VQA batch independently selects LingoQA or
    PAI. The schedule is shared by ranks, while each rank receives a disjoint
    DistributedSampler-style strided shard. PhysicalAI indices carry the
    scheduled global sample position as a visit ID, giving each repeat fresh timestamp
    and navigation draws while preserving exact resume of the sampling plan.
    """

    def __init__(
        self,
        data_source: TaskConcatDataset,
        *,
        batch_size: int,
        world_size: int,
        rank: int,
        seed: int,
        pai_weight: int = 1,
        lingoqa_weight: int = 1,
        pai_vqa_probability: float = 0.5,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid batch_size/world_size/rank")
        if not isinstance(pai_weight, int) or not isinstance(lingoqa_weight, int):
            raise TypeError("mixture weights must be integers")
        if pai_weight <= 0 or lingoqa_weight <= 0:
            raise ValueError("mixture weights must be positive integers")
        if not 0.0 <= pai_vqa_probability <= 1.0:
            raise ValueError("pai_vqa_probability must lie in [0, 1]")
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.weights = (pai_weight, lingoqa_weight)
        self.pai_vqa_probability = float(pai_vqa_probability)
        self.has_pai_vqa = data_source.pai_vqa is not None
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.sample_offset = 0
        self._source_lengths = data_source.source_lengths
        self._offsets = data_source.source_offsets
        self._batches_per_repeat = tuple(
            length // (self.world_size * self.batch_size)
            for length in self._source_lengths
        )
        if any(count == 0 for count in self._batches_per_repeat):
            raise ValueError("each mixture source needs at least one distributed batch")
        # Adding PAI VQA splits the existing VQA budget rather than adding a
        # third full-dataset traversal. This preserves the trajectory:VQA
        # exposure controlled by pai_weight and lingoqa_weight.
        self.num_batches = (
            pai_weight * self._batches_per_repeat[0]
            + lingoqa_weight * self._batches_per_repeat[1]
        )
        self.num_samples = self.num_batches * self.batch_size
        self.total_size = self.num_samples * self.world_size

    def _source_schedule(self) -> list[int]:
        trajectory_batches = self.weights[0] * self._batches_per_repeat[0]
        vqa_batches = self.weights[1] * self._batches_per_repeat[1]
        schedule = [0] * trajectory_batches
        if self.has_pai_vqa:
            choice_generator = torch.Generator().manual_seed(
                self.seed + self.epoch * 10_007 + 79_019
            )
            choose_pai = torch.rand(vqa_batches, generator=choice_generator).lt(
                self.pai_vqa_probability
            )
            schedule.extend(2 if selected else 1 for selected in choose_pai.tolist())
        else:
            schedule.extend([1] * vqa_batches)
        if self.shuffle:
            order_generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(schedule), generator=order_generator).tolist()
            schedule = [schedule[index] for index in order]
        return schedule

    @property
    def global_source_samples(self) -> tuple[int, ...]:
        schedule = self._source_schedule()
        global_batch_size = self.batch_size * self.world_size
        return tuple(
            schedule.count(source_id) * global_batch_size
            for source_id in range(len(self._source_lengths))
        )

    def __len__(self) -> int:
        return (self.num_samples - self.sample_offset) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.sample_offset = 0

    def set_sample_offset(self, sample_offset: int) -> None:
        sample_offset = int(sample_offset)
        if sample_offset % self.batch_size:
            raise ValueError("sample_offset must align to the per-rank batch size")
        if not 0 <= sample_offset <= self.num_samples:
            raise ValueError(f"sample_offset must be in [0, {self.num_samples}]")
        self.sample_offset = sample_offset

    def _rank_batches(self, source_id: int, repeat: int) -> list[list[int]]:
        length = self._source_lengths[source_id]
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 10_007 + source_id * 1_009 + repeat
        )
        indices = (
            torch.randperm(length, generator=generator).tolist()
            if self.shuffle
            else list(range(length))
        )
        usable = self._batches_per_repeat[source_id] * self.world_size * self.batch_size
        rank_indices = indices[:usable][self.rank:usable:self.world_size]
        offset = self._offsets[source_id]
        return [
            [offset + index for index in rank_indices[start : start + self.batch_size]]
            for start in range(0, len(rank_indices), self.batch_size)
        ]

    def __iter__(self) -> Iterator[list[SampleIndex]]:
        schedule = self._source_schedule()
        source_ordinals = [0] * len(self._source_lengths)
        batch_cache: dict[tuple[int, int], list[list[int]]] = {}
        batches: list[list[SampleIndex]] = []
        global_batch_size = self.batch_size * self.world_size
        for batch_id, source_id in enumerate(schedule):
            ordinal = source_ordinals[source_id]
            source_ordinals[source_id] += 1
            repeat, batch_index = divmod(ordinal, self._batches_per_repeat[source_id])
            cache_key = (source_id, repeat)
            if cache_key not in batch_cache:
                batch_cache[cache_key] = self._rank_batches(source_id, repeat)
            batch = batch_cache[cache_key][batch_index]
            batches.append(
                [
                    SampleVisit(
                        index,
                        batch_id * global_batch_size + position * self.world_size + self.rank,
                    )
                    if source_id != 1
                    else index
                    for position, index in enumerate(batch)
                ]
            )
        skipped_batches = self.sample_offset // self.batch_size
        return iter(batches[skipped_batches:])


def _single_record_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] != 1:
        raise ValueError(f"VQA processing expects one sample, got {value.shape}")
    return value[0].detach().contiguous()


def _single_tokenized(key: str, value: torch.Tensor) -> torch.Tensor:
    if key in _FLAT_MULTIMODAL_KEYS:
        return value.detach().contiguous()
    return _single_record_tensor(value)


def _messages(sample: dict[str, Any], answer: str | None = None) -> list[dict[str, Any]]:
    num_frames = int(sample["image_frames"].shape[1])
    messages = helper.create_vqa_message(
        frames=sample["image_frames"].flatten(0, 1),
        question=str(sample["question"]),
        camera_indices=sample["camera_indices"],
        num_frames_per_camera=num_frames,
    )
    if answer is not None:
        messages[-1]["content"][0]["text"] += f"{answer}<|answer_end|>"
    return messages


def _process_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return helper.to_device(result, device=device)


def _gt_record(
    model: DistillAlpamayo1_5,
    sample: dict[str, Any],
    processor: Any,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    prompt = _process_messages(processor, _messages(sample), device)
    full = _process_messages(processor, _messages(sample, str(sample["answer"])), device)
    prompt_length = int(prompt["input_ids"].shape[1])
    sequence = full["input_ids"]
    if not torch.equal(prompt["input_ids"], sequence[:, :prompt_length]):
        raise ValueError("LingoQA full answer does not preserve the tokenized prompt prefix")
    labels = torch.full_like(sequence, -100)
    labels[:, prompt_length:] = sequence[:, prompt_length:]
    loss_mask = torch.zeros_like(sequence, dtype=torch.bool)
    loss_mask[:, prompt_length - 1 : sequence.shape[1] - 1] = True
    prompt_mask = torch.zeros_like(sequence, dtype=torch.bool)
    prompt_attention_mask = prompt.get("attention_mask")
    if isinstance(prompt_attention_mask, torch.Tensor):
        prompt_mask[:, :prompt_length] = prompt_attention_mask.bool()
    else:
        prompt_mask[:, :prompt_length] = True
    return {
        "task_type": "vqa",
        "tokenized_data": {key: _single_tokenized(key, value) for key, value in full.items()},
        "labels": _single_record_tensor(labels),
        "loss_mask": _single_record_tensor(loss_mask),
        "prompt_mask": _single_record_tensor(prompt_mask),
        "question_id": str(sample.get("question_id", "")),
    }


def _collate_records(records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    batch = recursive_collate(records)
    for key in _FLAT_MULTIMODAL_KEYS:
        values = [
            record["tokenized_data"][key]
            for record in records
            if key in record["tokenized_data"]
        ]
        if values:
            batch["tokenized_data"][key] = torch.cat(values, dim=0)
    batch["task_type"] = "vqa"
    return helper.to_device(batch, device=device)


@torch.no_grad()
def build_lingoqa_gt_batch(
    student: DistillAlpamayo1_5,
    samples: list[dict[str, Any]],
    *,
    processor: Any,
) -> dict[str, Any]:
    """Build an off-policy VQA batch supervised only by ground truth answers."""
    records = [_gt_record(student, sample, processor) for sample in samples]
    batch = _collate_records(records, next(student.parameters()).device)
    batch["on_policy"] = False
    return batch


@torch.no_grad()
def build_pai_vqa_teacher_batch(
    teacher: DistillAlpamayo1_5,
    samples: list[dict[str, Any]],
    *,
    student: DistillAlpamayo1_5,
    teacher_processor: Any,
    student_processor: Any,
    top_p: float = 0.98,
    temperature: float = 0.6,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Generate Alpamayo VQA answers and score them for CE plus dense KD.

    Prompt construction and generation arguments mirror the public Alpamayo
    1.5 VQA inference notebook. PAI samples intentionally have no human VQA
    label, so the frozen teacher's generated answer supplies the CE target and
    its logits over that exact sequence supply the KD target.

    PAI's questions are open-ended, and the teacher occasionally rambles well
    past any reasonable token budget without ever closing the answer with
    ``<|answer_end|>``. Rather than discard the whole batch whenever that
    happens, the truncated text is used as the pseudo-label: a cut-off answer
    still supervises training, whereas an invalid-rollout skip supervises
    nothing.
    """

    teacher_device = next(teacher.parameters()).device
    stop_kwargs = helper.vqa_answer_stop_kwargs(teacher)
    answer_end_id = teacher.tokenizer.convert_tokens_to_ids(to_special_token("answer_end"))
    records = []
    teacher_answers = []
    for sample in samples:
        if str(sample.get("vqa_source")) != "pai":
            raise ValueError("PAI VQA teacher batch received a non-PAI sample")
        teacher_prompt = _process_messages(
            teacher_processor,
            _messages(sample),
            teacher_device,
        )
        input_ids = teacher_prompt["input_ids"]
        prompt_length = input_ids.shape[1]
        generation_config = copy.deepcopy(teacher.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = 1
        generation_config.max_new_tokens = max_new_tokens
        generation_config.pad_token_id = teacher.tokenizer.pad_token_id
        generation_config.return_dict_in_generate = True
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        generation_inputs = {
            key: value for key, value in teacher_prompt.items() if key != "input_ids"
        }
        generated = teacher.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            **stop_kwargs,
            **generation_inputs,
        )
        gen_tokens = generated.sequences[0, prompt_length:]
        end_locations = (gen_tokens == answer_end_id).nonzero(as_tuple=False).flatten()
        answer_tokens = gen_tokens[: int(end_locations[0].item())] if end_locations.numel() else gen_tokens
        answer = teacher.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        if not answer:
            raise InvalidVQAGenerationError(
                "Alpamayo teacher did not generate a non-empty PAI VQA answer"
            )
        teacher_answers.append(answer)
        pseudo_labeled_sample = dict(sample, answer=answer)
        student_record = _gt_record(student, pseudo_labeled_sample, student_processor)
        teacher_full = _process_messages(
            teacher_processor,
            _messages(pseudo_labeled_sample, answer),
            teacher_device,
        )
        student_sequence = student_record["tokenized_data"]["input_ids"].unsqueeze(0)
        teacher_sequence = teacher_full["input_ids"]
        if not torch.equal(student_sequence, teacher_sequence.to(student_sequence.device)):
            raise ValueError("teacher and student processed PAI VQA answer token IDs differ")
        teacher_outputs = teacher.vlm(
            **teacher_full,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        student_record["teacher_logits"] = _single_record_tensor(
            teacher_outputs.logits
        ).to(torch.bfloat16)
        records.append(student_record)

    batch = _collate_records(records, next(student.parameters()).device)
    batch["on_policy"] = False
    batch["teacher_answers"] = teacher_answers
    return batch


def _answer_end_offset(sequence: torch.Tensor, answer_end_id: int, prompt_length: int) -> int:
    locations = (sequence[0, prompt_length:] == answer_end_id).nonzero(as_tuple=False).flatten()
    if locations.numel() != 1:
        raise InvalidVQAGenerationError(
            "student VQA rollout must contain exactly one <|answer_end|>"
        )
    return prompt_length + int(locations[0].item())


@torch.no_grad()
def build_lingoqa_on_policy_batch(
    teacher: DistillAlpamayo1_5,
    samples: list[dict[str, Any]],
    *,
    student: DistillAlpamayo1_5,
    teacher_processor: Any,
    student_processor: Any,
    sample_temperature: float,
    sample_top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Sample student VQA answers and have the teacher score those exact answers."""
    records = []
    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device
    for sample in samples:
        student_prompt = _process_messages(student_processor, _messages(sample), student_device)
        teacher_prompt = _process_messages(teacher_processor, _messages(sample), teacher_device)
        student_prompt_ids = student_prompt["input_ids"]
        teacher_prompt_ids = teacher_prompt["input_ids"]
        if not torch.equal(student_prompt_ids, teacher_prompt_ids.to(student_device)):
            raise ValueError("teacher and student processed VQA prompt token IDs differ")
        prompt_length = student_prompt_ids.shape[1]
        answer_end_id = student.tokenizer.convert_tokens_to_ids(to_special_token("answer_end"))
        banned_names = (
            "traj_history_start", "traj_history_end", "traj_future_start",
            "traj_future_end", "question_start", "question_end", "answer_start",
        )
        banned_ids = [
            student.tokenizer.convert_tokens_to_ids(to_special_token(name))
            for name in banned_names
        ]
        generation_config = copy.deepcopy(student.vlm.generation_config)
        generation_config.do_sample = True
        generation_config.temperature = sample_temperature
        generation_config.top_p = sample_top_p
        generation_config.top_k = None
        generation_config.num_beams = 1
        generation_config.num_return_sequences = 1
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        generation_config.max_new_tokens = max_new_tokens + 1
        generation_config.pad_token_id = student.tokenizer.pad_token_id
        generation_config.return_dict_in_generate = True
        generation_inputs = {
            key: value
            for key, value in student_prompt.items()
            if key != "input_ids"
        }
        was_training = student.vlm.training
        student.vlm.eval()
        try:
            generated = student.vlm.generate(
                input_ids=student_prompt_ids,
                generation_config=generation_config,
                stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=answer_end_id)]),
                logits_processor=LogitsProcessorList([
                    ExpertLogitsProcessor(
                        student.config.traj_token_start_idx,
                        student.config.traj_vocab_size,
                    )
                ]),
                bad_words_ids=[
                    [int(token_id)]
                    for token_id in (*_QWEN_VISION_CONTROL_TOKEN_IDS, *banned_ids)
                ],
                **generation_inputs,
            )
        finally:
            if was_training:
                student.vlm.train()
        sequence = generated.sequences
        answer_end_offset = _answer_end_offset(sequence, answer_end_id, prompt_length)
        student_data = {
            key: value
            for key, value in student_prompt.items()
            if key not in {"input_ids", "attention_mask"}
        }
        student_data.update(input_ids=sequence, attention_mask=torch.ones_like(sequence))
        teacher_data = {
            key: value
            for key, value in teacher_prompt.items()
            if key not in {"input_ids", "attention_mask"}
        }
        teacher_sequence = sequence.to(teacher_device)
        teacher_data.update(
            input_ids=teacher_sequence,
            attention_mask=torch.ones_like(teacher_sequence),
        )
        teacher_outputs = teacher.vlm(
            input_ids=teacher_sequence,
            **{key: value for key, value in teacher_data.items() if key != "input_ids"},
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        kl_mask = torch.zeros_like(sequence, dtype=torch.bool)
        kl_mask[:, prompt_length : answer_end_offset + 1] = True
        records.append({
            "task_type": "vqa",
            "tokenized_data": {
                key: _single_tokenized(key, value)
                for key, value in student_data.items()
            },
            "labels": _single_record_tensor(torch.full_like(sequence, -100)),
            "loss_mask": _single_record_tensor(torch.zeros_like(sequence, dtype=torch.bool)),
            "on_policy_kl_mask": _single_record_tensor(kl_mask),
            "teacher_logits": _single_record_tensor(teacher_outputs.logits).to(torch.bfloat16),
            "question_id": str(sample.get("question_id", "")),
        })
    batch = _collate_records(records, student_device)
    batch["on_policy"] = True
    return batch


@torch.no_grad()
def build_lingoqa_teacher_guided_gt_batch(
    teacher: DistillAlpamayo1_5,
    samples: list[dict[str, Any]],
    *,
    student: DistillAlpamayo1_5,
    teacher_processor: Any,
    student_processor: Any,
) -> dict[str, Any]:
    """Teacher-score GT answers as a safe fallback for invalid VQA rollouts."""
    records = []
    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device
    for sample in samples:
        student_record = _gt_record(student, sample, student_processor)
        teacher_full = _process_messages(
            teacher_processor,
            _messages(sample, str(sample["answer"])),
            teacher_device,
        )
        sequence = student_record["tokenized_data"]["input_ids"].unsqueeze(0)
        teacher_sequence = teacher_full["input_ids"]
        if not torch.equal(sequence, teacher_sequence.to(student_device)):
            raise ValueError("teacher and student processed VQA answer token IDs differ")
        teacher_outputs = teacher.vlm(
            **teacher_full,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        student_record["on_policy_kl_mask"] = student_record["labels"].ne(-100)
        student_record["labels"].fill_(-100)
        student_record["loss_mask"].fill_(False)
        student_record["teacher_logits"] = _single_record_tensor(
            teacher_outputs.logits
        ).to(torch.bfloat16)
        records.append(student_record)
    batch = _collate_records(records, student_device)
    batch["on_policy"] = True
    return batch
