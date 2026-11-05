# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import pytest
import torch
from alpamayo1_5_distill.training import factory


class _GenerationInspected(RuntimeError):
    pass


def test_navigation_text_respects_dataset_condition_selection() -> None:
    assert factory._navigation_text({"nav_text": "Turn left"}) == "Turn left"
    assert factory._navigation_text({"nav_text": "Turn left", "nav_enabled": True}) == "Turn left"
    assert factory._navigation_text({"nav_text": "Turn left", "nav_enabled": False}) is None


def test_shared_sample_message_threads_navigation_to_prompt(monkeypatch) -> None:
    captured = {}

    def fake_create_message(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(factory.helper, "create_message", fake_create_message)
    messages = factory._create_sample_message(
        {
            "image_frames": torch.zeros(2, 3, 4, 5, 6),
            "camera_indices": torch.tensor([0, 1]),
            "nav_text": "Turn right in 25m",
            "nav_enabled": True,
        }
    )

    assert messages == []
    assert captured["nav_text"] == "Turn right in 25m"
    assert captured["frames"].shape == (6, 4, 5, 6)


def test_teacher_logit_targets_support_dense_and_sparse_kd() -> None:
    logits = torch.tensor([[[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]]])

    dense = factory._build_teacher_logit_targets(
        logits,
        top_k=-1,
        temperature=0.5,
    )
    sparse = factory._build_teacher_logit_targets(
        logits,
        top_k=2,
        temperature=0.5,
    )

    assert set(dense) == {"teacher_logits"}
    torch.testing.assert_close(dense["teacher_logits"], logits)
    assert set(sparse) == {
        "teacher_topk_indices",
        "teacher_topk_logits",
        "teacher_residual_mass",
    }
    assert sparse["teacher_topk_logits"].shape == (1, 2, 2)
    assert (sparse["teacher_residual_mass"] > 0).all()
    assert (sparse["teacher_residual_mass"] < 1).all()


@pytest.mark.parametrize("top_k", [0, -2])
def test_teacher_logit_targets_reject_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        factory._build_teacher_logit_targets(
            torch.zeros(1, 1, 4),
            top_k=top_k,
            temperature=0.5,
        )


def test_teacher_vector_disabled_skips_action_expert() -> None:
    class FakeVLM:
        def __init__(self) -> None:
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                logits=torch.ones(1, 3, 5),
                hidden_states=(torch.ones(1, 3, 7),),
            )

    class FakeModel:
        def __init__(self) -> None:
            self.vlm = FakeVLM()

        @staticmethod
        def forward_distill(batch):
            del batch
            pytest.fail("teacher action expert should not run")

        @staticmethod
        def fuse_traj_tokens(input_ids, history):
            assert history
            return input_ids

    model = FakeModel()
    outputs = factory._forward_teacher_distillation_targets(
        model,
        {
            "tokenized_data": {
                "input_ids": torch.ones(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }
        },
        {"ego_history_xyz": torch.zeros(1, 1, 1, 3)},
        compute_teacher_vector=False,
    )

    assert set(outputs) == {"student_logits", "student_hidden"}
    assert model.vlm.kwargs["use_cache"] is False
    assert model.vlm.kwargs["output_hidden_states"] is True


def test_build_student_inference_batch_collates_flat_multimodal_inputs(
    monkeypatch,
) -> None:
    class FakeModel:
        tokenizer = object()
        config = SimpleNamespace(vlm_name_or_path="unused")

        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(()))

    processor = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.ones(4, 3),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        }
    )
    monkeypatch.setattr(factory.helper, "create_message", lambda **kwargs: [])
    samples = [
        {
            "image_frames": torch.zeros(1, 1, 1),
            "camera_indices": torch.zeros(1, dtype=torch.long),
            "ego_history_xyz": torch.zeros(1, 1, 2, 3),
            "ego_history_rot": torch.zeros(1, 1, 2, 3, 3),
            "ego_future_xyz": torch.zeros(1, 1, 4, 3),
            "ego_future_rot": torch.zeros(1, 1, 4, 3, 3),
            "clip_id": f"clip-{index}",
            "t0_us": 5_100_000,
        }
        for index in range(2)
    ]

    batch = factory.build_student_inference_batch(
        FakeModel(),
        samples,
        processor=processor,
    )

    assert batch["tokenized_data"]["input_ids"].shape == (2, 2)
    assert batch["tokenized_data"]["pixel_values"].shape == (8, 3)
    assert batch["tokenized_data"]["image_grid_thw"].shape == (2, 3)
    assert batch["ego_future_xyz"].shape == (2, 1, 4, 3)


def test_online_teacher_generation_uses_safe_greedy_contract(monkeypatch) -> None:
    marker_id = 42
    chat_eos_id = 99

    class FakeTokenizer:
        pad_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            token_ids = {
                "<|traj_future_start|>": marker_id,
                "<|answer_start|>": 201,
                "<|answer_end|>": 202,
                "<|question_start|>": 203,
                "<|question_end|>": 204,
            }
            return token_ids[token]

    class FakeVLM:
        generation_config = SimpleNamespace(
            eos_token_id=chat_eos_id,
            forced_eos_token_id=chat_eos_id,
            do_sample=True,
        )
        config = SimpleNamespace(eos_token_id=chat_eos_id)

        @staticmethod
        def generate(**kwargs):
            config = kwargs["generation_config"]
            assert config.do_sample is False
            assert config.num_beams == 1
            assert config.top_p == 1.0
            assert config.top_k is None
            assert config.temperature == 1.0
            assert config.eos_token_id is None
            assert config.forced_eos_token_id is None
            assert config.max_new_tokens == 17
            assert kwargs["do_sample"] is False
            assert kwargs["num_beams"] == 1
            assert kwargs["eos_token_id"] is None
            assert kwargs["forced_eos_token_id"] is None
            assert kwargs["max_new_tokens"] == 17
            assert kwargs["output_logits"] is False
            banned = {ids[0] for ids in kwargs["bad_words_ids"]}
            assert {151652, 151653, 151654, 151655, chat_eos_id} <= banned
            assert {201, 202, 203, 204} <= banned
            raise _GenerationInspected

    class FakeModel:
        tokenizer = FakeTokenizer()
        vlm = FakeVLM()
        config = SimpleNamespace(
            vlm_name_or_path="unused",
            traj_token_start_idx=100,
            traj_vocab_size=8,
        )

        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(()))

        @staticmethod
        def fuse_traj_tokens(input_ids, history):
            del history
            return input_ids

    processor = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
    )
    monkeypatch.setattr(factory.helper, "create_message", lambda **kwargs: [])
    sample = {
        "image_frames": torch.zeros(1, 1, 1),
        "camera_indices": torch.zeros(1, dtype=torch.long),
        "ego_history_xyz": torch.zeros(1, 1, 1, 3),
        "ego_history_rot": torch.zeros(1, 1, 1, 3, 3),
    }

    with pytest.raises(_GenerationInspected):
        factory._build_teacher_targets(
            FakeModel(),
            sample,
            max_new_tokens=16,
            processor=processor,
        )


def test_on_policy_student_generation_uses_sampling_contract() -> None:
    marker_id = 42
    chat_eos_id = 99

    class FakeTokenizer:
        pad_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            token_ids = {
                "<|traj_future_start|>": marker_id,
                "<|answer_start|>": 201,
                "<|answer_end|>": 202,
                "<|question_start|>": 203,
                "<|question_end|>": 204,
            }
            return token_ids[token]

    class FakeVLM:
        training = True
        generation_config = SimpleNamespace(eos_token_id=chat_eos_id)
        config = SimpleNamespace(eos_token_id=chat_eos_id)

        @staticmethod
        def eval():
            FakeVLM.training = False

        @staticmethod
        def train():
            FakeVLM.training = True

        @staticmethod
        def generate(**kwargs):
            config = kwargs["generation_config"]
            assert config.do_sample is True
            assert config.temperature == 0.7
            assert config.top_p == 0.9
            assert config.top_k is None
            assert config.max_new_tokens == 17
            assert kwargs["do_sample"] is True
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_p"] == 0.9
            assert kwargs["eos_token_id"] is None
            banned = {ids[0] for ids in kwargs["bad_words_ids"]}
            assert {151652, 151653, 151654, 151655, chat_eos_id} <= banned
            assert {201, 202, 203, 204} <= banned
            return SimpleNamespace(sequences=torch.tensor([[1, 2, marker_id, 7]]))

    model = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        vlm=FakeVLM(),
        config=SimpleNamespace(traj_token_start_idx=100, traj_vocab_size=8),
    )
    sequence = factory._generate_student_sequence(
        model,
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        },
        torch.tensor([[1, 2]]),
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=16,
    )

    torch.testing.assert_close(sequence, torch.tensor([[1, 2, marker_id, 7]]))
    assert FakeVLM.training


def test_invalid_teacher_generation_logs_sample_and_suffix(caplog) -> None:
    class FakeTokenizer:
        @staticmethod
        def decode(token_ids, **kwargs):
            assert token_ids == [10, 11, 12]
            assert kwargs["skip_special_tokens"] is False
            return "unfinished reasoning"

    model = SimpleNamespace(tokenizer=FakeTokenizer())
    sequence = torch.tensor([[1, 2, 10, 11, 12]])

    with caplog.at_level(logging.ERROR, logger=factory.__name__):
        factory._log_invalid_teacher_generation(
            model,
            {"clip_id": "clip-123", "t0_us": 5_100_000},
            sequence,
            prompt_length=2,
            max_new_tokens=256,
        )

    message = caplog.messages[-1]
    assert "clip_id='clip-123'" in message
    assert "t0_us=5100000" in message
    assert "generated_suffix_length=3" in message
    assert "configured_max_new_tokens=256" in message
    assert "generated_token_ids=[10, 11, 12]" in message
    assert "decoded_suffix='unfinished reasoning'" in message


def test_forbidden_teacher_tokens_are_checked_only_in_suffix() -> None:
    token_ids = {
        "<|answer_start|>": 10,
        "<|answer_end|>": 11,
        "<|question_start|>": 12,
        "<|question_end|>": 13,
    }
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: token_ids[token],
    )
    sequence = torch.tensor([[10, 12, 7, 11, 8, 13]])

    found = factory._forbidden_teacher_suffix_tokens(
        tokenizer,
        sequence,
        prompt_length=3,
    )

    assert found == {
        "answer_end": {"token_id": 11, "suffix_positions": [0]},
        "question_end": {"token_id": 13, "suffix_positions": [2]},
    }


def test_invalid_teacher_generation_logs_forbidden_token_details(caplog) -> None:
    model = SimpleNamespace(
        tokenizer=SimpleNamespace(
            decode=lambda *_args, **_kwargs: "<|answer_end|>",
        )
    )

    with caplog.at_level(logging.ERROR, logger=factory.__name__):
        factory._log_invalid_teacher_generation(
            model,
            {"clip_id": "clip-vqa", "t0_us": 42},
            torch.tensor([[1, 2, 11]]),
            prompt_length=2,
            max_new_tokens=512,
            reason="forbidden VQA token in generated suffix",
            forbidden_tokens={"answer_end": {"token_id": 11, "suffix_positions": [0]}},
        )

    message = caplog.messages[-1]
    assert "reason='forbidden VQA token in generated suffix'" in message
    assert "'answer_end': {'token_id': 11, 'suffix_positions': [0]}" in message
    assert "clip_id='clip-vqa'" in message
