# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the boundary between PAI samples and the native Edge processor."""

from types import SimpleNamespace

import pytest
import torch
from alpamayo2edge_sft import processor as processor_module
from alpamayo2edge_sft.label_mask import get_trajectory_mask
from alpamayo2edge_sft.processor import (
    Alpamayo2EdgeCollator,
    Alpamayo2EdgeProcessor,
    get_assistant_mask,
)
from transformers import Cosmos3EdgeImageProcessor


class _Tokenizer:
    padding_side = "left"
    eos_token_id = 3

    def convert_ids_to_tokens(self, token_id):
        assert token_id == 10
        return "<future>"

    def pad(self, samples, return_tensors, padding, max_length=None):
        length = (
            max_length if padding == "max_length" else max(len(x["input_ids"]) for x in samples)
        )
        return {
            key: torch.stack(
                [
                    torch.nn.functional.pad(sample[key], (length - len(sample[key]), 0))
                    for sample in samples
                ]
            )
            for key in ("input_ids", "attention_mask")
        }


class _NativeProcessor:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.image_processor = Cosmos3EdgeImageProcessor(
            size={"shortest_edge": 4, "longest_edge": 4}, patch_size=1, merge_size=2,
        )

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages[-1]["content"]) == 1
        assert kwargs["add_vision_id"] is False
        return "prefix" + messages[-1]["content"][0]["text"]

    def __call__(self, **kwargs):
        self.inputs = kwargs
        images = self.image_processor(
            images=kwargs["images"], return_tensors="pt", do_resize=False,
            do_rescale=kwargs["do_rescale"], do_normalize=kwargs["do_normalize"],
        )
        return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2), **images}


def _processor():
    processor = object.__new__(Alpamayo2EdgeProcessor)
    processor.model_config = SimpleNamespace(
        tokens_per_history_traj=45,
        tokens_per_future_traj=48,
        include_camera_ids=True,
        frame_label="relative_time",
        token_layout="camera_ts",
        traj_ids={"future_start": 10, "future_end": 11},
    )
    processor.processor = _NativeProcessor()
    return processor


@pytest.mark.parametrize("generation_mode", [False, True])
def test_processor_keeps_flat_images_and_times_aligned_and_preserves_native_image_processing(
    monkeypatch,
    generation_mode,
):
    captured = {}

    def conversation(**kwargs):
        captured.update(kwargs)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image, "camera_id": int(camera_id)}
                    for image, camera_id in zip(
                        kwargs["data"]["image_frames"],
                        kwargs["data"]["camera_indices"],
                        strict=True,
                    )
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<future>"},
                    {"type": "text", "text": "future-labels"},
                    {"type": "text", "text": "</future>"},
                ],
            },
        ]

    monkeypatch.setattr(processor_module, "build_conversation", conversation)
    processor = _processor()
    images = torch.tensor([3, 1, 2], dtype=torch.uint8).view(3, 1, 1, 1, 1).expand(3, 1, 3, 2, 2)
    processor.preprocess(
        data={
            "image_frames": images,
            "camera_indices": torch.tensor([1, 0, 1]),
            "absolute_timestamps": torch.tensor([5_100_000, 4_100_000, 4_100_000]),
        },
        components_order=["image", "traj_history", "prompt", "traj_future"],
        components_prompt=["traj_future"],
        label_components=["traj_future"],
        generation_mode=generation_mode,
    )
    assert captured["data"]["camera_indices"].tolist() == [0, 1, 1]
    assert captured["data"]["absolute_timestamps"].tolist() == [4_100_000, 4_100_000, 5_100_000]
    assert captured["frame_label"] == "relative_time"
    native_inputs = processor.processor.inputs
    assert all(image.dtype == torch.uint8 for image in native_inputs["images"])
    assert [image[0, 0, 0].item() for image in native_inputs["images"]] == [1, 2, 3]
    assert native_inputs["do_rescale"] is False
    assert native_inputs["do_normalize"] is False
    assert native_inputs["return_mm_token_type_ids"] is True
    expected = "prefix<future>" if generation_mode else "prefix<future>future-labels</future>"
    assert native_inputs["text"] == expected


def test_collator_masks_only_complete_future_trajectory_after_left_padding():
    collator = object.__new__(Alpamayo2EdgeCollator)
    collator.data_processor = _processor()
    collator.max_length = 8
    collator.pad_to_fixed_length = True
    samples = []
    for ids in ([2, 10, 12, 11, 3], [2, 4, 10, 12, 11, 3]):
        samples.append(
            {
                "tokenized_data": {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids)),
                    "pixel_values": torch.zeros(1, 3),
                    "_label_components": ["traj_future"],
                    "_generation_mode": False,
                },
                "ego_future_xyz": torch.zeros(1, 64, 3),
            }
        )
    batch = collator(samples)
    assert batch["labels_mask"].tolist() == [[False] * 4 + [True] * 4] * 2
    assert batch["traj_data"]["ego_future_xyz"].shape == (2, 1, 64, 3)
    assert batch["tokenized_data"]["pixel_values"].shape == (2, 3)


@pytest.mark.parametrize("ids", [[1, 10, 12], [1, 11, 10], [10, 12, 11, 10]])
def test_trajectory_mask_rejects_missing_or_duplicate_boundaries(ids):
    with pytest.raises(ValueError, match="complete future trajectory"):
        get_trajectory_mask(torch.tensor(ids), start_id=10, end_id=11)


@pytest.mark.parametrize("eos_token_id", [3, [3, 4]])
def test_trajectory_mask_includes_only_immediately_following_native_eos(eos_token_id):
    tokens = torch.tensor([1, 10, 12, 11, 3, 4])
    mask = get_trajectory_mask(tokens, start_id=10, end_id=11, eos_token_id=eos_token_id)
    assert mask.tolist() == [False, True, True, True, True, False]


def test_processor_rejects_right_padding_before_expert_batches_are_built(monkeypatch):
    monkeypatch.setattr(processor_module, "resolve_checkpoint_name_or_path", lambda config: "model")
    monkeypatch.setattr(
        processor_module,
        "build_alpamayo2_edge_tokenizer",
        lambda *args: _Tokenizer(),
    )
    config = SimpleNamespace(history_vocab_size=100, future_vocab_size=100, padding_side="right")
    with pytest.raises(ValueError, match="padding_side='left'"):
        Alpamayo2EdgeProcessor(config)


@pytest.mark.parametrize("order", [[2, 0, 1], [2, 0]])
@pytest.mark.parametrize("front_override", [True, False])
def test_camera_pixel_bounds_follow_message_order_and_native_grid_sizes(
    monkeypatch,
    order,
    front_override,
):
    def conversation(data, **kwargs):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": data["image_frames"][index],
                        "camera_id": int(data["camera_indices"][index]),
                    }
                    for index in order
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<future>x</future>"}]},
        ]

    monkeypatch.setattr(processor_module, "build_conversation", conversation)
    processor = _processor()
    native = Cosmos3EdgeImageProcessor(
        size={"shortest_edge": 163840, "longest_edge": 196608},
        patch_size=16,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    processor.processor.image_processor = native
    bounds = {0: [81920, 98304]}
    if front_override:
        bounds[1] = [163840, 196608]
    images = torch.stack(
        [torch.full((1, 3, 1080, 1920), value, dtype=torch.uint8) for value in [20, 30, 10]]
    )
    processor.preprocess(
        data={
            "image_frames": images,
            "camera_indices": torch.tensor([0, 1, 1]),
            "absolute_timestamps": torch.tensor([4_100_000, 4_100_000, 5_100_000]),
            "camera_pixel_bounds": bounds,
        },
        components_order=["image", "traj_history", "prompt", "traj_future"],
        components_prompt=["traj_future"],
        label_components=["traj_future"],
        generation_mode=False,
    )
    inputs = processor.processor.inputs
    assert inputs["do_resize"] is False
    assert inputs["do_rescale"] is False and inputs["do_normalize"] is False
    expected_shapes = [(224, 416), (320, 576), (320, 576)]
    assert [tuple(image.shape[-2:]) for image in inputs["images"]] == [
        expected_shapes[index] for index in order
    ]
    assert [int(image[0, 0, 0]) for image in inputs["images"]] == [[20, 30, 10][i] for i in order]
    processed = native(images=inputs["images"], do_resize=False, return_tensors="pt")
    visual_tokens = processed["image_grid_thw"].prod(dim=-1) // native.merge_size**2
    assert visual_tokens.tolist() == [[91, 180, 180][i] for i in order]
    assert processed["pixel_values"].is_floating_point()


def test_camera_pixel_bounds_reject_message_images_without_camera_identity(monkeypatch):
    def conversation(data, **kwargs):
        return [
            {"role": "user", "content": [{"type": "image", "image": data["image_frames"][0]}]},
            {"role": "assistant", "content": [{"type": "text", "text": "<future>x</future>"}]},
        ]

    monkeypatch.setattr(processor_module, "build_conversation", conversation)
    with pytest.raises(ValueError, match="camera_id"):
        _processor().preprocess(
            data={
                "image_frames": torch.zeros((1, 1, 3, 8, 8), dtype=torch.uint8),
                "camera_indices": torch.tensor([0]),
                "absolute_timestamps": torch.tensor([5_100_000]),
                "camera_pixel_bounds": {0: [81920, 98304]},
            },
            components_order=["image", "traj_history", "prompt", "traj_future"],
            components_prompt=["traj_future"],
            label_components=["traj_future"],
            generation_mode=False,
        )


def test_collator_left_pads_native_multimodal_token_types_with_the_text():
    collator = object.__new__(Alpamayo2EdgeCollator)
    collator.data_processor = _processor()
    collator.max_length = 8
    collator.pad_to_fixed_length = True
    samples = []
    for ids, mm_ids in [
        ([2, 10, 12, 11, 3], [1, 0, 0, 0, 0]),
        ([2, 4, 10, 12, 11, 3], [0, 1, 0, 0, 0, 0]),
    ]:
        samples.append(
            {
                "tokenized_data": {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                    "mm_token_type_ids": torch.tensor([mm_ids]),
                    "_label_components": ["traj_future"],
                    "_generation_mode": False,
                }
            }
        )
    batch = collator(samples)["tokenized_data"]
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape == (2, 8)
    assert batch["mm_token_type_ids"].tolist() == [[0, 0, 0, 1, 0, 0, 0, 0]] * 2
    assert not batch["mm_token_type_ids"][batch["attention_mask"] == 0].any()


@pytest.mark.parametrize("mm_ids", [torch.zeros(1, 3), None])
def test_collator_rejects_missing_or_misaligned_native_multimodal_token_types(mm_ids):
    collator = object.__new__(Alpamayo2EdgeCollator)
    collator.data_processor = _processor()
    collator.max_length = 8
    collator.pad_to_fixed_length = True
    samples = []
    for value in (torch.zeros(1, 4), mm_ids):
        tokenized = {
            "input_ids": torch.tensor([[2, 10, 12, 11]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "_label_components": ["traj_future"],
            "_generation_mode": False,
        }
        if value is not None:
            tokenized["mm_token_type_ids"] = value
        samples.append({"tokenized_data": tokenized})
    with pytest.raises(ValueError, match="mm_token_type_ids must align"):
        collator(samples)


class _ChatMLTokenizer(_Tokenizer):
    def encode(self, text, add_special_tokens):
        assert text == "<|im_start|>assistant\n"
        assert not add_special_tokens
        return [1, 7, 8, 9]  # A multi-token assistant role, followed by newline.

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 3}[token]


def test_full_assistant_mask_preserves_cot_and_trajectory_targets_after_padding():
    collator = object.__new__(Alpamayo2EdgeCollator)
    collator.data_processor = _processor()
    collator.data_processor.processor.tokenizer = _ChatMLTokenizer()
    collator.max_length = 18
    collator.pad_to_fixed_length = True
    ids = [1, 4, 9, 5, 3, 1, 7, 8, 9, 21, 22, 10, 12, 11, 3]
    batch = collator(
        [
            {
                "tokenized_data": {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                    "_label_components": ["assistant"],
                    "_generation_mode": False,
                }
            }
        ]
    )
    assert batch["labels_mask"].tolist() == [[False] * 12 + [True] * 6]
    assert batch["tokenized_data"]["input_ids"][batch["labels_mask"]].tolist() == [
        21, 22, 10, 12, 11, 3
    ]


@pytest.mark.parametrize("ids", [[1, 7, 8, 9, 10, 11], [1, 4, 9, 10, 11, 3]])
def test_full_assistant_mask_rejects_incomplete_or_missing_assistant_messages(ids):
    with pytest.raises(ValueError, match="Full-assistant"):
        get_assistant_mask(torch.tensor(ids), _ChatMLTokenizer())


def test_processor_rejects_discarding_cot_supervision():
    with pytest.raises(ValueError, match="discard other assistant targets"):
        _processor().preprocess(
            data={},
            components_order=[
                "image", "traj_history", "nav_instruction", "prompt", "cot", "traj_future"
            ],
            components_prompt=["cot", "traj_future"],
            label_components=["traj_future"],
            generation_mode=False,
        )


@pytest.mark.parametrize("include_reference", [False, True])
def test_cot_generation_omits_targets_before_building_conversation(monkeypatch, include_reference):
    processor = _processor()
    data = {
        "image_frames": torch.zeros((1, 1, 3, 8, 8), dtype=torch.uint8),
        "camera_indices": torch.tensor([0]),
        "absolute_timestamps": torch.tensor([5_100_000]),
        "nav_text": ["Turn left."],
    }
    if include_reference:
        # A non-string reference fails if the conversation tries to render it.
        data["cot"] = object()

    def conversation(data, components_order, **kwargs):
        assert components_order == ["image", "traj_history", "nav_instruction", "prompt"]
        return [{"role": "user", "content": [
            {"type": "image", "image": data["image_frames"][0]},
            {"type": "text", "text": data["nav_text"][0]},
        ]}]

    def template(messages, **kwargs):
        assert [message["role"] for message in messages] == ["user"]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["continue_final_message"] is False
        assert messages[0]["content"][1]["text"] == "Turn left."
        return "Turn left.<|im_start|>assistant\n"

    monkeypatch.setattr(processor_module, "build_conversation", conversation)
    monkeypatch.setattr(processor.processor, "apply_chat_template", template)
    processor.preprocess(
        data=data,
        components_order=[
            "image", "traj_history", "nav_instruction", "prompt", "cot", "traj_future"
        ],
        components_prompt=["cot", "traj_future"],
        label_components=["assistant"],
        generation_mode=True,
    )
    assert processor.processor.inputs["text"] == "Turn left.<|im_start|>assistant\n"


def test_generation_collator_keeps_stable_event_ids_outside_model_inputs():
    collator = object.__new__(Alpamayo2EdgeCollator)
    collator.data_processor = _processor()
    collator.max_length = 8
    collator.pad_to_fixed_length = False
    sample = {
        "clip_id": "example", "requested_t0_us": 2000000, "t0_us": 1999000,
        "tokenized_data": {
            "input_ids": torch.tensor([[2, 10]]), "attention_mask": torch.ones(1, 2),
            "_label_components": ["traj_future"], "_generation_mode": True,
        },
    }
    batch = collator([sample])
    assert batch["sample_metadata"] == [{"clip_id": "example", "requested_t0_us": 2000000, "t0_us": 1999000}]
    assert "clip_id" not in batch["tokenized_data"]
    assert "clip_id" not in batch["traj_data"]


def test_cosmos3_patch_normalization_matches_checkpoint_channel_layout():
    """Unequal channels expose the C2/C3 flattened-patch layout difference."""
    processor = Cosmos3EdgeImageProcessor(
        patch_size=2, merge_size=1, temporal_patch_size=1,
        image_mean=[0.1, 0.5, 0.9], image_std=[0.2, 0.4, 0.8],
    )
    pixels = torch.tensor([
        [[10, 20], [30, 40]], [[50, 60], [70, 80]], [[90, 100], [110, 120]],
    ], dtype=torch.uint8)
    patches = processor(
        images=[pixels], do_resize=False, do_rescale=False, do_normalize=False,
        return_tensors="pt",
    )["pixel_values"]
    assert patches.tolist() == [[10, 50, 90, 20, 60, 100, 30, 70, 110, 40, 80, 120]]
    actual = processor_module.normalize_image_patches(patches, processor)
    scale = 1.0 / processor.rescale_factor
    expected = pixels.float().permute(1, 2, 0)
    expected = (expected - torch.tensor(processor.image_mean) * scale) / (
        torch.tensor(processor.image_std) * scale
    )
    torch.testing.assert_close(actual, expected.reshape(1, -1), rtol=0, atol=0)


@pytest.mark.parametrize("patches", [torch.zeros(1, 12), torch.zeros(1, 24, dtype=torch.uint8)])
def test_cosmos3_patch_normalization_rejects_wrong_dtype_or_layout(patches):
    processor = Cosmos3EdgeImageProcessor(patch_size=2)
    with pytest.raises(ValueError, match="uint8 image patches"):
        processor_module.normalize_image_patches(patches, processor)
