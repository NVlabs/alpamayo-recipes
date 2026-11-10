# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration checks against the installed Edge inference package, without API doubles."""

import os
from pathlib import Path

import pytest
import torch


@pytest.fixture
def release_conversations(request):
    if request.config.getoption("--edge-release-stub"):
        pytest.skip("This integration check requires the real Edge inference package")
    from alpamayo2_edge.chat_template import build_conversation as inference_conversation
    from alpamayo2edge_sft.processor import build_conversation as recipe_conversation

    return inference_conversation, recipe_conversation


def _sample():
    return {
        "image_frames": torch.zeros((3, 1, 3, 32, 32), dtype=torch.uint8),
        "camera_indices": torch.tensor([0, 1, 1]),
        "absolute_timestamps": torch.tensor([4_100_000, 4_100_000, 5_100_000]),
        "camera_pixel_bounds": {0: [1024, 1024], 1: [1024, 1024]},
        "nav_text": ["Continue straight."],
        "cot": "The lane ahead is clear.",
    }


def _text(message):
    return "".join(part["text"] for part in message["content"] if part["type"] == "text")


def _options(**kwargs):
    return {
        "num_tokens_per_history_traj": 45,
        "num_tokens_per_future_traj": 128,
        "components_order": ["image", "traj_history", "prompt", "traj_future"],
        "components_prompt": ["traj_future"],
        "generation_mode": False,
        "include_camera_ids": True,
        "frame_label": "relative_time",
        **kwargs,
    }


def test_nav_cot_sft_prefix_matches_actual_inference_conversation(release_conversations):
    inference, recipe = release_conversations
    sample = _sample()
    inference_messages = inference(sample, num_tokens_per_history_traj=45)
    messages = recipe(
        data=sample,
        **_options(
            components_order=[
                "image", "traj_history", "nav_instruction", "prompt", "cot", "traj_future"
            ],
            components_prompt=["cot", "traj_future"],
        ),
    )
    assert [_text(message) for message in messages[:2]] == [
        _text(message) for message in inference_messages
    ]
    inference_images = [p for p in inference_messages[1]["content"] if p["type"] == "image"]
    sft_images = [p for p in messages[1]["content"] if p["type"] == "image"]
    assert [p["camera_id"] for p in sft_images] == [0, 1, 1]
    for left, right in zip(inference_images, sft_images, strict=True):
        torch.testing.assert_close(left["image"], right["image"])
    assert _text(messages[-1]) == (
        "The lane ahead is clear.\n<|traj_future_start|>"
        + "<|traj_future|>" * 128 + "<|traj_future_end|>"
    )


def test_pai_sft_keeps_trajectory_only_task_and_exact_camera_labels(release_conversations):
    _, recipe = release_conversations
    sample = _sample()
    del sample["nav_text"], sample["cot"]
    messages = recipe(data=sample, **_options())
    assert _text(messages[1]) == (
        "Front left camera: <t=-1.0s>Front camera: <t=-1.0s><t=0.0s>\n"
        "<|traj_history_start|>" + "<|traj_history|>" * 45
        + "<|traj_history_end|>\nYour task: output the future trajectory."
    )
    assert _text(messages[-1]) == (
        "<|traj_future_start|>" + "<|traj_future|>" * 128 + "<|traj_future_end|>"
    )
    assert len(messages) == 3


@pytest.mark.parametrize("frame_label", ["none", "frame_num"])
def test_alternative_checkpoint_frame_labels_preserve_camera_grouping(
    release_conversations, frame_label
):
    _, recipe = release_conversations
    messages = recipe(data=_sample(), **_options(frame_label=frame_label))
    labels = _text(messages[1]).split("\n", 1)[0]
    expected = "Front left camera: Front camera: "
    if frame_label == "frame_num":
        expected = "Front left camera: frame 0 Front camera: frame 0 frame 1 "
    assert labels == expected


def test_conversation_honors_row_system_prompt_and_rejects_unknown_components(
    release_conversations,
):
    _, recipe = release_conversations
    sample = {**_sample(), "system_prompt": "Row-specific system instructions."}
    messages = recipe(data=sample, **_options(system_prompt="Configured instructions."))
    assert _text(messages[0]) == sample["system_prompt"]
    with pytest.raises(ValueError, match="Unsupported.*component"):
        recipe(
            data=sample,
            **_options(components_order=["missing_component", "prompt", "traj_future"]),
        )


@pytest.mark.parametrize("generation_mode", [False, True])
def test_actual_checkpoint_processor_preserves_trajectory_boundaries(
    release_conversations, generation_mode
):
    """Optional offline integration with complete local tokenizer/processor assets."""
    checkpoint = os.environ.get("ALPAMAYO2_EDGE_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set ALPAMAYO2_EDGE_TEST_CHECKPOINT to a complete local checkpoint")
    from alpamayo2_edge.config import Alpamayo2EdgeConfig
    from alpamayo2edge_sft.processor import Alpamayo2EdgeProcessor

    checkpoint = Path(checkpoint)
    assert checkpoint.is_dir(), "The requested local checkpoint must exist"
    config = Alpamayo2EdgeConfig.from_pretrained(checkpoint, local_files_only=True)
    # Model.from_pretrained supplies this resolved path; this test loads config only.
    config.name_or_path = str(checkpoint)
    processor = Alpamayo2EdgeProcessor(config)
    tokenized = processor.preprocess(
        data=_sample(),
        components_order=["image", "traj_history", "prompt", "traj_future"],
        components_prompt=["traj_future"],
        label_components=["traj_future"],
        generation_mode=generation_mode,
    )
    tokenizer = processor.processor.tokenizer
    ids = tokenized["input_ids"][0]
    future_start = tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    future_pad = tokenizer.convert_tokens_to_ids("<|traj_future|>")
    assert (ids == future_start).sum().item() == 1
    expected_future_tokens = 0 if generation_mode else config.tokens_per_future_traj
    assert (ids == future_pad).sum().item() == expected_future_tokens
    if generation_mode:
        assert ids[-1].item() == future_start
    assert tokenized["image_grid_thw"].shape == (3, 3)
    assert tokenized["pixel_values"].is_floating_point()
    assert tokenized["mm_token_type_ids"].shape == tokenized["input_ids"].shape
    assert processor.processor.image_processor.size == {
        "shortest_edge": config.min_pixels,
        "longest_edge": config.max_pixels,
    }


@pytest.mark.parametrize("navigation_input", [False, True])
def test_checkpoint_cot_generation_has_native_fresh_assistant_prompt(
    release_conversations, navigation_input
):
    checkpoint = os.environ.get("ALPAMAYO2_EDGE_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set ALPAMAYO2_EDGE_TEST_CHECKPOINT to a complete local checkpoint")
    from alpamayo2_edge.config import Alpamayo2EdgeConfig
    from alpamayo2edge_sft.processor import Alpamayo2EdgeProcessor

    config = Alpamayo2EdgeConfig.from_pretrained(checkpoint, local_files_only=True)
    config.name_or_path = checkpoint
    processor = Alpamayo2EdgeProcessor(config)
    sample = _sample()
    reference_cot = sample.pop("cot")
    components = ["image", "traj_history"]
    if navigation_input:
        components.append("nav_instruction")
    components.extend(["prompt", "cot", "traj_future"])
    tokenized = processor.preprocess(
        data=sample,
        components_order=components,
        components_prompt=["cot", "traj_future"],
        label_components=["assistant"],
        generation_mode=True,
    )
    tokenizer = processor.processor.tokenizer
    ids = tokenized["input_ids"][0]
    text = tokenizer.decode(ids)
    assert text.endswith("<|im_start|>assistant\n<think></think>")
    assert reference_cot not in text
    assert "<|traj_future" not in text
    assert ("Continue straight." in text) is navigation_input
    assert (ids == config.traj_ids["history_pad"]).sum() == config.tokens_per_history_traj
    if navigation_input:
        inference, _ = release_conversations
        messages = inference(sample, num_tokens_per_history_traj=config.tokens_per_history_traj)
        expected = processor.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=False,
            enable_thinking=False,
        )
        assert text == expected


@pytest.mark.parametrize("variant", ["nav", "cot", "nav_cot"])
@pytest.mark.parametrize("generation_mode", [False, True])
def test_checkpoint_processor_optional_tasks_supervise_text_without_generation_leakage(
    release_conversations, variant, generation_mode
):
    checkpoint = os.environ.get("ALPAMAYO2_EDGE_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set ALPAMAYO2_EDGE_TEST_CHECKPOINT to a complete local checkpoint")
    from alpamayo2_edge.config import Alpamayo2EdgeConfig
    from alpamayo2edge_sft.processor import Alpamayo2EdgeCollator, Alpamayo2EdgeProcessor

    config = Alpamayo2EdgeConfig.from_pretrained(checkpoint, local_files_only=True)
    config.name_or_path = checkpoint
    processor = Alpamayo2EdgeProcessor(config)
    nav, cot = "nav" in variant, "cot" in variant
    sample = _sample()
    options = {
        "components_order": ["image", "traj_history"]
        + (["nav_instruction"] if nav else [])
        + ["prompt"] + (["cot"] if cot else []) + ["traj_future"],
        "components_prompt": (["cot"] if cot else []) + ["traj_future"],
        "label_components": ["assistant"] if cot else ["traj_future"],
        "generation_mode": generation_mode,
    }
    tokenized = processor.preprocess(data=sample, **options)
    tokenizer = processor.processor.tokenizer
    ids = tokenized["input_ids"][0]
    rendered = tokenizer.decode(ids)
    assert (sample["nav_text"][0] in rendered) is nav
    assert (sample["cot"] in rendered) is (cot and not generation_mode)
    if generation_mode:
        # Generation must work even when the evaluation annotations omit CoT.
        without_cot = {key: value for key, value in sample.items() if key != "cot"}
        other = processor.preprocess(data=without_cot, **options)
        torch.testing.assert_close(tokenized["input_ids"], other["input_ids"], rtol=0, atol=0)
        assert not ids.eq(config.traj_ids["future_pad"]).any()
        if cot:
            assert not ids.eq(config.traj_ids["future_start"]).any()
            assert rendered.endswith("<|im_start|>assistant\n<think></think>")
        else:
            assert ids[-1] == config.traj_ids["future_start"]
    collator = Alpamayo2EdgeCollator(config, max_length=2048, pad_to_fixed_length=False)
    batch = collator([{**sample, "tokenized_data": tokenized}])
    supervised = tokenizer.decode(batch["tokenized_data"]["input_ids"][batch["labels_mask"]])
    assert sample["nav_text"][0] not in supervised
    assert (sample["cot"] in supervised) is (cot and not generation_mode)
    assert bool(batch["labels_mask"].any()) is (not generation_mode)


@pytest.mark.parametrize("variant", ["trajectory", "nav", "cot", "nav_cot"])
def test_every_task_prefix_is_compared_with_native_inference(release_conversations, variant):
    """Only native Nav+CoT is a like-for-like base-checkpoint prompt."""
    inference, recipe = release_conversations
    nav, cot = "nav" in variant, "cot" in variant
    components = ["image", "traj_history"] + (["nav_instruction"] if nav else [])
    components += ["prompt"] + (["cot"] if cot else []) + ["traj_future"]
    native = inference(_sample(), num_tokens_per_history_traj=45)
    actual = recipe(data=_sample(), **_options(
        components_order=components,
        components_prompt=(["cot"] if cot else []) + ["traj_future"],
    ))
    same_prefix = [_text(message) for message in actual[:2]] == [_text(message) for message in native]
    assert same_prefix is (variant == "nav_cot")
    # All variants retain the same camera/history encoding; only the declared
    # navigation/task content may adapt the native prompt.
    assert _text(actual[0]) == _text(native[0])
    assert _text(actual[1]).split("<|traj_history_end|>")[0] == _text(native[1]).split("<|traj_history_end|>")[0]


def test_checkpoint_c3_inputs_match_public_inference_exactly(release_conversations):
    """Check prompt, modality IDs and normalized SigLIP2 patches at the API boundary."""
    checkpoint = os.environ.get("ALPAMAYO2_EDGE_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set ALPAMAYO2_EDGE_TEST_CHECKPOINT to a complete local checkpoint")
    from alpamayo2_edge.config import Alpamayo2EdgeConfig
    from alpamayo2_edge.helper import prepare_model_inputs
    from alpamayo2edge_sft.processor import Alpamayo2EdgeProcessor

    config = Alpamayo2EdgeConfig.from_pretrained(checkpoint, local_files_only=True)
    config.name_or_path = checkpoint
    processor = Alpamayo2EdgeProcessor(config)
    sample = _sample()
    # Unequal channels and varying spatial values expose patch-order mistakes.
    sample["image_frames"] = torch.arange(22 * 3 * 32 * 32).remainder(251).to(
        torch.uint8
    ).reshape(22, 1, 3, 32, 32)
    sample["camera_indices"] = torch.tensor([0] * 3 + [1] * 5 + [2] * 3 + [3] * 3 + [5] * 3 + [6] * 5)
    sample["absolute_timestamps"] = torch.arange(22, dtype=torch.int64) * 100_000
    sample["camera_pixel_bounds"] = {camera: [1024, 1024] for camera in (0, 1, 2, 3, 5, 6)}
    # Compare both sort boundaries as well as the complete 22-frame API schema.
    for name in ("image_frames", "camera_indices", "absolute_timestamps"):
        sample[name] = sample[name].flip(0)
    sample["ego_history_xyz"] = torch.zeros(1, config.history_num_steps, 3)
    sample["ego_history_rot"] = torch.eye(3).expand(1, config.history_num_steps, 3, 3).clone()
    actual = processor.preprocess(
        data=sample,
        components_order=["image", "traj_history", "nav_instruction", "prompt", "cot", "traj_future"],
        components_prompt=["cot", "traj_future"],
        label_components=["assistant"], generation_mode=True,
    )
    expected = prepare_model_inputs(sample, config, processor.processor.tokenizer)["tokenized_data"]
    assert set(actual) - {"_label_components", "_generation_mode"} == set(expected)
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0, msg=name)
    prompt = processor.processor.tokenizer.decode(actual["input_ids"][0])
    assert prompt.endswith("<|im_start|>assistant\n<think></think>")
    assert prompt.count("<think></think>") == 1
