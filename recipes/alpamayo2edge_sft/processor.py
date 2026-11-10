# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data preprocessing for Alpamayo 2 Edge SFT."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torchvision.transforms.functional import InterpolationMode, resize
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from alpamayo2edge_sft.dataset import validate_camera_pixel_bounds
from alpamayo2edge_sft.label_mask import get_trajectory_mask

_TRAJECTORY_KEYS = (
    "ego_history_xyz",
    "ego_history_rot",
    "ego_future_xyz",
    "ego_future_rot",
)


def get_assistant_mask(tokens: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    """Supervise complete ChatML assistant messages, including their closing EOS.

    Derive the header from the tokenizer so multi-token role names are supported.
    """
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not header or header[0] != start_id or start_id == end_id:
        raise ValueError("Full-assistant supervision requires a ChatML tokenizer")
    starts = (tokens == start_id).nonzero(as_tuple=True)[0].tolist()
    ends = (tokens == end_id).nonzero(as_tuple=True)[0].tolist()
    if len(starts) != len(ends) or any(start >= end for start, end in zip(starts, ends)):
        raise ValueError("Full-assistant supervision requires complete message boundaries")
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for start, end in zip(starts, ends, strict=True):
        if tokens[start : start + len(header)].tolist() == header:
            mask[start + len(header) : end + 1] = True
    if not mask.any():
        raise ValueError("Full-assistant supervision found no assistant response tokens")
    return mask


def resolve_checkpoint_name_or_path(model_config: Any) -> str | None:
    """Resolve tokenizer and processor assets through the inference package."""
    from alpamayo2_edge.config import resolve_checkpoint_name_or_path as resolve_public_checkpoint

    return resolve_public_checkpoint(model_config)


def build_alpamayo2_edge_tokenizer(checkpoint_path, history_vocab_size, future_vocab_size):
    """Import the public tokenizer builder only for public checkpoints."""
    from alpamayo2_edge.config import build_alpamayo2_edge_tokenizer as build_public_tokenizer

    return build_public_tokenizer(checkpoint_path, history_vocab_size, future_vocab_size)


def get_processor(tokenizer: Any, model_config: Any):
    """Load the checkpoint processor through the inference package."""
    from alpamayo2_edge.helper import get_processor as get_public_processor

    processor = get_public_processor(tokenizer, model_config)
    # Legacy exports can retain the base VLM's size alongside the trained bounds.
    # Apply the checkpoint bounds when no per-camera override is supplied.
    processor.image_processor.size["shortest_edge"] = model_config.min_pixels
    processor.image_processor.size["longest_edge"] = model_config.max_pixels
    return processor


def build_conversation(**kwargs: Any):
    """Build supervised conversations with the released Edge chat template."""
    from alpamayo2edge_sft.conversation import build_conversation as build_sft_conversation

    return build_sft_conversation(**kwargs)


def normalize_image_patches(patches: torch.Tensor, image_processor: Any) -> torch.Tensor:
    """Normalize Cosmos 3 Edge uint8 patches in native height/width/channel/time layout."""
    channels = len(image_processor.image_mean)
    temporal = image_processor.temporal_patch_size
    patch_dim = image_processor.patch_size**2 * channels * temporal
    if patches.dtype != torch.uint8 or patches.ndim != 2 or patches.shape[1] != patch_dim:
        raise ValueError(f"Cosmos3-Edge processor must return uint8 image patches [N, {patch_dim}]")
    # Match the checkpoint's fused normalization after resizing uint8 frames.
    scale = 1.0 / image_processor.rescale_factor
    mean = patches.new_tensor(image_processor.image_mean, dtype=torch.float32) * scale
    std = patches.new_tensor(image_processor.image_std, dtype=torch.float32) * scale
    result = patches.float().reshape(-1, image_processor.patch_size**2, channels, temporal)
    result.sub_(mean.reshape(1, 1, channels, 1)).div_(std.reshape(1, 1, channels, 1))
    return result.reshape(-1, patch_dim)


class Alpamayo2EdgeProcessor:
    """Build training inputs with the released model's tokenizer and chat template."""

    def __init__(self, model_config: Any) -> None:
        checkpoint_path = resolve_checkpoint_name_or_path(model_config)
        if checkpoint_path is None:
            raise ValueError("model_config must identify an Alpamayo 2 Edge checkpoint")
        tokenizer = build_alpamayo2_edge_tokenizer(
            checkpoint_path,
            model_config.history_vocab_size,
            model_config.future_vocab_size,
        )
        if model_config.padding_side != "left":
            raise ValueError("Edge SFT requires checkpoint padding_side='left' for expert batches")
        tokenizer.padding_side = "left"
        self.model_config = model_config
        self.processor = get_processor(tokenizer, model_config)

    def preprocess(
        self,
        data: dict[str, Any],
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
    ) -> dict[str, Any]:
        """Convert one PhysicalAI-AV sample into release-model inputs."""
        if label_components not in (["traj_future"], ["assistant"]):
            raise ValueError("Use label_components=[traj_future] or [assistant]")
        if "prompt" in components_order:
            assistant_components = components_order[components_order.index("prompt") + 1 :]
        elif "question" in components_order:
            assistant_components = components_order[components_order.index("question") + 1 :]
        else:
            raise ValueError("The conversation requires a prompt or question component")
        if label_components == ["traj_future"] and assistant_components != ["traj_future"]:
            raise ValueError(
                "Trajectory-only labels would discard other assistant targets such as CoT; "
                "use label_components=[assistant] to supervise the full native response"
            )
        if generation_mode and assistant_components not in (["traj_future"], ["cot", "traj_future"]):
            raise ValueError("Generation supports trajectory-only or CoT followed by trajectory")
        generate_cot = generation_mode and assistant_components == ["cot", "traj_future"]
        conversation_components = components_order
        if generate_cot:
            # Omit targets before constructing messages: validation samples need
            # neither reference CoT nor future text to generate a response.
            conversation_components = components_order[: -len(assistant_components)]
        if getattr(self.model_config, "token_layout", "camera_ts") != "camera_ts":
            raise ValueError("The Edge recipe requires camera-major token_layout='camera_ts'")
        if getattr(self.model_config, "encode_as_video", False):
            raise ValueError("The Edge recipe supplies individual selected images, not videos")
        images = data["image_frames"]
        camera_indices = data["camera_indices"]
        timestamps = data["absolute_timestamps"].reshape(-1)
        if images.ndim != 5 or images.shape[1:3] != (1, 3):
            raise ValueError("image_frames must have shape [N, 1, 3, H, W]")
        if camera_indices.shape != (len(images),) or timestamps.shape != (len(images),):
            raise ValueError("Each flat image needs one camera ID and one timestamp")
        if timestamps.dtype != torch.int64:
            raise TypeError("absolute_timestamps must use int64 microseconds")
        # Sort time first, then stably group by camera, keeping labels aligned.
        sort_indices = torch.argsort(timestamps, stable=True)
        sort_indices = sort_indices[torch.argsort(camera_indices[sort_indices], stable=True)]
        images = images[sort_indices]
        sample = dict(data)
        sample["image_frames"] = images
        sample["camera_indices"] = camera_indices[sort_indices]
        sample["absolute_timestamps"] = timestamps[sort_indices]

        messages = build_conversation(
            data=sample,
            num_tokens_per_history_traj=self.model_config.tokens_per_history_traj,
            num_tokens_per_future_traj=self.model_config.tokens_per_future_traj,
            components_order=conversation_components,
            components_prompt=components_prompt,
            generation_mode=generation_mode,
            include_camera_ids=self.model_config.include_camera_ids,
            token_layout="camera_ts",
            frame_label=getattr(self.model_config, "frame_label", "relative_time"),
            encode_as_video=False,
            system_prompt=getattr(self.model_config, "system_prompt", None),
        )
        for message in messages:
            if message["role"] == "assistant":
                # The native Edge template reads one assistant text part.
                message["content"] = [
                    {
                        "type": "text",
                        "text": "".join(part["text"] for part in message["content"]),
                    }
                ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=generate_cot,
            add_vision_id=False,
            continue_final_message=generation_mode and not generate_cot,
            enable_thinking=False,
        )
        if generation_mode and not generate_cot:
            future_start = self.processor.tokenizer.convert_ids_to_tokens(
                self.model_config.traj_ids["future_start"]
            )
            if text.count(future_start) != 1:
                raise ValueError("Generation prompt must contain exactly one future-start token")
            text = text.split(future_start)[0] + future_start
        image_entries = [
            entry
            for message in messages
            for entry in message["content"]
            if entry.get("type") == "image"
        ]
        if not image_entries:
            raise ValueError("The Edge conversation must contain its input images")
        # Message order determines image placeholders; templates may filter or reorder frames.
        message_images = torch.cat(
            [
                entry["image"] if entry["image"].ndim == 4 else entry["image"].unsqueeze(0)
                for entry in image_entries
            ]
        )
        camera_pixel_bounds = sample.get("camera_pixel_bounds")
        bounds = (
            validate_camera_pixel_bounds(camera_pixel_bounds)
            if camera_pixel_bounds is not None
            else {}
        )
        if bounds and any(type(entry.get("camera_id")) is not int for entry in image_entries):
            raise ValueError("Per-camera pixel bounds require camera_id on each image entry")
        if message_images.dtype != torch.uint8 or message_images.device.type != "cpu":
            raise ValueError("Cosmos3-Edge resizing requires CPU uint8 image frames")
        image_processor = self.processor.image_processor
        default_bounds = (
            image_processor.size["shortest_edge"],
            image_processor.size["longest_edge"],
        )
        resized_images = []
        for frame, entry in zip(message_images, image_entries, strict=True):
            lower, upper = bounds.get(entry.get("camera_id"), default_bounds)
            height, width = smart_resize(
                *frame.shape[-2:],
                factor=image_processor.patch_size * image_processor.merge_size,
                min_pixels=lower,
                max_pixels=upper,
            )
            resized_images.append(
                resize(frame, [height, width], interpolation=InterpolationMode.BICUBIC)
            )
        # Keep resize and normalization in the same order as released inference.
        # The native processor packs uint8 patches before fused normalization.
        tokenized_data = dict(
            self.processor(
                text=text,
                images=resized_images,
                padding=False,
                return_tensors="pt",
                return_mm_token_type_ids=True,
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
            )
        )
        tokenized_data["pixel_values"] = normalize_image_patches(
            tokenized_data["pixel_values"], image_processor
        )
        tokenized_data["_label_components"] = label_components
        tokenized_data["_generation_mode"] = generation_mode
        return tokenized_data


class Alpamayo2EdgeCollator:
    """Pad token inputs and form the nested release-model batch contract."""

    def __init__(
        self,
        model_config: Any,
        max_length: int,
        pad_to_fixed_length: bool = True,
    ) -> None:
        self.data_processor = Alpamayo2EdgeProcessor(model_config)
        self.max_length = max_length
        self.pad_to_fixed_length = pad_to_fixed_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into token inputs, trajectories, and a label mask."""
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        tokenized_samples = [dict(sample["tokenized_data"]) for sample in samples]
        label_components = tokenized_samples[0].pop("_label_components")
        generation_mode = tokenized_samples[0].pop("_generation_mode")
        for tokenized_sample in tokenized_samples[1:]:
            if tokenized_sample.pop("_label_components") != label_components:
                raise ValueError("A batch cannot mix label component sets")
            if tokenized_sample.pop("_generation_mode") != generation_mode:
                raise ValueError("A batch cannot mix training and generation samples")

        text_inputs = [
            {
                "input_ids": tokenized_sample.pop("input_ids")[0],
                "attention_mask": tokenized_sample.pop("attention_mask")[0],
            }
            for tokenized_sample in tokenized_samples
        ]
        longest_sequence = max(text_input["input_ids"].shape[0] for text_input in text_inputs)
        if longest_sequence > self.max_length:
            raise ValueError(
                f"Tokenized sample length {longest_sequence} exceeds max_length={self.max_length}"
            )
        padding_kwargs: dict[str, Any] = {"padding": True}
        if self.pad_to_fixed_length:
            padding_kwargs = {"padding": "max_length", "max_length": self.max_length}
        tokenized_data = dict(
            self.data_processor.processor.tokenizer.pad(
                text_inputs,
                return_tensors="pt",
                **padding_kwargs,
            )
        )
        mm_token_type_ids = [sample.pop("mm_token_type_ids", None) for sample in tokenized_samples]
        if any(value is not None for value in mm_token_type_ids):
            padded_length = tokenized_data["input_ids"].shape[1]
            padded_mm_ids = []
            for value, text_input in zip(mm_token_type_ids, text_inputs, strict=True):
                if value is None or value.shape != (1, text_input["input_ids"].shape[0]):
                    raise ValueError(
                        "Every sample's mm_token_type_ids must align with its input_ids"
                    )
                padded_mm_ids.append(
                    torch.nn.functional.pad(value[0], (padded_length - value.shape[1], 0), value=0)
                )
            tokenized_data["mm_token_type_ids"] = torch.stack(padded_mm_ids)
        for key in tokenized_samples[0]:
            tokenized_data[key] = torch.cat(
                [tokenized_sample[key] for tokenized_sample in tokenized_samples]
            )

        if generation_mode:
            labels_mask = torch.zeros_like(tokenized_data["input_ids"], dtype=torch.bool)
        elif label_components == ["assistant"]:
            labels_mask = torch.stack(
                [
                    get_assistant_mask(input_ids, self.data_processor.processor.tokenizer)
                    for input_ids in tokenized_data["input_ids"]
                ]
            )
        else:
            labels_mask = torch.stack(
                [
                    get_trajectory_mask(
                        tokens=input_ids,
                        start_id=self.data_processor.model_config.traj_ids["future_start"],
                        end_id=self.data_processor.model_config.traj_ids["future_end"],
                        eos_token_id=self.data_processor.processor.tokenizer.eos_token_id,
                    )
                    for input_ids in tokenized_data["input_ids"]
                ]
            )
        traj_data = {
            key: torch.stack([sample[key] for sample in samples])
            for key in _TRAJECTORY_KEYS
            if key in samples[0]
        }
        batch = {
            "tokenized_data": tokenized_data,
            "traj_data": traj_data,
            "labels_mask": labels_mask,
        }
        if generation_mode:
            batch["sample_metadata"] = [
                {key: sample[key] for key in ("clip_id", "requested_t0_us", "t0_us") if key in sample}
                for sample in samples
            ]
        return batch


def get_preprocess_data_fn_from_model_config(
    model_config: Any,
    components_order: list[str],
    components_prompt: list[str],
    label_components: list[str],
    generation_mode: bool = False,
) -> Callable[..., dict[str, Any]]:
    """Build the preprocessing callable instantiated by PhysicalAI datasets."""
    data_processor = Alpamayo2EdgeProcessor(model_config)
    return partial(
        data_processor.preprocess,
        components_order=components_order,
        components_prompt=components_prompt,
        label_components=label_components,
        generation_mode=generation_mode,
    )
