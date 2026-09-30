# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data preprocessing for Alpamayo 2 Super SFT."""

from functools import partial
from typing import Any, Callable

import torch

from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.config import (
    Alpamayo2SuperConfig,
    build_alpamayo2_super_tokenizer,
    resolve_checkpoint_name_or_path,
)
from alpamayo2_super.helper import get_processor
from alpamayo2_sft.label_mask import get_assistant_mask


_TRAJECTORY_KEYS = (
    "ego_history_xyz",
    "ego_history_rot",
    "ego_future_xyz",
    "ego_future_rot",
)


class Alpamayo2SuperProcessor:
    """Build training inputs with the released model's tokenizer and chat template."""

    def __init__(self, model_config: Alpamayo2SuperConfig) -> None:
        checkpoint_path = resolve_checkpoint_name_or_path(model_config)
        if checkpoint_path is None:
            raise ValueError("model_config must identify an Alpamayo 2 Super checkpoint")
        tokenizer = build_alpamayo2_super_tokenizer(
            checkpoint_path,
            model_config.history_vocab_size,
            model_config.future_vocab_size,
        )
        tokenizer.padding_side = model_config.padding_side
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
        sort_indices = torch.argsort(data["camera_indices"], stable=True)
        images = data["image_frames"][sort_indices]
        camera_indices = data["camera_indices"][sort_indices]
        sample = dict(data)
        sample["image_frames"] = images
        sample["camera_indices"] = camera_indices

        messages = build_conversation(
            data=sample,
            num_tokens_per_history_traj=self.model_config.tokens_per_history_traj,
            num_tokens_per_future_traj=self.model_config.tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            generation_mode=generation_mode,
            include_camera_ids=self.model_config.include_camera_ids,
            camera_ids=camera_indices,
            include_frame_nums=self.model_config.frame_label == "frame_num",
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
            continue_final_message=generation_mode,
        )
        images = images.flatten(0, 1)
        images = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
        tokenized_data = dict(
            self.processor(
                text=text,
                images=images,
                videos=None,
                padding=False,
                return_tensors="pt",
                do_rescale=False,
            )
        )
        tokenized_data["_label_components"] = label_components
        tokenized_data["_generation_mode"] = generation_mode
        return tokenized_data


class Alpamayo2SuperCollator:
    """Pad token inputs and form the nested release-model batch contract."""

    def __init__(
        self,
        model_config: Alpamayo2SuperConfig,
        max_length: int,
        pad_to_fixed_length: bool = True,
    ) -> None:
        self.data_processor = Alpamayo2SuperProcessor(model_config)
        self.max_length = max_length
        self.pad_to_fixed_length = pad_to_fixed_length

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate samples into token inputs, trajectories, and a label mask."""
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
        for key in tokenized_samples[0]:
            tokenized_data[key] = torch.cat(
                [tokenized_sample[key] for tokenized_sample in tokenized_samples]
            )

        if generation_mode:
            labels_mask = torch.zeros_like(tokenized_data["input_ids"], dtype=torch.bool)
        else:
            labels_mask = torch.stack(
                [
                    get_assistant_mask(
                        tokenizer=self.data_processor.processor.tokenizer,
                        tokens=input_ids,
                    )
                    for input_ids in tokenized_data["input_ids"]
                ]
            )
        traj_data = {
            key: torch.stack([sample[key] for sample in samples])
            for key in _TRAJECTORY_KEYS
            if key in samples[0]
        }
        return {
            "tokenized_data": tokenized_data,
            "traj_data": traj_data,
            "labels_mask": labels_mask,
        }


def get_preprocess_data_fn_from_model_config(
    model_config: Alpamayo2SuperConfig,
    components_order: list[str],
    components_prompt: list[str],
    label_components: list[str],
    generation_mode: bool = False,
) -> Callable[..., dict[str, Any]]:
    """Build the preprocessing callable instantiated by PhysicalAI datasets."""
    data_processor = Alpamayo2SuperProcessor(model_config)
    return partial(
        data_processor.preprocess,
        components_order=components_order,
        components_prompt=components_prompt,
        label_components=label_components,
        generation_mode=generation_mode,
    )
