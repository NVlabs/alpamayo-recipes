from typing import Any, Callable
from functools import partial
import os
import torch

from alpamayo1_distill.processors.base_processor import BaseVLMProcessor
from transformers import AutoProcessor
from alpamayo1_distill.vla_constant import TRAJ_TOKEN, SPECIAL_TOKENS
from alpamayo1_distill.utils.get_label_mask import get_label_mask, get_role_eos_mask
from alpamayo1_distill.data.conversation import build_conversation
from alpamayo1_distill.data.data_utils import sort_images_by_camera_ids
from alpamayo1_distill.data.collation import basic_collation_fn


class QwenProcessor(BaseVLMProcessor):
    """Processor implementation for Qwen VLM models."""

    def build_processor(self) -> AutoProcessor:
        """Build the Qwen processor."""
        processor_kwargs = {}
        if self._min_pixels is not None:
            processor_kwargs["min_pixels"] = self._min_pixels
        if self._max_pixels is not None:
            processor_kwargs["max_pixels"] = self._max_pixels
        #print("in build processor:", self.vlm_name_or_path)
        # local_files_only=True triggers HF hub repo-ID validation on newer huggingface_hub
        # versions, which rejects absolute local paths. For local dirs the flag is redundant.
        _local_files_only = not os.path.isdir(self.vlm_name_or_path)
        processor = AutoProcessor.from_pretrained(self.vlm_name_or_path, local_files_only=_local_files_only, **processor_kwargs)
        #processor = AutoProcessor.from_pretrained(self.vlm_name_or_path, **processor_kwargs)
        tokenizer = processor.tokenizer

        # Add traj tokens to the tokenizer
        if self.traj_vocab_size is not None:
            discrete_tokens = [f"<i{v}>" for v in range(self.traj_vocab_size)]
            num_new_tokens = tokenizer.add_tokens(discrete_tokens)
            assert len(discrete_tokens) == num_new_tokens
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
            tokenizer.traj_token_end_idx = tokenizer.convert_tokens_to_ids(
                f"<i{self.traj_vocab_size - 1}>"
            )

        # Add all special tokens to the tokenizer
        special_tokens = list(SPECIAL_TOKENS.values())
        tokenizer.add_tokens(special_tokens, special_tokens=True)

        # Add mapping from traj token names to ids
        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
        }

        return processor

    def get_preprocess_data_fn(
        self,
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        """Get the preprocess data function for the Qwen VLM model."""
        processor = self.processor  # This will call build_processor() if needed
        return partial(
            self._preprocess_data,
            processor=processor,
            num_tokens_per_history_traj=num_tokens_per_history_traj,
            num_tokens_per_future_traj=num_tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            label_components=label_components,
            generation_mode=generation_mode,
        )

    def collate_fn(self, data: list[dict[str, Any]], padding_side: str = "left") -> dict[str, Any]:
        """Collate function for Qwen data."""
        # raw image_frames may not have the same size, so we don't stack it
        batchec_data: dict[str, Any] = basic_collation_fn(data, unstackable_keys=["image_frames"])

        # stack items with the same dimension in tokenized_data
        tokenized_data = {}
        for k in batchec_data["tokenized_data"][0].keys():
            #print("k:" , k)
            if k not in ["text"]:
                tokenized_data[k] = torch.cat([row[k] for row in batchec_data["tokenized_data"]])

        # tokenize the text (with padding) and update the tokenized_data
        batch_text = [instance["text"] for instance in batchec_data["tokenized_data"]]
        processed_inputs = self.processor.tokenizer(
            batch_text, return_tensors="pt", padding_side=padding_side, padding=True
        )
        tokenized_data.update(processed_inputs)
        batchec_data["tokenized_data"] = tokenized_data

        # assert label_components is the same for all instances in the batch
        #label_components = batchec_data["label_components"][0]
        #assert all(l_i == label_components for l_i in batchec_data["label_components"]), (
        #    "label_components is not the same for all instances in the batch."
        #)

        # generate mask of labels for loss computation
        batchec_data["labels_mask"] = get_label_mask(
            input_ids=tokenized_data["input_ids"],
            tokenizer=self.processor.tokenizer,
            label_components=[],
        )

        # set the mask of assistant's eos token as True
        eos_mask_assistant = get_role_eos_mask(
            input_ids=tokenized_data["input_ids"],
            tokenizer=self.processor.tokenizer,
            bos_token="<|im_start|>",
            eos_token="<|im_end|>",
            role="assistant",
        )
        batchec_data["labels_mask"] |= eos_mask_assistant
        return batchec_data

    def _preprocess_data(
        self,
        data: dict[str, Any],
        processor: AutoProcessor,
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        label_components: list[str],
        generation_mode: bool,
    ) -> dict[str, Any]:
        """Preprocess data for the Qwen VLM model."""
        # 1. generate image inputs
        #print("before sort")
        #print("camera_indices:", data['camera_indices'])
        #print("camera_indices shape:", data['camera_indices'].shape)
        images, camera_ids = sort_images_by_camera_ids(
            data["image_frames"], data["camera_indices"], return_camera_ids=True
        )
        data["image_frames"] = images
        data["camera_indices"] = camera_ids
        data["label_components"] = label_components
        #print("camera_ids",  camera_ids)
        #print("images shape",  images.shape)
        # Convert to float for bicubic interpolation, normalize to [0, 1]
        # Handle both uint8 (0-255) and float (0-1) inputs
        if images.dtype == torch.uint8:
            images_for_vlm = images.float() / 255.0
        else:
            images_for_vlm = images.float()
        #print("images_for_vlm shape:", images_for_vlm.flatten(0, 1).shape)
        image_inputs = processor.image_processor(
            images=images_for_vlm.flatten(0, 1), do_rescale=False
        )
        #print("image_inputs: ", image_inputs)
        #print("pixel values shape:", image_inputs["pixel_values"].shape)

        # 2. build the chat template
        messages = build_conversation(
            data=data,
            num_tokens_per_history_traj=num_tokens_per_history_traj,
            num_tokens_per_future_traj=num_tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            generation_mode=generation_mode,
            include_camera_ids=self.include_camera_ids,
            camera_ids=camera_ids,
            include_frame_nums=self.include_frame_nums,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=generation_mode,
        )

        # expand each image to the multiple image tokens
        # TODO: this part can be included in the conversation building step
        index = 0
        merge_length = processor.image_processor.merge_size**2
        #print("text: ", text)
        #print("processor.image_token:", processor.image_token)
        while processor.image_token in text:
            text = text.replace(
                processor.image_token,
                "<|placeholder|>" * (image_inputs["image_grid_thw"][index].prod() // merge_length),
                1,
            )
            index += 1
        text = text.replace("<|placeholder|>", processor.image_token)
        #print("text after while loop:", text)
        #print("index:", index)
        #print("image_grid_thw len:", len(image_inputs["image_grid_thw"]))
        #print("image_grid_thw:", image_inputs["image_grid_thw"])
        assert index == len(image_inputs["image_grid_thw"])

        tokenized_data = {"text": text, **image_inputs}

        return tokenized_data
