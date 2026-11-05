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

"""Alpamayo 1.5 multimodal prompt, frozen target, and greedy decoding contract.

Uses the alpamayo1_5 model package. Never imports
DeepSpec, a workspace module, or a scheduler integration.
"""

from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING

import torch
from transformers import AutoProcessor

from .checkpoint import check_loading, public_config
from .contracts import digest, file_digest

if TYPE_CHECKING:
    from .data import InputData
    from .runtime import Runtime


class Target:
    """Frozen multimodal target with a fingerprinted no-navigation CoC contract."""

    def __init__(self, args: Namespace, runtime: Runtime) -> None:
        from alpamayo1_5 import helper
        from alpamayo1_5.models.base_model import ReasoningVLA

        self.device = runtime.device
        self.helper = helper
        for directory in (args.checkpoint, args.base_vlm, args.processor):
            if not directory.is_dir():
                raise FileNotFoundError(directory)
        # Config construction itself loads VLM metadata, so override BEFORE __init__.
        raw = public_config(json.loads((args.checkpoint / "config.json").read_text()))
        raw["vlm_name_or_path"] = str(args.base_vlm.resolve())
        raw["attn_implementation"] = args.target_attention
        config_class = ReasoningVLA.config_class
        # HF's config repr otherwise constructs a default config that can fetch a
        # different remote VLM. This config requires the provided local metadata.
        config_class.has_no_defaults_at_init = True
        config = config_class(**raw)
        self.model, audit = ReasoningVLA.from_pretrained(
            args.checkpoint,
            config=config,
            dtype=torch.bfloat16,
            # The outer wrapper implements no attention and does not advertise
            # SDPA/FA2 support to HF. Its public config.attn_implementation above
            # is independently forwarded to the Qwen3-VL backbone constructor.
            # Keep HF's wrapper dispatch eager; do not falsify support flags or
            # silently switch the actual VLM away from the requested backend.
            attn_implementation="eager",
            local_files_only=True,
            output_loading_info=True,
        )
        self.ignored_expert_keys = check_loading(audit)
        self.attention_backends = {
            "text": self.model.vlm.config.text_config._attn_implementation,
            "vision": self.model.vlm.config.vision_config._attn_implementation,
        }
        if any(value != args.target_attention for value in self.attention_backends.values()):
            raise ValueError(
                f"Requested {args.target_attention} but VLM backends are {self.attention_backends}"
            )
        if runtime.rank == 0:
            print(f"Target attention: wrapper=eager VLM={self.attention_backends}", flush=True)
            print(
                f"Target VLM loaded; unused expert tensors={len(self.ignored_expert_keys)}",
                flush=True,
            )
        # The future tokenizer determines the history vocabulary offset during
        # construction, but no trajectory expert is instantiated or trained.
        self.model.traj_tokenizer = None
        self.model.requires_grad_(False).to(self.device).eval()
        self.vlm = self.model.vlm
        pixel_limits = {
            key: int(getattr(config, key))
            for key in ("min_pixels", "max_pixels")
            if getattr(config, key, None) is not None
        }
        self.processor = AutoProcessor.from_pretrained(
            args.processor,
            local_files_only=True,
            **pixel_limits,
        )
        self.processor.tokenizer = self.model.tokenizer
        self.tokenizer = self.model.tokenizer
        self.stop_id = self.token_id("<|traj_future_start|>")
        self.cot_end_id = self.token_id("<|cot_end|>")
        # Reuse a token absent from this CoC-only task, avoiding vocabulary
        # resizing and changes to the frozen target's embedding/head. This is
        # not suitable for VQA prompts; prompt and cached-CoC checks guard it.
        self.mask_id = self.token_id("<|answer_end|>")
        self.text_config = self.vlm.config.get_text_config()
        self.max_new_tokens = args.max_new_tokens
        self.contract = self._contract(args)

    def token_id(self, token: str) -> int:
        value = self.tokenizer.convert_tokens_to_ids(token)
        if value is None or self.tokenizer.convert_ids_to_tokens(value) != token:
            raise ValueError(f"Target is missing special token {token}")
        return int(value)

    def _contract(self, args):
        return {
            "checkpoint": str(args.checkpoint.resolve()),
            "config_sha256": file_digest(args.checkpoint / "config.json"),
            "processor_pixels": {
                key: getattr(self.processor.image_processor, key, None)
                for key in ("min_pixels", "max_pixels", "size")
            },
            "image_preprocessing": "float32_0_1_before_resize",
            "target_attention": args.target_attention,
            "stop_id": self.stop_id,
            "cot_end_id": self.cot_end_id,
            "vocab_size": int(self.text_config.vocab_size),
            "max_new_tokens": self.max_new_tokens,
        }

    def prepare(self, row: dict, data: InputData) -> tuple[dict[str, torch.Tensor], str]:
        """Load frames/history, fuse the prompt, and return inputs plus their hash."""
        frames, ego = data.load(row)
        messages = self.helper.create_message(
            frames,
            camera_indices=torch.tensor([0, 1, 2, 6]),
            num_frames_per_camera=4,
            nav_text=None,
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        # Match SFT's QwenProcessor: interpolate floating-point RGB, not uint8.
        # Resizing uint8 first introduces rounding that changes the VLM input.
        inputs = self.processor(
            text=[text],
            images=frames.float() / 255.0,
            do_rescale=False,
            return_tensors="pt",
        )
        if len(inputs["image_grid_thw"]) != 16:
            raise ValueError("Expected 16 images after processor tokenization")
        ego = {key: value.to(self.device) for key, value in ego.items()}
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            inputs["input_ids"] = self.model.fuse_traj_tokens(inputs["input_ids"], ego)
        placeholder = int(self.model.config.traj_token_ids["history"])
        if bool((inputs["input_ids"] == placeholder).any()):
            raise ValueError("Unfused trajectory-history placeholders")
        if bool((inputs["input_ids"] == self.mask_id).any()):
            raise ValueError("Draft mask token appears in the prompt")
        fingerprint = digest(
            {
                "fused_prompt_ids": inputs["input_ids"][0].tolist(),
                "image_grid_thw": inputs["image_grid_thw"].tolist(),
            }
        )
        return inputs, fingerprint

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Exclude future-trajectory vocabulary from language-token generation."""
        offset = int(self.model.config.traj_token_start_idx)
        logits[..., offset : offset + int(self.model.config.traj_vocab_size)] = -torch.inf
        return logits

    @torch.no_grad()
    def prefill_features(
        self, inputs: dict[str, torch.Tensor], gen_ids: list[int], layers: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-force cached CoCs; return token IDs, features and CoC loss mask."""
        ids = torch.cat((inputs["input_ids"], torch.tensor([gen_ids], device=self.device)), dim=1)
        prompt_length = inputs["input_ids"].shape[1]
        kwargs = {
            key: value
            for key, value in inputs.items()
            if key not in ("input_ids", "attention_mask")
        }
        # Each sample is unpadded and processed independently, so M-RoPE is rebuilt.
        self.vlm.model.rope_deltas = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.vlm.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                use_cache=False,
                **kwargs,
            )
        hidden = torch.cat([output.hidden_states[layer + 1] for layer in layers], dim=-1)
        mask = torch.zeros_like(ids, dtype=torch.float32)
        mask[:, prompt_length:] = 1
        return ids, hidden, mask

    def decode_session(self, inputs: dict[str, torch.Tensor]) -> DecodeSession:
        return DecodeSession(self, inputs)


class DecodeSession:
    """Inject images only at prefill; let Qwen3-VL retain its multimodal RoPE offsets."""

    def __init__(self, target, inputs):
        self.target, self.inputs = target, inputs
        self.first = True
        target.vlm.model.rope_deltas = None

    @torch.no_grad()
    def __call__(self, ids, cache, *, hidden=False, last_only=False):
        if self.first:
            kwargs = {key: value for key, value in self.inputs.items() if key != "input_ids"}
            self.first = False
        else:
            length = cache.get_seq_length()
            kwargs = {
                "cache_position": torch.arange(length, length + ids.shape[1], device=ids.device)
            }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.target.vlm(
                input_ids=ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=hidden,
                logits_to_keep=1 if last_only else 0,
                **kwargs,
            )
        output.logits = self.target.mask_logits(output.logits)
        return output


@torch.no_grad()
def greedy_decode(target: Target, inputs: dict[str, torch.Tensor]) -> list[int]:
    """Generate unsampled CoC tokens up to the trajectory boundary or token budget."""
    from transformers import DynamicCache

    session, cache = target.decode_session(inputs), DynamicCache()
    current, result = inputs["input_ids"], []
    for _ in range(target.max_new_tokens):
        output = session(current, cache, last_only=True)
        current = output.logits[:, -1:].argmax(-1)
        result.append(int(current.item()))
        if result[-1] == target.stop_id:
            break
    return result
