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

"""Alpamayo 1.5 model wrapper for single-clip inference."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from alpamayo.processor.qwen_processor import build_processor
from alpamayo.utils.checkpoint_utils import remap_targets
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

_CAMERA_DISPLAY_NAMES: dict[int, str] = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_NUM_TRAJ_HISTORY_TOKENS: int = 48

# Mirrors the target-remapping table in scripts/convert_checkpoint.py.
_A15_TO_A1_TARGETS: dict[str, str] = {
    "alpamayo1_5.models.action_in_proj.": "alpamayo_r1.models.action_in_proj.",
    "alpamayo1_5.models.delta_tokenizer.": "alpamayo_r1.models.delta_tokenizer.",
    "alpamayo1_5.action_space.": "alpamayo_r1.action_space.",
    "alpamayo1_5.diffusion.": "alpamayo_r1.diffusion.",
}


@dataclass
class AlpamayoModelConfig:
    """Configuration for loading Alpamayo 1.5 for single-clip inference.

    Args:
        model_name: HuggingFace model identifier or local path for the
            Alpamayo 1.5 model weights (default: ``nvidia/Alpamayo-1.5-10B``).
            Both the released A1.5 checkpoint (``model_type='alpamayo1_5'``) and
            an A1-format converted checkpoint are accepted; the config is
            auto-patched when needed.
        vlm_name_or_path: HuggingFace model identifier or local path for the
            VLM backbone processor (e.g. ``Qwen/Qwen3-VL-8B-Instruct`` or a
            local copy).  When ``None`` (default) the path is read from the
            loaded model's ``config.vlm_name_or_path``.  Override this when
            network access is unavailable and the model config points to a
            HuggingFace repo ID.
        device: Target device string (``"cuda"`` or ``"cpu"``).
        dtype: Floating-point precision as a string.  Must be one of
            ``"bfloat16"``, ``"float16"``, or ``"float32"``.
        attn_implementation: Attention backend override.  Pass
            ``"flash_attention_2"`` when flash-attn is available, or
            ``"sdpa"`` as a fallback.  ``None`` lets the model choose.
        min_pixels: Minimum pixel count for image resizing.
        max_pixels: Maximum pixel count for image resizing.
    """

    model_name: str = "nvidia/Alpamayo-1.5-10B"
    vlm_name_or_path: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    attn_implementation: str | None = None
    min_pixels: int = 163_840
    max_pixels: int = 196_608
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)


class AlpamayoSingleClipModel:
    """Alpamayo 1.5 wrapper for single-clip trajectory inference.

    Loads :class:`alpamayo_r1.models.alpamayo_r1.AlpamayoR1` via
    ``from_pretrained`` and exposes a single :meth:`infer` method that returns
    predicted trajectory tensors.
    """

    def __init__(self, config: AlpamayoModelConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.torch_dtype = _DTYPE_MAP[config.dtype]

        self.model = self._load_model()
        self.processor = self._load_processor()

    @staticmethod
    def _is_local_path(model_name: str) -> bool:
        """Return True if *model_name* is a local filesystem path."""
        return Path(model_name).exists()

    @staticmethod
    def _load_config_dict(model_name: str) -> dict[str, Any]:
        """Load ``config.json`` from a local directory or a HuggingFace Hub repo."""
        if AlpamayoSingleClipModel._is_local_path(model_name):
            config_path = Path(model_name) / "config.json"
            with config_path.open(encoding="utf-8") as f:
                return json.load(f)
        else:
            from huggingface_hub import hf_hub_download

            config_file = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(config_file, encoding="utf-8") as f:
                return json.load(f)

    @staticmethod
    def _needs_conversion(config_dict: dict[str, Any]) -> bool:
        """Return True when the checkpoint uses ``model_type='alpamayo1_5'``."""
        return config_dict.get("model_type") == "alpamayo1_5"

    @staticmethod
    def _hf_snapshot_local_dir(model_name: str) -> str:
        """Return the local cache directory for a HuggingFace Hub snapshot.

        Downloads the entire repository if it is not already cached.  Uses the
        same ``HF_HOME`` / ``HF_HUB_CACHE`` environment variables as the rest
        of the HuggingFace tooling, so previously cached files are reused.
        """
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_name)

    @staticmethod
    def _make_converted_tempdir(
        local_dir: str,
        vlm_name_or_path_override: str | None = None,
    ) -> str:
        """Write a patched ``config.json`` and symlink weights into a temp dir.

        Equivalent to running ``python scripts/convert_checkpoint.py to-a1``
        but without writing any files to the original checkpoint location.
        The caller is responsible for deleting the temp dir after loading.

        When *vlm_name_or_path_override* is given it replaces
        ``config["vlm_name_or_path"]`` so that ``alpamayo_r1``'s internal
        processor loader uses a local directory instead of a HuggingFace Hub
        ID, avoiding network access.
        """
        src = Path(local_dir).resolve()
        tmp = tempfile.mkdtemp(prefix="alpamayo_r1_converted_")

        config_path = src / "config.json"
        with config_path.open(encoding="utf-8") as f:
            config_dict = json.load(f)

        patched = copy.deepcopy(config_dict)
        remap_targets(patched, _A15_TO_A1_TARGETS)
        patched["model_type"] = "alpamayo_r1"
        patched["architectures"] = ["AlpamayoR1"]
        if vlm_name_or_path_override is not None:
            patched["vlm_name_or_path"] = vlm_name_or_path_override

        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(patched, f, indent=2)

        for item in src.iterdir():
            if item.name == "config.json":
                continue
            dst = Path(tmp) / item.name
            if item.suffix == ".json":
                shutil.copy2(item, dst)
            else:
                os.symlink(item, dst)

        return tmp

    def _load_model(self) -> AlpamayoR1:
        """Load :class:`AlpamayoR1` from a local path or HuggingFace Hub ID.

        When the checkpoint has ``model_type='alpamayo1_5'``, a patched
        ``config.json`` is written to a temporary directory and weights are
        symlinked there so that transformers uses the correct ``alpamayo_r1``
        class throughout, avoiding the "model type mismatch" code path.
        """
        kwargs: dict[str, Any] = {"dtype": self.torch_dtype, **self.config.extra_model_kwargs}
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        config_dict = self._load_config_dict(self.config.model_name)
        tmp_dir: str | None = None

        if self._needs_conversion(config_dict):
            if self._is_local_path(self.config.model_name):
                local_dir = self.config.model_name
            else:
                local_dir = self._hf_snapshot_local_dir(self.config.model_name)

            warnings.warn(
                f"'{self.config.model_name}' has model_type='alpamayo1_5'. "
                "Writing a patched config to a temporary directory and symlinking "
                "weights to load via AlpamayoR1. For repeated evaluation, do a "
                "one-time permanent conversion:\n"
                "  python scripts/convert_checkpoint.py to-a1 "
                f"--input {local_dir} --output <converted_dir>\n"
                "Then pass <converted_dir> as --model_name to skip this step.",
                stacklevel=4,
            )
            tmp_dir = self._make_converted_tempdir(
                local_dir,
                vlm_name_or_path_override=self.config.vlm_name_or_path,
            )
            load_path = tmp_dir
        else:
            load_path = self.config.model_name

        try:
            model: AlpamayoR1 = AlpamayoR1.from_pretrained(load_path, **kwargs)
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        model.to(self.device)
        model.eval()
        return model

    def _load_processor(self) -> Any:
        """Build the VLM processor with the model's trajectory vocabulary.

        Uses :attr:`AlpamayoModelConfig.vlm_name_or_path` when set, otherwise
        falls back to ``model.config.vlm_name_or_path``.  Providing a local
        path avoids network access when the model config points to a HuggingFace
        Hub ID.
        """
        traj_vocab_size = getattr(self.model.config, "traj_vocab_size", None)
        vlm_path = self.config.vlm_name_or_path or self.model.config.vlm_name_or_path
        return build_processor(
            vlm_name_or_path=vlm_path,
            traj_vocab_size=traj_vocab_size,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
            chat_template_version="r1_5",
        )

    def _build_image_content(
        self,
        frames: torch.Tensor,
        camera_indices: torch.Tensor,
        num_frames_per_camera: int = 4,
    ) -> list[dict[str, Any]]:
        """Build the list of image/text content blocks for the user message.

        Args:
            frames: ``(num_cameras * num_frames_per_camera, C, H, W)`` tensor.
            camera_indices: ``(num_cameras,)`` int64 camera-id tensor.
            num_frames_per_camera: Number of frames per camera.

        Returns:
            List of ``{"type": "text"|"image", ...}`` content dicts.
        """
        if frames.ndim != 4:
            raise ValueError(f"frames must be 4-D, got shape {tuple(frames.shape)}")

        expanded_cam_ids = camera_indices.repeat_interleave(num_frames_per_camera)
        content: list[dict[str, Any]] = []
        prev_cam_id: int | None = None
        frame_idx = 0

        for i, frame in enumerate(frames):
            cam_id = int(expanded_cam_ids[i].item())
            if prev_cam_id is not None and cam_id != prev_cam_id:
                frame_idx = 0
            if frame_idx == 0:
                cam_name = _CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
                content.append({"type": "text", "text": f"{cam_name}: "})
            content.append({"type": "text", "text": f"frame {frame_idx} "})
            content.append({"type": "image", "image": frame})
            prev_cam_id = cam_id
            frame_idx += 1

        return content

    def _create_messages(
        self,
        frames: torch.Tensor,
        camera_indices: torch.Tensor,
        num_frames_per_camera: int = 4,
    ) -> list[dict[str, Any]]:
        """Build the chat-template messages for VLM rollout inference.

        The message structure follows the Alpamayo R1.5 convention:
        system → user (images + history placeholder + prompt) → assistant prefix.
        """
        hist_placeholder = (
            "<|traj_history_start|>"
            + "<|traj_history|>" * _NUM_TRAJ_HISTORY_TOKENS
            + "<|traj_history_end|>"
        )
        prompt = (
            "output the chain-of-thought reasoning of the driving process, "
            "then output the future trajectory."
        )
        image_content = self._build_image_content(
            frames=frames,
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames_per_camera,
        )
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a driving assistant that generates safe and accurate actions.",
                    }
                ],
            },
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": f"{hist_placeholder}{prompt}"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<|cot_start|>"}],
            },
        ]

    def _to_device(self, data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if hasattr(data, "to") and callable(data.to):
            try:
                return data.to(self.device)
            except Exception:
                pass
        if isinstance(data, Mapping):
            return {k: self._to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._to_device(v) for v in data)
        return data

    @torch.inference_mode()
    def infer(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        temperature: float = 0.6,
        num_traj_samples: int = 1,
        max_generation_length: int = 256,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run VLM rollout inference on a single clip.

        Args:
            data: Output of :meth:`SingleClipDatasetLoader.load`.
            top_p: Nucleus sampling probability.
            temperature: Sampling temperature.
            num_traj_samples: Number of trajectory candidates to sample (K).
            max_generation_length: Maximum number of tokens to generate.
            seed: CUDA random seed for reproducibility.

        Returns:
            Dictionary with keys ``pred_xyz`` ``(1, 1, K, T, 3)``,
            ``pred_rot`` ``(1, 1, K, T, 3, 3)``, and ``extra``
            (contains ``"cot"`` if the model emits a chain-of-thought).
        """
        frames = data["image_frames"].flatten(0, 1)
        camera_indices = data["camera_indices"]

        messages = self._create_messages(
            frames=frames,
            camera_indices=camera_indices,
            num_frames_per_camera=4,
        )

        tokenized_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_inputs = self._to_device(
            {
                "tokenized_data": tokenized_inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }
        )

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.torch_dtype,
            enabled=self.device.type == "cuda",
        ):
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_traj_samples=num_traj_samples,
                max_generation_length=max_generation_length,
                return_extra=True,
            )

        return {"pred_xyz": pred_xyz, "pred_rot": pred_rot, "extra": extra}
