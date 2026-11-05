# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Base Reasoning VLA model implementation for Alpamayo R1 release."""

import logging
import os
from typing import Any

import einops
import hydra.utils as hyu
import torch
from transformers import (
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)
from alpamayo1_distill.tokenizer.trajectory import traj_tokenizer_helper
from alpamayo1_distill.defaults import DEFAULT_TEACHER_VLM_MODEL

# Constants
IGNORE_INDEX = -100
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}
SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]
SPECIAL_TOKENS = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}


def _recursive_setattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively set attribute on object and all its children."""
    setattr(obj, attr, value)
    for child in getattr(obj, "children", lambda: [])():
        _recursive_setattr(child, attr, value)


def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Replace pad tokens in input_ids with new token values."""
    mask = input_ids == pad_idx
    #print("in replace_pad_token and mask length", len(mask))
    return input_ids.masked_scatter(mask, new_ids)


def tokenize_history_trajectory(
    tokenizer: Any, traj_data: dict[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the history trajectory with prefix shape of (B, n_traj, ...).

    Args:
        tokenizer: Trajectory tokenizer with encode method
        traj_data: dict containing "ego_history_xyz" and "ego_history_rot"
        start_idx: start of token index of the history trajectory tokens

    Returns:
        torch.Tensor: [B, n_traj * tokens_per_history_traj]
    """
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4, "ego_history_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,  # note hist_xyz is passed to fut_xyz as it's encoding history.
            fut_rot=hist_rot,
        )
        + start_idx
    )  # [B*n_traj, tokens_per_history_traj]
    hist_idx = einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)

    return hist_idx


class TrajectoryFusionMixin:
    """Mixin class providing trajectory fusion functionality.

    This class can be mixed into Reasoning VLA models to provide trajectory token
    fusion capabilities while keeping the core model logic separate.

    Required Attributes:
        The class using this mixin must provide the following attributes:

        - hist_traj_tokenizer: Tokenizer for history trajectories
        - traj_tokenizer: Tokenizer for future trajectories
        - hist_token_start_idx (int): Starting index for history tokens
        - future_token_start_idx (int): Starting index for future tokens
        - config: Configurations

    Example:
        class ReasoningVLAModel(BaseModel, TrajectoryFusionMixin):
            def __init__(self, config):
                super().__init__(config)
                self.hist_traj_tokenizer = HistoryTrajectoryTokenizer()
                self.traj_tokenizer = TrajectoryTokenizer()
                self.hist_token_start_idx = 1000
                self.future_token_start_idx = 2000
                self.config = config
    """

    def _validate_mixin_requirements(self, require_future: bool = False) -> dict[str, Any]:
        """Validate that all required mixin attributes are present.

        Args:
            require_future: Whether future trajectory attributes are required.

        Returns:
            dict: Dictionary containing validated attributes.

        Raises:
            AttributeError: If required mixin attributes are not found.
        """
        # Get required attributes with proper error handling
        hist_traj_tokenizer = getattr(self, "hist_traj_tokenizer", None)
        if hist_traj_tokenizer is None:
            raise AttributeError("TrajectoryFusionMixin requires 'hist_traj_tokenizer' attribute")

        hist_token_start_idx = getattr(self, "hist_token_start_idx", None)
        if hist_token_start_idx is None:
            raise AttributeError("TrajectoryFusionMixin requires 'hist_token_start_idx' attribute")

        config = getattr(self, "config", None)
        if config is None or not hasattr(config, "traj_token_ids"):
            raise AttributeError("TrajectoryFusionMixin requires 'config' with 'traj_token_ids'")

        result = {
            "hist_traj_tokenizer": hist_traj_tokenizer,
            "hist_token_start_idx": hist_token_start_idx,
            "config": config,
        }

        if require_future:
            traj_tokenizer = getattr(self, "traj_tokenizer", None)
            if traj_tokenizer is None:
                raise AttributeError("Requires 'traj_tokenizer' attribute for future trajectories")

            future_token_start_idx = getattr(self, "future_token_start_idx", None)
            if future_token_start_idx is None:
                raise AttributeError(
                    "Requires 'future_token_start_idx' attribute for future trajectories"
                )

            result.update(
                {
                    "traj_tokenizer": traj_tokenizer,
                    "future_token_start_idx": future_token_start_idx,
                }
            )

        return result

    def fuse_traj_tokens(
        self, input_ids: torch.Tensor, traj_data: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Fuse the trajectory tokens into the input ids; interleaved trajectory is supported.

        Args:
            input_ids: [B, n_token]
            traj_data: dict
                ego_history_xyz: torch.Tensor, [B, n_traj, Th, 3]
                ego_history_rot: torch.Tensor, [B, n_traj, Th, 4]
                ego_future_xyz: torch.Tensor | None, [B, n_traj, Tf, 3]
                ego_future_rot: torch.Tensor | None, [B, n_traj, Tf, 4]

        Returns:
            input_ids: [B, n_token]
            labels: [B, n_token]

        Raises:
            AttributeError: If required mixin attributes are not found.
        """
        if (
            traj_data is None
            or traj_data["ego_history_xyz"] is None
            or traj_data["ego_history_rot"] is None
        ):
            return input_ids

        # Validate required attributes for history trajectories
        has_future = "ego_future_xyz" in traj_data
        attrs = self._validate_mixin_requirements(require_future=has_future)
        #print("history:", attrs["config"].traj_token_ids["history"])
        hist_idx = traj_tokenizer_helper.tokenize_history_trajectory(
            attrs["hist_traj_tokenizer"], traj_data, attrs["hist_token_start_idx"]
        )
        input_ids = replace_pad_token(
            input_ids, hist_idx, attrs["config"].traj_token_ids["history"]
        )

        if has_future:
            future_idx = traj_tokenizer_helper.tokenize_future_trajectory(
                attrs["traj_tokenizer"], traj_data, attrs["future_token_start_idx"]
            )
            #print("feature:", attrs["config"].traj_token_ids["future"])
            input_ids = replace_pad_token(
                input_ids, future_idx, attrs["config"].traj_token_ids["future"]
            )

        return input_ids

class ReasoningVLAConfig(PretrainedConfig):
    """Configuration for the ReasoningVLA model.

    This configuration class provides a flexible interface for different VLM backends
    and trajectory tokenization methods.

    IMPORTANT: logprob in sample_trajectories_from_data is a placeholder now.
    """

    model_type = "alpamayo_reasoning_vla"

    # NOTE:
    # `transformers.PretrainedConfig` may call `self.__class__()` internally when serializing
    # configs with `use_diff=True` (e.g. `save_pretrained()`, `to_json_string()`), to obtain
    # the class "default" config for diffing.
    #
    # Our `__init__` triggers `_initialize_vlm_config()` which can be heavy and may require
    # accessing HF configs/processors (sometimes network / cached artifacts). Setting this
    # flag tells transformers to *not* instantiate a fresh default config during serialization,
    # avoiding unintended side-effects and failures when saving checkpoints/logging.
    has_no_defaults_at_init = True

    def __init__(
        self,
        vlm_name_or_path: str = DEFAULT_TEACHER_VLM_MODEL,
        vlm_backend: str = "qwen",  # New: specify VLM backend type
        traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_tokenizer_cfg: dict[str, Any] | None = None,
        hist_traj_embed_cfg: dict[str, Any] | None = None,
        traj_vocab_size: int = 768,
        tokens_per_history_traj: int = 16,
        tokens_per_future_traj: int = 64,
        model_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        legacy_inference_image_input_format: bool = False,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
        loss_weights: dict[str, float] = {"future_traj": 1.0, "others": 1.0},
        vocab_size: int | None = None,
        traj_token_start_idx: int | None = None,
        traj_token_ids: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model configuration.

        Args:
            vlm_name_or_path: Path or name of the Vision Language Model to use as backbone.
            vlm_backend: Type of VLM backend ('qwen', 'internvl', etc.).
            traj_tokenizer_cfg: Configuration for the trajectory tokenizer.
            hist_traj_tokenizer_cfg: Configuration for the history trajectory tokenizer, if
                different from main tokenizer.
            hist_traj_embed_cfg: Configuration for the history trajectory embedding, if using a
                separate embedding method.
            traj_vocab_size: Size of the trajectory vocabulary.
            tokens_per_history_traj: Number of tokens allocated for encoding history trajectory.
            tokens_per_future_traj: Number of tokens allocated for encoding future trajectory.
            model_dtype: Data type for model parameters (e.g., 'bfloat16').
            attn_implementation: Attention implementation to use (e.g., 'flash_attention_2').
            legacy_inference_image_input_format: Whether to use the legacy image input format.
            min_pixels: Minimum number of pixels for image resizing.
            max_pixels: Maximum number of pixels for image resizing.
            include_camera_ids: Whether to include camera IDs as text before images.
            include_frame_nums: Whether to include frame numbers as text before images.
        """
        super().__init__(**kwargs)

        # Core configuration
        self.vlm_name_or_path = vlm_name_or_path
        self.vlm_backend = vlm_backend.lower()
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation
        # TODO (yurongy): check if this is still needed
        self.legacy_inference_image_input_format = legacy_inference_image_input_format

        # Trajectory tokenization configuration
        # A trajectory tokenizer is model-specific and must come from the model
        # configuration. The previous fallback referenced an external module
        # and a test checkpoint that are not part of this recipe.
        self.traj_tokenizer_cfg = traj_tokenizer_cfg
        self.hist_traj_tokenizer_cfg = hist_traj_tokenizer_cfg
        self.hist_traj_embed_cfg = hist_traj_embed_cfg

        # Token configuration
        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_history_traj = tokens_per_history_traj
        self.tokens_per_future_traj = tokens_per_future_traj
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums
        self.loss_weights = loss_weights
        self.vocab_size = vocab_size
        self.traj_token_start_idx = traj_token_start_idx
        self.traj_token_ids = traj_token_ids
        self._initialize_vlm_config()

    def _initialize_vlm_config(self) -> None:
        """Initialize VLM-specific configuration based on backend type."""
        if self.vlm_name_or_path is None:
            return
        if not os.path.isdir(self.vlm_name_or_path):
            # Preserve Hugging Face repository IDs without resolving or replacing
            # them during lightweight config construction.  Callers that need
            # tokenizer-derived fields pass a local/cache-resolved processor path.
            return

        # Initialize based on VLM backend
        if self.vlm_backend == "qwen":
            self._initialize_qwen_config()
        elif self.vlm_backend == "qwenvl3":
            self._initialize_qwenvl3_config()
        elif self.vlm_backend == "internvl":
            self._initialize_internvl_config()
        else:
            raise NotImplementedError(f"VLM backend '{self.vlm_backend}' not yet supported")

    def _initialize_qwen_config(self) -> None:
        """Initialize Qwen2.5-VL specific configuration."""
        # Move instantiation to yaml
        # NOTE make sure the same processor is passed to dataloader to preprocess data accordingly.
        from alpamayo1_distill.processors.qwen_processor import QwenProcessor

        if self.vlm_name_or_path is None:
            raise ValueError("vlm_name_or_path must be specified")
        processor_instance = QwenProcessor(
            vlm_name_or_path=self.vlm_name_or_path,
            traj_vocab_size=self.traj_vocab_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            include_camera_ids=self.include_camera_ids,
            include_frame_nums=self.include_frame_nums,
        )
        processor = processor_instance.processor
        self.vocab_size = len(processor.tokenizer)
        self.traj_token_start_idx = processor.tokenizer.traj_token_start_idx
        self.traj_token_ids = processor.tokenizer.traj_token_ids

    def _initialize_qwenvl3_config(self) -> None:
        """Initialize Qwen2-VL (QwenVL3) specific configuration."""
        self._initialize_qwen_config()

    def _initialize_internvl_config(self) -> None:
        """Initialize InternVL-specific configuration."""
        # For now, use Qwen processor as a fallback until InternVL processor is fully implemented
        # TODO: Replace with actual InternVL processor when ready
        from alpamayo1_distill.processors.qwen_processor import QwenProcessor

        if self.vlm_name_or_path is None:
            raise ValueError("vlm_name_or_path must be specified")
        processor_instance = QwenProcessor(
            vlm_name_or_path=self.vlm_name_or_path,
            traj_vocab_size=self.traj_vocab_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            include_camera_ids=self.include_camera_ids,
            include_frame_nums=self.include_frame_nums,
        )
        processor = processor_instance.processor
        self.vocab_size = len(processor.tokenizer)
        self.traj_token_start_idx = processor.tokenizer.traj_token_start_idx
        self.traj_token_ids = processor.tokenizer.traj_token_ids

    def get_processor_class(self) -> type[Any]:
        """Get the appropriate processor class for this configuration."""
        if self.vlm_backend == "qwenvl3":
            # Qwen3-VL uses the same processor as Qwen2.5-VL
            from alpamayo1_distill.processors.qwen_processor import QwenProcessor

            return QwenProcessor


            return InternVLProcessor
        else:
            raise NotImplementedError(f"Processor for '{self.vlm_backend}' not implemented")

    def get_llm_config(self) -> Any:
        """Return the underlying language model config for the VLM.

        This mirrors composite configs used in Cosmos that provide a get_llm_config()
        helper for downstream components (e.g., vLLM overrides) that need LLM fields
        like num_attention_heads, num_key_value_heads, hidden_size, etc.
        """
        # Cache to avoid repeated HF config loads (vLLM may call this multiple
        # times during initialization).
        cached: Any = getattr(self, "_cached_llm_config", None)
        if cached is not None:
            # NOTE: Some checkpoints accidentally serialize `_cached_llm_config`
            # into `config.json` as a plain dict. Downstream components (Cosmos
            # weight mapper, vLLM overrides) expect a `PretrainedConfig` object
            # with attributes like `text_config.num_key_value_heads`.
            # If we see a dict here, reconstruct a proper config object and
            # re-cache it.
            if isinstance(cached, dict):
                from transformers import AutoConfig

                model_type = cached.get("model_type", None)
                rebuilt = None
                # Best effort: rebuild from the dict itself.
                if model_type is not None:
                    try:
                        rebuilt = AutoConfig.for_model(model_type, **cached)
                    except Exception:
                        rebuilt = None
                # Fallback: reload from the backing VLM path.
                if rebuilt is None:
                    rebuilt = AutoConfig.from_pretrained(
                        self.vlm_name_or_path,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                setattr(self, "_cached_llm_config", rebuilt)
                return rebuilt
            return cached

        from transformers import AutoConfig

        base_cfg = AutoConfig.from_pretrained(
            self.vlm_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        setattr(self, "_cached_llm_config", base_cfg)
        return base_cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize config without transient caches.

        `_cached_llm_config` can be very large and, if saved, will be loaded back
        as a plain dict (breaking attribute-based consumers). Exclude it from
        serialization to keep checkpoints stable and minimal.
        """
        d: dict[str, Any] = super().to_dict()
        d.pop("_cached_llm_config", None)
        return d

    def get_text_config(self) -> Any:
        """Return the underlying *text* config.

        vLLM uses `config.get_text_config()` to populate `hf_text_config` and
        derive critical fields like `hidden_size`. For our composite config, the
        text config lives under the backing VLM config.
        """
        llm_cfg = self.get_llm_config()
        # Many multimodal configs (e.g., Qwen3-VL) expose a `get_text_config()`
        # that returns the nested text config (with hidden_size, heads, ...).
        if hasattr(llm_cfg, "get_text_config"):
            return llm_cfg.get_text_config()
        return llm_cfg


class ReasoningVLA(PreTrainedModel, TrajectoryFusionMixin):
    """Reasoning Vision-Language-Action model."""

    config_class: type[ReasoningVLAConfig] = ReasoningVLAConfig
    base_model_prefix: str = "vlm"

    def __init__(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        super().__init__(config)

        if pretrained_modules is not None:
            for module in pretrained_modules.values():
                if not isinstance(module, torch.nn.Module):
                    continue
                _recursive_setattr(module, "_is_hf_initialized", True)
        else:
            pretrained_modules = {}

        # Initialize VLM backbone
        self._initialize_vlm_backbone(config, pretrained_modules, original_vocab_size)

        # Initialize trajectory tokenizers
        self._initialize_trajectory_tokenizers(config, pretrained_modules)

        # Build tokenizer
        self.tokenizer = self._build_tokenizer(config)
        self.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()
        }

        # Log parameter count
        if print_param_count:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

    def _build_tokenizer(self, config: ReasoningVLAConfig) -> Any:
        """Build tokenizer with trajectory tokens."""
        processor_kwargs = {}
        if config.min_pixels is not None:
            processor_kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels is not None:
            processor_kwargs["max_pixels"] = config.max_pixels

        _local_files_only = not os.path.isdir(config.vlm_name_or_path)
        processor = AutoProcessor.from_pretrained(config.vlm_name_or_path, local_files_only=_local_files_only, **processor_kwargs)
        tokenizer = processor.tokenizer

        if config.traj_vocab_size is not None:
            discrete_tokens = [f"<i{v}>" for v in range(config.traj_vocab_size)]
            tokenizer.add_tokens(discrete_tokens)
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")

        if config.add_special_tokens:
            tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
        else:
            tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)

        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
        }

        return tokenizer

    def _initialize_vlm_backbone(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, Any],
        original_vocab_size: int | None = None,
    ) -> None:
        """Initialize the VLM backbone based on configuration."""
        if "vlm" in pretrained_modules:
            self.vlm = pretrained_modules["vlm"]
            self.original_vocab_size = original_vocab_size
        else:
            self._initialize_qwenvl3_vlm(config)

    def _initialize_qwenvl3_vlm(self, config: ReasoningVLAConfig) -> None:
        """Initialize Qwen3-VL VLM backbone.

        Qwen3-VL uses Qwen3VLForConditionalGeneration from transformers.
        See: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
        """
        if config.vlm_name_or_path is None:
            raise ValueError("config.vlm_name_or_path must be set before initializing Qwen3-VL backbone")
        vlm_config = Qwen3VLConfig.from_pretrained(
            config.vlm_name_or_path,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
            local_files_only=True,
        )
        self.original_vocab_size = vlm_config.text_config.vocab_size
        vlm_config.text_config.vocab_size = config.vocab_size
        vlm_config.vocab_size = config.vocab_size
        self.vlm = Qwen3VLForConditionalGeneration(vlm_config)

    def _initialize_trajectory_tokenizers(
        self, config: ReasoningVLAConfig, pretrained_modules: dict[str, Any]
    ) -> None:
        """Initialize trajectory tokenizers."""
        if "traj_tokenizer" in pretrained_modules:
            self.traj_tokenizer = pretrained_modules["traj_tokenizer"]
        elif config.traj_tokenizer_cfg is not None:
            self.traj_tokenizer = hyu.instantiate(config.traj_tokenizer_cfg, load_weights=False)
        else:
            self.traj_tokenizer = None

        self.future_token_start_idx = self.hist_token_start_idx = config.traj_token_start_idx

        if config.hist_traj_tokenizer_cfg is not None:
            self.hist_traj_tokenizer = hyu.instantiate(config.hist_traj_tokenizer_cfg)
            if self.traj_tokenizer is not None:
                self.hist_token_start_idx += self.traj_tokenizer.vocab_size
        else:
            self.hist_traj_tokenizer = self.traj_tokenizer

    @classmethod
    def from_pretrained_submodules(
        cls,
        config: ReasoningVLAConfig,
    ) -> "ReasoningVLA":
        """Load submodules with pretrained submodules and initialize the model."""
        pretrained_modules = {}

        # Load VLM
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_name_or_path,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
            local_files_only=True,
        )

        original_vocab_size = vlm.config.text_config.vocab_size
        vlm.resize_token_embeddings(config.vocab_size)
        vlm.config.text_config.vocab_size = config.vocab_size
        vlm.config.vocab_size = config.vocab_size
        pretrained_modules["vlm"] = vlm

        if config.traj_tokenizer_cfg is not None:
            traj_tokenizer = hyu.instantiate(config.traj_tokenizer_cfg)
            pretrained_modules["traj_tokenizer"] = traj_tokenizer

        return cls(
            config,
            pretrained_modules=pretrained_modules,
            original_vocab_size=original_vocab_size,
        )

    def get_output_embeddings(self) -> torch.nn.Module:
        """Get the output embeddings of the model."""
        return self.vlm.get_output_embeddings()

    def get_input_embeddings(self) -> torch.nn.Module:
        """Get the input embeddings of the model."""
        return self.vlm.language_model.embed_tokens

    def tie_weights(self) -> None:
        """Delegate weight tying to the nested VLM model."""
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()
