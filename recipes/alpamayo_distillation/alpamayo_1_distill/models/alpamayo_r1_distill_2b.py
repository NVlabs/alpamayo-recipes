import copy
import logging
from typing import Any
from contextlib import nullcontext

import einops
from dataclasses import dataclass
import hydra.utils as hyu
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    StoppingCriteriaList,
    PreTrainedModel,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,

)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.utils import ModelOutput

from alpamayo1_distill.action_space import ActionSpace
from alpamayo1_distill.models.base_model import TrajectoryFusionMixin, ReasoningVLAConfig
from alpamayo1_distill.models.alpamayo_r1 import AlpamayoR1
from alpamayo1_distill.commons import logging

from alpamayo1_distill.config import (
    AlpamayoR1Config,
    ExpertModelConfig,
    validate_supported_student_vlm_config,
)
from alpamayo1_distill.diffusion.base import BaseDiffusion
from alpamayo1_distill.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo1_distill.losses import (
    OnPolicyDistillConfig,
    masked_normalized_hidden_distance,
    sequence_distillation_loss,
)
from alpamayo1_distill.models.expert_context import (
    build_expert_attention_mask,
    build_expert_position_ids,
    extend_prompt_attention_mask,
    find_future_start_offsets,
    find_prompt_end_offsets,
)
from alpamayo1_distill.vla_constant import IGNORE_INDEX, SPECIAL_TOKENS
from alpamayo1_distill.processors.qwen_processor import QwenProcessor
from alpamayo1_distill.tokenizer.trajectory import traj_tokenizer_helper
from alpamayo1_distill.tokenizer.trajectory.base_trajectory_tokenizer import BaseTrajectoryTokenizer
from alpamayo1_distill.utils.extract_token import extract_traj_tokens
from alpamayo1_distill.processors.logit_processor import MaskDiscreteTrajectoryLogitsProcessor
from alpamayo1_distill.utils.model import get_param_count, to
from alpamayo1_distill.defaults import (
    DEFAULT_COSMOS_STUDENT_MODEL,
    SUPPORTED_ACTION_STACK_PARAMETER_COUNT,
)

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")

def _recursive_setattr(module: torch.nn.Module, attr: str, value: Any) -> None:
    """Set an attribute recursively on all children of a module."""
    for child in module.children():
        _recursive_setattr(child, attr, value)
    setattr(module, attr, value)



@dataclass
class ReasoningVLAOutput(ModelOutput):
    """Output of the ReasoningVLA model."""
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    loss_dict: dict[str, torch.FloatTensor] | None = None

def _shape_str(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if isinstance(x, (list, tuple)):
        # show container length + first element shape if tensor
        if len(x) == 0:
            return f"{type(x).__name__}[0]"
        first = x[0]
        if isinstance(first, torch.Tensor):
            return f"{type(x).__name__}[{len(x)}] of Tensor{tuple(first.shape)} dtype={first.dtype} device={first.device}"
        return f"{type(x).__name__}[{len(x)}] of {type(first).__name__}"
    if isinstance(x, dict):
        return f"dict(keys={list(x.keys())[:8]}{'...' if len(x) > 8 else ''})"
    return f"{type(x).__name__}: {str(x)[:120]}"

def print_outputs_shapes(outputs, prefix="vlm_outputs"):
    # Transformers often return ModelOutput (dict-like) or tuple
    if hasattr(outputs, "to_dict"):
        outputs = outputs.to_dict()

    if isinstance(outputs, dict):
        print(f"{prefix}: dict with keys={list(outputs.keys())}")
        for k, v in outputs.items():
            print(f"  - {k}: {_shape_str(v)}")
    elif isinstance(outputs, (list, tuple)):
        print(f"{prefix}: {type(outputs).__name__} len={len(outputs)}")
        for i, v in enumerate(outputs):
            print(f"  [{i}]: {_shape_str(v)}")
    else:
        print(f"{prefix}: {_shape_str(outputs)}")

class ReasoningVLA(PreTrainedModel, TrajectoryFusionMixin):
    """Distillation Reasoning Vision-Language-Action model.

    This model combines a Vision Language Model (VLM) backbone with trajectory
    tokenization capabilities for autonomous vehicle planning tasks.
    """

    config_class: type[ReasoningVLAConfig] = ReasoningVLAConfig
    base_model_prefix: str = "vlm"

    def __init__(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        """Initialize the model."""
        super().__init__(config)

        self.legacy_inference_image_input_format = config.legacy_inference_image_input_format

        if pretrained_modules is not None:
            # NOTE: we need to set _is_hf_initialized to True otherwise it will be re-initialized
            # by `post_init`.
            for module in pretrained_modules.values():
                if not isinstance(module, torch.nn.Module):
                    continue
                _recursive_setattr(module, "_is_hf_initialized", True)
        else:
            pretrained_modules = {}

        # 1. Initialize the VLM backbone
        self._initialize_vlm_backbone(config, pretrained_modules, original_vocab_size)

        # 2. Initialize the trajectory tokenizers
        self._initialize_trajectory_tokenizers(config, pretrained_modules)

        # 3. Log parameter count
        if print_param_count:
            logger.info("Model parameter count:")
            param_count = get_param_count(self)
            for key, value in param_count.items():
                logger.info(f"{key}: {value:,}")

        # 4. get tokenizer for processing generated sequences
        # NOTE: here we hard code the processor to QwenProcessor
        #print("in init qwen processor:", config.vlm_name_or_path)
        self.processor = QwenProcessor(
            vlm_name_or_path=config.vlm_name_or_path,
            traj_vocab_size=config.traj_vocab_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            include_camera_ids=config.include_camera_ids,
            include_frame_nums=config.include_frame_nums,
        )
        self.tokenizer = self.processor.processor.tokenizer
        self.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()
        }

    def _get_generation_mode_tokenized_data_online(
        self, data: dict[str, Any], last_component: str = "traj_future"
    ) -> dict[str, Any]:
        """Get `tokenized_data` for online inference when it is not prepared by the dataloader."""
        if self.config.legacy_inference_image_input_format:
            # The legacy image input format is in the range [-1, 1]
            # We need to convert it to the range [0, 1]
            data["image_frames"] = (data["image_frames"] + 1.0) / 2.0
        num_tokens_per_history_traj = self.config.tokens_per_history_traj
        num_tokens_per_future_traj = self.config.tokens_per_future_traj
        if last_component == "traj_future":
            components_order = ["image", "traj_history", "prompt", "traj_future"]
            components_prompt = ["traj_future"]
        elif last_component == "cot":
            components_order = ["image", "traj_history", "prompt", "cot"]
            components_prompt = ["cot", "traj_future"]
        else:
            raise ValueError(f"Unsupported last_component: {last_component}")

        preprocess_data_fn = self.processor.get_preprocess_data_fn(
            num_tokens_per_history_traj=num_tokens_per_history_traj,
            num_tokens_per_future_traj=num_tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            label_components=[],
            generation_mode=True,
        )
        batch_size = data["image_frames"].shape[0]
        tokenized_data = []
        for i in range(batch_size):
            _data = {
                "image_frames": data["image_frames"][i],
                "camera_indices": data["camera_indices"][i],
            }
            tokenized_data.append({"tokenized_data": preprocess_data_fn(data=_data)})
        batched_tokenized_data: dict[str, Any] = to(
            self.processor.collate_fn(tokenized_data),
            device=data["image_frames"].device,
        )["tokenized_data"]
        return batched_tokenized_data

    def _get_generation_mode_tokenized_data_online_distill(
        self, data: dict[str, Any], helper, processor, device
    ) -> dict[str, Any]:
        """Get `tokenized_data` for online inference when it is not prepared by the dataloader."""
        messages = helper.create_message(data["image_frames"].squeeze(0).flatten(0, 1))
        tokenized_data = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(tokenized_data, "data"):
            td = tokenized_data.data
        else:
            td = tokenized_data

        for k, v in list(td.items()):
            if torch.is_tensor(v):
                td[k] = v.to(device, non_blocking=True)
        return tokenized_data

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
            validate_supported_student_vlm_config(self.vlm.config)
        else:
            self._initialize_cosmos_vlm(config)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable gradient checkpointing for the model.

        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments for gradient checkpointing.
        """
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the model."""
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def _initialize_cosmos_vlm(self, config: ReasoningVLAConfig) -> None:
        """Initialize the Cosmos-Reason2-2B Qwen3-VL backbone skeleton."""
        vlm_config = Qwen3VLConfig.from_pretrained(
            config.vlm_name_or_path,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
            local_files_only=True,
        )
        validate_supported_student_vlm_config(vlm_config)
        self.original_vocab_size = vlm_config.text_config.vocab_size
        vlm_config.text_config.vocab_size = config.vocab_size
        vlm_config.vocab_size = config.vocab_size
        self.vlm = Qwen3VLForConditionalGeneration(vlm_config)

    def _initialize_trajectory_tokenizers(
        self, config: ReasoningVLAConfig, pretrained_modules: dict[str, Any]
    ) -> None:
        """Initialize trajectory tokenizers."""
        # Main trajectory tokenizer
        self.traj_tokenizer: BaseTrajectoryTokenizer
        if "traj_tokenizer" in pretrained_modules:
            self.traj_tokenizer = pretrained_modules["traj_tokenizer"]
        else:
            #print(config.traj_tokenizer_cfg)
            self.traj_tokenizer = hyu.instantiate(config.traj_tokenizer_cfg, load_weights=False)

        # Set up token indices
        self.future_token_start_idx = self.hist_token_start_idx = config.traj_token_start_idx
        total_traj_vocab_size = self.traj_tokenizer.vocab_size
        #print("total_traj_vocab_size:", total_traj_vocab_size)
        # History trajectory tokenizer (optional separate tokenizer)
        if config.hist_traj_tokenizer_cfg is not None:
            #print("hist_traj_tokenizer_cfg is not None")
            self.hist_traj_tokenizer = hyu.instantiate(config.hist_traj_tokenizer_cfg)
            self.hist_token_start_idx += self.traj_tokenizer.vocab_size
            total_traj_vocab_size += self.hist_traj_tokenizer.vocab_size
        else:
            #print("hist_traj_tokenizer_cfg is None")
            self.hist_traj_tokenizer = self.traj_tokenizer
        #print("total_traj_vocab_size:, ", total_traj_vocab_size)
        #print(total_traj_vocab_size, config.traj_vocab_size)
        assert total_traj_vocab_size == config.traj_vocab_size

        # Handle history trajectory embedding (alternative to tokenizer)
        if config.hist_traj_embed_cfg is not None:
            # When using embedding instead of tokenization, disable the tokenizer
            self.hist_traj_tokenizer = None

    @classmethod
    def from_pretrained_submodules(
        cls,
        config: ReasoningVLAConfig,
    ) -> "ReasoningVLA":
        """Load submodules with pretrained submodules and initialize the model.

        Args:
            config: ReasoningVLAConfig

        Returns:
            ReasoningVLA: The initialized model.
        """
        pretrained_modules = {}

        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm_name_or_path,
            dtype=config.model_dtype,
            attn_implementation=config.attn_implementation,
            local_files_only=True,
        )
        validate_supported_student_vlm_config(vlm.config)
        original_vocab_size = vlm.config.text_config.vocab_size
        vlm.resize_token_embeddings(config.vocab_size)
        vlm.config.text_config.vocab_size = config.vocab_size
        vlm.config.vocab_size = config.vocab_size
        pretrained_modules["vlm"] = vlm

        # Load trajectory tokenizer
        traj_tokenizer: BaseTrajectoryTokenizer = hyu.instantiate(config.traj_tokenizer_cfg)
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
        """Delegate weight tying to the nested VLM model.

        This method is automatically called by PreTrainedModel.post_init() during initialization.

        To control weight tying behavior, modify the Cosmos VLM config before
        model creation.
        """
        # Delegate to the nested VLM model - it knows its own config
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    def freeze_base_model_except_embeddings(self) -> None:
        """Only train the embeddings for new tokens."""
        for param in self.parameters():
            param.requires_grad = False

        self.vlm.language_model.embed_tokens.weight.requires_grad = True

        def reset_grad(grad: torch.Tensor) -> torch.Tensor:
            grad[: self.original_vocab_size] = 0
            return grad

        self.vlm.language_model.embed_tokens.weight.register_hook(reset_grad)

    @torch._dynamo.disable
    def _compute_next_token_loss(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for the next token prediction.

        Args:
            outputs: ModelOutput containing logits of shape (B, L, V)
            labels: [B, L]
            labels_mask: [B, L], indicates which tokens in the sequence are valid for loss
                computation
            token_mask: [V], indicates which tokens ids are valid for logits computation

        Returns:
            torch.Tensor: (,) loss value
        """
        if labels_mask is None:
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
        if labels_mask[:, 1:].sum() == 0:
            return torch.tensor(0.0, device=labels.device)
        # Shift labels to the left by 1 position (predict next token)
        shift_labels = labels[..., 1:]
        shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
        # The logits should also be trimmed to match the shifted labels. Avoid cloning
        # the full [B, L, V] tensor unless token_mask needs an in-place update.
        shift_logits = outputs.logits[..., :-1, :]
        shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        if token_mask is not None:
            shift_logits = shift_logits.clone()
            shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
        loss = torch.nan_to_num(
            torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean"
            ),
            nan=0.0,
        )
        return loss

    def _distill_loss(
            self,
            teacher_output: ModelOutput,
            student_output: ModelOutput,
        ):
        loss = 0
        return loss

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """Forward pass of the model."""
        # 1. tokenize trajectory and fuse into input_ids
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 2. get labels
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # 3. vlm forward pass
        outputs = self.vlm(input_ids=input_ids, labels=labels, **tokenized_data)

        losses = {}
        # Identify trajectory tokens (tokens between traj_future and next special token)
        traj_mask = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        losses["future_traj"] = self._compute_next_token_loss(
            outputs, labels, traj_mask
        ) * self.config.loss_weights.get("future_traj", 1.0)
        labels[traj_mask] = IGNORE_INDEX

        # Include all other tokens in the loss
        losses["others"] = self._compute_next_token_loss(
            outputs, labels, labels != IGNORE_INDEX
        ) * self.config.loss_weights.get("others", 1.0)

        # Replace the original loss
        outputs.loss = sum(losses.values())

        return ReasoningVLAOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )

    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        last_component: str = "traj_future",
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[Any, Any, Any, dict[str, list[str]]]
    ):
        """Sample trajectories from the data.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        if "tokenized_data" not in data:
            tokenized_data = self._get_generation_mode_tokenized_data_online(
                data, last_component=last_component
            )
        else:
            tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        n_samples_total = num_traj_samples * num_traj_sets
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        assert max_generation_length >= self.config.tokens_per_future_traj
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = n_samples_total
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generated = self.vlm.generate(
            input_ids=input_ids, **tokenized_data, generation_config=generation_config
        )
        # remove input ids from the generated sequences
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        # extract trajectory tokens from generated sequences
        traj_token_ids = extract_traj_tokens(
            generated_tokens,
            self.special_token_ids,
            self.config.tokens_per_future_traj,
            self.future_token_start_idx,
            self.traj_tokenizer.vocab_size,
        )

        pred_xyz, pred_rot, _ = self.traj_tokenizer.decode(
            hist_xyz=einops.repeat(
                ego_history_xyz[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            hist_rot=einops.repeat(
                ego_history_rot[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            tokens=traj_token_ids,
        )
        pred_xyz = einops.rearrange(
            pred_xyz,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        pred_rot = einops.rearrange(
            pred_rot,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        # TODO: compute logprob (i.e. generated.logits needs to have valid shape)
        # logits = torch.stack(generated.logits, dim=1)
        # logprob = F.log_softmax(logits, dim=-1).gather(-1, generated_ids[:, :, None]).squeeze(-1)
        # logprob = einops.rearrange(
        #     logprob, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        # )
        logprob = torch.zeros_like(pred_xyz[..., 0])

        # return additional information
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, generated_tokens)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, logprob, extra
        return pred_xyz, pred_rot, logprob

    def sample_text_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Sample text from the data.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            extra: The extra information including the text outputs.
        """
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")

        n_samples_total = num_traj_samples * num_traj_sets
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        assert max_generation_length >= self.config.tokens_per_future_traj
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = n_samples_total
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        logits_processor = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        generated = self.vlm.generate(
            input_ids=input_ids,
            **tokenized_data,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
        # remove input ids from the generated sequences
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        # return additional information
        extra: dict[str, Any] = extract_text_tokens(self.tokenizer, generated_tokens)
        # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
        for text_tokens, value in extra.items():
            extra[text_tokens] = np.array(value).reshape(
                [input_ids.shape[0], num_traj_sets, num_traj_samples]
            )
        return extra


# Register the model with Auto classes
AutoConfig.register("alpamayo_reasoning_vla", ReasoningVLAConfig)
AutoModel.register(ReasoningVLAConfig, ReasoningVLA)


class ExpertModelDistill(ReasoningVLA):
    """Expert model for reasoning VLA."""

    config_class: type[ExpertModelConfig] = ExpertModelConfig
    base_model_prefix = "vlm"

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Any, *model_args: Any, **kwargs: Any
    ) -> Any:
        """Load a complete distillation checkpoint without reloading its base VLM.

        Transformers constructs ``from_pretrained`` models on the meta device.
        Loading and resizing another pretrained VLM inside that context performs
        data-dependent initialization on meta tensors. A complete distillation
        checkpoint already contains those VLM weights, so only its architecture
        should be constructed before Transformers materializes the saved state.
        """
        kwargs["initialize_from_base_vlm"] = False
        validate_materialization = kwargs.get("device_map") is None
        loaded = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model = loaded[0] if isinstance(loaded, tuple) else loaded

        meta_tensors = []
        if validate_materialization:
            meta_tensors.extend(
                f"parameter:{name}"
                for name, tensor in model.named_parameters()
                if tensor.is_meta
            )
            meta_tensors.extend(
                f"buffer:{name}" for name, tensor in model.named_buffers() if tensor.is_meta
            )
        if meta_tensors:
            preview = ", ".join(meta_tensors[:10])
            if len(meta_tensors) > 10:
                preview += f", ... ({len(meta_tensors)} total)"
            raise RuntimeError(
                "Checkpoint loading left unmaterialized meta tensors and cannot "
                f"continue safely: {preview}"
            )
        return loaded

    def __init__(
        self,
        config: ExpertModelConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
        teacher_model: AlpamayoR1 | None = None,
        student_model_path: str = DEFAULT_COSMOS_STUDENT_MODEL,
        processor_model_path: str = DEFAULT_COSMOS_STUDENT_MODEL,
        distill_T: float = 0.5,
        initialize_from_base_vlm: bool = True,
    ):
        config.validate_supported_architecture()
        config.vlm_name_or_path = student_model_path

        # Fresh training must start from the pretrained student VLM. Full checkpoint
        # loading instead builds an empty, final-vocabulary skeleton and lets the
        # outer from_pretrained call restore every saved tensor.
        if initialize_from_base_vlm and (
            pretrained_modules is None or "vlm" not in pretrained_modules
        ):
            pretrained_modules = dict(pretrained_modules) if pretrained_modules else {}
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                student_model_path,
                dtype=config.model_dtype,
                attn_implementation=config.attn_implementation,
                local_files_only=True,
            )
            validate_supported_student_vlm_config(vlm.config)
            original_vocab_size = vlm.config.text_config.vocab_size
            vlm.resize_token_embeddings(config.vocab_size)
            vlm.config.text_config.vocab_size = config.vocab_size
            vlm.config.vocab_size = config.vocab_size
            pretrained_modules["vlm"] = vlm

        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        self.processor = QwenProcessor(
            vlm_name_or_path=processor_model_path,
            traj_vocab_size=config.traj_vocab_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            include_camera_ids=config.include_camera_ids,
            include_frame_nums=config.include_frame_nums,
        )
        self.distill_T = distill_T
        self.tokenizer = self.processor.processor.tokenizer
        self.special_token_ids = {
            k: self.tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL_TOKENS.items()
        }

        # we only need the text config for the expert model
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # we don't need the embed_tokens of the expert model
        del self.expert.embed_tokens
        # TODO: pi0.5 use a different normalization layer for each of the decoder layer in the
        # expert model. We can update this later.
        # Training supplies the teacher explicitly. Student-only inference leaves
        # it unset, avoiding an implicit 10B model load and keeping evaluation
        # independent of training-only weights.
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)  # freeze everything

            student_vocab = self.vlm.get_input_embeddings().num_embeddings
            teacher_vocab = self.teacher_model.vlm.get_input_embeddings().num_embeddings
            if student_vocab != teacher_vocab:
                raise ValueError(
                    f"Vocabulary size mismatch: student has {student_vocab} tokens, "
                    f"teacher has {teacher_vocab} tokens. KL distillation requires "
                    f"identical vocabularies (same tokenizer + same special tokens)."
                )

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )
        if self.diffusion.use_classifier_free_guidance:
            raise ValueError("Using classifier free guidance in diffusion is not fully tested yet.")

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )
        if config.expert_hist_traj_tokenizer_cfg is not None:
            self.expert_hist_traj_tokenizer = hyu.instantiate(
                config.expert_hist_traj_tokenizer_cfg, outdim=expert_config.hidden_size
            )
        else:
            self.expert_hist_traj_tokenizer = None

        action_stack_modules = [self.expert, self.action_in_proj, self.action_out_proj]
        if self.expert_hist_traj_tokenizer is not None:
            action_stack_modules.append(self.expert_hist_traj_tokenizer)
        self.action_stack_parameter_count = sum(
            parameter.numel()
            for module in action_stack_modules
            for parameter in module.parameters()
        )
        if self.action_stack_parameter_count != SUPPORTED_ACTION_STACK_PARAMETER_COUNT:
            raise RuntimeError(
                "Constructed action stack does not match the supported approximately "
                "100M expert: "
                f"expected={SUPPORTED_ACTION_STACK_PARAMETER_COUNT:,}, "
                f"found={self.action_stack_parameter_count:,}"
            )

        # Convert action-related modules to the same dtype as expert
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)
            if self.expert_hist_traj_tokenizer is not None:
                self.expert_hist_traj_tokenizer = self.expert_hist_traj_tokenizer.to(
                    dtype=expert_dtype
                )

        # Hidden-state distillation projection (student_hidden → teacher_hidden).
        # Only created when hidden_distill_weight > 0 and teacher is available.
        self.hidden_proj: nn.Linear | None = None
        if self.teacher_model is not None and getattr(config, "hidden_distill_weight", 0.0) > 0:
            student_h = self.vlm.config.text_config.hidden_size
            teacher_h = self.teacher_model.vlm.config.text_config.hidden_size
            self.hidden_proj = nn.Linear(student_h, teacher_h, bias=False).to(dtype=expert_dtype)
            nn.init.normal_(self.hidden_proj.weight, std=student_h ** -0.5)

        # freeze the vlm model
        if not self.config.cotrain_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False

        # On-policy distillation is off unless the trainer installs a config.
        # Kept off the model config because it only affects training-time behaviour
        # and must not be serialised into checkpoints.
        self.on_policy_cfg = OnPolicyDistillConfig()

        # print the param count
        logger.info("Expert parameter count:")
        param_count = get_param_count(self)
        for key, value in param_count.items():
            logger.info(f"{key}: {value:,}")
        self.post_init()

    def train(self, mode: bool = True) -> "ExpertModelDistill":
        super().train(mode)
        # Keep the teacher permanently in eval mode regardless of outer train() calls.
        if hasattr(self, "teacher_model") and self.teacher_model is not None:
            self.teacher_model.eval()
        return self

    # `from_pretrained` can construct instances through paths that bypass our
    # __init__, so both on-policy attributes resolve lazily to a disabled default.
    @property
    def on_policy_cfg(self) -> OnPolicyDistillConfig:
        """Training-time on-policy distillation settings (disabled by default)."""
        cfg = getattr(self, "_on_policy_cfg", None)
        if cfg is None:
            cfg = OnPolicyDistillConfig()
            self._on_policy_cfg = cfg
        return cfg

    @on_policy_cfg.setter
    def on_policy_cfg(self, value: OnPolicyDistillConfig) -> None:
        if not isinstance(value, OnPolicyDistillConfig):
            raise TypeError(f"on_policy_cfg must be an OnPolicyDistillConfig, got {type(value)}")
        self._on_policy_cfg = value

    @property
    def _on_policy_fallbacks(self) -> int:
        """Count of on-policy steps that fell back to the teacher (malformed rollout)."""
        return getattr(self, "_on_policy_fallback_count", 0)

    @_on_policy_fallbacks.setter
    def _on_policy_fallbacks(self, value: int) -> None:
        self._on_policy_fallback_count = int(value)

    def _process_traj_future_training(self, traj_data: dict[str, Any]) -> dict[str, Any]:
        """Process the trajectory future data for training."""
        ego_history_xyz = traj_data["ego_history_xyz"]
        ego_history_rot = traj_data["ego_history_rot"]
        ego_future_xyz = traj_data["ego_future_xyz"]
        ego_future_rot = traj_data["ego_future_rot"]
        action = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz,
            traj_history_rot=ego_history_rot,
            traj_future_xyz=ego_future_xyz,
            traj_future_rot=ego_future_rot,
        )
        action = action.reshape(-1, *self.action_space.get_action_space_dims())
        training_data: dict[str, Any] = self.diffusion.construct_training_data(action)
        return training_data

    def _build_gt_future_traj_token_suffix(
        self, traj_data: dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        """Tokenize GT future trajectory as SFT-style VLM supervision tokens."""
        future_tokens = traj_tokenizer_helper.tokenize_future_trajectory(
            self.traj_tokenizer,
            traj_data,
            self.future_token_start_idx,
        ).to(device=device, dtype=torch.long)
        future_end = torch.full(
            (future_tokens.shape[0], 1),
            self.special_token_ids["traj_future_end"],
            dtype=future_tokens.dtype,
            device=device,
        )
        return torch.cat([future_tokens, future_end], dim=1)

    def _compute_future_traj_vlm_loss_from_prefix(
        self,
        vlm_outputs: Any,
        seq: torch.Tensor,
        attention_mask: torch.Tensor,
        future_start_offsets: torch.LongTensor,
        gt_future_suffix: torch.Tensor,
        tokenized_data: dict[str, Any],
    ) -> torch.Tensor:
        """Compute SFT-style GT future-token CE without re-forwarding the prefix."""
        batch_size, suffix_len = gt_future_suffix.shape
        device = seq.device
        if suffix_len <= 0:
            return seq.sum() * 0.0

        batch_idx = torch.arange(batch_size, device=device)
        future_start_offsets = future_start_offsets.to(device=device)
        marker_positions = future_start_offsets - 1
        prev_positions = marker_positions - 1
        if bool((prev_positions < 0).any()):
            bad = (prev_positions < 0).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                f"future trajectory aux loss: marker has no previous token for batch indices {bad}"
            )

        # Match SFT's traj_mask: include <traj_future_start>, every GT future
        # discrete token, and <traj_future_end>. The marker itself is predicted
        # from the previous CoC token and is also part of the regular VLM loss.
        marker_labels = torch.full(
            (batch_size,),
            self.special_token_ids["traj_future_start"],
            dtype=torch.long,
            device=device,
        )
        marker_logits = vlm_outputs.logits[batch_idx, prev_positions, :].float()
        loss_sum = F.cross_entropy(marker_logits, marker_labels, reduction="sum")

        first_future_labels = gt_future_suffix[:, 0].to(device=device, dtype=torch.long)
        first_future_logits = vlm_outputs.logits[batch_idx, marker_positions, :].float()
        loss_sum = loss_sum + F.cross_entropy(
            first_future_logits, first_future_labels, reduction="sum"
        )
        token_count = torch.tensor(
            batch_size * 2,
            device=device,
            dtype=loss_sum.dtype,
        )

        if suffix_len == 1:
            return loss_sum / token_count.clamp_min(1.0)

        kv_cache = vlm_outputs.past_key_values
        prefill_seq_len = kv_cache.get_seq_length()
        suffix_input_ids = gt_future_suffix[:, :-1].to(device=device, dtype=torch.long)
        suffix_labels = gt_future_suffix[:, 1:].to(device=device, dtype=torch.long)
        suffix_input_len = suffix_input_ids.shape[1]

        cache_positions = torch.arange(prefill_seq_len, device=device)
        visible_cache = cache_positions.unsqueeze(0) < future_start_offsets.unsqueeze(1)
        visible_cache &= attention_mask[:, :prefill_seq_len].to(device=device, dtype=torch.bool)
        suffix_visible = torch.ones(
            (batch_size, suffix_input_len),
            dtype=torch.bool,
            device=device,
        )
        suffix_attention_mask = torch.cat((visible_cache, suffix_visible), dim=1)
        suffix_attention_mask = suffix_attention_mask.to(dtype=attention_mask.dtype)

        suffix_position_ids = build_expert_position_ids(
            future_start_offsets,
            vlm_outputs.rope_deltas,
            query_seq_len=suffix_input_len,
            device=device,
        )

        suffix_td = dict(tokenized_data)
        suffix_td["attention_mask"] = suffix_attention_mask
        suffix_td["position_ids"] = suffix_position_ids
        for k in (
            "cache_position",
            "rope_deltas",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_per_grid_ts",
        ):
            suffix_td.pop(k, None)

        suffix_outputs = self.vlm(
            input_ids=suffix_input_ids,
            past_key_values=kv_cache,
            use_cache=False,
            **suffix_td,
        )
        suffix_logits = suffix_outputs.logits.float()
        loss_sum = loss_sum + F.cross_entropy(
            suffix_logits.reshape(-1, suffix_logits.shape[-1]),
            suffix_labels.reshape(-1),
            reduction="sum",
        )
        suffix_outputs.logits = None

        # use_cache=False should leave the prefix cache untouched. This guard keeps
        # the expert path inference-aligned even if a backend mutates the cache.
        if kv_cache.get_seq_length() != prefill_seq_len:
            kv_cache.crop(prefill_seq_len)

        token_count = token_count + suffix_labels.numel()
        return loss_sum / token_count.clamp_min(1.0)

    def teacher_model_forward(
        self,
        input_ids: torch.Tensor,
        tokenized_data: dict[str, Any] | None = None,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        td = {} if tokenized_data is None else dict(tokenized_data)
        td.update(kwargs)

        td["input_ids"] = input_ids
        td.pop("labels", None)          # IMPORTANT: teacher should not compute CE loss
        td.pop("token_type_ids", None)  # optional
        if output_hidden_states:
            td["output_hidden_states"] = True

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.teacher_model.vlm(**td)

        # extra safety
        if hasattr(out, "logits") and isinstance(out.logits, torch.Tensor):
            out.logits = out.logits.detach()
        return out

    def teacher_model_generate(
        self,
        input_ids: torch.Tensor,
        tokenized_data: dict[str, Any] | None = None,
        **kwargs,
    ):
        ban_ids = [151652, 151653, 151654, 151655]  # vision_start/end/pad + image_pad
        bad_words_ids = [[i] for i in ban_ids]
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.reasoning_max_generation_length
        )
        logits_processor = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        generation_config = copy.deepcopy(self.teacher_model.vlm.generation_config)
        generation_config.top_p = 1.0
        generation_config.temperature = 0.0
        generation_config.do_sample = False
        generation_config.num_beams = 1
        generation_config.num_return_sequences = 1
        # The semantic boundary is <traj_future_start>, not the base model's ordinary
        # EOS token. StopAfterEOS below owns per-row termination.
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        # Reserve one decode step so a marker emitted at the reasoning limit follows
        # the same stop-after-marker contract as student rollout.
        generation_config.max_new_tokens = max_generation_length + 1
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = False
        generation_config.top_k = None
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future_start"])
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        generation_inputs = {} if tokenized_data is None else dict(tokenized_data)
        generation_inputs.pop("input_ids", None)
        # Direct kwargs are applied after Transformers 4.57.1 merges model defaults
        # back into a supplied GenerationConfig. Keep every deliberate default-valued
        # override here (especially ordinary EOS and greedy decoding).
        generation_inputs.update(
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=max_generation_length + 1,
            eos_token_id=None,
            forced_eos_token_id=None,
            output_logits=False,
            return_dict_in_generate=False,
        )
        sequences = self.teacher_model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            bad_words_ids=bad_words_ids,
            **generation_inputs,
        )
        return sequences

    def student_model_generate(
        self,
        input_ids: torch.Tensor,
        tokenized_data: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ):
        """Sample a reasoning sequence from the *student* for on-policy distillation.

        Mirrors :meth:`teacher_model_generate` — same banned vision tokens, same
        trajectory-token logits processor, same stop-after-``<traj_future_start>``
        contract, same reserved trailing slot — but decodes from ``self.vlm`` with
        sampling instead of greedily from the teacher. Sampling (rather than greedy)
        matters: the point is to visit the states the student actually reaches at
        inference, and the defaults mirror the evaluation rollout settings.

        Runs under ``no_grad``: the sampled tokens are training *data*: gradients
        reach the student through the subsequent scored forward pass, not through
        the sampling itself.
        """
        cfg = self.on_policy_cfg
        temperature = cfg.sample_temperature if temperature is None else temperature
        top_p = cfg.sample_top_p if top_p is None else top_p

        ban_ids = [151652, 151653, 151654, 151655]  # vision_start/end/pad + image_pad
        bad_words_ids = [[i] for i in ban_ids]
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.reasoning_max_generation_length
        )
        logits_processor = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        generation_config = copy.deepcopy(self.vlm.generation_config)
        generation_config.do_sample = True
        generation_config.temperature = temperature
        generation_config.top_p = top_p
        generation_config.top_k = None
        generation_config.num_beams = 1
        generation_config.num_return_sequences = 1
        # As on the teacher path, <traj_future_start> is the semantic boundary rather
        # than the base model's ordinary EOS; StopAfterEOS owns per-row termination.
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        # Reserve one decode step so a marker emitted at the reasoning limit still has
        # a trailing token to flush it into the KV cache.
        generation_config.max_new_tokens = max_generation_length + 1
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = False
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future_start"])
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        generation_inputs = {} if tokenized_data is None else dict(tokenized_data)
        generation_inputs.pop("input_ids", None)
        generation_inputs.pop("labels", None)
        # Transformers 4.57.1 merges model defaults back into a supplied
        # GenerationConfig, so restate every deliberate override as a direct kwarg.
        generation_inputs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=max_generation_length + 1,
            eos_token_id=None,
            forced_eos_token_id=None,
            output_logits=False,
            return_dict_in_generate=False,
        )
        was_training = self.vlm.training
        self.vlm.eval()
        try:
            with torch.no_grad():
                sequences = self.vlm.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    logits_processor=logits_processor,
                    bad_words_ids=bad_words_ids,
                    **generation_inputs,
                )
        finally:
            if was_training:
                self.vlm.train()
        return sequences

    def _sequence_has_valid_marker(self, seq: torch.Tensor, prompt_len: int) -> bool:
        """Whether every row carries a usable ``<traj_future_start>`` boundary.

        The downstream expert context builder raises when the marker is missing or
        lands in the final slot. A student rollout can legitimately fail either check
        (it is sampled, and early in training it may not have learned the format), so
        the on-policy path probes first and falls back rather than killing the run.
        """
        try:
            find_future_start_offsets(
                seq,
                self.special_token_ids["traj_future_start"],
                search_start=prompt_len,
                cache_seq_len=seq.shape[1],
                require_trailing_token=True,
                context="on-policy student rollout",
            )
        except ValueError:
            return False
        return True

    def _build_distillation_sequence(
        self,
        input_ids: torch.Tensor,
        student_td: dict[str, Any],
        teacher_td: dict[str, Any],
        on_policy: bool,
    ) -> tuple[torch.Tensor, bool]:
        """Produce the sequence to distil on, and report which policy generated it.

        Returns ``(seq, used_on_policy)``. An on-policy attempt that yields a
        malformed rollout silently falls back to the teacher for that step, so a
        rollout the pipeline cannot consume costs throughput rather than the job.
        """
        prompt_len = input_ids.shape[1]

        if on_policy:
            seq = self.student_model_generate(input_ids=input_ids, tokenized_data=student_td)
            if self._sequence_has_valid_marker(seq, prompt_len):
                return seq, True
            self._on_policy_fallbacks += 1
            if self._on_policy_fallbacks in (1, 10, 100) or self._on_policy_fallbacks % 1000 == 0:
                logger.warning(
                    "on-policy rollout lacked a usable <traj_future_start>; falling back to "
                    "the teacher for this step (%d fallbacks so far)",
                    self._on_policy_fallbacks,
                )

        with torch.no_grad():
            seq = self.teacher_model_generate(input_ids=input_ids, tokenized_data=teacher_td)
        return seq, False


    def distill_loss(
        self,
        student_logits: torch.Tensor,   # (B,S,V)
        teacher_logits: torch.Tensor,   # (B,S,V)
        labels: torch.Tensor | None = None,    # (B,S) with -100 ignore
        attention_mask: torch.Tensor | None = None,  # (B,S) optional
        T: float = 0.5,
        alpha: float = 0.5,             # weight for hard CE
    ):
        """Off-policy KL distillation against the teacher's logits.

        Thin wrapper over :func:`alpamayo1_distill.losses.sequence_distillation_loss`; the
        math lives in ``alpamayo1_distill.losses`` so the on-policy path can reuse it. Note
        there is no ``T**2`` Hinton rescaling — magnitude is controlled externally via
        ``--distill_weight``, so ``T`` only changes target softness.
        """
        return sequence_distillation_loss(
            student_logits,
            teacher_logits,
            labels=labels,
            attention_mask=attention_mask,
            temperature=T,
            mode="forward_kl",
            ce_weight=0.0 if labels is None else alpha,
        )


    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        teacher_tokenized_data: dict[str, Any] | None = None,
        on_policy: bool = False,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """
        Args:
            on_policy: Sample the distilled sequence from the student instead of the
                teacher. The step is then supervised by the divergence alone — hard CE
                against tokens the student itself sampled would merely reinforce them —
                and the loss is reported under ``on_policy_kl`` rather than
                ``distill_loss``/``vlm_loss``. Falls back to the teacher for any
                step whose rollout lacks a usable ``<traj_future_start>``.

        Returns:
        ReasoningVLAOutput with:
            - .loss: total loss tensor
            - .loss_dict: dict of component losses
        """
        if self.teacher_model is None:
            raise RuntimeError(
                "Distillation forward requires a teacher model. Pass teacher_model "
                "when constructing ExpertModelDistill; teacherless construction is "
                "supported only for student inference."
            )
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        td = dict(tokenized_data)  # don't mutate caller dict
        input_ids = td.pop("input_ids")

        traj_data_vlm = {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        # Use separate teacher tokenization when teacher and student image
        # processors differ. The fused input IDs remain shared.
        if teacher_tokenized_data is not None:
            t_td = dict(teacher_tokenized_data)
            t_td.pop("input_ids", None)
        else:
            t_td = td
        # Student rollouts must use student image features, not teacher features.
        seq, used_on_policy = self._build_distillation_sequence(
            input_ids=input_ids,
            student_td=td,
            teacher_td=t_td,
            on_policy=on_policy,
        )
        prompt_len = input_ids.shape[1]

        future_start_id = self.special_token_ids["traj_future_start"]
        future_start_offsets = find_future_start_offsets(
            seq,
            future_start_id,
            search_start=prompt_len,
            cache_seq_len=seq.shape[1],
            require_trailing_token=True,
            context="distillation training",
        )

        same = (seq[:, :prompt_len] == input_ids).all()
        if not same:
            # count mismatches
            mism = (seq[:, :prompt_len] != input_ids).sum().item()
            raise ValueError(
                "Teacher generate prefix mismatch vs input_ids. "
                f"mismatched tokens={mism}"
            )

        reasoning_labels = seq.clone()
        reasoning_labels[:, :prompt_len] = -100

        # Match Hugging Face generation: preserve prompt padding, but append a visible
        # mask slot for every generated step, including forced pad tokens in rows that
        # stopped early. This keeps Qwen's rope_deltas anchored to the prompt. The
        # supervised objectives and expert cache are still cut off at the marker below.
        new_attn_mask = extend_prompt_attention_mask(
            td.get("attention_mask"),
            batch_size=seq.shape[0],
            prompt_seq_len=prompt_len,
            total_seq_len=seq.shape[1],
            device=seq.device,
        )
        token_positions = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        objective_valid = new_attn_mask.to(dtype=torch.bool) & (
            token_positions < future_start_offsets.unsqueeze(1)
        )
        # On-policy supervision covers only the tokens the student actually generated.
        # The prompt is identical under both policies, so including it would add a
        # near-constant term — and at this vocabulary size those ~3k prompt positions
        # dominate the divergence's memory cost.
        generated_valid = objective_valid & (token_positions >= prompt_len)
        reasoning_labels[~objective_valid] = -100
        td["attention_mask"] = new_attn_mask

        # drop stale positional/index tensors if present
        for k in ("position_ids", "cache_position", "rope_deltas"):
            td.pop(k, None)

        # keep t_td in sync when it's a separate dict (teacher uses its own image processor)
        if t_td is not td:
            t_td["attention_mask"] = new_attn_mask
            for k in ("position_ids", "cache_position", "rope_deltas"):
                t_td.pop(k, None)

        # vocab sanity check (catches IndexKernel asserts early)
        vocab = int(self.vlm.get_input_embeddings().num_embeddings)
        mx = int(seq.max().item())
        mn = int(seq.min().item())
        if mn < 0 or mx >= vocab:
            raise ValueError(f"seq out-of-vocab: min={mn}, max={mx}, vocab={vocab}")

        _need_hidden = self.hidden_proj is not None

        # Hard CE is meaningful only against teacher tokens; on the student's own
        # samples it would just reinforce whatever was drawn.
        student_labels = None if used_on_policy else reasoning_labels
        with (nullcontext() if self.config.cotrain_vlm else torch.no_grad()):
            vlm_outputs = self.vlm(input_ids=seq, labels=student_labels,
                                   output_hidden_states=_need_hidden, **td)

        # Capture student last hidden state before freeing outputs (keep grad if cotrain_vlm)
        student_last_h: torch.Tensor | None = None
        if _need_hidden and vlm_outputs.hidden_states is not None:
            student_last_h = vlm_outputs.hidden_states[-1]  # (B, L, H_s)
            vlm_outputs.hidden_states = None  # free intermediate layers

        # 5) distillation loss (scalar) on all teacher forward logits
        with torch.no_grad():
            vlm_outputs_teacher = self.teacher_model_forward(
                input_ids=seq,
                tokenized_data=t_td,
                output_hidden_states=_need_hidden,
            )
            teacher_logs = vlm_outputs_teacher.logits
            teacher_last_h: torch.Tensor | None = None
            if _need_hidden and vlm_outputs_teacher.hidden_states is not None:
                teacher_last_h = vlm_outputs_teacher.hidden_states[-1].detach().float()
            del vlm_outputs_teacher

        # Both regimes minimise a divergence against the same teacher logits; they
        # differ in which distribution the sequence — and therefore the expectation —
        # was drawn from. Reported under separate keys so the logs stay attributable.
        # Detached, so the inactive branch's placeholder never pins the (B, S, V)
        # logits in the autograd graph. Safe for DDP: both regimes drive gradients
        # into the same parameters through the divergence they *do* compute.
        zero = torch.zeros((), device=vlm_outputs.logits.device, dtype=torch.float32)
        if used_on_policy:
            on_policy_cfg = self.on_policy_cfg
            vlm_distill_loss = zero
            on_policy_kl = sequence_distillation_loss(
                vlm_outputs.logits,
                teacher_logs,
                labels=None,
                attention_mask=generated_valid,
                temperature=on_policy_cfg.temperature,
                mode=on_policy_cfg.divergence,
                beta=on_policy_cfg.beta,
            )
        else:
            vlm_distill_loss = self.distill_loss(
                student_logits=vlm_outputs.logits,
                teacher_logits=teacher_logs,
                labels=None,  # pure KL; CE is tracked separately as vlm_loss
                attention_mask=objective_valid,
                T=self.distill_T,
            )
            on_policy_kl = zero
        teacher_logs = None

        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        future_traj_vlm_loss = torch.tensor(0.0, device=seq.device, dtype=torch.float32)
        ftv_w = getattr(self.config, "future_traj_vlm_loss_weight", 1.0)
        if ftv_w != 0.0:
            # Auxiliary SFT-style future-trajectory CE for the student VLM. Continue
            # only from the GT future suffix instead of re-forwarding the full prefix,
            # while the action expert keeps the original inference-aligned cache.
            gt_future_suffix = self._build_gt_future_traj_token_suffix(traj_data, device=seq.device)
            with (nullcontext() if self.config.cotrain_vlm else torch.no_grad()):
                future_traj_vlm_loss = self._compute_future_traj_vlm_loss_from_prefix(
                    vlm_outputs,
                    seq,
                    new_attn_mask,
                    future_start_offsets,
                    gt_future_suffix,
                    td,
                )
        # free logit tensors before expert forward; past_key_values is still intact
        vlm_outputs.logits = None

        # Hidden-state distillation: squared L2 distance on L2-normalized states.
        # Per token this is 2 - 2 * cosine_similarity, independent of hidden width.
        hidden_distill_loss = torch.tensor(0.0, device=seq.device, dtype=torch.float32)
        if _need_hidden and student_last_h is not None and teacher_last_h is not None:
            s_proj = self.hidden_proj(student_last_h.float())  # (B, L, H_t)
            hidden_distill_loss = masked_normalized_hidden_distance(
                s_proj,
                teacher_last_h,
                objective_valid,
            )
            student_last_h = None
            teacher_last_h = None

        # 6) diffusion/expert training target
        future_traj_data = self._process_traj_future_training(traj_data)

        # history embeds (optional)
        if self.expert_hist_traj_tokenizer is not None:
            assert ego_history_xyz is not None and ego_history_rot is not None
            action_history_embeds = self.expert_hist_traj_tokenizer(
                xyz=ego_history_xyz.squeeze(1),
                rot=ego_history_rot.squeeze(1),
            ).unsqueeze(1)
        else:
            action_history_embeds = None

        # noisy future action embeds
        action_embeds = self.action_in_proj(
            future_traj_data["noisy_x"], future_traj_data["timesteps"]
        )

        expert_embeds = (
            torch.cat([action_history_embeds, action_embeds], dim=1)
            if action_history_embeds is not None
            else action_embeds
        )

        # 7) run expert on top of student kv-cache
        # The teacher generation used StopAfterEOS, which appends one extra token after
        # <traj_future_start> so that the VLM can compute <traj_future_start>'s hidden state.
        # At inference that extra token is masked out per-sample (see inference path ~line 1676).
        # Here we do the same so train and inference contexts are identical.
        kv_cache = vlm_outputs.past_key_values
        prefill_seq_len = kv_cache.get_seq_length()  # includes the extra token

        if prefill_seq_len != seq.shape[1]:
            raise ValueError(
                "distillation training: student KV-cache length does not match the sequence "
                f"length ({prefill_seq_len} != {seq.shape[1]})"
            )
        offset = future_start_offsets

        if self.config.stop_grad_from_vlm:
            for layer in kv_cache.layers:
                layer.keys = layer.keys.detach()
                layer.values = layer.values.detach()

        # Per-sample RoPE: diffusion tokens start right after <traj_future_start>,
        # matching inference.
        position_ids = build_expert_position_ids(
            offset,
            vlm_outputs.rope_deltas,
            query_seq_len=expert_embeds.shape[1],
            device=expert_embeds.device,
        )

        # Hide every cached key after the marker. FlashAttention requires a 2-D boolean
        # key-padding mask; eager/SDPA require the equivalent 4-D additive mask.
        attention_mask = build_expert_attention_mask(
            offset.to(expert_embeds.device),
            cache_seq_len=prefill_seq_len,
            query_seq_len=expert_embeds.shape[1],
            attention_implementation=(
                getattr(self.expert.config, "_attn_implementation", None)
                or self.config.attn_implementation
            ),
            cache_valid_mask=new_attn_mask[:, :prefill_seq_len],
            dtype=expert_embeds.dtype,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        expert_outputs = self.expert(
            inputs_embeds=expert_embeds,
            position_ids=position_ids,
            past_key_values=kv_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )

        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
        pred = self.action_out_proj(diffusion_out)
        pred = pred.view(-1, *self.action_space.get_action_space_dims())

        future_traj_loss = (
            self.diffusion.compute_loss_from_pred(
                training_data=future_traj_data,
                pred=pred,
            )
            * self.config.traj_loss_weight
        )

        # 8) compose loss dict
        # distill_loss       = KL(teacher_soft || student)  — soft logit matching
        # vlm_loss           = CE(student, greedy teacher tokens) — hard sequence supervision
        # future_traj_vlm_loss = CE(student, GT future trajectory tokens) - SFT-style aux
        # hidden_distill_loss = 2 - 2*cos on normalized last VLM states — feature alignment
        hw = getattr(self.config, "hidden_distill_weight", 0.0)
        # `vlm_outputs.loss` is None on on-policy steps (no labels were passed). Keep
        # the key present with a graph-connected zero so the loss-key set stays
        # identical across steps and ranks — _sync_loss_keys fixes it from the first
        # batch, and DDP requires every rank to build the same graph.
        vlm_ce_loss = zero if vlm_outputs.loss is None else vlm_outputs.loss
        loss_dict: dict[str, torch.Tensor] = {
            "future_traj_loss": future_traj_loss,
            "future_traj_vlm_loss": future_traj_vlm_loss,
            "distill_loss": vlm_distill_loss,
            "vlm_loss": vlm_ce_loss,
            "on_policy_kl": on_policy_kl,
        }
        if _need_hidden:
            loss_dict["hidden_distill_loss"] = hidden_distill_loss

        total_loss = (
            loss_dict["future_traj_loss"]
            + ftv_w * loss_dict["future_traj_vlm_loss"]
            + loss_dict["distill_loss"]
            + loss_dict["vlm_loss"]
            + self.on_policy_cfg.kl_weight * loss_dict["on_policy_kl"]
            + hw * hidden_distill_loss
        )
        loss_dict["total_loss"] = total_loss

        return ReasoningVLAOutput(
            loss=total_loss,
            logits=None,
            loss_dict=loss_dict,
        )

    def sample_trajectories_from_data(  # type: ignore[override]
        self,
        data: dict[str, Any],
        with_vlm_rollout: bool = True,
        last_component: str = "traj_future",
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories from the data.

        Args:
            with_vlm_rollout: Whether to use VLM rollout.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if "tokenized_data" not in data:
            data["tokenized_data"] = self._get_generation_mode_tokenized_data_online(
                data, last_component=last_component
            )
        # Select the path from the actual prompt, including caller-supplied prompts.
        # A prompt that already ends at <traj_future_start> must go directly to the
        # expert; a CoT prompt still needs the VLM rollout. Mixed prompt types cannot
        # share one generation call safely.
        if with_vlm_rollout:
            prompt_data = data["tokenized_data"]
            prompt_input_ids = prompt_data["input_ids"]
            prompt_end_offsets = find_prompt_end_offsets(
                prompt_input_ids,
                prompt_data.get("attention_mask"),
                context="trajectory sampling prompt",
            )
            terminal_ids = prompt_input_ids.gather(
                1, (prompt_end_offsets - 1).unsqueeze(1)
            ).squeeze(1)
            prompt_at_expert_boundary = terminal_ids.eq(
                self.config.traj_token_ids["future_start"]
            )
            if bool(prompt_at_expert_boundary.any()) and not bool(
                prompt_at_expert_boundary.all()
            ):
                raise ValueError(
                    "trajectory sampling prompt: batch mixes prompts that do and do not "
                    "end at <traj_future_start>"
                )
            if bool(prompt_at_expert_boundary.all()):
                with_vlm_rollout = False
        if with_vlm_rollout:
            return self._sample_trajectories_from_data_with_vlm_rollout(
                data,
                *args,
                **kwargs,
            )
        else:
            return self._sample_trajectories_from_data_without_vlm_rollout(
                data,
                *args,
                **kwargs,
            )

    def _sample_trajectories_from_data_without_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories from the data.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = dict(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        prompt_end_offsets = find_prompt_end_offsets(
            input_ids,
            tokenized_data.get("attention_mask"),
            context="expert-only trajectory sampling",
        )
        terminal_ids = input_ids.gather(
            1, (prompt_end_offsets - 1).unsqueeze(1)
        ).squeeze(1)
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        if torch.any(terminal_ids != self.config.traj_token_ids["future_start"]):
            raise NotImplementedError("Expert + CoT generation is not implemented yet.")

        device = input_ids.device

        vlm_outputs = self.vlm.model(
            input_ids=input_ids,
            **tokenized_data,
            use_cache=True,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()
        if prefill_seq_len != input_ids.shape[1]:
            raise ValueError(
                "expert prompt KV-cache length does not match input length "
                f"({prefill_seq_len} != {input_ids.shape[1]})"
            )

        if self.expert_hist_traj_tokenizer is not None:
            action_history_embeds = self.expert_hist_traj_tokenizer(
                xyz=ego_history_xyz.squeeze(1),
                rot=ego_history_rot.squeeze(1),
            ).unsqueeze(1)
        else:
            action_history_embeds = None

        # repeat the batch dimension
        prompt_cache.batch_repeat_interleave(n_samples_total)
        if action_history_embeds is not None:
            action_history_embeds = torch.repeat_interleave(
                action_history_embeds, n_samples_total, dim=0
            )
        vlm_outputs.rope_deltas = torch.repeat_interleave(
            vlm_outputs.rope_deltas, n_samples_total, dim=0
        )
        prompt_end_offsets = torch.repeat_interleave(
            prompt_end_offsets, n_samples_total, dim=0
        )
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        history_tokens = (
            action_history_embeds.shape[1] if action_history_embeds is not None else 0
        )
        expert_seq_len = history_tokens + n_diffusion_tokens
        position_ids = build_expert_position_ids(
            torch.full_like(prompt_end_offsets, prefill_seq_len),
            vlm_outputs.rope_deltas,
            query_seq_len=expert_seq_len,
            device=device,
        )
        prompt_valid_mask = tokenized_data.get("attention_mask")
        if prompt_valid_mask is not None:
            prompt_valid_mask = torch.repeat_interleave(
                prompt_valid_mask.to(dtype=torch.bool), n_samples_total, dim=0
            )
        attention_mask = build_expert_attention_mask(
            prompt_end_offsets,
            cache_seq_len=prefill_seq_len,
            query_seq_len=expert_seq_len,
            attention_implementation=(
                getattr(self.expert.config, "_attn_implementation", None)
                or self.config.attn_implementation
            ),
            cache_valid_mask=prompt_valid_mask,
            dtype=self.expert.dtype,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            guidance: Any | None = None,
            is_drop_guidance: torch.BoolTensor | None = None,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # add the history tokens
            future_token_embeds = (
                torch.cat([action_history_embeds, future_token_embeds], dim=1)
                if action_history_embeds is not None
                else future_token_embeds
            )

            # Run expert with cached prefill, only on the future tokens
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            # remove the history token
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            guidance=prompt_cache,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # Repeat history to align with num_traj_samples
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # Return dummy logprob for now (same shape as (B, S, N))
        logprob = torch.zeros_like(pred_xyz[..., 0])

        if kwargs.get("return_extra", False):
            repeated_prompts = torch.repeat_interleave(
                input_ids, n_samples_total, dim=0
            )
            extra = extract_text_tokens(self.tokenizer, repeated_prompts)
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, logprob, extra
        return pred_xyz, pred_rot, logprob

    def _sample_trajectories_from_data_with_vlm_rollout(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories from the data with VLM rollout.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = dict(data["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        prompt_attention_mask = tokenized_data.get("attention_mask")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # run autoregressive generation for VLM tokens
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.reasoning_max_generation_length
        )
        generation_config = copy.deepcopy(self.vlm.generation_config)
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        do_sample = kwargs.get("do_sample", True)
        generation_config.do_sample = do_sample
        generation_config.num_beams = 1
        generation_config.num_return_sequences = n_samples_total
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        # Reserve one decode step so a marker emitted at the reasoning limit is
        # represented in the returned KV cache.
        generation_config.max_new_tokens = max_generation_length + 1
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        logits_processor = LogitsProcessorList(
            [
                MaskDiscreteTrajectoryLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )

        # use custom stopping criteria to stop after EOS token + one more token,
        # because the KV cache is updated after the next token is generated
        eos_token_id = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["traj_future_start"])
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        generation_inputs = dict(tokenized_data)
        generation_inputs.pop("input_ids", None)
        # Direct kwargs are applied after Transformers 4.57.1 merges model defaults
        # back into a supplied GenerationConfig. This preserves do_sample=False for
        # greedy validation even when the checkpoint defaults to sampling.
        generation_inputs.update(
            do_sample=do_sample,
            num_beams=1,
            num_return_sequences=n_samples_total,
            max_new_tokens=max_generation_length + 1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=None,
            forced_eos_token_id=None,
            output_logits=False,
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **generation_inputs,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        # get the action history embeds
        if self.expert_hist_traj_tokenizer is not None:
            action_history_embeds = self.expert_hist_traj_tokenizer(
                xyz=ego_history_xyz.squeeze(1),
                rot=ego_history_rot.squeeze(1),
            ).unsqueeze(1)
        else:
            action_history_embeds = None

        # repeat the batch dimension
        if action_history_embeds is not None:
            action_history_embeds = torch.repeat_interleave(
                action_history_embeds, n_samples_total, dim=0
            )

        # Require a real generated marker and require that it is already represented in
        # the returned cache. There is no semantically valid sequence-length fallback.
        b_star = vlm_outputs.sequences.shape[0]
        offset = find_future_start_offsets(
            vlm_outputs.sequences,
            eos_token_id,
            search_start=input_ids.shape[1],
            cache_seq_len=prefill_seq_len,
            context="student VLM rollout",
        )

        # modify the position ids to remove padding tokens
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        history_tokens = (
            action_history_embeds.shape[1] if action_history_embeds is not None else 0
        )
        expert_seq_len = history_tokens + n_diffusion_tokens
        position_ids = build_expert_position_ids(
            offset,
            vlm_outputs.rope_deltas,
            query_seq_len=expert_seq_len,
            device=device,
        )

        prompt_seq_len = input_ids.shape[1]
        if prefill_seq_len < prompt_seq_len:
            raise ValueError(
                "student rollout KV cache is shorter than its prompt "
                f"({prefill_seq_len} < {prompt_seq_len})"
            )
        expanded_prompt_mask = (
            None
            if prompt_attention_mask is None
            else torch.repeat_interleave(
                prompt_attention_mask, n_samples_total, dim=0
            )
        )
        cache_valid_mask = extend_prompt_attention_mask(
            expanded_prompt_mask,
            batch_size=b_star,
            prompt_seq_len=prompt_seq_len,
            total_seq_len=prefill_seq_len,
            device=device,
        )
        attention_mask = build_expert_attention_mask(
            offset.to(device),
            cache_seq_len=prefill_seq_len,
            query_seq_len=expert_seq_len,
            attention_implementation=(
                getattr(self.expert.config, "_attn_implementation", None)
                or self.config.attn_implementation
            ),
            cache_valid_mask=cache_valid_mask,
            dtype=self.expert.dtype,
        )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            guidance: Any | None = None,
            is_drop_guidance: torch.BoolTensor | None = None,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # add the history tokens
            future_token_embeds = (
                torch.cat([action_history_embeds, future_token_embeds], dim=1)
                if action_history_embeds is not None
                else future_token_embeds
            )

            # Run expert with cached prefill, only on the future tokens
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            # remove the history token
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            guidance=prompt_cache,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # Repeat history to align with num_traj_samples
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # Return dummy logprob for now (same shape as (B, S, N))
        logprob = torch.zeros_like(pred_xyz[..., 0])

        # return the text tokens generated by the VLM
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, logprob, extra
        return pred_xyz, pred_rot, logprob


# Register the model with Auto classes
AutoConfig.register("alpamayo_reasoning_vla_expert_distill", ExpertModelConfig)
AutoModel.register(ExpertModelConfig, ExpertModelDistill)
