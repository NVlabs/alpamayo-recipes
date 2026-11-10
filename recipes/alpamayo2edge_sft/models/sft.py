# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint SFT and component evaluation for the released Edge inference model."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import torch
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.integrations.deepspeed import (
    _load_state_dict_into_zero3_model,
    is_deepspeed_zero3_enabled,
)
from transformers.utils import ModelOutput

from alpamayo2edge_sft.models.action_precision import joint_sft_action_precision
from alpamayo2edge_sft.models.qk_gradient_layout import configure_joint_qk_gradient_layout
from alpamayo2edge_sft.models.expert_masking import noncausal_expert_mask
from alpamayo2edge_sft.models.geometry import CPUCholesky
from alpamayo2edge_sft.models.utils import (
    IGNORE_INDEX,
    compute_token_loss,
    fuse_traj_tokens,
    future_traj_loss_mask,
)
from alpamayo2edge_sft.models.vision_precision import joint_sft_vision_precision


@dataclass
class Alpamayo2EdgeSFTOutput(ModelOutput):
    """Training output with separate trajectory and text losses."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    supervised_logits: torch.FloatTensor | None = None
    future_traj_ce: torch.FloatTensor | None = None
    others_ce: torch.FloatTensor | None = None
    expert_diffusion: torch.FloatTensor | None = None
    vlm_ce: torch.FloatTensor | None = None


class Alpamayo2EdgeSFTMixin:
    """Add VLM and expert SFT behavior to an Edge inference model."""

    inference_architecture = "Alpamayo2Edge"
    sft_backend = "public"
    accepts_loss_kwargs = False

    def __init__(self, config, training_stage: str | None = None) -> None:
        super().__init__(config)
        self.config.sft_backend = self.sft_backend
        self._validate_objective_weights()
        self.set_training_stage(training_stage or getattr(config, "training_stage", "joint"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Self:
        """Load a release checkpoint and optionally overlay a Stage-1 VLM checkpoint."""
        stage1_vlm_checkpoint_path = kwargs.pop("stage1_vlm_checkpoint_path", None)
        # Training-only fields may be absent from inference configs; do not let
        # HF forward unknown config overrides to the inference constructor.
        training_overrides = {
            key: kwargs.pop(key)
            for key in ("loss_weights", "lm_token_loss_weights", "qk_gradient_layout")
            if key in kwargs
        }
        require_complete_weights = kwargs.pop("require_complete_weights", False)
        if kwargs.pop("output_loading_info", False):
            raise ValueError(
                "SFT from_pretrained returns a model; use require_complete_weights=True "
                "to validate loading."
            )
        if require_complete_weights:
            kwargs["output_loading_info"] = True
        loaded = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        if require_complete_weights:
            model, loading_info = loaded
            problems = {
                key: loading_info[key]
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(key)
            }
            if problems:
                raise RuntimeError(f"Checkpoint did not load completely: {problems}")
        else:
            model = loaded
        for key, value in training_overrides.items():
            setattr(model.config, key, value)
        model._validate_objective_weights()
        if stage1_vlm_checkpoint_path is not None:
            if model.training_stage == "joint":
                raise ValueError(
                    "Joint phase continuation requires the complete checkpoint: set "
                    "pretrained_model_name_or_path to the previous phase checkpoint and "
                    "leave stage1_vlm_checkpoint_path unset."
                )
            model.load_stage1_vlm(stage1_vlm_checkpoint_path)
        model._restore_training_buffers()
        model.set_training_stage(model.training_stage)
        return model

    def _validate_objective_weights(self) -> None:
        """Resolve defaults once and persist the actual objective in the checkpoint."""
        defaults = {
            "loss_weights": {"future_traj": 1.0, "others": 1.0},
            "lm_token_loss_weights": {"future_traj": 1.0, "text": 0.2},
        }
        for field, fallback in defaults.items():
            values = getattr(self.config, field, fallback)
            # Older SFT exports explicitly stored null to mean unweighted CE.
            # Preserve that objective when reloading for evaluation/continuation.
            if field == "lm_token_loss_weights" and values is None:
                continue
            if values is None:
                values = {}
            if not isinstance(values, Mapping):
                raise ValueError(f"{field} must be a mapping of nonnegative finite weights")
            if set(values) - set(fallback):
                raise ValueError(f"Unknown {field} keys: {sorted(set(values) - set(fallback))}")
            resolved = dict(fallback)
            for name, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{field}.{name} must be a nonnegative finite number")
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{field}.{name} must be a nonnegative finite number")
                resolved[name] = float(value)
            setattr(self.config, field, resolved)

    def _apply(self, fn, recurse: bool = True) -> Self:
        """Preserve exact FP32 math constants through model dtype and device changes."""
        super()._apply(fn, recurse=recurse)
        # DeepSpeed casts the whole model after checkpoint loading. Reconstruct
        # derived buffers after that cast; upcasting their rounded values cannot
        # recover the trajectory normalization or Fourier frequencies.
        self._restore_training_buffers()
        return self

    def _restore_training_buffers(self) -> None:
        """Rebuild nonpersistent math constants that checkpoint dtype casts can round."""
        # Cosmos3-Edge text and expert rotary buffers are nonpersistent.
        # Recompute from config; merely calling .float() retains BF16 rounding.
        for module in self.modules():
            inv_freq = getattr(module, "inv_freq", None)
            if not isinstance(inv_freq, torch.Tensor):
                continue
            rope_type = getattr(module, "rope_type", None)
            if rope_type == "axial" and hasattr(module, "compute_axial_rope_parameters"):
                initialize = module.compute_axial_rope_parameters
            elif rope_type == "default" and hasattr(module, "compute_default_rope_parameters"):
                initialize = module.compute_default_rope_parameters
            else:
                initialize = ROPE_INIT_FUNCTIONS.get(rope_type)
            if initialize is None:
                raise ValueError(f"Cannot restore FP32 RoPE for {type(module).__name__}")
            # Compute small constants on CPU even during meta initialization;
            # pass no deprecated device argument to Transformers' initializers.
            with torch.device("cpu"):
                values, scaling = initialize(
                    module.config, seq_len=getattr(module, "max_seq_len_cached", None)
                )
                if hasattr(module, "original_inv_freq"):
                    original, _ = initialize(
                        module.config, seq_len=getattr(module, "original_max_seq_len", None)
                    )
                    module.original_inv_freq = original.to(inv_freq.device, dtype=torch.float32)
            module.inv_freq = values.to(inv_freq.device, dtype=torch.float32)
            module.attention_scaling = scaling

        action_spaces = []
        if hasattr(self, "expert") and hasattr(self.expert, "action_space"):
            action_spaces.append(
                (self.expert.action_space, self.config.expert_config.action_space_cfg)
            )
            projection = self.expert.action_in_proj
            for encoder in [*projection.sinus, projection.timestep_fourier_encoder]:
                # These constants are not in safetensors. Upcasting a BF16 buffer
                # retains rounding from construction instead of restoring FP32 math.
                max_freq = getattr(encoder, "max_freq", None)
                if max_freq is None:
                    max_freq = self.config.expert_config.action_in_proj_cfg.get("max_freq", 100.0)
                encoder.freqs = torch.logspace(
                    0,
                    math.log10(max_freq),
                    steps=encoder.out_dim // 2,
                    device=encoder.freqs.device,
                    dtype=torch.float32,
                )[None, :]
        future_tokenizer = getattr(self, "future_traj_tokenizer", None)
        if hasattr(future_tokenizer, "action_space"):
            action_spaces.append(
                (
                    future_tokenizer.action_space,
                    self.config.future_traj_tokenizer_cfg["action_space_cfg"],
                )
            )
        for action_space, config in action_spaces:
            for name in ("accel_mean", "accel_std", "curvature_mean", "curvature_std"):
                buffer = getattr(action_space, name)
                device = buffer.device
                if (
                    buffer.is_meta
                    and action_space is getattr(future_tokenizer, "action_space", None)
                    and not isinstance(future_tokenizer, torch.nn.Module)
                ):
                    # This tokenizer is not a registered child module, so HF's
                    # meta loading never materializes its constants. Action-space
                    # methods move these small CPU tensors to the input device.
                    device = torch.device("cpu")
                default = 1.0 if name.endswith("std") else 0.0
                setattr(
                    action_space,
                    name,
                    torch.tensor(config.get(name, default), device=device, dtype=torch.float32),
                )

    def set_training_stage(self, training_stage: str) -> None:
        """Select the trainable parameter set and forward path."""
        if training_stage not in {"joint", "vlm", "expert"}:
            raise ValueError(
                f"training_stage must be 'joint', 'vlm' or 'expert', got {training_stage!r}"
            )
        if training_stage in {"joint", "expert"} and not self.config.enable_expert:
            raise ValueError("Joint/expert training requires an expert-enabled release checkpoint")

        self.training_stage = training_stage
        self.config.training_stage = training_stage
        self.config.cotrain_expert_vlm = training_stage == "joint"
        self.config.cotrain_detach_expert_kv = True
        self.vlm.requires_grad_(training_stage in {"joint", "vlm"})
        if hasattr(self, "expert"):
            self.expert.requires_grad_(training_stage in {"joint", "expert"})
        if training_stage == "joint":
            is_cosmos3_edge = (
                getattr(getattr(self.config, "vlm_config", None), "model_type", None)
                == "cosmos3_edge"
            )
            enabled = getattr(self.config, "qk_gradient_layout", not is_cosmos3_edge)
            if is_cosmos3_edge and enabled:
                raise ValueError(
                    "Cosmos3-Edge attention has no Q/K normalization; "
                    "set qk_gradient_layout=false"
                )
            configure_joint_qk_gradient_layout(self, enabled=enabled)
            self.config.qk_gradient_layout = enabled

    def train(self, mode: bool = True) -> Self:
        """Keep the frozen component deterministic when Trainer enables training."""
        super().train(mode)
        if self.training_stage == "expert":
            self.vlm.eval()
        elif self.training_stage == "vlm" and hasattr(self, "expert"):
            self.expert.eval()
        return self

    def save_pretrained(self, save_directory, *args: Any, **kwargs: Any) -> None:
        """Save weights and assets that the inference model can load directly."""
        # HF5 can reverse load-time key mappings during saving. Stage-1 overlays
        # require the native keys of the instantiated VLM.
        kwargs.setdefault("save_original_format", False)
        super().save_pretrained(save_directory, *args, **kwargs)
        if kwargs.get("is_main_process", True):
            self.config.architectures = [self.inference_architecture]
            self.config.save_pretrained(save_directory)
            self.tokenizer.save_pretrained(save_directory)

    def load_stage1_vlm(self, checkpoint_path: str) -> None:
        """Overlay VLM tensors saved by Hugging Face Trainer."""
        state_dict = self._load_checkpoint_state_dict(checkpoint_path)
        vlm_state_dict = {
            key.removeprefix("vlm."): value
            for key, value in state_dict.items()
            if key.startswith("vlm.")
        }
        if not vlm_state_dict:
            raise ValueError(f"No vlm.* tensors found in {checkpoint_path}")
        # Safetensors omits duplicate storage for tied embeddings/LM heads. Restore
        # only aliases that the live VLM actually ties before strict validation.
        tied_parameters: dict[int, list[str]] = {}
        for name, parameter in self.vlm.named_parameters(remove_duplicate=False):
            tied_parameters.setdefault(id(parameter), []).append(name)
        for names in tied_parameters.values():
            saved_names = [name for name in names if name in vlm_state_dict]
            if saved_names:
                for name in names:
                    vlm_state_dict.setdefault(name, vlm_state_dict[saved_names[0]])
        if is_deepspeed_zero3_enabled():
            expected_keys = set(self.vlm.state_dict())
            checkpoint_keys = set(vlm_state_dict)
            missing_keys = sorted(expected_keys - checkpoint_keys)
            unexpected_keys = sorted(checkpoint_keys - expected_keys)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Stage-1 VLM checkpoint keys do not match the release VLM: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}"
                )
            error_messages, unloaded_keys = _load_state_dict_into_zero3_model(
                self.vlm, vlm_state_dict
            )
            if error_messages or unloaded_keys:
                raise RuntimeError(
                    "Stage-1 VLM ZeRO-3 load failed: "
                    f"errors={error_messages}, unloaded={sorted(unloaded_keys)}"
                )
        else:
            self.vlm.load_state_dict(vlm_state_dict, strict=True)

    @staticmethod
    def _load_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
        """Read a single-file or sharded safetensors checkpoint on CPU."""
        checkpoint_dir = Path(checkpoint_path).resolve()
        index_path = checkpoint_dir / "model.safetensors.index.json"
        single_path = checkpoint_dir / "model.safetensors"

        if single_path.is_file():
            return load_file(str(single_path), device="cpu")
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Expected model.safetensors or model.safetensors.index.json in {checkpoint_dir}"
            )

        with index_path.open(encoding="utf-8") as file:
            weight_map = json.load(file)["weight_map"]
        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            state_dict.update(load_file(str(checkpoint_dir / shard_name), device="cpu"))
        return state_dict

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
        every_n_layers: int = 1,
        offload: bool = False,
    ) -> None:
        """Enable checkpointing for the legacy VLM-only path."""
        if self.training_stage == "joint":
            raise ValueError(
                "Joint training requires VLM KV output; set trainer.gradient_checkpointing=false. "
                "The native inference VLM cannot export its cache with gradient checkpointing."
            )
        self.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            every_n_layers=every_n_layers,
            offload=offload,
        )

    def gradient_checkpointing_disable(self) -> None:
        """Disable checkpointing on the nested VLM."""
        self.vlm.gradient_checkpointing_disable()

    def _prepare_vlm_inputs(self, tokenized_data: dict[str, Any]) -> dict[str, Any]:
        """Restore modality IDs omitted by older packed-data producers."""
        tokenized_data = dict(tokenized_data)
        has_multimodal = any(
            tokenized_data.get(name) is not None for name in ("image_grid_thw", "video_grid_thw")
        )
        if has_multimodal and tokenized_data.get("mm_token_type_ids") is None:
            input_ids = tokenized_data["input_ids"]
            modality_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            # Cosmos3-Edge's processor assigns modalities only to placeholder
            # tokens; video timestamps and vision delimiters remain text.
            modality_ids[input_ids == self.config.vlm_config.image_token_id] = 1
            modality_ids[input_ids == self.config.vlm_config.video_token_id] = 2
            tokenized_data["mm_token_type_ids"] = modality_ids
        return tokenized_data

    def forward(
        self,
        tokenized_data: dict[str, Any],
        traj_data: dict[str, torch.Tensor] | list[dict[str, torch.Tensor] | None] | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Alpamayo2EdgeSFTOutput:
        """Run joint training or an explicitly selected legacy stage."""
        if self.training_stage == "joint":
            with (
                joint_sft_vision_precision(self.vlm),
                joint_sft_action_precision(self.expert.action_in_proj),
            ):
                starts = tokenized_data["input_ids"].eq(self.config.traj_ids["future_start"])
                if isinstance(traj_data, dict) and bool(starts.sum(dim=1).eq(1).all()):
                    return self._forward_joint(tokenized_data, traj_data, labels_mask)
                return self._forward_joint_mixed(tokenized_data, traj_data, labels_mask)
        if self.training_stage == "expert":
            return self._forward_expert(
                tokenized_data=tokenized_data,
                traj_data=traj_data,
                labels_mask=labels_mask,
                **kwargs,
            )

        tokenized_data = self._prepare_vlm_inputs(tokenized_data)
        tokenized_data["input_ids"] = self._fuse_training_trajectories(
            tokenized_data["input_ids"], traj_data
        )
        outputs = self.vlm.model(**tokenized_data, use_cache=False)
        return self._vlm_loss(outputs, tokenized_data, labels_mask)

    def _fuse_training_trajectories(self, input_ids, traj_data):
        with CPUCholesky():
            return fuse_traj_tokens(
                self.history_traj_tokenizer, self.future_traj_tokenizer,
                input_ids, traj_data, self.config.traj_ids,
            )

    def _vlm_loss(self, outputs, tokenized_data, labels_mask, *, joint=False):
        """Project supervised predictors and compute the selected token objective."""
        labels = tokenized_data["input_ids"].clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)
        predictor_mask = labels[:, 1:].ne(IGNORE_INDEX)
        attention_mask = tokenized_data.get("attention_mask")
        if attention_mask is not None and not bool(attention_mask[:, :-1][predictor_mask].all()):
            raise ValueError("Supervised labels must be predicted from non-padding positions")
        # Selecting hidden states before projection matches the native training
        # geometry and avoids materializing image/prompt logits that are never used.
        logits = self.vlm.lm_head(outputs.last_hidden_state[:, :-1][predictor_mask])
        target_labels = labels[:, 1:][predictor_mask].contiguous()
        cap = float(getattr(self.config, "logit_cap", 0.0))
        if cap > 0.0:
            logits = cap * torch.tanh(logits / cap)

        traj_mask = future_traj_loss_mask(
            labels,
            self.config.traj_ids,
            self.config.future_vocab_size,
            self.config.vlm_config.text_config.eos_token_id,
        )
        target_traj_mask = traj_mask[:, 1:][predictor_mask].contiguous()
        if joint:
            # One mean across all supervised tokens, not a sum of bucket means.
            # Per-position weights change the numerator, never the denominator.
            token_weights = getattr(self.config, "lm_token_loss_weights", None)
            if target_labels.numel():
                token_losses = torch.nn.functional.cross_entropy(
                    logits.float(), target_labels, reduction="none"
                )
                if token_weights:
                    token_losses = token_losses * torch.where(
                        target_traj_mask, token_weights["future_traj"], token_weights["text"]
                    )
                loss = token_losses.mean()
            else:
                loss = logits.float().sum() * 0.0
            loss = loss * self.config.loss_weights.get("others", 1.0)
            with torch.no_grad():
                future_traj_loss = compute_token_loss(logits, target_labels, target_traj_mask)
                other_loss = compute_token_loss(logits, target_labels, ~target_traj_mask)
                future_traj_loss *= self.config.loss_weights.get("others", 1.0)
                other_loss *= self.config.loss_weights.get("others", 1.0)
        else:
            future_traj_loss = compute_token_loss(logits, target_labels, target_traj_mask)
            future_traj_loss *= self.config.loss_weights.get("future_traj", 1.0)
            other_loss = compute_token_loss(logits, target_labels, ~target_traj_mask)
            other_loss *= self.config.loss_weights.get("others", 1.0)
            loss = future_traj_loss + other_loss
        return Alpamayo2EdgeSFTOutput(
            loss=loss,
            supervised_logits=logits,
            future_traj_ce=future_traj_loss.detach(),
            others_ce=other_loss.detach(),
            vlm_ce=loss.detach(),
        )

    def _forward_joint_mixed(self, tokenized_data, traj_data, labels_mask):
        """Supervise mixed text/driving rows without skipping expert collectives.

        Rows need both trajectory labels and a future-start anchor to supervise
        the expert. Labelled navigation/grounding rows without that anchor still
        train their text objective. Trajectory-free batches execute a zero-weight
        expert segment so every data-parallel rank enters its backward graph.
        """
        if getattr(self.vlm, "is_gradient_checkpointing", False):
            raise ValueError("Joint training requires trainer.gradient_checkpointing=false")
        tokenized_data = self._prepare_vlm_inputs(tokenized_data)
        ids = tokenized_data["input_ids"]
        batch_size = ids.shape[0]
        if isinstance(traj_data, dict):
            present = list(range(batch_size))
            stacked = traj_data
        elif isinstance(traj_data, list):
            if len(traj_data) != batch_size:
                raise ValueError("Per-row trajectory data must match the input batch size")
            if any(row is not None and not isinstance(row, dict) for row in traj_data):
                raise ValueError("Each trajectory row must be a dictionary or None")
            present = [index for index, row in enumerate(traj_data) if row is not None]
            stacked = ({key: torch.stack([traj_data[index][key] for index in present])
                        for key in traj_data[present[0]]} if present else None)
        elif traj_data is None:
            present, stacked = [], None
        else:
            raise ValueError("Trajectory data must be a dictionary, per-row list, or None")
        starts = ids.eq(self.config.traj_ids["future_start"])
        counts = starts.sum(dim=1)
        if bool(counts.gt(1).any()):
            raise ValueError("Each joint sample may contain at most one future-start token")
        labelled = torch.zeros(batch_size, dtype=torch.bool, device=ids.device)
        labelled[present] = True
        if bool((counts.eq(1) & ~labelled).any()):
            raise ValueError("A future-start token requires trajectory data for that row")
        tokenized_data["input_ids"] = self._fuse_training_trajectories(ids, traj_data)
        ids = tokenized_data["input_ids"]
        valid = tokenized_data.get("attention_mask", torch.ones_like(ids)).bool()
        if bool((starts & ~valid).any()):
            raise ValueError("A future-start token must not be a padding position")
        positions = tokenized_data.get("position_ids")
        if positions is None:
            positions, rope_deltas = self.vlm.model.get_rope_index(
                input_ids=ids,
                mm_token_type_ids=tokenized_data.get("mm_token_type_ids", torch.zeros_like(ids)),
                image_grid_thw=tokenized_data.get("image_grid_thw"),
                video_grid_thw=tokenized_data.get("video_grid_thw"),
                attention_mask=valid,
            )
            self.vlm.model.rope_deltas = rope_deltas
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
        if positions.shape != (3, *ids.shape):
            raise ValueError("Joint training requires position_ids with shape [3, B, L]")
        tokenized_data["position_ids"] = positions
        outputs = self.vlm.model(**tokenized_data, use_cache=True)
        if outputs.past_key_values is None:
            raise RuntimeError("Joint training requires the native VLM to return its KV cache")
        vlm_output = self._vlm_loss(outputs, tokenized_data, labels_mask, joint=True)
        selected = counts[present].eq(1)
        rows = torch.tensor(present, dtype=torch.long, device=ids.device)[selected]
        if rows.numel():
            trajectories = {key: value[selected] if torch.is_tensor(value) else value
                            for key, value in stacked.items()}
            anchor_columns = starts[rows].long().argmax(dim=1)
            cache_length = int(anchor_columns.max().item()) + 1
            expert_cache = DynamicCache([
                (layer.keys[rows, ..., :cache_length, :].detach(),
                 layer.values[rows, ..., :cache_length, :].detach())
                for layer in outputs.past_key_values.layers
            ])
            visible = valid[rows, :cache_length] & (
                torch.arange(cache_length, device=ids.device)[None, :] <= anchor_columns[:, None]
            )
            expert_start = positions[:, rows, anchor_columns].amax(dim=0) + 1
            expert_output = self._run_expert_training(
                trajectories, SimpleNamespace(past_key_values=expert_cache), visible,
                expert_start_positions=expert_start,
            )
            expert_loss = expert_output.loss * self.config.loss_weights.get("future_traj", 1.0)
        else:
            expert_output = self._run_zero_weight_expert(ids.device)
            expert_loss = expert_output.loss
        return Alpamayo2EdgeSFTOutput(
            loss=vlm_output.loss + expert_loss,
            logits=expert_output.logits,
            supervised_logits=vlm_output.supervised_logits,
            future_traj_ce=vlm_output.future_traj_ce,
            others_ce=vlm_output.others_ce,
            expert_diffusion=expert_loss.detach(),
            vlm_ce=vlm_output.vlm_ce,
        )

    def _run_zero_weight_expert(self, device):
        """Run one self-attending dummy action segment with a zero-weight loss."""
        dims = self.expert.action_space.get_action_space_dims()
        with torch.autocast(device.type, enabled=False):
            zero_action = torch.zeros((1, *dims), device=device, dtype=torch.float32)
            # Reuse the FP32 sampler: ZeRO-3 initialization can cast the
            # inference diffusion object's cached Beta distribution to BF16.
            dummy = self._flow_matching_targets(zero_action)
        projection = self.expert.action_in_proj
        for encoder in [*projection.sinus, projection.timestep_fourier_encoder]:
            encoder.freqs = encoder.freqs.to(device=device, dtype=torch.float32)
        embeds = projection(dummy["noisy_x"], dummy["timesteps"])
        # Every dummy action position is zero; there is no visible VLM prefix.
        positions = torch.zeros(3, 1, embeds.shape[1], device=device, dtype=torch.long)
        noncausal = self.expert.config.expert_non_causal_attention
        forward_kwargs = {"is_causal": False} if noncausal else {}
        with noncausal_expert_mask(self.expert.expert.config, enabled=noncausal):
            outputs = self.expert.expert(
                inputs_embeds=embeds,
                position_ids=positions,
                past_key_values=None,
                attention_mask=None,
                use_cache=True,
                **forward_kwargs,
            )
        pred = self.expert.action_out_proj(outputs.last_hidden_state).view(-1, *dims)
        # Retain graph edges to every expert parameter, including if a prediction
        # is nonfinite. Returning a detached zero would break distributed backward.
        loss = torch.nan_to_num(pred.float(), nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
        return Alpamayo2EdgeSFTOutput(loss=loss, logits=pred)

    def _forward_joint(self, tokenized_data, traj_data, labels_mask):
        """Train both branches, exposing only detached prompt K/V to the expert."""
        if not isinstance(traj_data, dict):
            raise ValueError("Joint training requires a trajectory for every sample in the batch")
        if getattr(self.vlm, "is_gradient_checkpointing", False):
            raise ValueError("Joint training requires trainer.gradient_checkpointing=false")
        tokenized_data = self._prepare_vlm_inputs(tokenized_data)
        tokenized_data["input_ids"] = self._fuse_training_trajectories(
            tokenized_data["input_ids"], traj_data
        )
        ids = tokenized_data["input_ids"]
        valid = tokenized_data.get("attention_mask", torch.ones_like(ids)).bool()
        anchors = ids.eq(self.config.traj_ids["future_start"]) & valid
        if not torch.all(anchors.sum(dim=1) == 1):
            raise ValueError("Every joint sample must contain exactly one future-start token")
        anchor_columns = anchors.long().argmax(dim=1)
        positions = tokenized_data.get("position_ids")
        if positions is None:
            positions, rope_deltas = self.vlm.model.get_rope_index(
                input_ids=ids,
                mm_token_type_ids=tokenized_data.get("mm_token_type_ids", torch.zeros_like(ids)),
                image_grid_thw=tokenized_data.get("image_grid_thw"),
                video_grid_thw=tokenized_data.get("video_grid_thw"),
                attention_mask=valid,
            )
            self.vlm.model.rope_deltas = rope_deltas
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
        if positions.shape != (3, *ids.shape):
            raise ValueError("Joint training requires position_ids with shape [3, B, L]")
        tokenized_data["position_ids"] = positions
        outputs = self.vlm.model(**tokenized_data, use_cache=True)
        if outputs.past_key_values is None:
            raise RuntimeError("Joint training requires the native VLM to return its KV cache")
        vlm_output = self._vlm_loss(outputs, tokenized_data, labels_mask, joint=True)

        # The expert's DynamicCache is a separate object: its forward appends K/V.
        # Never detach/crop the live VLM cache used by the CE backward graph.
        cache_length = int(anchor_columns.max().item()) + 1
        expert_cache = DynamicCache([
            (layer.keys[..., :cache_length, :].detach(),
             layer.values[..., :cache_length, :].detach())
            for layer in outputs.past_key_values.layers
        ])
        visible = valid[:, :cache_length] & (
            torch.arange(cache_length, device=ids.device)[None, :] <= anchor_columns[:, None]
        )
        rows = torch.arange(ids.shape[0], device=ids.device)
        expert_start = positions[:, rows, anchor_columns].amax(dim=0) + 1
        expert_output = self._run_expert_training(
            traj_data, SimpleNamespace(past_key_values=expert_cache), visible,
            expert_start_positions=expert_start,
        )
        expert_loss = expert_output.loss * self.config.loss_weights.get("future_traj", 1.0)
        return Alpamayo2EdgeSFTOutput(
            loss=vlm_output.loss + expert_loss,
            logits=expert_output.logits,
            supervised_logits=vlm_output.supervised_logits,
            future_traj_ce=vlm_output.future_traj_ce,
            others_ce=vlm_output.others_ce,
            expert_diffusion=expert_loss.detach(),
            vlm_ce=vlm_output.vlm_ce,
        )

    def _forward_expert(
        self,
        tokenized_data: dict[str, Any],
        traj_data: dict[str, torch.Tensor],
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Alpamayo2EdgeSFTOutput:
        """Train the expert against a cropped, frozen VLM KV cache."""
        del labels_mask, kwargs
        if not isinstance(traj_data, dict):
            raise ValueError("Expert training requires a trajectory for every sample in the batch")
        tokenized_data = self._prepare_vlm_inputs(tokenized_data)
        tokenized_data["input_ids"] = self._fuse_training_trajectories(
            tokenized_data["input_ids"], traj_data
        )
        with torch.no_grad():
            vlm_outputs = self.vlm.model(**tokenized_data, use_cache=True)
            vlm_outputs["last_hidden_state"] = None
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        crop_length = self._future_start_crop_length(tokenized_data["input_ids"])
        vlm_outputs.past_key_values.crop(crop_length)
        for layer in vlm_outputs.past_key_values.layers:
            layer.keys = layer.keys.detach()
            layer.values = layer.values.detach()

        cache_attention_mask = tokenized_data.get("attention_mask")
        if cache_attention_mask is not None:
            cache_attention_mask = cache_attention_mask[:, :crop_length]
        expert_outputs = self._run_expert_training(
            traj_data=traj_data,
            vlm_outputs=vlm_outputs,
            cache_attention_mask=cache_attention_mask,
        )
        loss = expert_outputs.loss * self.config.loss_weights.get("future_traj", 1.0)
        logits = expert_outputs.logits
        cap = float(getattr(self.config, "logit_cap", 0.0))
        if cap > 0.0:
            logits = cap * torch.tanh(logits / cap)
        return Alpamayo2EdgeSFTOutput(loss=loss, logits=logits, expert_diffusion=loss.detach())

    def _run_expert_training(
        self,
        traj_data: dict[str, torch.Tensor],
        vlm_outputs: Any,
        cache_attention_mask: torch.Tensor | None,
        expert_start_positions: torch.Tensor | None = None,
    ) -> Alpamayo2EdgeSFTOutput:
        """Run expert training while keeping trajectory fitting and beta sampling in float32."""
        float_traj_data = {
            key: value.float() if torch.is_floating_point(value) else value
            for key, value in traj_data.items()
        }
        with CPUCholesky(), torch.autocast(
            float_traj_data["ego_history_xyz"].device.type, enabled=False
        ):
            action = (
                self.expert.action_space.traj_to_action(
                    traj_history_xyz=float_traj_data["ego_history_xyz"],
                    traj_history_rot=float_traj_data["ego_history_rot"],
                    traj_future_xyz=float_traj_data["ego_future_xyz"],
                    traj_future_rot=float_traj_data["ego_future_rot"],
                )
                .reshape(-1, *self.expert.action_space.get_action_space_dims())
                .float()
            )
            future_traj_data = self._flow_matching_targets(action)

        batch_size = future_traj_data["noisy_x"].shape[0]
        action_device = future_traj_data["noisy_x"].device
        action_in_proj = self.expert.action_in_proj
        for encoder in action_in_proj.sinus:
            encoder.freqs = encoder.freqs.to(device=action_device, dtype=torch.float32)
        action_in_proj.timestep_fourier_encoder.freqs = (
            action_in_proj.timestep_fourier_encoder.freqs.to(
                device=action_device,
                dtype=torch.float32,
            )
        )
        expert_embeds = self.expert.action_in_proj(
            future_traj_data["noisy_x"],
            future_traj_data["timesteps"],
        )
        position_ids = torch.arange(expert_embeds.shape[1], device=expert_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).clone()
        # HF5 measures mRoPE deltas against valid tokens. Expert positions use
        # physical cache columns, so remove the left padding at this boundary.
        if expert_start_positions is None:
            delta = vlm_outputs.rope_deltas
            if cache_attention_mask is not None:
                padding = cache_attention_mask.eq(0).sum(dim=-1, keepdim=True)
                delta = delta - padding.to(delta.device)
            delta = delta + vlm_outputs.past_key_values.get_seq_length()
        else:
            delta = expert_start_positions[:, None]
        position_ids += delta.to(position_ids.device)
        attention_mask = None
        if cache_attention_mask is not None:
            expert_token_mask = cache_attention_mask.new_ones((batch_size, expert_embeds.shape[1]))
            attention_mask = torch.cat([cache_attention_mask, expert_token_mask], dim=1)

        forward_kwargs = {}
        noncausal = self.expert.config.expert_non_causal_attention
        if noncausal:
            forward_kwargs["is_causal"] = False
        # HF5 constructs the mask before its attention kernel reads is_causal.
        # A 2D key-validity mask preserves padding and per-example cache bounds;
        # HF can omit its materialized SDPA mask when every key is visible.
        with noncausal_expert_mask(self.expert.expert.config, enabled=noncausal):
            expert_outputs = self.expert.expert(
                inputs_embeds=expert_embeds,
                position_ids=position_ids,
                past_key_values=vlm_outputs.past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
        pred = self.expert.action_out_proj(expert_outputs.last_hidden_state)
        pred = pred.view(-1, *self.expert.action_space.get_action_space_dims())
        target = future_traj_data["x"] - future_traj_data["noise"]
        loss = torch.nn.functional.mse_loss(target, pred.float())
        return Alpamayo2EdgeSFTOutput(loss=loss, logits=pred)

    def _flow_matching_targets(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Construct the Edge flow-matching training objective."""
        diffusion_cfg = self.config.expert_config.diffusion_cfg
        sampler = diffusion_cfg.get("train_timestep_sampler", "beta")
        if sampler == "beta":
            beta = torch.distributions.Beta(
                torch.tensor(1.5, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)
            )
            timesteps = beta.sample((action.shape[0],)).to(action.device)
            timesteps = 0.999 - timesteps * 0.999
        elif sampler == "uniform":
            timesteps = torch.rand(action.shape[0], device=action.device)
        else:
            raise ValueError(f"Unsupported flow-matching timestep sampler: {sampler!r}")
        timesteps = timesteps.reshape(-1, *([1] * (action.ndim - 1)))
        noise = torch.randn_like(action)
        return {
            "x": action,
            "noisy_x": timesteps * action + (1 - timesteps) * noise,
            "timesteps": timesteps,
            "noise": noise,
        }

    def _future_start_crop_length(self, input_ids: torch.Tensor) -> int:
        """Return the shared cache length ending after each future-start token."""
        positions = (input_ids == self.config.traj_ids["future_start"]).nonzero(as_tuple=False)
        if not torch.all((input_ids == self.config.traj_ids["future_start"]).sum(dim=1) == 1):
            raise ValueError("Every sample must contain exactly one future-start token")
        if torch.any(positions[:, 1] != positions[0, 1]):
            raise ValueError("Expert batches require left padding")
        return int(positions[0, 1].item() + 1)
