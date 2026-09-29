# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training extension for the released Alpamayo 2 Super model."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Self

import torch
from safetensors.torch import load_file
from transformers.integrations.deepspeed import (
    _load_state_dict_into_zero3_model,
    is_deepspeed_zero3_enabled,
)
from transformers.utils import ModelOutput

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
from alpamayo2_super.models.utils import IGNORE_INDEX, compute_next_token_loss, fuse_traj_tokens


@dataclass
class Alpamayo2SuperSFTOutput(ModelOutput):
    """Training output with separate trajectory and text losses."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    future_traj_ce: torch.FloatTensor | None = None
    others_ce: torch.FloatTensor | None = None


class TrainableAlpamayo2Super(Alpamayo2Super):
    """Add VLM and expert SFT behavior to Alpamayo 2 Super."""

    def __init__(self, config, training_stage: str = "vlm") -> None:
        super().__init__(config)
        self.set_training_stage(training_stage)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Self:
        """Load a release checkpoint and optionally overlay a Stage-1 VLM checkpoint."""
        stage1_vlm_checkpoint_path = kwargs.pop("stage1_vlm_checkpoint_path", None)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )
        if stage1_vlm_checkpoint_path is not None:
            model.load_stage1_vlm(stage1_vlm_checkpoint_path)
        model.set_training_stage(model.training_stage)
        return model

    def set_training_stage(self, training_stage: str) -> None:
        """Select the trainable parameter set and forward path."""
        if training_stage not in {"vlm", "expert"}:
            raise ValueError(f"training_stage must be 'vlm' or 'expert', got {training_stage!r}")
        if training_stage == "expert" and not self.config.enable_expert:
            raise ValueError("Expert training requires an expert-enabled release checkpoint")

        self.training_stage = training_stage
        self.vlm.requires_grad_(training_stage == "vlm")
        if hasattr(self, "expert"):
            self.expert.requires_grad_(training_stage == "expert")

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
            error_messages = _load_state_dict_into_zero3_model(self.vlm, vlm_state_dict)
            if error_messages:
                raise RuntimeError("\n".join(error_messages))
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
    ) -> None:
        """Enable checkpointing on the nested VLM."""
        self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable checkpointing on the nested VLM."""
        self.vlm.gradient_checkpointing_disable()

    def forward(
        self,
        tokenized_data: dict[str, Any],
        traj_data: dict[str, torch.Tensor],
        labels_mask: torch.Tensor | None = None,
        expert_skip_vlm_lm_head: bool = False,
        **kwargs: Any,
    ) -> Alpamayo2SuperSFTOutput:
        """Run the selected SFT stage."""
        if self.training_stage == "expert":
            return self._forward_expert(
                tokenized_data=tokenized_data,
                traj_data=traj_data,
                labels_mask=labels_mask,
                expert_skip_vlm_lm_head=expert_skip_vlm_lm_head,
                **kwargs,
            )
        if expert_skip_vlm_lm_head:
            raise ValueError("VLM LM-head bypass requires frozen-VLM expert training")

        tokenized_data = dict(tokenized_data)
        tokenized_data["input_ids"] = fuse_traj_tokens(
            self.history_traj_tokenizer,
            self.future_traj_tokenizer,
            tokenized_data["input_ids"],
            traj_data,
            self.config.traj_ids,
        )
        labels = tokenized_data["input_ids"].clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        outputs = self.vlm(**tokenized_data, use_cache=False)
        logits = outputs.logits
        cap = float(getattr(self.config, "logit_cap", 0.0))
        if cap > 0.0:
            logits = cap * torch.tanh(logits / cap)

        traj_mask = (
            (
                (labels >= self.config.traj_ids["future_id0"])
                & (labels < self.config.traj_ids["future_id0"] + self.config.future_vocab_size)
            )
            | (labels == self.config.traj_ids["future_start"])
            | (labels == self.config.traj_ids["future_end"])
        )
        future_traj_loss = compute_next_token_loss(logits, labels, traj_mask)
        future_traj_loss *= self.config.loss_weights.get("future_traj", 1.0)
        labels[traj_mask] = IGNORE_INDEX
        other_loss = compute_next_token_loss(logits, labels, labels != IGNORE_INDEX)
        other_loss *= self.config.loss_weights.get("others", 1.0)
        return Alpamayo2SuperSFTOutput(
            loss=future_traj_loss + other_loss,
            logits=logits,
            future_traj_ce=future_traj_loss.detach(),
            others_ce=other_loss.detach(),
        )

    def _forward_expert(
        self,
        tokenized_data: dict[str, Any],
        traj_data: dict[str, torch.Tensor],
        labels_mask: torch.Tensor | None = None,
        expert_skip_vlm_lm_head: bool = False,
        **kwargs: Any,
    ) -> Alpamayo2SuperSFTOutput:
        """Train the expert against a cropped, frozen VLM KV cache."""
        if expert_skip_vlm_lm_head and self.config.cotrain_expert_vlm:
            raise ValueError("VLM LM-head bypass requires a frozen VLM")

        tokenized_data = dict(tokenized_data)
        tokenized_data["input_ids"] = fuse_traj_tokens(
            self.history_traj_tokenizer,
            self.future_traj_tokenizer,
            tokenized_data["input_ids"],
            traj_data,
            self.config.traj_ids,
        )
        labels = tokenized_data["input_ids"].clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        context = nullcontext() if self.config.cotrain_expert_vlm else torch.no_grad()
        with context:
            if expert_skip_vlm_lm_head:
                vlm_outputs = self.vlm.model(**tokenized_data, use_cache=True, **kwargs)
                vlm_outputs["last_hidden_state"] = None
            else:
                vlm_outputs = self.vlm(**tokenized_data, use_cache=True, **kwargs)

        crop_length = self._future_start_crop_length(tokenized_data["input_ids"])
        vlm_outputs.past_key_values.crop(crop_length)
        if not self.config.cotrain_expert_vlm:
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
        if self.config.cotrain_expert_vlm:
            loss += compute_next_token_loss(
                vlm_outputs.logits,
                labels,
                labels != IGNORE_INDEX,
            ) * self.config.loss_weights.get("others", 1.0)
        return Alpamayo2SuperSFTOutput(loss=loss, logits=expert_outputs.logits)

    def _run_expert_training(
        self,
        traj_data: dict[str, torch.Tensor],
        vlm_outputs: Any,
        cache_attention_mask: torch.Tensor | None,
    ) -> Alpamayo2SuperSFTOutput:
        """Run expert training while keeping trajectory fitting and beta sampling in float32."""
        float_traj_data = {
            key: value.float() if torch.is_floating_point(value) else value
            for key, value in traj_data.items()
        }
        with torch.autocast("cuda", enabled=False):
            diffusion = self.expert.diffusion
            if (
                diffusion.train_timestep_sampler == "beta"
                and diffusion.beta_dist.concentration1.dtype != torch.float32
            ):
                diffusion.beta_dist = torch.distributions.Beta(
                    diffusion.beta_dist.concentration1.float().cpu(),
                    diffusion.beta_dist.concentration0.float().cpu(),
                )
            future_traj_data = self.expert._process_traj_future_training(float_traj_data)

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
        position_ids = self.expert._process_mrope_position_ids(
            vlm_outputs,
            batch_size,
            expert_embeds.shape[1],
            expert_embeds.device,
        )
        attention_mask = None
        if cache_attention_mask is not None:
            expert_token_mask = cache_attention_mask.new_ones(
                (batch_size, expert_embeds.shape[1])
            )
            attention_mask = torch.cat([cache_attention_mask, expert_token_mask], dim=1)

        forward_kwargs = {}
        if self.expert.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
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
        loss = self.expert.diffusion.compute_loss_from_pred(future_traj_data, pred.float())
        return Alpamayo2SuperSFTOutput(loss=loss, logits=pred)

    def _future_start_crop_length(self, input_ids: torch.Tensor) -> int:
        """Return the shared cache length ending after each future-start token."""
        positions = (input_ids == self.config.traj_ids["future_start"]).nonzero(as_tuple=False)
        if positions.shape[0] != input_ids.shape[0]:
            raise ValueError("Every sample must contain exactly one future-start token")
        if torch.any(positions[:, 1] != positions[0, 1]):
            raise ValueError("Expert batches require left padding")
        return int(positions[0, 1].item() + 1)
