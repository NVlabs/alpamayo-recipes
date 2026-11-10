# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face trainer extensions for Alpamayo 2 Edge."""

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback
from transformers import TrainingArguments as HuggingFaceTrainingArguments

from .deepspeed_clipping import configure_joint_deepspeed_clipping
from .deepspeed_reduction import configure_joint_deepspeed_reduction
from .global_norm import (
    clip_fp32_with_norm_, fp32_global_norm,
    fsdp_norm_unsupported_reason, validate_norm_algorithm,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(HuggingFaceTrainingArguments):
    """Add per-module learning rates and independent expert training RNG streams."""

    lr_multiplier: dict[str, float] | None = field(
        default=None,
        metadata={"help": "Longest-prefix learning-rate multipliers for named parameters."},
    )
    independent_dp_rng: bool = field(
        default=False,
        metadata={"help": "Seed fresh expert flow draws with seed + data-parallel rank."},
    )
    gradient_norm_algorithm: str = field(
        default="auto",
        metadata={"help": "Joint norm arithmetic: auto matches FSDP FP32 on supported GPU DP backends; fsdp_fp32 requires it; engine retains backend arithmetic."},
    )

    gradient_norm_policy: dict | None = field(default=None, init=False)

    def __post_init__(self):
        validate_norm_algorithm(self.gradient_norm_algorithm)
        super().__post_init__()


class IndependentDPRNGCallback(TrainerCallback):
    """Assign independent expert flow draws after model and Trainer initialization."""

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if (
            not args.independent_dp_rng
            or state.global_step != 0
            or getattr(model, "training_stage", None) not in {"joint", "expert"}
        ):
            return control
        # Beta timesteps use the CPU RNG; Gaussian action noise uses the local
        # CUDA RNG. manual_seed seeds both. In the supported data-parallel
        # configurations, process_index is the data-parallel rank.
        # HF restores saved rank RNG states later when resuming a checkpoint.
        rank_seed = (args.seed + args.process_index) % (2**63 - 1)
        torch.manual_seed(rank_seed)
        logger.info("Expert flow RNG: rank=%s seed=%s", args.process_index, rank_seed)
        return control


class DatasetEpochCallback(TrainerCallback):
    """Propagate epochs even when Accelerate only updates its nested sampler."""

    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(math.floor(state.epoch or 0))
        return control


class Alpamayo2EdgeTrainer(Trainer):
    """Trainer extension that applies per-module learning-rate multipliers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(IndependentDPRNGCallback)
        self.add_callback(DatasetEpochCallback(self.train_dataset))

    def training_step(self, model, inputs, num_items_in_batch=None):
        # HF has constructed/wrapped and restored the actual engine by this
        # point. Accelerate can clip and update inside backward, so setup must
        # precede super().training_step rather than an optimizer callback.
        if self.is_deepspeed_enabled:
            configure_joint_deepspeed_reduction(model)
            configure_joint_deepspeed_clipping(
                model, norm_algorithm=getattr(self.args, "gradient_norm_algorithm", "auto")
            )
            policy = getattr(model.optimizer, "_alpamayo2edge_norm_policy", None)
            if policy is not None:
                self._record_gradient_norm_policy(policy)
        return super().training_step(model, inputs, num_items_in_batch)

    def _record_gradient_norm_policy(self, policy):
        """Persist runtime resolution alongside launch config and saved training args."""
        if getattr(self.args, "gradient_norm_policy", None) == policy:
            return
        self.args.gradient_norm_policy = dict(policy)
        logger.info("Resolved joint gradient norm policy: %s", policy)
        if self.is_world_process_zero():
            destination = Path(self.args.output_dir) / "runtime_policy.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"gradient_norm": policy}, indent=2) + "\n")

    def _clip_grad_norm(self, model):
        """Use the same model-ordered foreach norm on supported joint FSDP2."""
        algorithm = validate_norm_algorithm(getattr(self.args, "gradient_norm_algorithm", "auto"))
        if getattr(model, "training_stage", None) != "joint":
            return super()._clip_grad_norm(model)
        if algorithm == "engine":
            self._record_gradient_norm_policy(dict(requested=algorithm, resolved="engine", fallback_reason=None))
            return super()._clip_grad_norm(model)
        if self.is_deepspeed_enabled:
            # DS clips inside backward; the installation above owns that path.
            return super()._clip_grad_norm(model)
        plugin = self.accelerator.state.fsdp_plugin if self.is_fsdp_enabled else None
        reason = None
        if plugin is None or plugin.fsdp_version != 2:
            reason = "the matched distributed norm requires ZeRO3 or FSDP2"
        else:
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            reason = fsdp_norm_unsupported_reason(gradients)
        if reason is not None:
            if algorithm == "fsdp_fp32":
                raise ValueError(f'fsdp_fp32 gradient norm is unsupported: {reason}')
            if not getattr(self, "_norm_fallback_warned", False):
                logger.warning('Joint gradient norm uses the engine algorithm: %s', reason)
                self._norm_fallback_warned = True
            self._record_gradient_norm_policy(dict(requested=algorithm, resolved="engine", fallback_reason=reason))
            return super()._clip_grad_norm(model)
        self._record_gradient_norm_policy(dict(requested=algorithm, resolved="fsdp_fp32", fallback_reason=None))
        self.accelerator.unscale_gradients()
        norm = fp32_global_norm(gradients)
        clip_fp32_with_norm_(gradients, self.args.max_grad_norm, norm)
        return norm

    def _prepare_input(self, data):
        """Move inputs to the device while keeping geometric calculations in FP32.

        HF's DeepSpeed preparation casts every floating tensor to the engine
        dtype. That loses trajectory precision before action fitting/tokenization.
        Keep the original FP32 values here; autocast and the model's explicit
        vision/expert casts still select the neural-network compute dtype.
        """
        if isinstance(data, Mapping):
            return type(data)({key: self._prepare_input(value) for key, value in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(value) for value in data)
        if isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if torch.is_floating_point(data):
                kwargs["dtype"] = torch.float32
            return data.to(**kwargs)
        return data

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Track the two objectives separately across the logging interval."""
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        phase = "train" if model.training else "eval"
        if not hasattr(self, "_component_losses"):
            self._component_losses = {}
        totals = self._component_losses.setdefault(phase, {})
        names = [name for name in ("vlm_ce", "expert_diffusion", "future_traj_ce", "others_ce")
                 if outputs.get(name) is not None]
        if names:
            values = torch.stack([outputs[name].detach().float() for name in names])
            if phase == "eval":
                from transformers.trainer_pt_utils import find_batch_size

                # Match Trainer's sample-weighted loss and discard duplicated
                # padding samples in the final distributed evaluation batch.
                batch_size = find_batch_size(inputs) or 1
                values = self.accelerator.gather_for_metrics(
                    values.unsqueeze(0).expand(batch_size, -1)
                )
                count = values.shape[0]
                values = values.sum(dim=0)
            else:
                count = 1
            for name, value in zip(names, values, strict=True):
                previous, previous_count = totals.get(name, (torch.zeros_like(value), 0))
                totals[name] = (previous + value, previous_count + count)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        """Report mean branch losses across data-parallel ranks to Trainer/W&B."""
        phase = "eval" if "eval_loss" in logs else "train" if "loss" in logs else None
        totals = getattr(self, "_component_losses", {}).pop(phase, {})
        for name, (total, count) in totals.items():
            values = torch.stack((total, total.new_tensor(count)))
            values = self.accelerator.reduce(values, reduction="sum")
            key = f"eval_{name}" if phase == "eval" else name
            logs[key] = (values[0] / values[1]).item()
        return super().log(logs, start_time=start_time)

    def create_accelerator_and_postprocess(self) -> None:
        """Keep FSDP2 masters and each accumulated gradient in FP32."""
        super().create_accelerator_and_postprocess()
        if not self.is_fsdp_enabled:
            return
        plugin = self.accelerator.state.fsdp_plugin
        if plugin.fsdp_version != 2:
            return
        if not self.args.bf16 or self.args.fp16:
            raise ValueError("Edge FSDP2 training requires bf16=true and fp16=false")

        from torch.distributed.fsdp import MixedPrecisionPolicy

        # Preserve FP32 trajectory fitting and Fourier inputs at the model
        # boundary; FSDP materializes parameter copies in BF16 for computation.
        plugin.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=False,
        )
        # Reduce every microbatch, then accumulate FP32 gradients
        # before each optimizer step. no_sync would instead accumulate BF16
        # materialized gradients, even when running FSDP2 on a single GPU.
        self.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True
        self.args.accelerator_config.gradient_accumulation_kwargs = {
            **(self.args.accelerator_config.gradient_accumulation_kwargs or {}),
            "sync_each_batch": True,
        }

    def get_decay_parameter_names(self, model):
        """Apply AdamW decay to every trainable parameter."""
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        """Create the configured optimizer once."""
        if self.optimizer is not None:
            return self.optimizer
        optimizer_model = self.model if model is None else model
        if not self.args.deepspeed and not self.is_fsdp_enabled:
            low_precision = [
                name
                for name, parameter in optimizer_model.named_parameters()
                if parameter.requires_grad and parameter.dtype != torch.float32
            ]
            if low_precision:
                raise ValueError(
                    "Plain AdamW training requires FP32 parameters and optimizer states. "
                    "Load the checkpoint with model.dtype=float32; trainer.bf16=true "
                    "still enables BF16 computation. Low-precision trainable parameter: "
                    f"{low_precision[0]}"
                )
        if self.args.lr_multiplier is None:
            return super().create_optimizer(model=optimizer_model)

        decay_names = self.get_decay_parameter_names(optimizer_model)
        grouped_parameters: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        for name, parameter in optimizer_model.named_parameters():
            if not parameter.requires_grad:
                continue
            multiplier = self._learning_rate_multiplier(name)
            weight_decay = self.args.weight_decay if name in decay_names else 0.0
            key = (self.args.learning_rate * multiplier, weight_decay)
            grouped_parameters.setdefault(key, []).append(parameter)

        # HF logs scheduler.get_last_lr()[0]. Put the unscaled language-model
        # group first so learning_rate represents the configured base LR.
        ordered_groups = sorted(
            grouped_parameters.items(),
            key=lambda item: item[0][0] != self.args.learning_rate,
        )
        optimizer_groups = [
            {"params": parameters, "lr": learning_rate, "weight_decay": weight_decay}
            for (learning_rate, weight_decay), parameters in ordered_groups
        ]
        optimizer_class, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args,
            optimizer_model,
        )
        self.optimizer = optimizer_class(optimizer_groups, **optimizer_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Create the cosine schedule with an inclusive warmup endpoint."""
        if self.args.lr_scheduler_type != "cosine_with_min_lr":
            return super().create_scheduler(num_training_steps, optimizer=optimizer)
        if self.lr_scheduler is None:
            options = self.args.lr_scheduler_kwargs or {}
            if set(options) - {"min_lr_rate"}:
                raise ValueError("Cosine scheduling accepts only min_lr_rate")
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer or self.optimizer,
                partial(
                    cosine_with_warmup_multiplier,
                    total_steps=num_training_steps,
                    warmup_steps=min(self.args.get_warmup_steps(num_training_steps), num_training_steps),
                    min_lr_factor=options.get("min_lr_rate", 0.01),
                ),
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def evaluate(self, *args, **kwargs):
        """Reuse validation noise without advancing the training RNG stream."""
        devices = [self.args.device.index] if self.args.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.args.seed)
            return super().evaluate(*args, **kwargs)

    def _learning_rate_multiplier(self, parameter_name: str) -> float:
        """Return the longest matching parameter-prefix multiplier."""
        matches = [
            (prefix, multiplier)
            for prefix, multiplier in self.args.lr_multiplier.items()
            if parameter_name.startswith(prefix)
        ]
        if not matches:
            return 1.0
        return max(matches, key=lambda item: len(item[0]))[1]


def cosine_with_warmup_multiplier(current_step, total_steps, warmup_steps, min_lr_factor):
    """Return the zero-indexed cosine multiplier used by LambdaLR."""
    if current_step < warmup_steps:
        return 1.0 if warmup_steps == 1 else current_step / (warmup_steps - 1)
    if current_step <= warmup_steps:
        return 1.0
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * progress))
