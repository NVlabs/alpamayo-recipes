# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face trainer extensions for Alpamayo 2 Super."""

from dataclasses import dataclass, field

import torch
from transformers import Trainer
from transformers import TrainingArguments as HuggingFaceTrainingArguments


@dataclass
class TrainingArguments(HuggingFaceTrainingArguments):
    """Add per-module learning-rate multipliers."""

    lr_multiplier: dict[str, float] | None = field(
        default=None,
        metadata={"help": "Longest-prefix learning-rate multipliers for named parameters."},
    )


class Alpamayo2SuperTrainer(Trainer):
    """Trainer extension that applies per-module learning-rate multipliers."""

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the configured optimizer once."""
        if self.optimizer is not None or self.args.lr_multiplier is None:
            return super().create_optimizer()

        decay_names = self.get_decay_parameter_names(self.model)
        grouped_parameters: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            multiplier = self._learning_rate_multiplier(name)
            weight_decay = self.args.weight_decay if name in decay_names else 0.0
            key = (self.args.learning_rate * multiplier, weight_decay)
            grouped_parameters.setdefault(key, []).append(parameter)

        optimizer_groups = [
            {"params": parameters, "lr": learning_rate, "weight_decay": weight_decay}
            for (learning_rate, weight_decay), parameters in grouped_parameters.items()
        ]
        optimizer_class, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args,
            self.model,
        )
        self.optimizer = optimizer_class(optimizer_groups, **optimizer_kwargs)
        return self.optimizer

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
