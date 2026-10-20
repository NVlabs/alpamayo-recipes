# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a saved distillation checkpoint on the configured validation set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from alpamayo1_5_distill.data.physical_aiav_dataset import (
    OnlinePhysicalAIAVDataset,
    identity_collate,
)
from alpamayo1_5_distill.models.modeling_distill import DistillAlpamayo1_5
from alpamayo1_5_distill.train import DistillTrainModule
from alpamayo1_5_distill.training import processor
from alpamayo1_5_distill.training.common import import_callable, load_config
from alpamayo1_5_distill.training.losses import (
    DistillationLoss,
    DistillationLossConfig,
)
from alpamayo1_5_distill.training.on_policy import OnPolicyDistillConfig
from alpamayo1_5_distill.training.validation import RankShardSampler, run_validation


def evaluate(config: dict[str, Any]) -> dict[str, float]:
    """Load student and teacher models, then run the shared validation loop."""
    accelerator = Accelerator(
        mixed_precision=str(config.get("evaluation", {}).get("mixed_precision", "no"))
    )
    checkpoint = config.get("evaluation", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("evaluation.checkpoint is required")
    checkpoint_path = Path(checkpoint)
    student_path = checkpoint_path / "student"
    if not student_path.is_dir():
        student_path = checkpoint_path
    state_root = checkpoint_path.parent if checkpoint_path.name == "student" else checkpoint_path

    model = DistillAlpamayo1_5.from_pretrained(
        student_path, dtype=getattr(torch, config["model"]["kwargs"].get("dtype", "bfloat16"))
    ).to(accelerator.device)
    teacher_spec = config["teacher"]
    teacher = import_callable(teacher_spec["factory"])(**teacher_spec.get("kwargs", {})).to(
        accelerator.device
    )
    teacher.eval().requires_grad_(False)

    objective = DistillationLoss(
        DistillationLossConfig(**config.get("loss", {})),
        OnPolicyDistillConfig(**config.get("on_policy", {})),
    ).to(accelerator.device)
    objective_path = state_root / "objective.pt"
    if objective_path.is_file():
        objective.load_state_dict(
            torch.load(objective_path, map_location=accelerator.device, weights_only=True)
        )
    elif any(True for _ in objective.parameters()):
        raise FileNotFoundError(f"checkpoint is missing learned objective state: {objective_path}")
    module = DistillTrainModule(model, objective).to(accelerator.device)

    validation = config.get("validation", {})
    data = config["data"]
    nav_config = dict(validation.get("nav", {}))
    nav_config["root"] = nav_config.get("root") or os.getenv("ALPAMAYO_NAV_DATASET_DIR")
    dataset = OnlinePhysicalAIAVDataset(
        local_dir=data.get("local_dir") or os.getenv("ALPAMAYO_DATASET_DIR"),
        token=os.getenv("HF_TOKEN"),
        split=str(validation.get("split", "val")),
        t0_us_list=[int(validation.get("t0_us", 5_100_000))],
        random_t0=False,
        samples_per_clip=1,
        margin_us=int(validation.get("t0_margin_us", 0)),
        seed=int(config.get("training", {}).get("seed", 42)),
        max_clip_retries=int(validation.get("max_clip_retries", 0)),
        loader_kwargs={"maybe_stream": False},
        clip_allowlist_path=validation.get("clip_allowlist_path"),
        exclude_clip_ids_path=validation.get("exclude_clip_ids_path"),
        nav_config=nav_config,
    )
    workers = int(validation.get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {}
    if workers:
        loader_kwargs["prefetch_factor"] = int(validation.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(validation.get("persistent_workers", True))
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(
        dataset,
        batch_size=int(validation.get("batch_size", 1)),
        sampler=RankShardSampler(
            len(dataset), accelerator.process_index, accelerator.num_processes
        ),
        num_workers=workers,
        pin_memory=bool(validation.get("pin_memory", True)),
        collate_fn=identity_collate,
        **loader_kwargs,
    )
    metrics = run_validation(
        module=module,
        teacher=teacher,
        loader=loader,
        accelerator=accelerator,
        teacher_processor=processor.get_processor(
            teacher.tokenizer, getattr(teacher.config, "vlm_name_or_path", None)
        ),
        student_processor=processor.get_processor(
            model.tokenizer, getattr(model.config, "vlm_name_or_path", None)
        ),
        online_teacher_config=config.get("online_teacher", {}),
        loss_temperature=float(config["loss"].get("temperature", 0.5)),
        compute_minade=bool(validation.get("compute_minade", True)),
        stochastic_samples=int(validation.get("stochastic_samples", 6)),
        top_p=float(validation.get("top_p", 0.98)),
        sampling_temperature=float(validation.get("temperature", 0.6)),
        max_generation_length=int(validation.get("max_generation_length", 256)),
    )
    if accelerator.is_main_process:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    evaluate(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
