# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration, run tracking, and memory reporting for SFT entry points."""

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import wandb
from omegaconf import DictConfig, OmegaConf


def save_resolved_config(config: DictConfig, path: Path) -> None:
    """Save the resolved recipe config without a legacy W&B API key."""
    saved = deepcopy(config)
    if saved.get("wandb") is not None and "key" in saved.wandb:
        # Redact before resolving: an environment interpolation can contain a key.
        saved.wandb.key = "***"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(saved, path, resolve=True)


@contextmanager
def tracking_run(config: DictConfig, trainer):
    """Initialize W&B for HF's normal callback and flush it on success or failure.

    Environment settings support the ordinary ``report_to=wandb`` workflow. The
    optional legacy ``wandb`` config remains supported; prefer environment-based
    authentication because Hydra records command-line overrides verbatim.
    """
    reports = trainer.args.report_to
    enabled = reports == "wandb" or isinstance(reports, (list, tuple)) and "wandb" in reports
    if not enabled or not trainer.is_world_process_zero():
        yield
        return

    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = config.get("wandb") or {}
    if settings.get("key") is not None:
        wandb.login(key=settings["key"])
    id_path = output_dir / ".wandb_id"
    resume = bool(getattr(trainer.args, "resume_from_checkpoint", None))
    # A reused output directory is not an instruction to merge unrelated runs.
    run_id = (
        os.environ.get("WANDB_RUN_ID")
        or (id_path.read_text().strip() if resume and id_path.is_file() else None)
        or wandb.util.generate_id()
    )
    options = {
        "id": run_id,
        "project": settings.get("project") or os.environ.get("WANDB_PROJECT", "huggingface"),
        "name": settings.get("name") or trainer.args.run_name,
        "dir": str(output_dir),
        "resume": "allow" if resume else "never",
    }
    if settings.get("team") is not None:
        options["entity"] = settings["team"]
    if settings.get("group") is not None:
        options["group"] = settings["group"]
    run = wandb.init(**options)
    exit_code = 1
    try:
        id_path.write_text(run.id + "\n")
        mode = run.settings.mode
        metadata = {
            "id": run.id,
            "url": run.url if mode not in {"offline", "disabled"} else None,
            "project": run.project,
            "entity": run.entity,
            "mode": mode,
        }
        (output_dir / "wandb_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        yield
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)


def reset_peak_cuda_memory() -> None:
    """Start peak tracking after model setup, retaining currently allocated memory."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_cuda_memory_metrics(prefix: str) -> dict[str, float]:
    """Report this process's peak PyTorch allocator usage in GiB."""
    if not torch.cuda.is_available():
        return {}
    return {
        f"{prefix}_peak_cuda_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        f"{prefix}_peak_cuda_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
