# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for persisting the fully resolved recipe configuration."""

from copy import deepcopy
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict


def save_run_config(config: DictConfig, output_dir: Path) -> None:
    """Write ``config.yaml`` under ``output_dir`` with every interpolation resolved.

    Hydra's own config node is dropped on purpose: ``hydra.sweep.subdir`` resolves
    ``${hydra.job.num}``, which is mandatory-missing outside multirun, so persisting
    that node makes ``resolve=True`` raise on every single-run launch.
    """
    save_config = deepcopy(config)
    with open_dict(save_config):
        save_config.pop("hydra", None)
    # Keep the API key out of the persisted config; Hydra still records CLI
    # overrides verbatim, so prefer `wandb login` over `wandb.key=...`.
    if save_config.get("wandb") is not None and save_config.wandb.get("key") is not None:
        save_config.wandb.key = "***"
    OmegaConf.save(
        config=save_config,
        f=output_dir / "config.yaml",
        resolve=True,
    )
