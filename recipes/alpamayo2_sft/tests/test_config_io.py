# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for saving the run config on single-run launches."""

from pathlib import Path

from omegaconf import OmegaConf

from alpamayo2_sft.config_io import save_run_config


def _single_run_config(output_dir: Path) -> OmegaConf:
    """Config with the Hydra node that single-run launches inject."""
    config = OmegaConf.create(
        {
            "camera_features": ["camera_front_wide_120fov", "camera_rear_tele_30fov"],
            "num_frames": 4,
            "paths": {"output_dir": str(output_dir)},
            "data": {
                "train_dataset": {
                    "camera_features": "${camera_features}",
                    "num_frames": "${num_frames}",
                }
            },
            "wandb": {"key": "secret"},
            "hydra": {
                "job": {"num": "???"},
                "sweep": {"subdir": "${hydra.job.num}"},
            },
        }
    )
    OmegaConf.set_struct(config, True)
    return config


def test_save_run_config_resolves_single_run_config(tmp_path: Path) -> None:
    save_run_config(_single_run_config(tmp_path), tmp_path)

    saved = OmegaConf.load(tmp_path / "config.yaml")

    assert "hydra" not in saved
    assert saved.data.train_dataset.camera_features == [
        "camera_front_wide_120fov",
        "camera_rear_tele_30fov",
    ]
    assert saved.data.train_dataset.num_frames == 4
    assert saved.paths.output_dir == str(tmp_path)
    assert saved.wandb.key == "***"
