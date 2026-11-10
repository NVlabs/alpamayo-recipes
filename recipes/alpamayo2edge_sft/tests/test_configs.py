# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composition checks for joint, evaluation, and debug configurations."""

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "name",
    [
        "sft_joint",
        "sft_eval_loss",
        "sft_eval_expert_loss",
        "sft_eval_trajectory",
        "sft_joint_single_node_debug",
    ],
)
def test_configs_compose_with_consistent_data_contracts(name):
    directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name=name,
            overrides=[
                "data.train_dataset.local_dir=/tmp/pai",
                "data.val_dataset.local_dir=/tmp/pai",
                "seed=123",
            ],
        )
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert config.trainer.seed == config.trainer.data_seed == 123
    assert config.data.train_dataset.data_seed == 123
    assert config.data.train_dataset.frame_offsets[-1] == 0
    assert config.data.val_dataset.front_frame_offsets[-1] == 0
    assert config.model.pretrained_model_name_or_path == "nvidia/Alpamayo2-Edge"
    assert config.model.qk_gradient_layout is False
    expected_stage = "expert" if name in {"sft_eval_expert_loss", "sft_eval_trajectory"} else "joint"
    assert config.model.training_stage == expected_stage
    assert config.trainer.gradient_checkpointing is False
    assert config.trainer.independent_dp_rng is True
    assert config.trainer.learning_rate == pytest.approx(3.5e-6)
    assert OmegaConf.to_container(config.trainer.lr_multiplier) == {
        "vlm.model.visual": 0.1,
        "vlm.model.projector": 0.1,
        "expert": 4.0,
    }
    assert "stage1_vlm_checkpoint_path" not in config.model
    assert OmegaConf.to_container(config.model.loss_weights) == {"future_traj": 1.0, "others": 1.0}
    assert OmegaConf.to_container(config.model.lm_token_loss_weights) == {"future_traj": 1.0, "text": 0.2}
    if config.evaluate.mode == "loss" and "eval" in name:
        assert not config.data.val_dataset.preprocess.generation_mode
    if config.evaluate.mode == "trajectory":
        assert config.data.val_dataset.preprocess.generation_mode
        assert config.trainer.deepspeed is None
    if "debug" in name:
        assert config.trainer.max_steps == 2
        assert config.trainer.deepspeed is None
        assert config.trainer.warmup_steps == 0
    else:
        assert config.trainer.max_steps == -1
        assert config.trainer.num_train_epochs == 3
        assert config.trainer.warmup_steps == 0.04
        assert config.trainer.save_total_limit == 3
        assert config.trainer.save_strategy == "epoch"
        assert config.trainer.eval_strategy == "epoch"
        assert config.trainer.eval_on_start is True
        assert config.data.collator.pad_to_fixed_length is False
        assert config.evaluate.max_eval_steps == -1
        assert config.data.train_dataset.split == "train"
        assert config.data.val_dataset.split == "val"
        assert config.data.train_dataset.random_keyframes is True
        assert config.data.train_dataset.chunk_ids == "729-744"
        assert config.data.val_dataset.chunk_ids == "1134-1145"
    if config.trainer.deepspeed:
        assert (directory.parent / config.trainer.deepspeed).is_file()


def test_training_dataset_follows_data_seed_independently_from_model_seed():
    directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name="sft_joint",
            overrides=[
                "data.train_dataset.local_dir=/tmp/pai",
                "data.val_dataset.local_dir=/tmp/pai",
                "seed=123",
                "trainer.seed=456",
                "trainer.data_seed=789",
            ],
        )
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert config.seed == 123
    assert config.trainer.seed == 456
    assert config.trainer.data_seed == config.data.train_dataset.data_seed == 789


def test_joint_loads_full_checkpoint_and_accepts_a_different_training_dataset():
    directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name="sft_joint",
            overrides=[
                "model.pretrained_model_name_or_path=/tmp/joint/checkpoint-500",
                "data.train_dataset.local_dir=/tmp/pai-new",
                "data.train_dataset.chunk_ids=100-110",
                "data.val_dataset.local_dir=/tmp/pai",
            ],
        )
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert config.model.training_stage == "joint"
    assert config.model.pretrained_model_name_or_path == "/tmp/joint/checkpoint-500"
    assert "stage1_vlm_checkpoint_path" not in config.model
    assert config.data.train_dataset.chunk_ids == "100-110"
    assert config.data.val_dataset.chunk_ids == "1134-1145"


def test_joint_objective_and_token_weights_are_independent_overrides():
    directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name="sft_joint",
            overrides=[
                "model.loss_weights.future_traj=2.0",
                "model.loss_weights.others=0.5",
                "model.lm_token_loss_weights={future_traj:1.0,text:0.2}",
            ],
        )
    assert OmegaConf.to_container(config.model.loss_weights) == {"future_traj": 2.0, "others": 0.5}
    assert OmegaConf.to_container(config.model.lm_token_loss_weights) == {"future_traj": 1.0, "text": 0.2}


@pytest.mark.parametrize("filename", ["zero3.json", "zero3_optimizer_offload.json"])
def test_zero3_preserves_fp32_accumulation_and_standard_linear_arithmetic(filename):
    path = Path(__file__).resolve().parents[1] / "configs" / "deepspeed" / filename
    config = json.loads(path.read_text())
    assert config["zero_optimization"]["stage"] == 3
    assert config["zero_optimization"]["memory_efficient_linear"] is False
    assert config["data_types"]["grad_accum_dtype"] == "fp32"
    assert config["communication_data_type"] == "fp32"


@pytest.mark.parametrize("dataset", ["once", "tartan"])
@pytest.mark.parametrize("evaluation", [False, True])
def test_driving_configs_replace_pai_dataset_contract(dataset, evaluation):
    """An OOD override must not forward inherited PAI-only kwargs to its loader."""
    directory = Path(__file__).resolve().parents[1] / "configs"
    name = f"sft_eval_trajectory_{dataset}" if evaluation else f"sft_joint_{dataset}"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name=name,
            overrides=[
                "driving.data_root=/tmp/driving",
                "driving.split_manifest=/tmp/split.json",
                "seed=123",
                "trainer.data_seed=789",
            ],
        )
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    class_name = "ONCEDataset" if dataset == "once" else "TartanDriveDataset"
    forbidden = {
        "local_dir", "chunk_ids", "camera_features", "front_frame_offsets",
        "include_calibration", "use_default_keyframe", "random_keyframes", "annotations_path",
    }
    for split, loader in [("train", config.data.train_dataset), ("val", config.data.val_dataset)]:
        assert loader._target_ == f"alpamayo2edge_sft.driving_datasets.{class_name}"
        assert not forbidden.intersection(loader)
        assert loader.data_root == "/tmp/driving"
        assert loader.split_manifest == "/tmp/split.json"
        assert loader.split == split
        assert loader.data_seed == (789 if split == "train" else 42)
        assert loader.num_history_steps is None
        assert loader.num_future_steps == 64
        assert loader.time_step == 0.1
        assert loader.max_samples is None
        assert list(loader.frame_offsets) == [-2.0, -1.0, -0.5, -0.1, 0.0]
        assert list(loader.camera_pixel_bounds) == [1]
        assert list(loader.preprocess.components_order) == ["image", "traj_history", "prompt", "traj_future"]
        assert loader.preprocess.generation_mode is (evaluation and split == "val")
    assert config.trainer.seed == 123
    assert config.trainer.max_steps == -1
    assert config.trainer.num_train_epochs == 3
    assert config.trainer.learning_rate == pytest.approx(3.5e-6)
    assert config.trainer.eval_on_start is True
    assert config.trainer.eval_strategy == config.trainer.save_strategy == "epoch"
    assert config.model.training_stage == ("expert" if evaluation else "joint")
    assert config.evaluate.mode == ("trajectory" if evaluation else "loss")
    assert config.evaluate.max_eval_steps == -1
    if evaluation:
        assert config.trainer.deepspeed is None
        assert config.evaluate.num_traj_samples == 6
    else:
        assert config.trainer.deepspeed == "configs/deepspeed/zero3.json"


@pytest.mark.parametrize("name", ["sft_joint_once", "sft_joint_tartan"])
@pytest.mark.parametrize("missing", ["data_root", "split_manifest"])
def test_driving_configs_require_explicit_data_and_split(name, missing):
    from omegaconf.errors import InterpolationToMissingValueError

    directory = Path(__file__).resolve().parents[1] / "configs"
    provided = "split_manifest" if missing == "data_root" else "data_root"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(config_name=name, overrides=[f"driving.{provided}=/tmp/provided"])
    with pytest.raises(InterpolationToMissingValueError, match=f"driving.{missing}"):
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


@pytest.mark.parametrize("dataset", ["once", "tartan"])
def test_driving_eval_sampling_seed_does_not_change_dataset_protocol(dataset):
    """Repeated trajectory samples must retain an identical validation protocol."""
    directory = Path(__file__).resolve().parents[1] / "configs"
    configs = []
    for seed in (42, 43):
        with initialize_config_dir(version_base=None, config_dir=str(directory)):
            config = compose(
                config_name=f"sft_eval_trajectory_{dataset}",
                overrides=[
                    "driving.data_root=/tmp/driving",
                    "driving.split_manifest=/tmp/split.json",
                    f"seed={seed}",
                ],
            )
        configs.append(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    first, second = configs
    assert first["seed"] == 42 and second["seed"] == 43
    assert first["data"]["train_dataset"]["data_seed"] == 42
    assert second["data"]["train_dataset"]["data_seed"] == 43
    assert first["data"]["val_dataset"] == second["data"]["val_dataset"]
    assert first["data"]["collator"] == second["data"]["collator"]
    assert first["data"]["val_dataset"]["data_seed"] == 42


@pytest.mark.parametrize("task", ["nav", "cot", "nav_cot"])
@pytest.mark.parametrize("evaluation", [False, True])
def test_documented_annotation_overrides_compose_without_add_prefix(task, evaluation):
    directory = Path(__file__).resolve().parents[1] / "configs"
    name = f"sft_eval_trajectory_{task}" if evaluation else f"sft_joint_{task}"
    splits = ("val",) if evaluation else ("train", "val")
    overrides = [
        "model.pretrained_model_name_or_path=/tmp/checkpoint",
        "data.train_dataset.local_dir=/tmp/pai",
        "data.val_dataset.local_dir=/tmp/pai",
        "paths.output_dir=/tmp/annotation-run",
        "trainer.report_to=wandb",
        "trainer.run_name=annotation-run",
    ]
    for split in splits:
        overrides += [
            f"data.{split}_dataset.annotations_path=/tmp/{split}.json",
            f"data.{split}_dataset.validate_navigation_format=false",
            f"data.{split}_dataset.clip_index_metadata=/tmp/{split}.parquet",
        ]
    overrides += (["evaluate.eval_ckpt=/tmp/checkpoint", "evaluate.max_eval_steps=-1", "seed=43"]
                  if evaluation else ["trainer.dataloader_num_workers=12",
                                      "+trainer.dataloader_persistent_workers=true",
                                      "+trainer.dataloader_prefetch_factor=2"])
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(config_name=name, overrides=overrides)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for split in splits:
        dataset = config.data[f"{split}_dataset"]
        assert dataset.validate_navigation_format is False
        assert dataset.clip_index_metadata == f"/tmp/{split}.parquet"
        assert dataset.annotations_path == f"/tmp/{split}.json"


@pytest.mark.parametrize("evaluation", [False, True])
def test_tartan_camera_timing_is_required_by_default_and_legacy_overrides_reach_both_splits(evaluation):
    directory = Path(__file__).resolve().parents[1] / "configs"
    name = "sft_eval_trajectory_tartan" if evaluation else "sft_joint_tartan"
    common = ["driving.data_root=/tmp/tartan", "driving.split_manifest=/tmp/split.json"]
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        default = compose(config_name=name, overrides=common)
        legacy = compose(config_name=name, overrides=[
            *common,
            "driving.allow_unverified_image_timestamps=true",
            "driving.legacy_image_timestamp_evidence='My export/version; image N assumed to match TartanVO row N; unverified'",
        ])
    for config in (default, legacy):
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for split in ("train_dataset", "val_dataset"):
        assert default.data[split].allow_unverified_image_timestamps is False
        assert default.data[split].legacy_image_timestamp_evidence is None
        assert legacy.data[split].allow_unverified_image_timestamps is True
        assert legacy.data[split].legacy_image_timestamp_evidence == (
            "My export/version; image N assumed to match TartanVO row N; unverified"
        )


@pytest.mark.parametrize("dataset", ["once", "tartan"])
@pytest.mark.parametrize("evaluation", [False, True])
def test_driving_split_audit_policy_is_shared_by_training_and_evaluation(dataset, evaluation):
    directory = Path(__file__).resolve().parents[1] / "configs"
    name = f"sft_eval_trajectory_{dataset}" if evaluation else f"sft_joint_{dataset}"
    common = ["driving.data_root=/tmp/driving", "driving.split_manifest=/tmp/split.json"]
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        default = compose(config_name=name, overrides=common)
        custom = compose(config_name=name, overrides=[
            *common,
            "driving.split_audit.temporal_only=false",
            "driving.split_audit.min_time_gap_seconds=172800",
            "driving.split_audit.min_location_gap_m=500",
        ])
    for config in (default, custom):
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    for split in ("train_dataset", "val_dataset"):
        assert OmegaConf.to_container(default.data[split].split_audit, resolve=True) == {
            "temporal_only": True, "min_time_gap_seconds": 86400, "min_location_gap_m": 250,
        }
        assert OmegaConf.to_container(custom.data[split].split_audit, resolve=True) == {
            "temporal_only": False, "min_time_gap_seconds": 172800, "min_location_gap_m": 500,
        }
