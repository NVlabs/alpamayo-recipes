# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional tasks must preserve text supervision and annotation-defined splits."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
VARIANTS = ("nav", "cot", "nav_cot")


def _compose(name, annotation_overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        return compose(config_name=name, overrides=[
            "data.train_dataset.local_dir=/tmp/pai",
            "data.val_dataset.local_dir=/tmp/pai",
            *annotation_overrides,
        ])


@pytest.mark.parametrize("variant", VARIANTS)
def test_optional_training_tasks_keep_annotations_and_supervision(variant):
    config = _compose(f"sft_joint_{variant}", [
        "data.train_dataset.annotations_path=/tmp/train.json",
        "data.val_dataset.annotations_path=/tmp/val.json",
    ])
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    nav, cot = "nav" in variant, "cot" in variant
    assert config.model.training_stage == "joint"
    assert config.model.pretrained_model_name_or_path == "nvidia/Alpamayo2-Edge"
    assert config.model.qk_gradient_layout is False
    for split in ("train_dataset", "val_dataset"):
        dataset = config.data[split]
        assert dataset._target_.endswith(".PhysicalAIAnnotatedDataset")
        # Baseline chunk splits must not silently drop annotated events.
        assert dataset.chunk_ids is None
        assert dataset.use_default_keyframe is False
        assert dataset.random_keyframes is False
        assert dataset.split is None
        assert dataset.require_navigation is nav
        assert dataset.require_cot is cot
        preprocess = dataset.preprocess
        assert preprocess.generation_mode is False  # Validation loss needs targets too.
        assert ("nav_instruction" in preprocess.components_order) is nav
        assert ("cot" in preprocess.components_order) is cot
        assert list(preprocess.label_components) == (["assistant"] if cot else ["traj_future"])
        assert list(preprocess.components_prompt) == (["cot", "traj_future"] if cot else ["traj_future"])
    assert config.data.train_dataset.annotations_path != config.data.val_dataset.annotations_path
    assert len(config.camera_features) == 6
    assert config.trainer.independent_dp_rng is True
    assert config.trainer.gradient_checkpointing is False
    assert config.trainer.learning_rate == pytest.approx(3.5e-6)
    assert config.trainer.lr_multiplier["vlm.model.projector"] == 0.1
    assert config.trainer.lr_multiplier.expert == 4.0
    assert "stage1_vlm_checkpoint_path" not in config.model


@pytest.mark.parametrize("variant", VARIANTS)
def test_generation_config_does_not_require_ground_truth_reasoning(variant):
    config = _compose(f"sft_eval_trajectory_{variant}", [
        "data.val_dataset.annotations_path=/tmp/val.json",
    ])
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    dataset = config.data.val_dataset
    assert dataset.chunk_ids is None
    assert dataset.require_navigation is ("nav" in variant)
    assert dataset.require_cot is False
    assert dataset.preprocess.generation_mode is True
    assert config.evaluate.generate_cot is ("cot" in variant)
    assert config.evaluate.mode == "trajectory"
    assert config.trainer.deepspeed is None
    assert config.model.training_stage == "expert"


@pytest.mark.parametrize("variant", VARIANTS)
def test_annotation_training_cannot_silently_reuse_train_as_validation(variant):
    config = _compose(f"sft_joint_{variant}", [
        "data.train_dataset.annotations_path=/tmp/train.json",
    ])
    with pytest.raises(MissingMandatoryValue, match="annotations_path"):
        OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("selected_index", [False, True])
def test_annotated_configs_load_default_and_selected_local_indexes(
    tmp_path, variant, evaluation, selected_index,
):
    """Composed configs must reach the real local reader with a usable index path."""
    import json

    import pandas as pd
    from hydra.utils import instantiate

    pd.DataFrame({
        "feature": ["egomotion"],
        "directory": ["labels"],
        "clip_files_in_zip": ["{}"],
    }).to_csv(tmp_path / "features.csv", index=False)
    index = pd.DataFrame({"chunk": [1, 2]}, index=["clip-a", "clip-b"])
    index.to_parquet(tmp_path / "clip_index.parquet")
    index.loc[["clip-b"]].to_parquet(tmp_path / "selected.parquet")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps([
        {"clip_id": clip, "t0_relative": 5_000_000,
         "nav_text": "Turn left in 25m", "cot": "Yield before turning."}
        for clip in index.index
    ]))
    overrides = [f"data.val_dataset.annotations_path={annotations}"]
    if selected_index:
        overrides.append("data.val_dataset.clip_index_metadata=selected.parquet")
    name = f"sft_eval_trajectory_{variant}" if evaluation else f"sft_joint_{variant}"
    config = _compose(name, overrides)
    dataset = instantiate(
        config.data.val_dataset,
        local_dir=str(tmp_path),
        preprocess=None,
        num_history_steps=16,
        _convert_="partial",
    )
    assert dataset.clip_ids == (["clip-b"] if selected_index else ["clip-a", "clip-b"])
