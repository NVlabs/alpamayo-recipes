# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep live HF5 configuration objects out of Hydra's dataclass conversion."""

from types import SimpleNamespace

import pytest
from alpamayo2edge_sft import dataset as dataset_module
from alpamayo2edge_sft import evaluate_hf as evaluate_module
from alpamayo2edge_sft import processor as processor_module
from alpamayo2edge_sft import train_hf as train_module
from omegaconf import OmegaConf
from transformers import Cosmos3EdgeConfig


@pytest.mark.parametrize("entrypoint", ["train", "evaluate"])
def test_native_hf5_config_survives_dataset_preprocess_and_collator_construction(
    monkeypatch, tmp_path, entrypoint
):
    # HF5's real Cosmos3-Edge config is a dataclass. Passing it as an instantiate kwarg
    # makes OmegaConf inspect HF annotations (including unresolved torch names).
    model_config = Cosmos3EdgeConfig()
    model_config.tokens_per_history_traj = 45
    model = SimpleNamespace(config=model_config, training_stage="vlm")
    monkeypatch.setattr(train_module, "regression_model_factory", lambda **kwargs: model, raising=False)
    monkeypatch.setattr(
        dataset_module,
        "_create_public_dataset_interface",
        lambda **kwargs: SimpleNamespace(get_all_clip_ids=lambda: ["clip"], reasoning_db=None),
    )

    class NativeProcessorBoundary:
        def __init__(self, model_config):
            self.model_config = model_config
            self.processor = SimpleNamespace()

        def preprocess(self, data, **kwargs):
            return self.model_config

    monkeypatch.setattr(processor_module, "Alpamayo2EdgeProcessor", NativeProcessorBoundary)
    observed = []

    class TrainerBoundary:
        def __init__(self, **kwargs):
            self.inputs = kwargs
            self.args = kwargs["args"]
            self.saved_metrics = None
            self.saved_state = False
            observed.append(self)

        def is_world_process_zero(self):
            return True

        def train(self, resume_from_checkpoint=None):
            assert resume_from_checkpoint is None
            return SimpleNamespace(metrics={"train_loss": 0.5, "train_runtime": 2.0})

        def save_metrics(self, split, metrics):
            self.saved_metrics = (split, metrics)

        def save_state(self):
            self.saved_state = True

    entry_module = train_module if entrypoint == "train" else evaluate_module
    monkeypatch.setattr(entry_module, "Alpamayo2EdgeTrainer", TrainerBoundary)
    monkeypatch.setattr(
        entry_module,
        "training_arguments",
        lambda config: SimpleNamespace(
            deepspeed=None,
            resume_from_checkpoint=None,
            report_to=[],
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            world_size=1,
            max_steps=-1,
            num_train_epochs=3.0,
        ),
    )
    # Dataset parsing is mocked, but evaluation hashes the real metadata bytes.
    (tmp_path / "clip_index.parquet").write_bytes(b"fixture clip index\n")
    dataset_config = {
        "_target_": "alpamayo2edge_sft.dataset.PhysicalAIDataset",
        "_recursive_": False,
        "local_dir": str(tmp_path),
        "preprocess": {
            "_target_": "alpamayo2edge_sft.processor.get_preprocess_data_fn_from_model_config",
            "components_order": ["image", "traj_history", "prompt", "traj_future"],
            "components_prompt": ["traj_future"],
            "label_components": ["traj_future"],
            "generation_mode": False,
        },
    }
    config = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "_target_": "alpamayo2edge_sft.train_hf.regression_model_factory",
                "pretrained_model_name_or_path": "checkpoint",
            },
            "trainer": {"output_dir": str(tmp_path)},
            "data": {
                "train_dataset": dataset_config,
                "val_dataset": dataset_config,
                "collator": {
                    "_target_": "alpamayo2edge_sft.processor.Alpamayo2EdgeCollator",
                    "max_length": 64,
                },
            },
            "callbacks": {},
            "paths": {"output_dir": str(tmp_path)},
            "evaluate": {"eval_ckpt": None, "max_eval_steps": -1, "mode": "loss"},
        }
    )

    if entrypoint == "train":
        train_module.train.__wrapped__(config)
        saved_config = OmegaConf.load(tmp_path / "config.yaml")
        assert saved_config.seed == 42
        assert "hydra" not in saved_config
    else:
        evaluate_module.build_trainer(config)

    assert len(observed) == 1
    trainer = observed[0]
    if entrypoint == "train":
        assert trainer.saved_metrics == ("train", {"train_loss": 0.5, "train_runtime": 2.0})
        assert trainer.saved_state
    dataset_keys = ["eval_dataset"]
    if entrypoint == "train":
        dataset_keys.append("train_dataset")
    for key in dataset_keys:
        dataset = trainer.inputs[key]
        assert dataset.num_history_steps == 16
        # Exercise the real nested preprocessing factory, rather than only
        # checking the config passed to the outer dataset constructor.
        assert dataset.preprocess(data={}) is model_config
    assert trainer.inputs["data_collator"].data_processor.model_config is model_config
    assert trainer.inputs["model"].config is model_config
