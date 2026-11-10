# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check policy configuration; actual FSDP materialization requires GPU validation."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer

from alpamayo2edge_sft.trainer import Alpamayo2EdgeTrainer


def _trainer_shell(monkeypatch, fsdp_version, bf16=True):
    trainer = object.__new__(Alpamayo2EdgeTrainer)
    trainer.args = SimpleNamespace(
        bf16=bf16, fp16=False,
        accelerator_config=SimpleNamespace(gradient_accumulation_kwargs={"adjust_scheduler": False}),
    )

    def initialize(self):
        self.is_fsdp_enabled = fsdp_version is not None
        self.accelerator = SimpleNamespace(
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(
                fsdp_version=fsdp_version, mixed_precision_policy="original"
            )),
            gradient_state=SimpleNamespace(plugin_kwargs={"adjust_scheduler": False}),
        )

    monkeypatch.setattr(Trainer, "create_accelerator_and_postprocess", initialize)
    return trainer


def test_fsdp2_policy_preserves_input_precision_and_reduces_every_microbatch(monkeypatch):
    trainer = _trainer_shell(monkeypatch, 2)
    trainer.create_accelerator_and_postprocess()
    policy = trainer.accelerator.state.fsdp_plugin.mixed_precision_policy
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype is None
    assert policy.cast_forward_inputs is False
    assert trainer.accelerator.gradient_state.plugin_kwargs == {
        "adjust_scheduler": False, "sync_each_batch": True
    }
    assert trainer.args.accelerator_config.gradient_accumulation_kwargs == {
        "adjust_scheduler": False, "sync_each_batch": True
    }


@pytest.mark.parametrize("fsdp_version", [None, 1])
def test_non_fsdp2_precision_is_unchanged(monkeypatch, fsdp_version):
    trainer = _trainer_shell(monkeypatch, fsdp_version, bf16=False)
    trainer.create_accelerator_and_postprocess()
    assert trainer.accelerator.state.fsdp_plugin.mixed_precision_policy == "original"
    assert trainer.accelerator.gradient_state.plugin_kwargs == {"adjust_scheduler": False}


def test_fsdp2_rejects_conflicting_compute_precision(monkeypatch):
    trainer = _trainer_shell(monkeypatch, 2, bf16=False)
    with pytest.raises(ValueError, match="bf16=true"):
        trainer.create_accelerator_and_postprocess()
