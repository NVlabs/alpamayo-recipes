from __future__ import annotations

import copy
import inspect
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import alpamayo1_distill.dataloader.physical_aiavdataset as pai_dataset_module
from alpamayo1_distill.defaults import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_COSMOS_STUDENT_MODEL,
    DEFAULT_GOLD_CLIP_IDS_PATH,
    PAI_GOLD_EVAL_SIZE,
)
from alpamayo1_distill.dataloader.physical_aiavdataset import physicalAiAvDataset
from alpamayo1_distill.models.alpamayo_r1_distill_2b import ExpertModelDistill
from alpamayo1_distill.train import (
    CosineWarmupScheduler,
    parse_args,
    resolve_student_config_source,
)
from alpamayo1_distill.training import combine_training_loss
from alpamayo1_distill.training_resume import capture_rng_state, restore_rng_state


def test_training_cli_exposes_only_live_teacher_data_path(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["alpamayo1_distill.train"])
    args = parse_args()

    for removed_option in (
        "cfg_dir",
        "offline_gt_dir",
        "reduce_dataset",
        "small_dataset",
        "use_cosmos_2b",
        "use_online_dataset",
        "use_precompute_dataset",
    ):
        assert not hasattr(args, removed_option)
    assert args.student_vlm_path == DEFAULT_COSMOS_STUDENT_MODEL


@pytest.mark.parametrize("removed_flag", ["--small_dataset", "--use_cosmos_2b", "--cfg_dir"])
def test_training_cli_rejects_removed_flags(monkeypatch, removed_flag: str) -> None:
    monkeypatch.setattr(sys, "argv", ["alpamayo1_distill.train", removed_flag])
    with pytest.raises(SystemExit):
        parse_args()


def test_student_config_source_uses_checkpoint_architecture_when_loading(
    tmp_path,
) -> None:
    resume_dir = tmp_path / "resume"
    init_dir = tmp_path / "init"
    resume_dir.mkdir()
    init_dir.mkdir()
    (resume_dir / "config.json").write_text("{}")
    (init_dir / "config.json").write_text("{}")

    assert resolve_student_config_source(None, None) == DEFAULT_CONFIG_PATH
    assert resolve_student_config_source(str(resume_dir), None) == resume_dir.resolve()
    assert resolve_student_config_source(None, str(init_dir)) == init_dir.resolve()
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_student_config_source(str(resume_dir), str(init_dir))
    with pytest.raises(FileNotFoundError, match="config.json"):
        resolve_student_config_source(str(tmp_path / "missing"), None)


def test_distillation_model_has_no_precomputed_target_inputs() -> None:
    constructor_parameters = inspect.signature(ExpertModelDistill.__init__).parameters
    forward_parameters = inspect.signature(ExpertModelDistill.forward).parameters

    assert "use_precompute_dataset" not in constructor_parameters
    assert "seq_input" not in forward_parameters
    assert "teacher_logits" not in forward_parameters


def test_dataset_selector_has_no_debug_subset_interface() -> None:
    parameters = inspect.signature(physicalAiAvDataset.__init__).parameters

    assert "small_dataset" not in parameters
    assert "clip_ids_file" not in parameters
    assert "gold_eval" in parameters
    assert "exclude_gold_eval" not in parameters


def test_dataset_exposes_only_canonical_train_gold_and_full_val_cohorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gold_ids = DEFAULT_GOLD_CLIP_IDS_PATH.read_text().splitlines()
    assert len(gold_ids) == PAI_GOLD_EVAL_SIZE
    held_out_id = gold_ids[0]
    avdi = SimpleNamespace(
        clip_index=pd.DataFrame(
            {"split": ["train"] * len(gold_ids) + ["train", "val"]},
            index=gold_ids + ["train-only", "val-only"],
        )
    )
    monkeypatch.setattr(pai_dataset_module, "PAI_TRAIN_SPLIT_SIZE", len(gold_ids) + 1)
    monkeypatch.setattr(pai_dataset_module, "PAI_GOLD_TRAIN_OVERLAP", len(gold_ids))
    monkeypatch.setattr(pai_dataset_module, "PAI_TRAIN_SIZE", 1)
    monkeypatch.setattr(pai_dataset_module, "PAI_FULL_VAL_SIZE", 1)

    train = physicalAiAvDataset(
        split="train",
        avdi=avdi,
    )
    full_val = physicalAiAvDataset(split="val", avdi=avdi)
    gold_val = physicalAiAvDataset(split="val", avdi=avdi, gold_eval=True)

    assert train.clip_ids == ["train-only"]
    assert full_val.clip_ids == ["val-only"]
    assert gold_val.clip_ids == gold_ids
    with pytest.raises(ValueError, match="only valid for split='val'"):
        physicalAiAvDataset(split="train", avdi=avdi, gold_eval=True)
    with pytest.raises(ValueError, match="split must be 'train' or 'val'"):
        physicalAiAvDataset(split="custom", avdi=avdi)


def test_dataset_rejects_partial_pai_train_and_val_indexes() -> None:
    gold_ids = DEFAULT_GOLD_CLIP_IDS_PATH.read_text().splitlines()
    partial_train = SimpleNamespace(
        clip_index=pd.DataFrame(
            {"split": ["train"] * len(gold_ids)},
            index=gold_ids,
        )
    )
    partial_val = SimpleNamespace(
        clip_index=pd.DataFrame(
            {"split": ["val"] * len(gold_ids)},
            index=gold_ids,
        )
    )

    with pytest.raises(ValueError, match="train split has the wrong cardinality"):
        physicalAiAvDataset(split="train", avdi=partial_train)
    with pytest.raises(ValueError, match="val split has the wrong cardinality"):
        physicalAiAvDataset(split="val", avdi=partial_val)


def test_distillation_forward_requires_live_teacher() -> None:
    model = ExpertModelDistill.__new__(ExpertModelDistill)
    torch.nn.Module.__init__(model)
    model.teacher_model = None

    with pytest.raises(RuntimeError, match="requires a teacher model"):
        model.forward(tokenized_data={})


def test_reference_loss_weights_drive_expected_backward_gradients() -> None:
    names = (
        "future_traj_loss",
        "future_traj_vlm_loss",
        "distill_loss",
        "vlm_loss",
        "on_policy_kl",
        "hidden_distill_loss",
    )
    values = {name: torch.tensor(1.0, requires_grad=True) for name in names}
    fallback = torch.tensor(-1.0, requires_grad=True)

    loss = combine_training_loss(
        fallback,
        values,
        future_traj_weight=2.0,
        future_traj_vlm_weight=1.0,
        distill_weight=1.0,
        vlm_loss_weight=1.0,
        on_policy_kl_weight=1.0,
        hidden_distill_weight=0.0001,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(6.0001), rtol=0.0, atol=1e-6)
    expected = {
        "future_traj_loss": 2.0,
        "future_traj_vlm_loss": 1.0,
        "distill_loss": 1.0,
        "vlm_loss": 1.0,
        "on_policy_kl": 1.0,
        "hidden_distill_loss": 0.0001,
    }
    for name, expected_gradient in expected.items():
        assert values[name].grad is not None
        torch.testing.assert_close(
            values[name].grad,
            torch.tensor(expected_gradient),
            rtol=0.0,
            atol=1e-8,
        )
    assert fallback.grad is None


def test_default_loss_contract_returns_model_aggregate_unchanged() -> None:
    fallback = torch.tensor(3.0, requires_grad=True)
    result = combine_training_loss(
        fallback,
        {},
        future_traj_weight=1.0,
        future_traj_vlm_weight=1.0,
        distill_weight=1.0,
        vlm_loss_weight=1.0,
        on_policy_kl_weight=1.0,
        hidden_distill_weight=0.0,
    )
    assert result is fallback


def _run_toy_step(model, optimizer, scheduler) -> torch.Tensor:
    features = torch.randn(2, 3)
    target = torch.randn(2, 1)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(features), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return loss.detach()


def _new_toy_training_state(initial_state):
    model = torch.nn.Linear(3, 1)
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.1e-4, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=1,
        total_steps=4,
        min_lr=1e-5,
        vlm_min_lr=1e-6,
        num_vlm_groups=0,
    )
    return model, optimizer, scheduler


def _assert_state_equal(left, right) -> None:
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_state_equal(left_item, right_item)
    else:
        assert left == right


def test_two_step_save_resume_matches_continuous_cpu_training() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    initial_model = torch.nn.Linear(3, 1).state_dict()
    initial_rng = capture_rng_state(torch.device("cpu"))

    continuous_model, continuous_optimizer, continuous_scheduler = _new_toy_training_state(
        initial_model
    )
    restore_rng_state(initial_rng, torch.device("cpu"))
    _run_toy_step(continuous_model, continuous_optimizer, continuous_scheduler)
    continuous_second_loss = _run_toy_step(
        continuous_model, continuous_optimizer, continuous_scheduler
    )

    resumed_model, resumed_optimizer, resumed_scheduler = _new_toy_training_state(initial_model)
    restore_rng_state(initial_rng, torch.device("cpu"))
    _run_toy_step(resumed_model, resumed_optimizer, resumed_scheduler)
    checkpoint = {
        "model": copy.deepcopy(resumed_model.state_dict()),
        "optimizer": copy.deepcopy(resumed_optimizer.state_dict()),
        "scheduler": copy.deepcopy(resumed_scheduler.state_dict()),
        "rng": capture_rng_state(torch.device("cpu")),
    }

    reloaded_model, reloaded_optimizer, reloaded_scheduler = _new_toy_training_state(initial_model)
    reloaded_model.load_state_dict(checkpoint["model"])
    reloaded_optimizer.load_state_dict(checkpoint["optimizer"])
    reloaded_scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng_state(checkpoint["rng"], torch.device("cpu"))
    resumed_second_loss = _run_toy_step(
        reloaded_model, reloaded_optimizer, reloaded_scheduler
    )

    assert torch.equal(continuous_second_loss, resumed_second_loss)
    for continuous, resumed in zip(continuous_model.parameters(), reloaded_model.parameters()):
        assert torch.equal(continuous, resumed)
    _assert_state_equal(continuous_optimizer.state_dict(), reloaded_optimizer.state_dict())
    _assert_state_equal(continuous_scheduler.state_dict(), reloaded_scheduler.state_dict())
