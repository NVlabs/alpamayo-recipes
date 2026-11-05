from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import alpamayo1_distill.eval_student as eval_student
from alpamayo1_distill.eval_student import (
    _minade_values,
    _seed_for_sample,
    _validate_loading_info,
    parse_args,
)
from alpamayo1_distill.eval_teacher import parse_args as parse_teacher_args


def test_minade_selects_best_candidate_for_each_sample() -> None:
    predictions = torch.tensor(
        [
            [
                [[1.0, 0.0, 7.0], [1.0, 0.0, 7.0]],
                [[0.0, 0.0, 9.0], [0.0, 0.0, 9.0]],
            ],
            [
                [[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
                [[6.0, 8.0, 0.0], [6.0, 8.0, 0.0]],
            ],
        ]
    )
    ground_truth = torch.zeros(2, 2, 2)

    assert _minade_values(predictions, ground_truth) == [0.0, 5.0]


def test_reference_seed_protocol_is_fixed_per_mode() -> None:
    assert _seed_for_sample(
        42, clip_id="clip-a", t0_us=5_100_000, greedy=False, strategy="reference"
    ) == 42
    assert _seed_for_sample(
        42, clip_id="clip-b", t0_us=7_000_000, greedy=False, strategy="reference"
    ) == 42
    assert _seed_for_sample(
        42, clip_id="clip-a", t0_us=5_100_000, greedy=True, strategy="reference"
    ) == 43


def test_evaluation_cli_defaults_match_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_student", "--ckpt_dir", "/checkpoint"])

    args = parse_args()

    assert args.eval_set == "gold"
    assert args.t0_us == 5_100_000
    assert args.seed == 42
    assert args.seed_strategy == "reference"
    assert args.num_traj_samples == 6
    assert args.top_p == 0.98
    assert args.temperature == 0.6
    for removed_option in (
        "small_dataset",
        "clip_ids_file",
        "limit",
        "split",
        "full_eval",
    ):
        assert not hasattr(args, removed_option)


def test_full_evaluation_is_the_only_alternative_to_gold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_student", "--ckpt_dir", "/checkpoint", "--eval_set", "full"],
    )

    assert parse_args().eval_set == "full"


def test_teacher_evaluation_exposes_only_gold_and_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_teacher", "--local_dir", "/dataset"],
    )
    args = parse_teacher_args()
    assert args.eval_set == "gold"
    assert not hasattr(args, "small_dataset")
    assert not hasattr(args, "split")

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_teacher", "--local_dir", "/dataset", "--eval_set", "full"],
    )
    assert parse_teacher_args().eval_set == "full"


def test_evaluation_clis_reject_removed_or_custom_dataset_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_student", "--ckpt_dir", "/checkpoint", "--small_dataset"],
    )
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_student", "--ckpt_dir", "/checkpoint", "--eval_set", "custom"],
    )
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_teacher", "--local_dir", "/dataset", "--small_dataset"],
    )
    with pytest.raises(SystemExit):
        parse_teacher_args()


def test_checkpoint_loading_contract_allows_only_frozen_teacher_extras() -> None:
    assert _validate_loading_info(
        {
            "missing_keys": [],
            "unexpected_keys": ["teacher_model.layer.weight", "hidden_proj.weight"],
            "mismatched_keys": [],
            "error_msgs": [],
        }
    ) == 2

    with pytest.raises(RuntimeError, match="did not load exactly"):
        _validate_loading_info(
            {
                "missing_keys": ["expert.layer.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )


@pytest.fixture
def gold_metadata_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, list[str]]:
    metadata_path = tmp_path / "1005_7cam_gold_eval_metadb_public.parquet"
    monkeypatch.setattr(eval_student, "DEFAULT_GOLD_PARQUET", metadata_path)
    clip_ids = [f"clip-{index}" for index in range(eval_student.PAI_GOLD_EVAL_SIZE)]
    return metadata_path, clip_ids


def test_missing_gold_metadata_explains_manual_download(gold_metadata_inputs) -> None:
    metadata_path, clip_ids = gold_metadata_inputs

    with pytest.raises(FileNotFoundError) as error:
        eval_student._validate_gold_assets(clip_ids)

    message = str(error.value)
    assert "Download it from" in message
    assert (
        "https://github.com/NVlabs/alpamayo-recipes/raw/refs/heads/main/"
        "data/1005_7cam_gold_eval_metadb_public.parquet"
    ) in message
    assert str(metadata_path) in message


def test_downloaded_gold_metadata_accepts_same_cohort(gold_metadata_inputs) -> None:
    metadata_path, clip_ids = gold_metadata_inputs
    pd.DataFrame({"key": list(reversed(clip_ids))}).to_parquet(metadata_path, index=False)

    eval_student._validate_gold_assets(clip_ids)


def test_downloaded_gold_metadata_rejects_different_cohort(gold_metadata_inputs) -> None:
    metadata_path, clip_ids = gold_metadata_inputs
    metadata_ids = ["unexpected-clip", *clip_ids[1:]]
    pd.DataFrame({"key": metadata_ids}).to_parquet(metadata_path, index=False)

    with pytest.raises(ValueError, match="does not match") as error:
        eval_student._validate_gold_assets(clip_ids)

    assert "unexpected-clip" in str(error.value)
    assert clip_ids[0] in str(error.value)
