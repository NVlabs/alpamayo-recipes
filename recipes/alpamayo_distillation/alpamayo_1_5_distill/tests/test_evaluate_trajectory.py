# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from importlib.resources import files

import pandas as pd
import pytest
import torch

import alpamayo1_5_distill.evaluate_trajectory as evaluate_trajectory
from alpamayo1_5_distill.evaluate_trajectory import (
    FIXED_T0_US,
    GOLD_SET_SIZE,
    ExactShardSampler,
    EvalSample,
    PhysicalAIEvalDataset,
    _gold_samples,
    _model_inputs,
    _nav_eligible_samples,
    _resolve_navigation_cohorts,
    _resolve_navigation_t0_mode,
    trajectory_minade,
)


def test_trajectory_minade_matches_training_validation() -> None:
    prediction = torch.tensor(
        [[[[[1.0, 0.0, 8.0], [3.0, 0.0, 9.0]], [[0.0, 2.0, 7.0], [0.0, 4.0, 6.0]]]]]
    )
    ground_truth = torch.zeros((1, 1, 2, 3))

    expected = float(evaluate_trajectory.minade_values(prediction, ground_truth).item())
    assert trajectory_minade(prediction, ground_truth) == expected


def test_model_inputs_reuse_training_builder_and_disable_navigation(monkeypatch) -> None:
    captured = {}

    def fake_builder(model, samples, *, processor):
        captured.update(model=model, samples=samples, processor=processor)
        return {"built": True}

    monkeypatch.setattr(
        evaluate_trajectory,
        "build_student_inference_batch",
        fake_builder,
    )
    model = object()
    processor = object()
    sample = {"clip_id": "clip", "nav_enabled": True, "nav_text": "turn left"}

    assert _model_inputs(model, processor, sample) == {"built": True}
    assert captured["model"] is model
    assert captured["processor"] is processor
    assert captured["samples"][0]["nav_enabled"] is False


def test_model_inputs_preserve_navigation_when_requested(monkeypatch) -> None:
    captured = {}

    def fake_builder(model, samples, *, processor):
        captured.update(model=model, samples=samples, processor=processor)
        return {"built": True}

    monkeypatch.setattr(
        evaluate_trajectory,
        "build_student_inference_batch",
        fake_builder,
    )
    sample = {
        "clip_id": "clip",
        "nav_enabled": True,
        "nav_text": "Turn left in 40 meters",
    }

    assert _model_inputs(
        object(),
        object(),
        sample,
        use_navigation=True,
    ) == {"built": True}
    assert captured["samples"][0]["nav_enabled"] is True
    assert captured["samples"][0]["nav_text"] == sample["nav_text"]


def test_nav_eligible_samples_require_sidecar_and_fixed_timestamp() -> None:
    class FakeNavigationStore:
        @staticmethod
        def available_clip_ids():
            return {"a", "b"}

        @staticmethod
        def valid_t0_us(clip_id):
            return (FIXED_T0_US,) if clip_id.removeprefix("clipgt-") == "a" else (0,)

    samples = [
        EvalSample(index=0, clip_id="clipgt-a"),
        EvalSample(index=1, clip_id="b"),
        EvalSample(index=2, clip_id="c"),
    ]

    assert _nav_eligible_samples(samples, FakeNavigationStore()) == samples[:1]


def test_nav_eligible_samples_choose_closest_timestamp_with_earlier_tie() -> None:
    class FakeNavigationStore:
        @staticmethod
        def available_clip_ids():
            return {"a", "b"}

        @staticmethod
        def valid_t0_us(clip_id):
            if clip_id == "a":
                return (4_900_000, 5_300_000)
            return (5_000_000, 5_200_000)

    samples = [
        EvalSample(index=0, clip_id="a"),
        EvalSample(index=1, clip_id="b"),
    ]

    selected = _nav_eligible_samples(
        samples,
        FakeNavigationStore(),
        t0_mode="closest",
    )

    assert [sample.t0_us for sample in selected] == [4_900_000, 5_000_000]


def test_eval_dataset_keeps_base_t0_fixed_and_loads_selected_nav_t0(
    monkeypatch,
    tmp_path,
) -> None:
    loaded_timestamps = []

    def fake_load(*, clip_id, t0_us, avdi, maybe_stream):
        del avdi, maybe_stream
        loaded_timestamps.append(t0_us)
        return {"clip_id": clip_id, "t0_us": t0_us}

    class FakeNavigationStore:
        enabled = True

        @staticmethod
        def attach(sample):
            return {**sample, "nav_enabled": True}

    monkeypatch.setattr(evaluate_trajectory, "load_physical_aiavdataset", fake_load)
    monkeypatch.setattr(
        evaluate_trajectory,
        "PhysicalAIAVDatasetInterfaceLocal",
        lambda **kwargs: object(),
    )
    dataset = PhysicalAIEvalDataset(
        [EvalSample(index=0, clip_id="clip-a")],
        str(tmp_path),
        nav_samples={0: EvalSample(index=0, clip_id="clip-a", t0_us=4_900_000)},
    )
    dataset.nav_store = FakeNavigationStore()

    item = dataset[0]

    assert loaded_timestamps == [FIXED_T0_US, 4_900_000]
    assert item["sample"]["t0_us"] == FIXED_T0_US
    assert item["nav_sample"]["t0_us"] == 4_900_000
    assert item["nav_sample"]["nav_enabled"] is True


def test_navigation_cohort_defaults_keep_644_strict_and_enable_mixed_90k() -> None:
    assert _resolve_navigation_cohorts("644", None, None) == ("gold644", None)
    assert _resolve_navigation_cohorts("90k", None, None) == (
        "eval",
        "checkpoint_val_5k,opd_unseen_val_70k",
    )
    assert _resolve_navigation_cohorts("90k", "custom", None) == ("custom", None)
    assert _resolve_navigation_t0_mode("644", None) == "exact"
    assert _resolve_navigation_t0_mode("90k", None) == "closest"
    assert _resolve_navigation_t0_mode("90k", "exact") == "exact"


def test_sampling_uses_training_missing_marker_fallback(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def sample_trajectories_from_data_with_vlm_rollout(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 1, 6, 2, 3)), None, None

    monkeypatch.setattr(evaluate_trajectory, "_reset_inference_seed", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())

    output = evaluate_trajectory._sample_predictions(
        FakeModel(),
        {"tokenized_data": {}},
        greedy=False,
        max_generation_length=512,
    )

    assert output.shape == (1, 1, 6, 2, 3)
    assert captured["allow_missing_future_start"] is True
    assert captured["max_generation_length"] == 512
    assert captured["num_traj_samples"] == 6


@pytest.fixture
def gold_metadb_file(tmp_path, monkeypatch):
    clip_ids = (
        files("alpamayo1_5_distill")
        .joinpath("assets/clip_ids_eval_gold.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    metadata = tmp_path / "gold.parquet"
    pd.DataFrame({"key": clip_ids}).to_parquet(metadata, index=False)
    monkeypatch.setattr(evaluate_trajectory, "GOLD_METADB", metadata)
    return metadata, clip_ids


def test_gold_samples_are_all_fixed_at_5_1_seconds(gold_metadb_file) -> None:
    _, clip_ids = gold_metadb_file
    samples = _gold_samples()
    assert len(samples) == GOLD_SET_SIZE
    assert len({sample.clip_id for sample in samples}) == GOLD_SET_SIZE
    assert [sample.clip_id for sample in samples] == clip_ids
    assert {sample.t0_us for sample in samples} == {FIXED_T0_US}


def test_gold_samples_explain_how_to_download_missing_metadata(tmp_path, monkeypatch) -> None:
    metadata = tmp_path / "missing.parquet"
    monkeypatch.setattr(evaluate_trajectory, "GOLD_METADB", metadata)

    with pytest.raises(FileNotFoundError) as error:
        _gold_samples()

    assert str(metadata) in str(error.value)
    assert evaluate_trajectory.GOLD_METADB_URL in str(error.value)
    assert "Download" in str(error.value)


@pytest.mark.parametrize("invalid_keys", ["duplicate", "missing", "extra"])
def test_gold_samples_reject_invalid_metadata(gold_metadb_file, invalid_keys) -> None:
    metadata, clip_ids = gold_metadb_file
    if invalid_keys == "duplicate":
        clip_ids[-1] = clip_ids[0]
    elif invalid_keys == "missing":
        clip_ids.pop()
    else:
        clip_ids.append("extra-clip")
    pd.DataFrame({"key": clip_ids}).to_parquet(metadata, index=False)

    with pytest.raises(ValueError, match="gold metadb must contain 644 unique clips"):
        _gold_samples()


@pytest.mark.parametrize("world_size", [1, 2, 8, 64, 128])
def test_exact_shards_cover_644_without_padding(world_size: int) -> None:
    shards = [
        list(ExactShardSampler(GOLD_SET_SIZE, rank, world_size))
        for rank in range(world_size)
    ]
    flattened = [index for shard in shards for index in shard]
    assert len(flattened) == GOLD_SET_SIZE
    assert sorted(flattened) == list(range(GOLD_SET_SIZE))
