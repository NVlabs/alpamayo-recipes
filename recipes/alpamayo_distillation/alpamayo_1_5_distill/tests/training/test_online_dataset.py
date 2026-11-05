# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from alpamayo1_5_distill.data import physical_aiav_dataset
from alpamayo1_5_distill.data.physical_aiav_dataset import (
    InfeasibleClipError,
    OnlinePhysicalAIAVDataset,
    identity_collate,
)
from alpamayo1_5_distill.data.physical_aiav_loader import _sample_t0_from_range
from alpamayo1_5_distill.data.sample_index import SampleVisit
from torch.utils.data import DataLoader


@pytest.fixture
def fake_runtime(monkeypatch):
    runtime = SimpleNamespace(
        interfaces=[],
        loads=[],
        feasible_ranges={
            "clip-a": (2_000_000, 3_000_000),
            "clip-b": (8_000_000, 9_000_000),
        },
        clip_index=pd.DataFrame({"split": ["train", "train"]}, index=["clip-a", "clip-b"]),
    )

    class FakeInterface:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.clip_index = runtime.clip_index
            runtime.interfaces.append(self)

    monkeypatch.setitem(
        sys.modules,
        "physical_ai_av",
        SimpleNamespace(PhysicalAIAVDatasetInterface=FakeInterface),
    )
    monkeypatch.setitem(
        sys.modules,
        "alpamayo1_5_distill.data.dataset_local",
        SimpleNamespace(PhysicalAIAVDatasetInterfaceLocal=FakeInterface),
    )

    def fake_load(**kwargs):
        loaded = dict(kwargs)
        if loaded["t0_us"] is None:
            lower, upper = runtime.feasible_ranges[loaded["clip_id"]]
            loaded["t0_us"] = _sample_t0_from_range(
                lower, upper, loaded["t0_random_bits"], loaded.get("t0_quantum_us")
            )
        runtime.loads.append(loaded)
        return {
            "clip_id": loaded["clip_id"],
            "t0_us": loaded["t0_us"],
        }

    monkeypatch.setattr(
        physical_aiav_dataset,
        "load_physical_aiavdataset",
        fake_load,
    )
    return runtime


def _write_nav_clip(root, clip_id, *, frame_count=24):
    nav_dir = root / "train"
    nav_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "clip_id": clip_id,
            "cohort": "train",
            "hf_split": "train",
            "t0_relative_us": frame_index * 100_000,
            "lookup_in_route_timestamp_range": True,
            "has_future_instruction": True,
            "nav_command": "TURN_RIGHT",
            "nav_distance_m": 25.0,
            "nav_text": "Turn right in 25m",
        }
        for frame_index in range(frame_count)
    ]
    (nav_dir / f"{clip_id}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mixture_config(**overrides):
    config = {
        "enabled": True,
        "basic_ratio": 0.5,
        "nav_ratio": 0.5,
        "nav_t0_retries": 3,
        "nav_sample_retries": 3,
        "fallback_nav_to_basic": True,
    }
    config.update(overrides)
    return config


def _nav_config(root):
    return {
        "enabled": True,
        "root": str(root),
        "cohort": "train",
        "max_distance_m": -1,
    }


class _SpawnTestDataset(OnlinePhysicalAIAVDataset):
    """Mock sensor I/O inside spawned workers while exercising real sampling."""

    def _interface(self):
        return SimpleNamespace(
            clip_index=pd.DataFrame({"split": ["train", "train"]}, index=["clip-a", "clip-b"])
        )

    @staticmethod
    def _load(**kwargs):
        t0_us = kwargs["t0_us"]
        if t0_us is None:
            t0_us = _sample_t0_from_range(
                2_000_000, 3_000_000, kwargs["t0_random_bits"], kwargs.get("t0_quantum_us")
            )
        return {"clip_id": kwargs["clip_id"], "t0_us": t0_us}

    def __getitem__(self, index):
        with patch.object(physical_aiav_dataset, "load_physical_aiavdataset", self._load):
            return super().__getitem__(index)


def test_fixed_timestamps_length_loading_and_worker_interface(fake_runtime):
    fake_runtime.clip_index = pd.DataFrame(
        {"split": ["train", "val", "train", "train"]},
        index=["clip-a", "clip-val", "clip-b", None],
    )
    dataset = OnlinePhysicalAIAVDataset(
        split="train",
        samples_per_clip=3,
        t0_us_list=[2_000_000, 5_100_000],
        local_dir="/data/av",
        token="secret",
        loader_kwargs={"maybe_stream": False},
    )

    assert len(dataset) == 6
    assert len(fake_runtime.interfaces) == 1
    assert [dataset[index]["t0_us"] for index in range(3)] == [
        2_000_000,
        5_100_000,
        2_000_000,
    ]
    assert dataset[3]["clip_id"] == "clip-b"
    assert len(fake_runtime.interfaces) == 2
    assert fake_runtime.interfaces[0].kwargs == {
        "local_dir": "/data/av",
        "token": "secret",
    }
    assert fake_runtime.loads[0]["avdi"] is fake_runtime.interfaces[1]
    assert fake_runtime.loads[0]["maybe_stream"] is False


def test_random_timestamps_are_deterministic_per_epoch_and_slot(fake_runtime):
    first = OnlinePhysicalAIAVDataset(
        samples_per_clip=4,
        random_t0=True,
        margin_us=100_000,
        seed=123,
    )
    second = OnlinePhysicalAIAVDataset(
        samples_per_clip=4,
        random_t0=True,
        margin_us=100_000,
        seed=123,
    )

    epoch_zero = [first[index]["t0_us"] for index in range(len(first))]
    assert epoch_zero == [second[index]["t0_us"] for index in range(len(second))]
    assert all(2_000_000 <= timestamp <= 3_000_000 for timestamp in epoch_zero[:4])
    assert all(8_000_000 <= timestamp <= 9_000_000 for timestamp in epoch_zero[4:])
    assert all(
        load["t0_margin_us"] == 100_000 and load["t0_random_bits"] is not None
        for load in fake_runtime.loads
    )

    first.set_epoch(7)
    epoch_seven = [first[index]["t0_us"] for index in range(len(first))]
    assert epoch_seven != epoch_zero
    first.set_epoch(0)
    assert [first[index]["t0_us"] for index in range(len(first))] == epoch_zero


def test_random_timestamps_refresh_per_visit_and_replay_in_any_order(fake_runtime):
    dataset = OnlinePhysicalAIAVDataset(random_t0=True, seed=123)
    keys = [SampleVisit(0, visit_id) for visit_id in range(16)]
    samples = [dataset[key] for key in keys]
    assert {sample["clip_id"] for sample in samples} == {"clip-a"}
    assert len({sample["t0_us"] for sample in samples}) > 1
    assert all(2_000_000 <= sample["t0_us"] <= 3_000_000 for sample in samples)
    assert [dataset[key] for key in reversed(keys)] == list(reversed(samples))
    assert dataset[pickle.loads(pickle.dumps(keys[0]))] == samples[0]
    assert dataset[SampleVisit(-len(dataset), 0)] == samples[0]
    with pytest.raises(IndexError):
        dataset[SampleVisit(len(dataset), 0)]

    dataset.set_epoch(1)
    assert [dataset[key] for key in keys] != samples
    dataset.set_epoch(0)
    assert [dataset[key] for key in keys] == samples


def test_visit_keys_preserve_explicit_fixed_timestamps(fake_runtime):
    dataset = OnlinePhysicalAIAVDataset(random_t0=False, t0_us_list=[2_500_000])
    assert dataset[SampleVisit(0, 1)]["t0_us"] == 2_500_000
    assert dataset[SampleVisit(0, 2)]["t0_us"] == 2_500_000


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples_per_clip": 0}, "samples_per_clip"),
        ({"t0_us_list": []}, "t0_us_list"),
        ({"margin_us": -1}, "margin_us"),
    ],
)
def test_configuration_validation(fake_runtime, kwargs, message):
    with pytest.raises(ValueError, match=message):
        OnlinePhysicalAIAVDataset(**kwargs)


def test_missing_split_column_and_empty_split(fake_runtime):
    fake_runtime.clip_index = pd.DataFrame({"other": ["train"]}, index=["clip-a"])
    with pytest.raises(ValueError, match="split"):
        OnlinePhysicalAIAVDataset()

    fake_runtime.clip_index = pd.DataFrame({"split": ["val"]}, index=["clip-a"])
    with pytest.raises(ValueError, match="no clips"):
        OnlinePhysicalAIAVDataset(split="train")


def test_allowlist_overrides_split_and_exclusions_apply(fake_runtime, tmp_path):
    fake_runtime.clip_index = pd.DataFrame(
        {"split": ["train", "train", "val"]},
        index=["clip-a", "clip-b", "clip-c"],
    )
    allowlist = tmp_path / "gold.txt"
    allowlist.write_text("clip-c\nclip-a\nclip-c\n", encoding="utf-8")

    dataset = OnlinePhysicalAIAVDataset(
        split="val",
        clip_allowlist_path=allowlist,
        exclude_clip_ids={"clip-a"},
    )

    assert dataset.clip_ids == ["clip-c"]
    assert dataset[0]["clip_id"] == "clip-c"


def test_allowlist_sources_are_mutually_exclusive(fake_runtime, tmp_path):
    allowlist = tmp_path / "gold.txt"
    allowlist.write_text("clip-a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only one"):
        OnlinePhysicalAIAVDataset(
            clip_allowlist=["clip-a"],
            clip_allowlist_path=allowlist,
        )


def test_navigation_builds_nav_pool_and_attaches_label(fake_runtime, tmp_path):
    nav_dir = tmp_path / "nav" / "train"
    nav_dir.mkdir(parents=True)
    rows = [
        {
            "clip_id": "clip-a",
            "cohort": "train",
            "hf_split": "train",
            "t0_relative_us": frame_index * 100_000,
            "lookup_in_route_timestamp_range": True,
            "has_future_instruction": True,
            "nav_command": "TURN_RIGHT",
            "nav_distance_m": 25.0,
            "nav_text": "Turn right in 25m",
        }
        for frame_index in range(21)
    ]
    (nav_dir / "clip-a.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    dataset = OnlinePhysicalAIAVDataset(
        t0_us_list=[2_000_000],
        nav_config={
            "enabled": True,
            "root": str(tmp_path / "nav"),
            "cohort": "train",
            "max_distance_m": -1,
        },
    )

    assert dataset.clip_ids == ["clip-a", "clip-b"]
    assert dataset.nav_clip_ids == ["clip-a"]
    sample = dataset[0]
    assert sample["nav_enabled"] is True
    assert sample["nav_text"] == "Turn right in 25m"


@pytest.mark.parametrize(
    "sampling_kwargs, message",
    [
        ({"random_t0": True}, "t0_quantum_us=100000"),
        ({"t0_us_list": [2_123_456]}, "every fixed t0_us value"),
    ],
)
def test_navigation_attachment_requires_exact_10hz_sampling(
    fake_runtime, tmp_path, sampling_kwargs, message
):
    nav_dir = tmp_path / "nav" / "train"
    nav_dir.mkdir(parents=True)
    (nav_dir / "clip-a.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        OnlinePhysicalAIAVDataset(
            **sampling_kwargs,
            nav_config={
                "enabled": True,
                "root": str(tmp_path / "nav"),
                "cohort": "train",
            },
        )


@pytest.mark.parametrize("random_t0", [False, True])
def test_mixture_base_timestamps_are_unrestricted_and_navigation_stays_aligned(
    fake_runtime, tmp_path, random_t0
):
    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    dataset = OnlinePhysicalAIAVDataset(
        clip_allowlist=["clip-a"],
        samples_per_clip=2,
        random_t0=random_t0,
        t0_us_list=[2_123_456],
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(),
    )
    base_keys = [SampleVisit(1, visit_id) for visit_id in range(16)]
    base_samples = [dataset[key] for key in base_keys]
    nav_samples = [dataset[SampleVisit(0, visit_id)] for visit_id in range(16)]

    assert all(not sample["nav_enabled"] for sample in base_samples)
    assert all(not sample["mixture_requested_nav"] for sample in base_samples)
    assert all(2_000_000 <= sample["t0_us"] <= 3_000_000 for sample in base_samples)
    assert any(sample["t0_us"] % 100_000 for sample in base_samples)
    if random_t0:
        assert len({sample["t0_us"] for sample in base_samples}) > 1
    else:
        assert {sample["t0_us"] for sample in base_samples} == {2_123_456}
    assert all(sample["nav_enabled"] and sample["mixture_requested_nav"] for sample in nav_samples)
    assert all(sample["t0_us"] % 100_000 == 0 for sample in nav_samples)
    assert all(sample["t0_us"] in dataset.nav_store.valid_t0_us("clip-a") for sample in nav_samples)
    assert [dataset[key] for key in reversed(base_keys)] == list(reversed(base_samples))


def test_random_sampling_retries_only_infeasible_clips(
    fake_runtime,
    monkeypatch,
    caplog,
):
    attempts = []

    def load_with_infeasible_clip(**kwargs):
        attempts.append(kwargs["clip_id"])
        if kwargs["clip_id"] == "clip-a":
            raise InfeasibleClipError("no common sensor window")
        return {"clip_id": kwargs["clip_id"], "t0_us": 8_500_000}

    monkeypatch.setattr(
        physical_aiav_dataset,
        "load_physical_aiavdataset",
        load_with_infeasible_clip,
    )
    dataset = OnlinePhysicalAIAVDataset(
        random_t0=True,
        max_clip_retries=1,
    )

    with caplog.at_level("WARNING"):
        assert dataset[0]["clip_id"] == "clip-b"
        assert dataset[0]["clip_id"] == "clip-b"

    assert attempts == ["clip-a", "clip-b", "clip-a", "clip-b"]
    messages = [
        record.getMessage()
        for record in caplog.records
        if "Skipping infeasible clip" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "clip-a" in messages[0]
    assert "no common sensor window" in messages[0]


def test_explicit_mixture_is_exact_and_basic_uses_full_pool(
    fake_runtime,
    tmp_path,
):
    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    first = OnlinePhysicalAIAVDataset(
        samples_per_clip=4,
        t0_us_list=[2_000_000],
        seed=19,
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(),
    )
    second = OnlinePhysicalAIAVDataset(
        samples_per_clip=4,
        t0_us_list=[2_000_000],
        seed=19,
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(),
    )

    first_samples = [first[index] for index in range(len(first))]
    second_samples = [second[index] for index in range(len(second))]
    assert len(first_samples) == 8
    assert sum(sample["mixture_requested_nav"] for sample in first_samples) == 4
    assert sum(sample["mixture_actual_nav"] for sample in first_samples) == 4
    basic_samples = [sample for sample in first_samples if not sample["mixture_requested_nav"]]
    assert {sample["clip_id"] for sample in basic_samples} == {"clip-a", "clip-b"}
    assert all(not sample["nav_enabled"] for sample in basic_samples)
    assert [
        (
            sample["clip_id"],
            sample["t0_us"],
            sample["mixture_requested_nav"],
        )
        for sample in first_samples
    ] == [
        (
            sample["clip_id"],
            sample["t0_us"],
            sample["mixture_requested_nav"],
        )
        for sample in second_samples
    ]

    first.set_epoch(1)
    epoch_one_modes = [first[index]["mixture_requested_nav"] for index in range(len(first))]
    assert epoch_one_modes == [not sample["mixture_requested_nav"] for sample in first_samples]


def test_navigation_retries_distinct_timestamps(
    fake_runtime,
    monkeypatch,
    tmp_path,
):
    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    attempts = []

    def fail_first_two_timestamps(**kwargs):
        attempts.append((kwargs["clip_id"], kwargs["t0_us"]))
        if len(attempts) <= 2:
            raise InfeasibleClipError("bad timestamp")
        return {"clip_id": kwargs["clip_id"], "t0_us": kwargs["t0_us"]}

    monkeypatch.setattr(
        physical_aiav_dataset,
        "load_physical_aiavdataset",
        fail_first_two_timestamps,
    )
    dataset = OnlinePhysicalAIAVDataset(
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(nav_sample_retries=1),
    )

    sample = dataset[0]

    assert len(attempts) == 3
    assert len({t0_us for _, t0_us in attempts}) == 3
    assert sample["mixture_nav_t0_retries"] == 2
    assert sample["mixture_nav_clip_retries"] == 0
    assert sample["nav_enabled"] is True


def test_navigation_refreshes_clip_and_row_per_visit(fake_runtime, tmp_path):
    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    _write_nav_clip(nav_root, "clip-b")
    dataset = OnlinePhysicalAIAVDataset(
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(),
        seed=19,
    )
    keys = [SampleVisit(0, visit_id) for visit_id in range(32)]
    samples = [dataset[key] for key in keys]
    assert {sample["clip_id"] for sample in samples} == {"clip-a", "clip-b"}
    for clip_id in ("clip-a", "clip-b"):
        assert len({sample["t0_us"] for sample in samples if sample["clip_id"] == clip_id}) > 1
    assert all(sample["nav_enabled"] and sample["mixture_requested_nav"] for sample in samples)
    assert all(sample["t0_us"] % 100_000 == 0 for sample in samples)
    assert [dataset[key] for key in reversed(keys)] == list(reversed(samples))
    # Visit randomness does not change the index-based requested mixture.
    for visit_id in (0, 1, 9):
        assert [
            dataset[SampleVisit(index, visit_id)]["mixture_requested_nav"]
            for index in range(len(dataset))
        ] == [True, False]


@pytest.mark.parametrize("pai_vqa_probability", [0.0, 1.0])
def test_weighted_loader_refreshes_windows_and_resumes_exactly(fake_runtime, pai_vqa_probability):
    from alpamayo1_5_distill.training.lingoqa import (
        PAIVQADataset,
        TaskConcatDataset,
        WeightedTaskDistributedBatchSampler,
    )

    pai = OnlinePhysicalAIAVDataset(random_t0=True, seed=123)
    lingoqa = [{"task_type": "vqa", "vqa_source": "lingoqa"}] * 2
    dataset = TaskConcatDataset(pai, lingoqa, PAIVQADataset(pai))
    sampler = WeightedTaskDistributedBatchSampler(
        dataset,
        batch_size=1,
        world_size=1,
        rank=0,
        seed=42,
        pai_weight=4,
        lingoqa_weight=2,
        pai_vqa_probability=pai_vqa_probability,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=identity_collate)
    full = list(loader)
    physicalai_samples = [batch[0] for batch in full if "clip_id" in batch[0]]
    for clip_id in pai.clip_ids:
        clip_samples = [sample for sample in physicalai_samples if sample["clip_id"] == clip_id]
        assert len(clip_samples) == (4 if pai_vqa_probability == 0.0 else 6)
        assert len({sample["t0_us"] for sample in clip_samples}) > 1
    assert list(loader) == full
    sampler.set_sample_offset(5)
    assert list(loader) == full[5:]


@pytest.mark.parametrize("navigation", [False, True])
def test_spawned_persistent_workers_replay_visits_and_follow_epoch(tmp_path, navigation):
    from alpamayo1_5_distill.training.lingoqa import (
        TaskConcatDataset,
        WeightedTaskDistributedBatchSampler,
    )

    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    pai = _SpawnTestDataset(
        samples_per_clip=2,
        random_t0=True,
        seed=123,
        nav_config=_nav_config(nav_root) if navigation else None,
        mixture_config=_mixture_config(basic_ratio=0.75, nav_ratio=0.25) if navigation else None,
    )
    dataset = TaskConcatDataset(pai, [{"task_type": "vqa", "vqa_source": "lingoqa"}])
    sampler = WeightedTaskDistributedBatchSampler(
        dataset,
        batch_size=1,
        world_size=1,
        rank=0,
        seed=42,
        pai_weight=4,
    )
    single = DataLoader(dataset, batch_sampler=sampler, collate_fn=identity_collate)
    workers = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=identity_collate,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    epoch_samples = []
    for epoch in (0, 1):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        expected = list(single)
        epoch_samples.append(expected)
        assert list(workers) == expected
        sampler.set_sample_offset(5)
        assert list(workers) == expected[5:]
        base_samples = [
            batch[0]
            for batch in expected
            if batch[0]["task_type"] == "trajectory" and not batch[0].get("mixture_actual_nav")
        ]
        assert any(sample["t0_us"] % 100_000 for sample in base_samples)
        if navigation:
            nav_samples = [batch[0] for batch in expected if batch[0].get("mixture_actual_nav")]
            assert len(nav_samples) == 4
            assert len({sample["t0_us"] for sample in nav_samples}) > 1
            assert all(sample["t0_us"] % 100_000 == 0 for sample in nav_samples)
    assert epoch_samples[0] != epoch_samples[1]


def test_multiple_visits_to_retry_clip_in_one_batch_get_independent_draws(
    fake_runtime, monkeypatch
):
    from alpamayo1_5_distill.training.lingoqa import (
        TaskConcatDataset,
        WeightedTaskDistributedBatchSampler,
    )

    load_basic = physical_aiav_dataset.load_physical_aiavdataset

    def reject_first_clip(**kwargs):
        if kwargs["clip_id"] == "clip-a":
            raise InfeasibleClipError("no common sensor window")
        return load_basic(**kwargs)

    monkeypatch.setattr(physical_aiav_dataset, "load_physical_aiavdataset", reject_first_clip)
    pai = OnlinePhysicalAIAVDataset(samples_per_clip=2, random_t0=True, max_clip_retries=1)
    dataset = TaskConcatDataset(pai, [{"task_type": "vqa"}] * 4)
    sampler = WeightedTaskDistributedBatchSampler(
        dataset,
        batch_size=4,
        world_size=1,
        rank=0,
        seed=42,
    )
    batches = list(DataLoader(dataset, batch_sampler=sampler, collate_fn=identity_collate))
    trajectory = next(batch for batch in batches if batch[0]["task_type"] == "trajectory")
    assert len(trajectory) == 4
    assert all(sample["clip_id"] == "clip-b" for sample in trajectory)
    assert len({load["t0_random_bits"] for load in fake_runtime.loads}) == 4


def test_navigation_retries_another_clip(
    fake_runtime,
    monkeypatch,
    tmp_path,
):
    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    _write_nav_clip(nav_root, "clip-b")
    attempts = []

    def fail_first_clip(**kwargs):
        attempts.append(kwargs["clip_id"])
        if len(attempts) == 1:
            raise InfeasibleClipError("bad clip")
        return {"clip_id": kwargs["clip_id"], "t0_us": kwargs["t0_us"]}

    monkeypatch.setattr(
        physical_aiav_dataset,
        "load_physical_aiavdataset",
        fail_first_clip,
    )
    dataset = OnlinePhysicalAIAVDataset(
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(
            nav_t0_retries=1,
            nav_sample_retries=2,
        ),
    )

    sample = dataset[0]

    assert len(attempts) == 2
    assert attempts[0] != attempts[1]
    assert sample["mixture_nav_clip_retries"] == 1
    assert sample["nav_enabled"] is True


def test_navigation_falls_back_to_full_pool_basic_sample(
    fake_runtime,
    monkeypatch,
    tmp_path,
):
    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    attempts = []

    def reject_navigation_timestamps(**kwargs):
        attempts.append((kwargs["clip_id"], kwargs["t0_us"]))
        if kwargs["t0_us"] is not None:
            raise InfeasibleClipError("bad navigation timestamp")
        return {"clip_id": kwargs["clip_id"], "t0_us": 2_500_000}

    monkeypatch.setattr(
        physical_aiav_dataset,
        "load_physical_aiavdataset",
        reject_navigation_timestamps,
    )
    dataset = OnlinePhysicalAIAVDataset(
        random_t0=True,
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(
            nav_t0_retries=3,
            nav_sample_retries=1,
        ),
    )

    sample = dataset[0]

    assert len(attempts) == 4
    assert len({t0_us for _, t0_us in attempts[:3]}) == 3
    assert attempts[-1][1] is None
    assert sample["mixture_requested_nav"] is True
    assert sample["mixture_actual_nav"] is False
    assert sample["mixture_nav_fallback"] is True
    assert sample["nav_enabled"] is False


def test_navigation_fallback_refreshes_basic_timestamp_per_visit(
    fake_runtime, monkeypatch, tmp_path
):
    nav_root = tmp_path / "nav"
    _write_nav_clip(nav_root, "clip-a")
    load_basic = physical_aiav_dataset.load_physical_aiavdataset

    def reject_navigation(**kwargs):
        if kwargs["t0_us"] is not None:
            raise InfeasibleClipError("bad navigation timestamp")
        return load_basic(**kwargs)

    monkeypatch.setattr(physical_aiav_dataset, "load_physical_aiavdataset", reject_navigation)
    dataset = OnlinePhysicalAIAVDataset(
        random_t0=True,
        seed=123,
        nav_config=_nav_config(nav_root),
        mixture_config=_mixture_config(),
    )
    samples = [dataset[SampleVisit(0, visit_id)] for visit_id in range(8)]
    assert all(sample["mixture_nav_fallback"] for sample in samples)
    assert all(not sample["mixture_actual_nav"] for sample in samples)
    assert len({sample["t0_us"] for sample in samples}) > 1
    assert any(sample["t0_us"] % 100_000 for sample in samples)
    assert dataset[SampleVisit(0, 0)] == samples[0]


def test_identity_collate_preserves_raw_samples():
    samples = [{"value": object()}, {"value": object()}]
    result = identity_collate(samples)

    assert result == samples
    assert result is not samples
    assert result[0] is samples[0]
