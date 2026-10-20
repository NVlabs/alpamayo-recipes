# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from alpamayo1_5_distill.training.resume import (
    ResumeDistributedSampler,
    load_trainer_state,
    make_trainer_state,
)
from torch.utils.data import DataLoader


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_sampler_is_deterministic_and_shards_all_ranks(world_size):
    dataset = list(range(11))
    samplers = [
        ResumeDistributedSampler(
            dataset,
            rank=rank,
            world_size=world_size,
            seed=29,
        )
        for rank in range(world_size)
    ]
    for sampler in samplers:
        sampler.set_epoch(5)

    generator = torch.Generator().manual_seed(29 + 5)
    expected = torch.randperm(len(dataset), generator=generator).tolist()
    expected = expected[: (len(dataset) // world_size) * world_size]
    shards = [list(sampler) for sampler in samplers]

    assert shards == [expected[rank::world_size] for rank in range(world_size)]
    assert len({index for shard in shards for index in shard}) == len(expected)


def test_sampler_resumes_at_rank_local_offset_and_epoch_reset():
    sampler = ResumeDistributedSampler(
        list(range(20)),
        rank=1,
        world_size=3,
        seed=7,
    )
    sampler.set_epoch(4)
    full_epoch = list(sampler)

    # Two completed per-rank micro-batches at batch_size=2 skip four samples.
    sampler.set_sample_offset(2 * 2)
    assert list(sampler) == full_epoch[4:]
    assert len(sampler) == len(full_epoch) - 4

    sampler.set_epoch(4)
    assert sampler.sample_offset == 0
    assert list(sampler) == full_epoch
    sampler.set_epoch(5)
    assert sampler.sample_offset == 0
    assert list(sampler) != full_epoch


def test_resumed_dataloader_does_not_replay_completed_batches():
    dataset = list(range(24))
    sampler = ResumeDistributedSampler(
        dataset,
        rank=2,
        world_size=4,
        seed=11,
    )
    sampler.set_epoch(6)
    full_batches = list(DataLoader(dataset, batch_size=2, sampler=sampler))

    sampler.set_epoch(6)
    sampler.set_sample_offset(2 * 2)
    resumed_batches = list(DataLoader(dataset, batch_size=2, sampler=sampler))

    assert len(full_batches) == 3
    assert len(resumed_batches) == 1
    assert torch.equal(resumed_batches[0], full_batches[2])


def _expected_state(**overrides):
    values = {
        "global_step": 0,
        "epoch": 0,
        "sample_offset": 0,
        "seed": 17,
        "world_size": 8,
        "dataset_length": 101,
        "per_rank_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "sampler_epoch": 0,
    }
    values.update(overrides)
    return make_trainer_state(**values)


def test_trainer_state_round_trip_contains_compatibility_metadata(tmp_path):
    saved = _expected_state(global_step=12, epoch=3, sample_offset=8, sampler_epoch=3)
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps(saved), encoding="utf-8")

    loaded = load_trainer_state(path, expected=_expected_state())

    assert loaded == {
        "global_step": 12,
        "epoch": 3,
        "sample_offset": 8,
        "sampler_epoch": 3,
    }
    assert set(saved) == {
        "global_step",
        "epoch",
        "sample_offset",
        "seed",
        "world_size",
        "dataset_length",
        "per_rank_batch_size",
        "gradient_accumulation_steps",
        "sampler_epoch",
    }


def test_legacy_batch_cursor_converts_using_current_batch_size(tmp_path):
    path = tmp_path / "trainer_state.json"
    path.write_text(
        json.dumps({"global_step": 9, "epoch": 2, "batch_in_epoch": 3}),
        encoding="utf-8",
    )

    loaded = load_trainer_state(path, expected=_expected_state(per_rank_batch_size=4))

    assert loaded == {
        "global_step": 9,
        "epoch": 2,
        "sample_offset": 12,
        "sampler_epoch": 2,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 18),
        ("world_size", 4),
        ("dataset_length", 100),
        ("per_rank_batch_size", 1),
        ("gradient_accumulation_steps", 2),
    ],
)
def test_trainer_state_rejects_incompatible_resume(tmp_path, field, value):
    saved = _expected_state(epoch=3, sampler_epoch=3)
    saved[field] = value
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_trainer_state(path, expected=_expected_state())


def test_trainer_state_rejects_mismatched_sampler_epoch(tmp_path):
    path = tmp_path / "trainer_state.json"
    path.write_text(
        json.dumps(_expected_state(epoch=3, sampler_epoch=2)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampler_epoch"):
        load_trainer_state(path, expected=_expected_state())
