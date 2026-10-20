from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alpamayo1_distill.training_resume import (
    CHECKPOINT_PHASE,
    CHECKPOINT_VERSION,
    ExactDistributedEvalSampler,
    ResumeDistributedSampler,
    capture_rng_state,
    gather_rank_states,
    iter_dataloader_without_advancing_model_rng,
    restore_rank_state,
    restore_rng_state,
    resume_cursor_from_state,
    validate_resume_metadata,
)


@pytest.mark.parametrize("dataset_size", [644, 90_928])
@pytest.mark.parametrize("world_size", [1, 8, 16, 128])
def test_exact_eval_sampler_covers_dataset_without_padding_or_dropping(
    dataset_size: int,
    world_size: int,
) -> None:
    dataset = range(dataset_size)
    shards = [
        list(
            ExactDistributedEvalSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
            )
        )
        for rank in range(world_size)
    ]

    combined = [index for shard in shards for index in shard]
    assert len(combined) == dataset_size
    assert len(set(combined)) == dataset_size
    assert sorted(combined) == list(dataset)
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_exact_eval_sampler_rejects_empty_rank() -> None:
    with pytest.raises(ValueError, match="at least one sample per rank"):
        ExactDistributedEvalSampler(range(1), num_replicas=2, rank=0)


def test_resume_sampler_skips_consumed_per_rank_batches() -> None:
    dataset = list(range(10))
    sampler = ResumeDistributedSampler(
        dataset,
        num_replicas=2,
        rank=0,
        shuffle=False,
    )

    assert list(sampler) == [0, 2, 4, 6, 8]
    sampler.set_start_batch(batch_index=1, batch_size=2)
    assert list(sampler) == [4, 6, 8]
    assert len(sampler) == 3

    with pytest.raises(ValueError, match="batch_index"):
        sampler.set_start_batch(batch_index=-1, batch_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        sampler.set_start_batch(batch_index=0, batch_size=0)


def test_resume_cursor_preserves_v2_and_rewinds_legacy_accumulation() -> None:
    current = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_phase": CHECKPOINT_PHASE,
        "epoch": 3,
        "step_in_epoch": 7,
    }
    assert resume_cursor_from_state(
        current,
        grad_accum_steps=4,
        steps_per_epoch=10,
    ) == (3, 7, False)

    legacy = {"epoch": 1, "step_in_epoch": 1}
    assert resume_cursor_from_state(
        legacy,
        grad_accum_steps=3,
        steps_per_epoch=10,
    ) == (0, 8, True)


def test_resume_metadata_accepts_exact_match_and_rejects_drift() -> None:
    expected = {"world_size": 2, "batch_size_per_gpu": 1, "random_t0": False}
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_phase": CHECKPOINT_PHASE,
        "resume_metadata": dict(expected),
    }
    validate_resume_metadata(checkpoint, expected)

    changed = dict(expected, world_size=4)
    with pytest.raises(ValueError, match="world_size"):
        validate_resume_metadata(checkpoint, changed)

    missing_metadata = dict(checkpoint)
    missing_metadata.pop("resume_metadata")
    with pytest.raises(ValueError, match="missing resume_metadata"):
        validate_resume_metadata(missing_metadata, expected)

    wrong_phase = dict(checkpoint, checkpoint_phase="pre_step")
    with pytest.raises(ValueError, match="checkpoint_phase"):
        validate_resume_metadata(wrong_phase, expected)

    # Unversioned checkpoints predate resume metadata and remain loadable.
    validate_resume_metadata({"checkpoint_version": 1}, expected)


def test_resume_metadata_normalizes_legacy_cosmos_default_paths() -> None:
    expected = {
        "world_size": 2,
        "student_vlm_path": "nvidia/Cosmos-Reason2-2B",
        "student_processor_path": "nvidia/Cosmos-Reason2-2B",
    }
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_phase": CHECKPOINT_PHASE,
        "resume_metadata": {
            "world_size": 2,
            "student_vlm_path": None,
            "student_processor_path": None,
            "use_cosmos_2b": True,
        },
    }

    validate_resume_metadata(checkpoint, expected)

    custom_checkpoint = {
        **checkpoint,
        "resume_metadata": {
            **checkpoint["resume_metadata"],
            "student_vlm_path": "/models/cosmos",
        },
    }
    with pytest.raises(ValueError, match="student_vlm_path"):
        validate_resume_metadata(custom_checkpoint, expected)


def test_resume_metadata_ignores_removed_saved_rollout_fields() -> None:
    expected = {"world_size": 2, "batch_size_per_gpu": 1}
    saved = {
        **expected,
        "use_online_dataset": True,
        "use_precompute_dataset": False,
        "reduce_dataset": False,
        "offline_gt_dir": None,
        "small_dataset": False,
    }
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_phase": CHECKPOINT_PHASE,
        "resume_metadata": saved,
    }

    validate_resume_metadata(checkpoint, expected)


@pytest.mark.parametrize(
    "legacy_fields",
    [
        {"use_online_dataset": False, "use_precompute_dataset": False},
        {"use_online_dataset": False, "use_precompute_dataset": True},
        {"use_online_dataset": True, "use_precompute_dataset": True},
    ],
)
def test_resume_metadata_rejects_removed_saved_rollout_mode(legacy_fields) -> None:
    expected = {"world_size": 2, "batch_size_per_gpu": 1}
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_phase": CHECKPOINT_PHASE,
        "resume_metadata": {**expected, **legacy_fields},
    }

    with pytest.raises(ValueError, match="removed saved-rollout dataset mode"):
        validate_resume_metadata(checkpoint, expected)


def test_resume_metadata_rejects_removed_debug_subset_mode() -> None:
    expected = {"world_size": 2, "batch_size_per_gpu": 1}
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_phase": CHECKPOINT_PHASE,
        "resume_metadata": {**expected, "small_dataset": True},
    }

    with pytest.raises(ValueError, match="removed debug-subset training mode"):
        validate_resume_metadata(checkpoint, expected)


def test_rng_capture_restore_replays_all_cpu_streams() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = capture_rng_state(torch.device("cpu"))

    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng_state(state, torch.device("cpu"))
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_rank_state_restore_checks_topology_and_restores_rng() -> None:
    torch.manual_seed(7)
    rank_state = capture_rng_state(torch.device("cpu"))
    expected = torch.rand(2)
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "world_size": 1,
        "rank_states": [rank_state],
    }

    restored = restore_rank_state(
        checkpoint,
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    assert restored is rank_state
    assert torch.equal(torch.rand(2), expected)

    with pytest.raises(ValueError, match="different world size"):
        restore_rank_state(
            checkpoint,
            rank=0,
            world_size=2,
            device=torch.device("cpu"),
        )


def test_single_rank_gather_and_loader_iterator_preserve_rng() -> None:
    local_state = {"micro_batch_cursor": 3}
    assert gather_rank_states(local_state, world_size=1) == [local_state]

    loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=1)
    torch.manual_seed(123)
    before_iterator = torch.get_rng_state()
    iterator = iter_dataloader_without_advancing_model_rng(loader)
    after_iterator = torch.get_rng_state()

    assert torch.equal(after_iterator, before_iterator)
    assert next(iterator)[0].item() == 0
