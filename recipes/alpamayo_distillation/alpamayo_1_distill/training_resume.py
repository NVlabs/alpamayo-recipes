"""Utilities for exact, distributed training resume."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Sampler

from alpamayo1_distill.defaults import DEFAULT_COSMOS_STUDENT_MODEL


CHECKPOINT_VERSION = 2
CHECKPOINT_PHASE = "post_step"


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices across ranks without padding or dropping.

    PyTorch's :class:`DistributedSampler` pads a non-divisible dataset when
    ``drop_last=False``. That is useful for synchronized training, but it
    evaluates duplicate clips and biases aggregate validation metrics. This
    sampler assigns ``rank, rank + world_size, ...`` instead, so the union of
    all rank-local shards is exactly the original dataset.

    Every rank must receive at least one sample because validation discovers
    and synchronizes the model's loss keys on the first local batch.
    """

    def __init__(self, dataset, *, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        if len(dataset) < num_replicas:
            raise ValueError(
                "Exact distributed evaluation requires at least one sample per "
                f"rank, but dataset_size={len(dataset)} and world_size={num_replicas}"
            )
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


class ResumeDistributedSampler(DistributedSampler):
    """A deterministic distributed sampler that can resume at a batch cursor.

    ``DistributedSampler`` yields sample indices, while trainer checkpoints store
    the number of consumed micro-batches. ``set_start_batch`` performs the unit
    conversion explicitly so resume also works when the per-rank batch size is
    greater than one.
    """

    def __init__(self, *args, start_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = int(start_index)

    def set_start_batch(self, batch_index: int, batch_size: int) -> None:
        if batch_index < 0:
            raise ValueError(f"batch_index must be non-negative, got {batch_index}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.start_index = int(batch_index) * int(batch_size)

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return max(0, super().__len__() - self.start_index)


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture every process-local RNG used by the training input/model path."""

    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    if state.get("numpy_rng_state") is not None:
        np.random.set_state(state["numpy_rng_state"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], device=device)


def gather_rank_states(local_state: dict[str, Any], world_size: int) -> list[dict[str, Any]]:
    """Gather process-local resume state in global-rank order on every rank."""

    if world_size == 1:
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    if any(state is None for state in gathered):
        raise RuntimeError("Failed to gather resume state from every distributed rank")
    return [state for state in gathered if state is not None]


def restore_rank_state(
    checkpoint: dict[str, Any],
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Restore this rank's RNG and return its saved running-window state.

    Unversioned checkpoints only contain rank 0's RNG. Restoring it on every
    worker would collapse all diffusion-noise streams, so legacy RNG is restored
    on rank 0 only. Exact rank-local continuation begins with version-2
    checkpoints.
    """

    rank_states = checkpoint.get("rank_states")
    if rank_states is not None:
        saved_world_size = int(checkpoint.get("world_size", len(rank_states)))
        if saved_world_size != world_size or len(rank_states) != world_size:
            raise ValueError(
                "Cannot exactly resume with a different world size: "
                f"checkpoint={saved_world_size}, current={world_size}. "
                "Use --init_ckpt for a new run with a different topology."
            )
        rank_state = rank_states[rank]
        restore_rng_state(rank_state, device)
        return rank_state

    if rank == 0:
        legacy_state = {
            "torch_rng_state": checkpoint.get("torch_rng_state"),
            "cuda_rng_state": None,
        }
        cuda_states = checkpoint.get("cuda_rng_state_all")
        if torch.cuda.is_available() and cuda_states:
            legacy_state["cuda_rng_state"] = cuda_states[0]
        restore_rng_state(legacy_state, device)
    return None


def resume_cursor_from_state(
    checkpoint: dict[str, Any],
    *,
    grad_accum_steps: int,
    steps_per_epoch: int,
) -> tuple[int, int, bool]:
    """Return ``(epoch, micro_batch_cursor, is_legacy)`` for a checkpoint.

    Legacy checkpoints were written after forward/backward but before the
    optimizer update. Their cursor includes an accumulation whose gradients were
    never saved, so rewind it and replay those micro-batches.
    """

    epoch = int(checkpoint.get("epoch", 0))
    cursor = int(checkpoint.get("step_in_epoch", 0))
    version = int(checkpoint.get("checkpoint_version", 1))
    phase = checkpoint.get("checkpoint_phase")
    is_legacy = version < CHECKPOINT_VERSION or phase != CHECKPOINT_PHASE

    if is_legacy and cursor > 0:
        absolute_cursor = epoch * steps_per_epoch + cursor
        absolute_cursor = max(0, absolute_cursor - grad_accum_steps)
        epoch, cursor = divmod(absolute_cursor, steps_per_epoch)

    return epoch, cursor, is_legacy


def validate_resume_metadata(
    checkpoint: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Reject configuration changes that would invalidate exact continuation."""

    if int(checkpoint.get("checkpoint_version", 1)) < CHECKPOINT_VERSION:
        return

    if checkpoint.get("checkpoint_phase") != CHECKPOINT_PHASE:
        raise ValueError(
            "Versioned checkpoint has an unsupported checkpoint_phase: "
            f"{checkpoint.get('checkpoint_phase')!r}"
        )

    saved = checkpoint.get("resume_metadata")
    if not isinstance(saved, dict):
        raise ValueError("Versioned checkpoint is missing resume_metadata")

    # Version-2 checkpoints from before the cleanup recorded which dataset path
    # produced their batches. Continue to accept the live-teacher/raw-clip mode,
    # but never reinterpret a saved-rollout cursor as a raw-dataset cursor.
    if (
        saved.get("use_online_dataset") is False
        or saved.get("use_precompute_dataset") is True
    ):
        raise ValueError(
            "Cannot resume a checkpoint created with the removed saved-rollout "
            "dataset mode. Use --init_ckpt to load its model weights into a new "
            "raw-data/live-teacher run instead."
        )
    if saved.get("small_dataset") is True:
        raise ValueError(
            "Cannot resume a checkpoint created with the removed debug-subset "
            "training mode. Use --init_ckpt to load its model weights into a new "
            "full-data run instead."
        )

    def _student_paths(metadata: dict[str, Any]) -> tuple[Any, Any]:
        # Early Cosmos runs recorded the unresolved CLI values (None) even
        # though --use_cosmos_2b resolved both paths to the canonical model.
        # Normalize that representation without relaxing custom-path checks.
        vlm_path = metadata.get("student_vlm_path") or DEFAULT_COSMOS_STUDENT_MODEL
        processor_path = metadata.get("student_processor_path") or vlm_path
        return vlm_path, processor_path

    saved_student_paths = _student_paths(saved)
    expected_student_paths = _student_paths(expected)

    mismatches = []
    for key, current_value in expected.items():
        if key not in saved:
            mismatches.append(f"{key}: missing (current={current_value!r})")
        elif key == "student_vlm_path":
            if saved_student_paths[0] == expected_student_paths[0]:
                continue
            mismatches.append(
                f"{key}: checkpoint={saved[key]!r}, current={current_value!r}"
            )
        elif key == "student_processor_path":
            if saved_student_paths[1] == expected_student_paths[1]:
                continue
            mismatches.append(
                f"{key}: checkpoint={saved[key]!r}, current={current_value!r}"
            )
        elif saved[key] != current_value:
            mismatches.append(
                f"{key}: checkpoint={saved[key]!r}, current={current_value!r}"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Resume configuration does not match the checkpoint: "
            f"{details}. Use --init_ckpt to start a new run with changed settings."
        )


def iter_dataloader_without_advancing_model_rng(loader):
    """Create a DataLoader iterator without consuming the model's CPU RNG.

    PyTorch normally draws a worker base seed from the global CPU generator when
    an iterator is created. A mid-epoch restart creates an extra iterator, which
    would otherwise shift subsequent model-side random draws.
    """

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    cpu_rng_state = torch.get_rng_state()
    try:
        return iter(loader)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(cpu_rng_state)
