# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU loss/fusion checks that do not require the prospective inference package."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_SPEC = importlib.util.spec_from_file_location(
    "edge_sft_loss_utils", Path(__file__).parents[1] / "models" / "utils.py"
)
utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(utils)


def test_next_token_loss_uses_shifted_supervised_positions() -> None:
    logits = torch.tensor([[[1.0, 2.0, 0.0], [4.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[2, 1, 0]])
    loss = utils.compute_next_token_loss(logits, labels, torch.tensor([[True, True, False]]))
    torch.testing.assert_close(loss, F.cross_entropy(logits[:, 0], labels[:, 1]))
    loss.backward()
    assert logits.grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[:, 1:]) == 0


@pytest.mark.parametrize("masked", [False, True])
def test_empty_loss_is_finite_and_preserves_zero_gradients(masked: bool) -> None:
    logits = torch.full((2, 3, 4), float("nan"), requires_grad=True)
    labels = torch.zeros((2, 3), dtype=torch.long) if masked else torch.full((2, 3), -100)
    mask = torch.zeros_like(labels, dtype=torch.bool) if masked else None
    loss = utils.compute_next_token_loss(logits, labels, mask)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_future_loss_routes_only_trajectory_closing_eos() -> None:
    labels = torch.tensor([[7, 4, 10, 5, 7, 1, 8]])
    mask = utils.future_traj_loss_mask(
        labels, {"future_id0": 10, "future_start": 4, "future_end": 5}, 3, [7, 8]
    )
    assert mask.tolist() == [[False, True, True, True, True, False, False]]


def test_aligned_loss_matches_native_ignored_bucket_and_nan_scalar_handling() -> None:
    logits = torch.randn(3, 5, requires_grad=True)
    ignored = torch.full((3,), utils.IGNORE_INDEX)
    loss = utils.compute_token_loss(logits, ignored)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    # Native CE sanitizes its scalar; this does not imply finite NaN-input gradients.
    nan_logits = torch.full((1, 5), float("nan"))
    assert utils.compute_token_loss(nan_logits, torch.tensor([1])).item() == 0


class _TrajectoryTokenizer:
    def __init__(self, tokens: list[list[int]], history: bool = False) -> None:
        self.tokens = torch.tensor(tokens)
        self.history = history

    def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
        for value in (hist_xyz, hist_rot, fut_xyz, fut_rot):
            assert value.dtype == torch.float32
        assert not torch.is_autocast_enabled("cpu")
        if self.history:
            assert hist_xyz.shape[1] == 1
            torch.testing.assert_close(hist_xyz, fut_xyz[:, :1])
        return self.tokens


def _trajectory_data() -> dict[str, torch.Tensor]:
    return {
        "ego_history_xyz": torch.zeros((2, 1, 3, 3), dtype=torch.bfloat16),
        "ego_history_rot": torch.eye(3, dtype=torch.bfloat16).expand(2, 1, 3, 3, 3),
        "ego_future_xyz": torch.zeros((2, 1, 2, 3), dtype=torch.bfloat16),
        "ego_future_rot": torch.eye(3, dtype=torch.bfloat16).expand(2, 1, 2, 3, 3),
    }


_TRAJ_IDS = {"history_pad": 5, "history_id0": 20, "future_pad": 6, "future_id0": 30}


def test_fusion_preserves_sample_boundaries_and_disables_autocast() -> None:
    input_ids = torch.tensor([[1, 1, 5, 5, 1, 1], [1, 1, 5, 5, 6, 6]])
    original = input_ids.clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        fused = utils.fuse_traj_tokens(
            _TrajectoryTokenizer([[0, 1], [2, 3]], history=True),
            _TrajectoryTokenizer([[4, 5], [6, 7]]),
            input_ids,
            _trajectory_data(),
            _TRAJ_IDS,
        )
    assert fused.tolist() == [[1, 1, 20, 21, 1, 1], [1, 1, 22, 23, 36, 37]]
    torch.testing.assert_close(input_ids, original)


def test_fusion_rejects_truncated_per_row_trajectory_blocks() -> None:
    with pytest.raises(ValueError, match="placeholder counts"):
        utils.fuse_traj_tokens(
            _TrajectoryTokenizer([[0, 1], [2, 3]]),
            None,
            torch.tensor([[1, 5, 1], [1, 5, 5]]),
            _trajectory_data(),
            _TRAJ_IDS,
        )


def test_mixed_vqa_and_trajectory_rows_fuse_only_present_targets() -> None:
    original = torch.tensor([[1, 1, 1, 1], [5, 5, 6, 6]])
    trajectory = {key: value[1] for key, value in _trajectory_data().items()}
    fused = utils.fuse_traj_tokens(
        _TrajectoryTokenizer([[2, 3]], history=True),
        _TrajectoryTokenizer([[6, 7]]),
        original,
        [None, trajectory],
        _TRAJ_IDS,
    )
    assert fused.tolist() == [[1, 1, 1, 1], [22, 23, 36, 37]]
    assert original.tolist() == [[1, 1, 1, 1], [5, 5, 6, 6]]


def test_all_vqa_rows_do_not_require_trajectory_tokenizers() -> None:
    ids = torch.tensor([[1, 2], [2, 3]])
    fused = utils.fuse_traj_tokens(None, None, ids, [None, None], _TRAJ_IDS)
    torch.testing.assert_close(fused, ids)
