# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from types import SimpleNamespace

import torch
from alpamayo1_5_distill.models.modeling_distill import DistillAlpamayo1_5
from alpamayo1_5_distill.models.trajectory_tokens import tokenize_future_trajectory


class _FakeTrajectoryTokenizer:
    def encode(self, *, hist_xyz, hist_rot, fut_xyz, fut_rot):
        del hist_rot, fut_rot
        batch_size = hist_xyz.shape[0]
        return torch.arange(3).expand(batch_size, -1).clone()


class _FakeCache:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.crop_calls: list[int] = []

    def get_seq_length(self) -> int:
        return self.seq_len

    def crop(self, seq_len: int) -> None:
        self.crop_calls.append(seq_len)
        self.seq_len = seq_len


class _FakeVLM:
    def __init__(self, cache: _FakeCache, vocab_size: int, *, fail: bool = False):
        self.cache = cache
        self.vocab_size = vocab_size
        self.fail = fail
        self.kwargs = None
        self.logits = torch.zeros(vocab_size, requires_grad=True)

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        batch_size, seq_len = kwargs["input_ids"].shape
        # Exercise the cache-restoration guard.
        self.cache.seq_len += seq_len
        if self.fail:
            raise RuntimeError("suffix forward failed")
        return SimpleNamespace(logits=self.logits.expand(batch_size, seq_len, -1))


def _trajectory_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "ego_history_xyz": torch.zeros(batch_size, 1, 2, 3),
        "ego_history_rot": torch.eye(3).expand(batch_size, 1, 2, 3, 3).clone(),
        "ego_future_xyz": torch.zeros(batch_size, 1, 3, 3),
        "ego_future_rot": torch.eye(3).expand(batch_size, 1, 3, 3, 3).clone(),
    }


class FutureTrajectoryTokenizationTest(unittest.TestCase):
    def test_tokenizes_and_offsets_future_trajectory(self) -> None:
        tokens = tokenize_future_trajectory(
            _FakeTrajectoryTokenizer(),
            _trajectory_batch(),
            start_idx=10,
        )
        torch.testing.assert_close(
            tokens,
            torch.tensor([[10, 11, 12], [10, 11, 12]]),
        )

    def test_builds_suffix_with_future_end_token(self) -> None:
        model = SimpleNamespace(
            traj_tokenizer=_FakeTrajectoryTokenizer(),
            future_token_start_idx=10,
            special_token_ids={"traj_future_end": 19},
        )
        suffix = DistillAlpamayo1_5._build_gt_future_traj_token_suffix(
            model,
            _trajectory_batch(),
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            suffix,
            torch.tensor([[10, 11, 12, 19], [10, 11, 12, 19]]),
        )


class FutureTrajectoryPrefixLossTest(unittest.TestCase):
    def test_reuses_prefix_cache_and_restores_its_length(self) -> None:
        vocab_size = 8
        cache = _FakeCache(seq_len=5)
        vlm = _FakeVLM(cache, vocab_size)
        model = SimpleNamespace(
            special_token_ids={"traj_future_start": 3},
            vlm=vlm,
        )
        outputs = SimpleNamespace(
            logits=torch.zeros(2, 5, vocab_size),
            past_key_values=cache,
            rope_deltas=torch.zeros(2, 1, dtype=torch.long),
        )
        seq = torch.zeros(2, 5, dtype=torch.long)
        seq[0, 2] = 3
        seq[1, 3] = 3

        loss = DistillAlpamayo1_5._compute_future_traj_vlm_loss_from_prefix(
            model,
            outputs,
            seq=seq,
            attention_mask=torch.ones(2, 5, dtype=torch.long),
            future_start_offsets=torch.tensor([3, 4]),
            gt_future_suffix=torch.tensor([[4, 5, 6], [4, 5, 6]]),
            tokenized_data={
                "attention_mask": torch.ones(2, 5, dtype=torch.long),
                "pixel_values": torch.ones(1),
            },
        )

        self.assertAlmostEqual(loss.item(), math.log(vocab_size), places=6)
        self.assertEqual(cache.crop_calls, [5])
        self.assertEqual(tuple(vlm.kwargs["attention_mask"].shape), (2, 7))
        self.assertEqual(tuple(vlm.kwargs["position_ids"].shape), (3, 2, 2))
        self.assertNotIn("pixel_values", vlm.kwargs)
        self.assertFalse(vlm.kwargs["use_cache"])

    def test_backpropagates_through_prefix_and_suffix_without_leaking_cache(self) -> None:
        cache = _FakeCache(seq_len=4)
        vlm = _FakeVLM(cache, vocab_size=8)
        prefix_logits = torch.zeros(1, 4, 8, requires_grad=True)
        outputs = SimpleNamespace(
            logits=prefix_logits,
            past_key_values=cache,
            rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        )
        seq = torch.tensor([[1, 2, 3, 0]])

        loss = DistillAlpamayo1_5._compute_future_traj_vlm_loss_from_prefix(
            SimpleNamespace(
                special_token_ids={"traj_future_start": 3},
                vlm=vlm,
            ),
            outputs,
            seq=seq,
            attention_mask=torch.ones_like(seq),
            future_start_offsets=torch.tensor([3]),
            gt_future_suffix=torch.tensor([[4, 5, 6]]),
            tokenized_data={"attention_mask": torch.ones_like(seq)},
        )
        loss.backward()

        self.assertEqual(cache.get_seq_length(), 4)
        self.assertIsNotNone(prefix_logits.grad)
        self.assertGreater(prefix_logits.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(vlm.logits.grad)
        self.assertGreater(vlm.logits.grad.abs().sum().item(), 0.0)

    def test_restores_cache_when_suffix_forward_raises(self) -> None:
        cache = _FakeCache(seq_len=4)
        outputs = SimpleNamespace(
            logits=torch.zeros(1, 4, 8),
            past_key_values=cache,
            rope_deltas=torch.zeros(1, 1, dtype=torch.long),
        )

        with self.assertRaisesRegex(RuntimeError, "suffix forward failed"):
            DistillAlpamayo1_5._compute_future_traj_vlm_loss_from_prefix(
                SimpleNamespace(
                    special_token_ids={"traj_future_start": 3},
                    vlm=_FakeVLM(cache, vocab_size=8, fail=True),
                ),
                outputs,
                seq=torch.tensor([[1, 2, 3, 0]]),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
                future_start_offsets=torch.tensor([3]),
                gt_future_suffix=torch.tensor([[4, 5]]),
                tokenized_data={},
            )

        self.assertEqual(cache.get_seq_length(), 4)
        self.assertEqual(cache.crop_calls, [4])

    def test_rejects_marker_without_preceding_token(self) -> None:
        model = SimpleNamespace(
            special_token_ids={"traj_future_start": 3},
            vlm=None,
        )
        outputs = SimpleNamespace(logits=torch.zeros(1, 2, 8))
        with self.assertRaisesRegex(ValueError, "previous token"):
            DistillAlpamayo1_5._compute_future_traj_vlm_loss_from_prefix(
                model,
                outputs,
                seq=torch.zeros(1, 2, dtype=torch.long),
                attention_mask=torch.ones(1, 2, dtype=torch.long),
                future_start_offsets=torch.tensor([1]),
                gt_future_suffix=torch.tensor([[4]]),
                tokenized_data={},
            )

    def test_missing_marker_skips_auxiliary_loss(self) -> None:
        model = SimpleNamespace(
            special_token_ids={"traj_future_start": 3},
            vlm=None,
        )
        outputs = SimpleNamespace(logits=torch.zeros(1, 3, 8, requires_grad=True))
        loss = DistillAlpamayo1_5._compute_future_traj_vlm_loss_from_prefix(
            model,
            outputs,
            seq=torch.zeros(1, 3, dtype=torch.long),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            future_start_offsets=torch.tensor([3]),
            gt_future_suffix=torch.tensor([[4]]),
            tokenized_data={},
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)


if __name__ == "__main__":
    unittest.main()
