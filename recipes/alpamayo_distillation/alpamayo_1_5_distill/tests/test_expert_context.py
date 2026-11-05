# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from alpamayo1_5_distill.models.expert_context import (
    append_suffix_after_offsets,
    build_expert_attention_mask,
    find_future_start_offsets,
)


class AppendSuffixAfterOffsetsTest(unittest.TestCase):
    def test_appends_suffix_per_row_and_masks_suffix_tokens(self) -> None:
        sequences = torch.tensor(
            [
                [10, 11, 12, 99, 77, 77],
                [20, 21, 99, 88, 88, 88],
            ]
        )
        offsets = torch.tensor([4, 3])
        suffix = torch.tensor(
            [
                [100, 101, 102],
                [200, 201, 202],
            ]
        )

        out, out_mask, suffix_mask = append_suffix_after_offsets(
            sequences,
            offsets,
            suffix,
            pad_token_id=0,
            attention_mask=torch.ones_like(sequences),
        )

        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    [10, 11, 12, 99, 100, 101, 102],
                    [20, 21, 99, 200, 201, 202, 0],
                ]
            ),
        )
        torch.testing.assert_close(
            out_mask,
            torch.tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0],
                ]
            ),
        )
        torch.testing.assert_close(
            suffix_mask,
            torch.tensor(
                [
                    [False, False, False, False, True, True, True],
                    [False, False, False, True, True, True, False],
                ]
            ),
        )

    def test_rejects_invalid_offset(self) -> None:
        with self.assertRaisesRegex(ValueError, "offsets"):
            append_suffix_after_offsets(
                torch.tensor([[1, 2]]),
                torch.tensor([3]),
                torch.tensor([[4]]),
                pad_token_id=0,
            )

    def test_marker_offsets_and_trailing_contract(self) -> None:
        sequences = torch.tensor([[1, 2, 9, 7], [3, 9, 7, 7]])
        offsets = find_future_start_offsets(
            sequences,
            9,
            search_start=1,
            cache_seq_len=4,
            require_trailing_token=True,
        )
        torch.testing.assert_close(offsets, torch.tensor([3, 2]))

        with self.assertRaisesRegex(ValueError, "trailing cache-flush token"):
            find_future_start_offsets(
                torch.tensor([[1, 2, 9]]),
                9,
                require_trailing_token=True,
            )

    def test_marker_search_supports_per_row_prompt_lengths(self) -> None:
        sequences = torch.tensor(
            [
                [9, 1, 2, 9, 7],
                [3, 9, 4, 9, 7],
            ]
        )
        offsets = find_future_start_offsets(
            sequences,
            9,
            search_start=torch.tensor([2, 3]),
            cache_seq_len=5,
            require_trailing_token=True,
        )
        torch.testing.assert_close(offsets, torch.tensor([4, 4]))

    def test_expert_mask_hides_cache_after_each_marker(self) -> None:
        mask = build_expert_attention_mask(
            torch.tensor([2, 3]),
            cache_seq_len=4,
            query_seq_len=2,
            attention_implementation="flash_attention_2",
            cache_valid_mask=torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]]),
        )
        torch.testing.assert_close(
            mask,
            torch.tensor(
                [
                    [True, True, False, False, True, True],
                    [False, True, True, False, True, True],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
