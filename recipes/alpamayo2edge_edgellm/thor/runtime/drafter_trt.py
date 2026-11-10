# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 drafter on Edge-LLM's exported draft engine (``host/export.py draft`` + ``thor/build_draft.py``).

Same interface as the torch ``Drafter``: ``publish(hidden_concat)`` queues the target's tap rows of newly committed
tokens, ``propose(anchor)`` runs one engine call that (1) folds the queued rows into the 4-layer paged draft K/V pool
through the ``DFlashTargetKVCacheUpdate`` plugin and (2) scores the block [anchor, mask x 15]: top-16 candidate ids and
unary logits per slot plus the selector-projected hidden; the low-rank transition walk (SpecForge ``greedy_path``)
over the 15 x 16 candidates runs on the host from one pinned copy. Profile 0 takes the whole prompt (bootstrap),
profile 1 the short continuation deltas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .engine import Engine

PAGE = 128


class TrtDrafter:
    def __init__(self, engine: Engine, engine_dir: Path, embedding: torch.Tensor, capacity: int):
        cfg = json.load(open(engine_dir / "config.json"))
        d = cfg["dflash_config"]
        if d.get("is_causal", False):
            raise ValueError("the recipe's drafter uses a non-causal block")
        self.engine = engine
        self.block, self.top_k, self.mask_id = (
            int(d["block_size"]),
            int(d["selector_top_k"]),
            int(d["mask_token_id"]),
        )
        self.taps = list(d["target_layer_ids"])
        self.layers, self.theta = cfg["num_hidden_layers"], float(cfg["rope_theta"])
        kv_heads, head_dim = cfg["num_key_value_heads"], cfg["head_dim"]
        selector = load_file(
            str(engine_dir / d.get("selector_file", "dflash2_selector.safetensors"))
        )
        self.pred, self.succ = (
            selector["predecessor_codebook"].cuda(),
            selector["successor_codebook"].cuda(),
        )
        self.embedding = embedding
        vocab_map = (
            engine_dir / "draft_vocab_map.safetensors"
        )  # reduced draft head: proposal index -> token id
        self.vocab_map = (
            load_file(str(vocab_map))["vocab_map"].long().cuda() if vocab_map.is_file() else None
        )
        self.pages = -(-capacity // PAGE)
        self.pool = [
            torch.zeros(2, self.pages, PAGE, kv_heads, head_dim, dtype=torch.float16, device="cuda")
            for _ in range(self.layers)
        ]
        i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
        ids = torch.arange(self.pages, dtype=torch.int32)
        self.const = {
            "query_start_offsets": i32(0, self.block),
            "query_lengths": i32(self.block),
            "state_indices": i32(0),
            "execution_phase_marker": torch.zeros(
                4, dtype=torch.int32, device="cuda"
            ),  # kSpecDraftProposal
            "context_sequence_count_carrier": torch.zeros(0, dtype=torch.int32, device="cuda"),
            "kv_page_table": torch.stack([ids, ids + self.pages]).cuda()[None],
            "packed_attention_mask": torch.full(
                (self.block, 1), (1 << self.block) - 1, dtype=torch.int32, device="cuda"
            ),
        }
        rotary = cfg["head_dim"]
        self.inv_freq = (
            1.0 / (self.theta ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary))
        ).cuda()
        P, K, R = self.block - 1, self.top_k, self.pred.shape[1]
        self.packed = torch.empty(
            P * K * 2 + P * R + 2 * P * K * R + R, dtype=torch.float32
        ).pin_memory()
        self.pending: list[torch.Tensor] = []
        self.length = 0

    def reset(self) -> None:
        self.pending, self.length = [], 0

    def publish(self, hidden_concat: torch.Tensor) -> None:
        """Tap rows [n, len(taps) * hidden] of the next committed tokens; folded into the pool by the next propose."""
        self.pending.append(hidden_concat.half())

    def rope(self, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.float()[:, None] * self.inv_freq[None]
        return torch.cat([angles.cos(), angles.sin()], dim=1)

    def propose(self, anchor: int) -> list[int]:
        delta = torch.cat(self.pending) if len(self.pending) > 1 else self.pending[0]
        self.pending = []
        n = delta.shape[0]
        target_len = self.length + n
        block = self.block
        if target_len + block > self.pages * PAGE:
            raise ValueError("draft K/V pool exceeded")
        delta_pos = torch.arange(self.length, target_len, dtype=torch.int32, device="cuda")
        block_pos = torch.arange(target_len, target_len + block, dtype=torch.int32, device="cuda")
        ids = torch.tensor([anchor] + [self.mask_id] * (block - 1), device="cuda")
        P, K, R = block - 1, self.top_k, self.pred.shape[1]
        support = torch.empty(1, P, K, dtype=torch.int32, device="cuda")
        unary = torch.empty(1, P, K, dtype=torch.float32, device="cuda")
        projected = torch.empty(1, P, R, dtype=torch.half, device="cuda")
        io = {
            "inputs_embeds": self.embedding[ids],
            "dflash_target_hidden_concat": delta,
            "rope_rotary_cos_sin": self.rope(block_pos),
            "positions": block_pos,
            "attention_position_ids": block_pos,
            "past_lengths": torch.tensor([target_len], dtype=torch.int32, device="cuda"),
            "attention_sequence_lengths": torch.tensor(
                [target_len + block], dtype=torch.int32, device="cuda"
            ),
            "dflash_delta_rope_cos_sin": self.rope(delta_pos),
            "dflash_delta_positions": delta_pos,
            "dflash_delta_token_to_sequence": torch.zeros(n, dtype=torch.int32, device="cuda"),
            "spec_proposal_support_ids": support,
            "spec_proposal_unary_values": unary,
            "spec_proposal_projected_hidden": projected,
            **self.const,
        }
        for i, pool in enumerate(self.pool):
            io[f"past_key_values_{i}"] = io[f"present_key_values_{i}"] = pool
        profile = 1 if (self.length > 0 and self.engine.profiles > 1) else 0
        self.engine(io, profile=profile)
        self.length = target_len
        cand = support[0].long()  # [15, 16]
        if self.vocab_map is not None:  # reduced draft head (host/export.py draft --vocab)
            cand = self.vocab_map[cand]
        packed = torch.cat(
            [
                cand.float().reshape(-1),
                unary[0].reshape(-1),
                projected[0].float().reshape(-1),
                self.succ[cand].float().reshape(-1),
                self.pred[cand].float().reshape(-1),
                self.pred[anchor].float(),
            ]
        )
        self.packed.copy_(packed, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        flat = self.packed.numpy()
        o = 0

        def take(count, shape):
            nonlocal o
            v = flat[o : o + count].reshape(shape)
            o += count
            return v

        cand = take(P * K, (P, K)).astype(np.int64)
        unary_v, proj = take(P * K, (P, K)), take(P * R, (P, R))
        succ, pred, prev = take(P * K * R, (P, K, R)), take(P * K * R, (P, K, R)), take(R, (R,))
        tokens = []
        for p in range(
            P
        ):  # SpecForge CandidateSelector.greedy_path: unary + <pred[prev] * proj[p], succ[cand]>
            scores = unary_v[p] + succ[p] @ (prev * proj[p])
            j = int(scores.argmax())
            tokens.append(int(cand[p, j]))
            prev = pred[p, j]
        return tokens
