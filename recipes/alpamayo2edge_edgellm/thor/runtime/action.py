# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Alpamayo2-Edge action expert on the engine built from host/action_graph.py.

Edge-LLM's attention plugin (tree mode) reads the language model's paged K/V pool in place; each candidate's 64
rows live in private pages at the top of the pool, so the only per-request copy is one partial page.
"""

from __future__ import annotations

import torch

from .engine import Engine
from .rope import text_cos_sin
from .llm import PAGE, UNUSED_PAGE as UNUSED


class TreeActionExpert:
    """Edge-LLM's attention plugin reads the language model's paged pool in
    place. Candidates are packed ``per_seq`` per plugin sequence (the tree kernel's 128-row block): candidate j of a
    sequence stores its 64 rows at cache positions length + 64 j, a block-diagonal mask keeps candidates apart. Each
    sequence's page-table row lists the shared prefix pages before the one holding row ``length``, then private pages:
    the first starts as a copy of that partial page, all of them receive the sequence's rows (rewritten every step).
    Private pages sit at the top of the pool, above the language model's pages."""

    def __init__(self, engine: Engine, config: dict, candidates: int):
        self.engine = engine
        self.layers = config["num_hidden_layers"]
        self.tokens = config["tokens"]
        self.steps = config["steps"]
        self.theta = float(config["rope_theta"])
        self.section = tuple(config["mrope_section"])
        if candidates != config["candidates"]:
            raise ValueError(
                f"the tree action engine was exported for {config['candidates']} candidates, not {candidates}"
            )
        self.candidates = candidates
        self.per_seq = config.get("candidates_per_sequence") or 1
        while candidates % self.per_seq:
            self.per_seq -= 1
        self.sequences = candidates // self.per_seq
        self.block = self.per_seq * self.tokens
        self.private_per_seq = (
            -(-(self.block + PAGE - 1) // PAGE) + 1
        )  # partial page copy + the block's pages
        self.max_pages = engine.profile_shape("kv_page_table")[2][2]
        self.int = {n: engine.dtype[n] for n in engine.inputs}
        self.pool: list[torch.Tensor] = []
        self.table = None

    def load_prefix(self, pool: list[torch.Tensor], pages: list[int], length: int) -> None:
        total, n = pool[0].shape[1], self.private_per_seq
        first = total - n * self.sequences
        private = [list(range(first + n * s, first + n * s + n)) for s in range(self.sequences)]
        if max(pages) >= first:
            raise ValueError(
                "the language model's pages overlap the action expert's private pages; enlarge the pool"
            )
        p0, r0 = length // PAGE, length % PAGE
        if p0 + n > self.max_pages:
            raise ValueError(
                f"prefix {length} needs {p0 + n} page-table entries, engine allows {self.max_pages}"
            )
        table = torch.full(
            (self.sequences, 2, self.max_pages), UNUSED, dtype=self.int["kv_page_table"]
        )
        for s in range(self.sequences):
            ids = torch.tensor(pages[:p0] + private[s])
            table[s, 0, : len(ids)] = ids
            table[s, 1, : len(ids)] = ids + total
        self.table = table.cuda()
        if r0:
            src = pages[p0]
            for layer in range(self.layers):
                for s in range(self.sequences):
                    pool[layer][:, private[s][0], :r0] = pool[layer][:, src, :r0]
        self.pool, self.length = pool, length

    def sample(self, length: int, rope_delta: int, steps: int, noise: torch.Tensor) -> torch.Tensor:
        if steps != self.steps:
            raise ValueError(
                f"the tree action engine runs {self.steps} Euler steps; re-export for {steps}"
            )
        if length != self.length:
            raise ValueError("load_prefix must run with the same length as sample")
        b, t, q = self.candidates, self.tokens, self.sequences
        i32 = lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")  # noqa: E731
        io = {
            "noise": noise.reshape(-1, 2).float().contiguous().cuda(),
            "rope": text_cos_sin(rope_delta + length, t, self.theta, section=self.section)
            .float()
            .cuda(),
            "past_lengths": i32([length] * q),
            "kv_page_table": self.table,
            "attention_pos_id": i32(
                [length + k for _ in range(q) for k in range(self.block)]
            ),  # row j*64+t -> length + j*64 + t
            "attention_sequence_lengths": i32([length + self.block] * q),
            "execution_phase_marker": torch.zeros(
                5, dtype=torch.int32, device="cuda"
            ),  # tree decoding
            "context_sequence_count_carrier": torch.zeros(0, dtype=torch.int32, device="cuda"),
        }  # only prefill/chunk phases count context sequences
        for i, pool in enumerate(self.pool):
            io[f"pool_{i}"] = pool
        for name in self.engine.outputs:  # present_<step>_<layer>: the same buffer as pool_<layer>
            if name.startswith("present_"):
                io[name] = self.pool[int(name.rsplit("_", 1)[1])]
        return self.engine(io)["actions"].reshape(b, t, 2).clone()


def action_expert(engine: Engine, config: dict, candidates: int):
    """The action engine built by thor/build_action.py from host/action_graph.py."""
    if "pool_0" not in engine.names:
        raise ValueError(f"{engine.path}: not an action-graph engine (no pool_0 binding)")
    return TreeActionExpert(engine, config, candidates)
