# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Cosmos3-Edge language model on the stock ``llm.engine`` (Edge-LLM v0.11 ragged contract).

One engine, two optimization profiles: 0 = context (prefill, up to max_input_len tokens), 1 = generation (one token
per sequence). Token rows are ragged (``query_start_offsets`` delimit sequences); every attention layer reads and
writes a fixed paged K/V pool ``[2, pages, 128, kv_heads, head_dim]`` through a per-sequence page table
``[N, 2, max_pages]`` (K page ids, then V ids = K ids + pages; -1 = unused). The shape-only ``execution_phase_marker``
selects the kernel path by its length (1 = context prefill, 3 = autoregressive decode) and
``context_sequence_count_carrier`` has one entry per sequence in the context phase, none in decode.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .engine import Engine
from .rope import cos_sin

PAGE = 128
PREFILL, CHUNK, DECODE = (
    1,
    2,
    3,
)  # ExecutionPhase marker lengths: kContextPrefill / kContextChunk / kAutoregressiveDecode
UNUSED_PAGE = -1


class LanguageModel:
    def __init__(self, engine: Engine, engine_dir: Path, config: dict):
        self.engine = engine
        self.layers = config["num_hidden_layers"]
        self.hidden = config["hidden_size"]
        self.theta = float(config["rope_theta"])
        self.section = tuple(config["rope_scaling"]["mrope_section"])
        self.past = [f"past_key_values_{i}" for i in range(self.layers)]
        self.present = [f"present_key_values_{i}" for i in range(self.layers)]
        for name in self.past + self.present:
            if name not in engine.names:
                raise ValueError(f"llm engine has no {name}")
        pool_shape = engine.shape(self.past[0])  # [2, pages, 128, kv_heads, head_dim]
        self.pages = pool_shape[1]
        self.max_pages = engine.shape("kv_page_table")[2]
        self.max_tokens = engine.profile_shape("inputs_embeds", 0)[2][0]
        self.pool = [
            torch.zeros(pool_shape, dtype=engine.dtype[self.past[0]], device="cuda")
            for _ in range(self.layers)
        ]
        self.embedding = load_file(str(engine_dir / "embedding.safetensors"))["embedding"].to(
            "cuda"
        )
        self.vocab = self.embedding.shape[0]
        # reduced output vocabulary (host/vocab.py): logits index -> token id
        vocab_map = engine_dir / "vocab_map.safetensors"
        self.vocab_map = (
            load_file(str(vocab_map))["vocab_map"].long().cuda() if vocab_map.is_file() else None
        )
        self.int = {n: engine.dtype[n] for n in engine.inputs}
        # host/export.py llm: ``hidden_states`` = the tap layers' outputs per token (drafter context)
        self.taps = (
            config.get("hidden_states_tap_layers") if "hidden_states" in engine.outputs else None
        )
        self.hidden_states = None  # [n, len(taps) * hidden] of the last forward
        self.max_logits = engine.profile_shape("logits_indices", 0)[2][0]
        # Per-step glue stays off the host: token ids and the eight int32 metadata values go through pinned staging
        # buffers (one copy each), positions / logits_indices are device aranges, the RoPE rows come from a per-request
        # table (``begin_request``); the shape-only carriers are constant per phase.
        rows = max(self.max_tokens, self.max_logits)
        self._ids_host = torch.empty(rows, dtype=torch.int64).pin_memory()
        self._ids_dev = torch.empty(rows, dtype=torch.int64, device="cuda")
        self._meta_host = torch.empty(8, dtype=torch.int32).pin_memory()
        self._meta_dev = torch.zeros(8, dtype=torch.int32, device="cuda")
        self._positions = torch.empty(rows, dtype=self.int["positions"], device="cuda")
        self._logits_indices = torch.empty(
            self.max_logits, dtype=self.int["logits_indices"], device="cuda"
        )
        self._phase = {
            phase: torch.zeros(phase, dtype=self.int["execution_phase_marker"], device="cuda")
            for phase in (PREFILL, CHUNK, DECODE)
        }
        self._carrier = {
            n: torch.zeros(n, dtype=self.int["context_sequence_count_carrier"], device="cuda")
            for n in (0, 1)
        }
        self._empty = {
            name: torch.zeros(0, dtype=self.int[name], device="cuda")
            for name in engine.inputs
            if name not in self.past
            and name
            not in (
                "inputs_embeds",
                "rope_rotary_cos_sin",
                "positions",
                "query_start_offsets",
                "query_lengths",
                "past_lengths",
                "attention_sequence_lengths",
                "state_indices",
                "execution_phase_marker",
                "context_sequence_count_carrier",
                "kv_page_table",
                "logits_indices",
            )
        }
        self.rope_table = None

    def begin_request(self, rope_delta: int, total: int) -> None:
        """RoPE rows for text positions rope_delta + 0 .. total - 1 (token index = row), computed once on the GPU."""
        pos = torch.arange(rope_delta, rope_delta + total, dtype=torch.int64, device="cuda")[
            None
        ].expand(3, -1)
        self.rope_table = cos_sin(pos, self.theta, self.section)

    def text_rope(self, first: int, count: int) -> torch.Tensor:
        """RoPE rows of the ``count`` text tokens at token indices first .. first + count - 1 (a view)."""
        if self.rope_table is None or first + count > self.rope_table.shape[0]:
            raise ValueError(
                f"rope rows {first}..{first + count} outside the request table; call begin_request"
            )
        return self.rope_table[first : first + count]

    def embed(self, token_ids) -> torch.Tensor:
        """Embedding rows of a Python list or tensor of token ids (ids staged through pinned memory)."""
        if isinstance(token_ids, torch.Tensor) and token_ids.is_cuda:
            return self.embedding[token_ids]
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        n = len(ids)
        self._ids_host[:n] = torch.tensor(ids, dtype=torch.int64)
        self._ids_dev[:n].copy_(self._ids_host[:n], non_blocking=True)
        return self.embedding[self._ids_dev[:n]]

    def page_table(self, pages: list[int]) -> torch.Tensor:
        row = torch.full((2, self.max_pages), UNUSED_PAGE, dtype=self.int["kv_page_table"])
        row[0, : len(pages)] = torch.tensor(pages)
        row[1, : len(pages)] = torch.tensor(pages) + self.pages
        return row[None].cuda()

    def argmax(self, logits: torch.Tensor) -> torch.Tensor:
        """Greedy token ids of ``logits`` [..., logits_vocab], mapped back to the full vocabulary."""
        idx = torch.argmax(logits, dim=-1)
        return self.vocab_map[idx] if self.vocab_map is not None else idx

    def forward(
        self,
        embeds: torch.Tensor,
        rope: torch.Tensor,
        page_table: torch.Tensor,
        past: int,
        phase: int,
        logits_rows: int = 1,
    ) -> torch.Tensor:
        """One sequence: ``embeds`` [n, hidden] fp16 at positions past .. past + n - 1 with their ``rope``
        [n, head_dim] rows; returns the logits of the last ``logits_rows`` rows, [logits_rows, vocab] float32
        (``[vocab]`` when logits_rows == 1). With a tapped engine ``self.hidden_states`` holds the tap outputs."""
        n = embeds.shape[0]
        if logits_rows > self.max_logits:
            raise ValueError(
                f"{logits_rows} logits rows exceed the engine's {self.max_logits} (llm_build --maxBatchSize)"
            )
        meta = self._meta_host
        meta[0], meta[1], meta[2], meta[3], meta[4], meta[5] = 0, n, n, past, past + n, 0
        self._meta_dev.copy_(meta, non_blocking=True)  # one copy for the six scalars below
        positions = self._positions[:n]
        torch.arange(past, past + n, dtype=positions.dtype, device="cuda", out=positions)
        logits_indices = self._logits_indices[:logits_rows]
        torch.arange(
            n - logits_rows, n, dtype=logits_indices.dtype, device="cuda", out=logits_indices
        )
        io = {
            "inputs_embeds": embeds,
            "rope_rotary_cos_sin": rope,
            "positions": positions,
            "query_start_offsets": self._meta_dev[0:2],
            "query_lengths": self._meta_dev[2:3],
            "past_lengths": self._meta_dev[3:4],
            "attention_sequence_lengths": self._meta_dev[4:5],
            "state_indices": self._meta_dev[5:6],
            "execution_phase_marker": self._phase[phase],
            "context_sequence_count_carrier": self._carrier[
                0 if phase == DECODE else 1
            ],  # context phases: 1 per sequence
            "kv_page_table": page_table,
            "logits_indices": logits_indices,
            **self._empty,
        }
        for past_name, present_name, pool in zip(self.past, self.present, self.pool):
            io[past_name] = io[present_name] = pool
        out = self.engine(io, profile=1 if phase == DECODE else 0)
        if self.taps is not None:
            self.hidden_states = out["hidden_states"]
        return out["logits"][0] if logits_rows == 1 else out["logits"]

    def prefill(self, embeds, rope, page_table):
        if embeds.shape[0] > self.max_tokens:
            raise ValueError(
                f"{embeds.shape[0]} prompt tokens exceed the engine's {self.max_tokens}"
            )
        return self.forward(embeds, rope, page_table, 0, PREFILL)

    def decode(self, embed, rope, page_table, past):
        return self.forward(embed, rope, page_table, past, DECODE)

    def verify(self, embeds, rope, page_table, past):
        """Append n tokens after ``past`` rows in one causal pass (chunked-prefill kernel path) and return every
        row's next-token logits [n, vocab]: the speculative-decoding verify step. The K/V rows of rejected
        tokens stay in the pool and are overwritten by the next pass."""
        return self.forward(embeds, rope, page_table, past, CHUNK, logits_rows=embeds.shape[0])
