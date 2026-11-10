# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding for the recipe runtime: drafter loading, DFlash2 rounds, verify replay.

Mixed into ``run.Runtime``; uses its ``lm`` (LanguageModel), ``stream`` and tokenizer attributes.
"""

from __future__ import annotations

import time
from pathlib import Path

import tensorrt as trt
import torch

from .drafter_trt import TrtDrafter
from .engine import Engine


class Speculative:
    drafter = None
    spec_ms: dict = {}

    def load_draft_engine(self, engine_dir: Path) -> None:
        """Edge-LLM's exported DFlash2 draft engine (thor/build_draft.py) instead of the torch drafter."""
        if self.lm.taps is None:
            raise ValueError(
                "speculative decoding needs the tapped LLM engine (host/export.py llm)"
            )
        engine = Engine(trt.Runtime(trt.Logger(trt.Logger.WARNING)), engine_dir / "draft.engine")
        self.drafter = TrtDrafter(engine, engine_dir, self.lm.embedding, self.lm.max_pages * 128)
        if self.drafter.taps != self.lm.taps:
            raise ValueError(f"drafter taps {self.drafter.taps} != engine taps {self.lm.taps}")
        if self.drafter.block > self.lm.max_logits:
            raise ValueError(
                f"drafter block {self.drafter.block} exceeds the engine's logits rows {self.lm.max_logits}"
            )

    def speculate(
        self,
        logits,
        past: int,
        rope_delta: int,
        table,
        future_start: int,
        max_tokens: int,
    ):
        """DFlash2 rounds: the drafter proposes block - 1 tokens after the anchor (the last committed token); one chunked
        pass scores [anchor, drafts]; the longest matching prefix plus the first disagreeing argmax are committed. The
        drafter's context = the target's tap hidden states of every committed row (prefill rows first)."""
        lm, drafter = self.lm, self.drafter
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        drafter.reset()
        drafter.publish(lm.hidden_states)  # prompt rows -> drafter context (once per request)
        anchor, generated, rounds = int(lm.argmax(logits)), [], []
        torch.cuda.synchronize()
        self.spec_ms = {
            "context_ms": round((time.perf_counter() - t0) * 1000, 2),
            "publish_ms": 0.0,
            "tail_decode_ms": 0.0,
        }
        generated.append(anchor)
        tail_needed = True
        stop = lambda t: t == future_start or t in self.eos
        while not stop(anchor) and len(generated) < max_tokens:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            drafts = drafter.propose(anchor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            block = [anchor] + drafts
            scores = lm.verify(lm.embed(block), lm.text_rope(past, len(block)), table, past)
            predicted = lm.argmax(scores).tolist()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            accepted = 0
            while (
                accepted < len(drafts)
                and drafts[accepted] == predicted[accepted]
                and not stop(drafts[accepted])
            ):
                accepted += 1
            # committed: the accepted drafts, then either the stop token the drafter proposed (the target agrees; its K/V
            # row was written by this pass, so no tail decode) or the target's correction (K/V written by the next pass)
            stop_from_draft = accepted < len(drafts) and drafts[accepted] == predicted[accepted]
            committed = drafts[:accepted] + [
                drafts[accepted] if stop_from_draft else predicted[accepted]
            ]
            rows = (
                1 + accepted + (1 if stop_from_draft else 0)
            )  # K/V rows now valid: anchor + accepted (+ drafted stop)
            drafter.publish(lm.hidden_states[:rows])
            torch.cuda.synchronize()
            self.spec_ms["publish_ms"] += (time.perf_counter() - t2) * 1000
            past += rows
            generated.extend(committed)
            rounds.append(
                {
                    "draft_ms": round((t1 - t0) * 1000, 2),
                    "verify_ms": round((t2 - t1) * 1000, 2),
                    "accepted": accepted,
                    "committed": len(committed),
                }
            )
            anchor = committed[-1]
            tail_needed = not stop_from_draft
        if stop(anchor) and anchor == future_start and tail_needed:
            t3 = time.perf_counter()
            lm.decode(
                lm.embed([anchor]), lm.text_rope(past, 1), table, past
            )  # its K/V for the expert
            past += 1
            torch.cuda.synchronize()
            self.spec_ms["tail_decode_ms"] = round((time.perf_counter() - t3) * 1000, 2)
        self.spec_ms["publish_ms"] = round(self.spec_ms["publish_ms"], 2)
        return generated, past, rounds
