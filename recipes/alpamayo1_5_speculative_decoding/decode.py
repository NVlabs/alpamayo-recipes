# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 The DeepSpec Authors
# SPDX-License-Identifier: Apache-2.0 AND MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions adapted from DeepSpec, distributed under the following license:
#
# MIT License
#
# Copyright (c) 2026 The DeepSpec Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Greedy speculative verification adapted from DeepSpec.

Domino shift-label proposes B tokens; DFlash2 proposes B-1. The same verifier
commits the longest matching prefix and one target token, trimming at the
accepted stop. Rejected stops never terminate generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from transformers import DynamicCache

from .contracts import trim_committed

if TYPE_CHECKING:
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel

    from .target import Target


@dataclass(frozen=True)
class _TargetHead:
    """The only target attribute consumed by the pinned SpecForge sampler."""

    lm_head: torch.nn.Module


def propose_dflash2(
    draft: DFlash2DraftModel,
    hidden: torch.Tensor,
    block: torch.Tensor,
    head: torch.nn.Module,
) -> torch.Tensor:
    """Isolate SpecForge's private selector API; return B-1 draft token IDs."""
    proposed = draft._sample_draft_tokens(_TargetHead(head), hidden, block)
    expected_shape = (block.shape[0], draft.block_size - 1)
    if tuple(proposed.shape) != expected_shape or proposed.dtype != torch.long:
        raise ValueError("SpecForge DFlash2 sampler returned an unexpected token layout")
    return proposed


def context_features(output, layers):
    return torch.cat([output.hidden_states[layer + 1] for layer in layers], dim=-1)


def propose_domino(draft, hidden, anchor, embedding, head):
    """Run the causal GRU on previously PROPOSED tokens, never ground truth."""
    if not draft.shift_label:
        raise ValueError("Domino decoding requires shift_label=True")
    base = head(hidden)
    tokens = [anchor]
    state = None
    for index in range(hidden.shape[1]):
        _, state = draft.prefix_gru(embedding(tokens[-1]), state)
        logits = base[:, index : index + 1]
        if index >= draft.suffix_start:
            logits = logits + draft.embed_proj(
                torch.cat((hidden[:, index : index + 1], state.transpose(0, 1)), dim=-1)
            )
        tokens.append(logits.argmax(-1))
    return torch.cat(tokens, dim=1)


@torch.no_grad()
def speculative_decode(
    target: Target, draft: torch.nn.Module, inputs: dict[str, torch.Tensor]
) -> tuple[list[int], list[dict]]:
    """Commit exact greedy prefixes and log proposals, acceptance and emitted tokens."""
    session = target.decode_session(inputs)
    target_cache, draft_cache = DynamicCache(), DynamicCache()
    prompt = inputs["input_ids"]
    output = session(prompt, target_cache, hidden=True, last_only=True)
    anchor = output.logits[:, -1:].argmax(-1)
    generated, rounds = [int(anchor.item())], []
    hidden = context_features(output, draft.target_layer_ids)
    start = prompt.shape[1]
    block_size = int(draft.block_size)
    embedding, head = target.vlm.get_input_embeddings(), target.vlm.get_output_embeddings()
    while generated[-1] != target.stop_id and len(generated) < target.max_new_tokens:
        block = torch.full(
            (1, block_size), int(draft.mask_token_id), device=prompt.device, dtype=torch.long
        )
        block[:, :1] = anchor
        positions = torch.arange(
            draft_cache.get_seq_length(), start + block_size, device=prompt.device
        )[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = draft(
                target_hidden=hidden,
                noise_embedding=embedding(block),
                position_ids=positions,
                past_key_values=draft_cache,
                use_cache=True,
                is_causal=False,
            )
            draft_cache.crop(start)
            if draft.config.specdec_method == "domino":
                proposal = propose_domino(draft, features, anchor, embedding, head)
            else:
                proposed = propose_dflash2(draft, features, block, head)
                proposal = torch.cat((anchor, proposed), dim=1)
        output = session(proposal, target_cache, hidden=True)
        posterior = output.logits.argmax(-1)
        accepted = int((proposal[:, 1:] == posterior[:, :-1]).cumprod(1).sum().item())
        new_ids = proposal[0, 1 : accepted + 1].tolist() + [int(posterior[0, accepted])]
        committed = trim_committed(new_ids, target.max_new_tokens - len(generated), target.stop_id)
        rounds.append(
            {
                "proposed_ids": proposal[0, 1:].tolist(),
                "accepted_draft_tokens": accepted,
                "acceptance_length": accepted + 1,
                "committed_ids": committed,
                "emitted_tokens": len(committed),
                "proposed_tokens": proposal.shape[1] - 1,
            }
        )
        generated.extend(committed)
        if generated[-1] == target.stop_id or len(generated) >= target.max_new_tokens:
            break
        start += accepted + 1
        target_cache.crop(start)
        hidden = context_features(output, draft.target_layer_ids)[:, : accepted + 1]
        anchor = posterior[:, accepted : accepted + 1]
    return generated, rounds
