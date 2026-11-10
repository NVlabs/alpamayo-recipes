# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Edge vision encoder (SigLIP2 tower + 2x2 patch merger) on the stock ``visual.engine``.

Packs images the way the stock runner does (``transposeToPatchQwenViT`` with channel-last patches, 2x2 merge-block
order) and precomputes the learned position-embedding taps (``fast_pos_embed_interpolate``: bilinear,
align_corners=False, over the 16x16 table). Encoded images are cached by name so a rolling runtime encodes only the
frames it has not seen (all cache misses go through one enqueue).
"""

from __future__ import annotations

import torch

from .engine import Engine

PATCH, MERGE = 16, 2
MEAN, STD = 0.5, 0.5
INPUTS = (
    "input",
    "cu_seqlens",
    "fast_pos_embed_idx",
    "fast_pos_embed_weight",
    "max_seqlen_carrier",
)


def patchify(image: torch.Tensor) -> torch.Tensor:
    """uint8 [H, W, 3] (H, W multiples of 32) -> fp16 [H/16 * W/16, 768] in (block row, block col, 2, 2) order,
    each patch flattened (channel, row, col).

    Channel-first is what the exported graph expects: ``tensorrt_edgellm`` permutes the SigLIP2 patch-embedding
    columns from the HF (row, col, channel) layout to (channel, row, col) at export
    (``_adapt_patch_embedding_weight_for_chw_input``). The stock C++ Cosmos3-Edge runner nevertheless packs
    channel-last (``vitPatchChannelLast``), which scrambles the patch embedding; measured against the HF vision
    tower on this clip, channel-last gives feature correlation 0.59, channel-first 0.995 and higher.
    """
    h, w, c = image.shape
    if h % (PATCH * MERGE) or w % (PATCH * MERGE):
        raise ValueError(f"image {h}x{w} is not a multiple of {PATCH * MERGE}")
    x = (image.float() / 255.0 - MEAN) / STD
    x = x.view(h // (PATCH * MERGE), MERGE, PATCH, w // (PATCH * MERGE), MERGE, PATCH, c)
    return x.permute(0, 3, 1, 4, 6, 2, 5).reshape(-1, PATCH * PATCH * c).half()


def pos_embed_taps(rows: int, cols: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """4-tap bilinear gather (idx [4, rows*cols] int64, weight [4, rows*cols] fp16) of the side x side learned
    table onto a rows x cols patch grid, emitted in 2x2 merge-block order (port of the exporter's reference)."""

    def axis(n):
        v = ((torch.arange(n, dtype=torch.float32) + 0.5) * (side / n) - 0.5).clamp(0, side - 1)
        lo = v.floor()
        return lo.long(), (lo + 1).clamp(max=side - 1).long(), v - lo

    h_lo, h_hi, dh = axis(rows)
    w_lo, w_hi, dw = axis(cols)
    mh, mw = rows // MERGE, cols // MERGE

    def blocks(vh, vw):
        return (vh.reshape(mh, 1, MERGE, 1) + vw.reshape(1, mw, 1, MERGE)).flatten()

    def weights(ah, aw):
        return (ah.reshape(mh, 1, MERGE, 1) * aw.reshape(1, mw, 1, MERGE)).flatten()

    idx = torch.stack(
        [
            blocks(h_lo * side, w_lo),
            blocks(h_lo * side, w_hi),
            blocks(h_hi * side, w_lo),
            blocks(h_hi * side, w_hi),
        ]
    )
    weight = torch.stack(
        [weights(1 - dh, 1 - dw), weights(1 - dh, dw), weights(dh, 1 - dw), weights(dh, dw)]
    )
    return idx, weight.half()


class VisionEncoder:
    def __init__(self, engine: Engine, config: dict):
        self.engine = engine
        missing = [n for n in INPUTS if n not in engine.inputs]
        if missing or engine.outputs != ["output"]:
            raise ValueError(
                f"visual engine bindings {engine.names} are not the Cosmos3-Edge contract ({missing})"
            )
        self.side = int(round(config["vision_config"]["num_patches"] ** 0.5))
        self.hidden = engine.shape("output")[1]
        self.max_patches = engine.profile_shape("input")[2][0]
        self.cache: dict[str, torch.Tensor] = {}
        self.taps: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.calls = 0

    def encode(self, images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """name -> uint8 [H, W, 3] GPU images; returns name -> [tokens, hidden] fp16 (cached names are not re-run)."""
        todo = [n for n in images if n not in self.cache]
        if todo:
            patches, idx, weight, lengths = [], [], [], []
            for name in todo:
                img = images[name]
                rows, cols = img.shape[0] // PATCH, img.shape[1] // PATCH
                if (rows, cols) not in self.taps:
                    self.taps[(rows, cols)] = tuple(
                        t.cuda() for t in pos_embed_taps(rows, cols, self.side)
                    )
                patches.append(patchify(img))
                idx.append(self.taps[(rows, cols)][0])
                weight.append(self.taps[(rows, cols)][1])
                lengths.append(rows * cols)
            total = sum(lengths)
            if total > self.max_patches:
                raise ValueError(
                    f"{total} patches exceed the visual engine's {self.max_patches}; encode fewer images"
                )
            cu = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).to(torch.int32).cuda()
            io = {
                "input": torch.cat(patches),
                "cu_seqlens": cu,
                "fast_pos_embed_idx": torch.cat(idx, dim=1),
                "fast_pos_embed_weight": torch.cat(weight, dim=1),
                "max_seqlen_carrier": torch.zeros(
                    max(lengths), dtype=self.engine.dtype["max_seqlen_carrier"], device="cuda"
                ),
            }
            out = self.engine(io)["output"]
            self.calls += 1
            start = 0
            for name, n in zip(todo, lengths):
                self.cache[name] = out[start // MERGE**2 : (start + n) // MERGE**2].clone()
                start += n
        return {n: self.cache[n] for n in images}

    def forget(self, names) -> None:
        for n in names:
            self.cache.pop(n, None)
