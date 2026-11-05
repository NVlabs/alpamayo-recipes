# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the speculative verifier to ``<output-dir>/verify/{q17,q1}``.

``q17`` scores the 16 drafted tokens plus the anchor against the persistent KV cache and returns
the greedy ``posterior`` [17] and the drafter ``features`` [1, 17, 10240]. ``q1`` appends the final
stop token to the cache (27 attention layers plus layer-27 K/V ``fresh_27`` [2, 8, 1, 128], which the
runner writes into the pool; no head).

Attention is TensorRT Edge-LLM's ``AttentionPlugin`` in tree mode (one sequence of 17 or 1 new rows,
causal among the new rows, FP16 Q/K/V) over the paged FP16 pool ``pool_i`` [2, 40, 128, 8, 128],
updated in place (``present_i`` aliases ``pool_i``; the plugin writes the round's K/V rows at
``past_lengths``). The page table is the constant 0..24 (the language rows). Projections are FP16;
with ``fp8``, ``q17`` runs the FFN and LM head in FP8 (weight scales amax / 448, activation scales
from the bundle's ``fp8-scales.json``). ``q1``, and everything with ``fp16``, is FP16 throughout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from onnx import TensorProto as T

from alpamayo1_x_edgellm.export.graph import (HEAD_DIM, HEADS, KV_HEADS, NUM_PAGES, POOL_SHAPE, Graph, TreeAttention,
                                                add_model_arguments, causal_mask, fp8_codes, open_bundle, open_checkpoint)

HIDDEN, LAYERS = 2048, 28
LANGUAGE_PAGES = 25  # pages 0..24: 3006 prompt rows + up to 128 generated rows
FEATURE_LAYERS = (1, 7, 13, 19, 25)
PREFIX = "vlm.model.language_model."


def export_verify(checkpoint: Path, scales: dict | None, output: Path, rows: int) -> Path:
    """``scales`` None exports the FP16 graph."""
    head = rows == 17  # q17 scores drafts (features + posterior); q1 only appends K/V
    fp8 = head and scales is not None
    if rows not in (17, 1):
        raise ValueError("verifier rows must be 17 or 1")
    ckpt = open_checkpoint(checkpoint)
    get = lambda key: ckpt.get_tensor(PREFIX + key)
    f32 = np.float32
    g = Graph(output)
    x = g.input("activation", [rows, HIDDEN])
    rope = g.input("rope", [rows, 128], T.FLOAT)
    attention_layers = LAYERS if head else LAYERS - 1
    pools = [g.input(f"pool_{i}", POOL_SHAPE) for i in range(attention_layers)]
    pages = np.arange(LANGUAGE_PAGES)
    attention = TreeAttention(g, 1, rows, LANGUAGE_PAGES, causal_mask(rows),
                              page_table=np.stack((pages, pages + NUM_PAGES)))
    cos, sin = (g.reshape(g.slice(rope, a, a + 64, 1), [rows, 1, 64]) for a in (0, 64))

    def kv(h: str, p: str, with_query: bool):
        roles = ("q", "k", "v") if with_query else ("k", "v")
        packed = g.linear(h, torch.cat([get(p + f"self_attn.{r}_proj.weight") for r in roles]))
        begin = HIDDEN if with_query else 0
        q = None
        if with_query:
            q = g.reshape(g.slice(packed, 0, HIDDEN, 1), [rows, HEADS, HEAD_DIM])
            q = g.rotate(g.rms_norm_fp32(q, get(p + "self_attn.q_norm.weight")), cos, sin)
        k = g.reshape(g.slice(packed, begin, begin + 1024, 1), [rows, KV_HEADS, HEAD_DIM])
        k = g.rotate(g.rms_norm_fp32(k, get(p + "self_attn.k_norm.weight")), cos, sin)
        v = g.reshape(g.slice(packed, begin + 1024, begin + 2048, 1), [rows, KV_HEADS, HEAD_DIM])
        return q, k, v

    features, presents = [], []
    for i in range(LAYERS):
        p = f"layers.{i}."
        h = g.rms_norm_fp32(x, get(p + "input_layernorm.weight"))
        q, k, v = kv(h, p, with_query=i < attention_layers)
        if i == attention_layers:  # q1: the last layer only publishes its K/V, [2, 8, rows, 128]
            k, v = (g.op("Transpose", g.reshape(t, [1, rows, KV_HEADS, HEAD_DIM]), perm=[0, 2, 1, 3]) for t in (k, v))
            g.output(g.op("Concat", k, v, axis=0), f"fresh_{i}", [2, KV_HEADS, rows, HEAD_DIM])
            break
        qkv = g.op("Concat", *(g.reshape(t, [rows, n * HEAD_DIM]) for t, n in ((q, HEADS), (k, KV_HEADS), (v, KV_HEADS))),
                   axis=1)
        # present_i is the in-place updated pool_i: a direct plugin output, never copied.
        out, present = attention(qkv, pools[i], f"present_{i}")
        presents.append(present)
        x = g.op("Add", x, g.linear(g.reshape(out, [rows, HIDDEN]), get(p + "self_attn.o_proj.weight")))

        h = g.rms_norm_fp32(x, get(p + "post_attention_layernorm.weight"))
        weights = [get(p + f"mlp.{r}_proj.weight") for r in ("gate", "up", "down")]
        if fp8:
            s_in, d = f32(scales["ffn_input"][i]), f32(scales["down_input"][i])
            (cg, sg), (cu, su), (cd, sd) = (fp8_codes(w) for w in weights)
            gate = g.fp8_codes_linear(h, cg, s_in, s_in * f32(sg))
            up = g.fp8_codes_linear(h, cu, s_in, s_in * f32(su) / d)  # up output pre-encoded by 1/d
            swiglu = g.op("Mul", g.op("Mul", gate, g.op("Sigmoid", gate)), up)
            down = g.cast(g.fp8_codes_linear(swiglu, cd, 1.0, d * f32(sd)), T.FLOAT16)
        else:
            gate, up = g.linear(h, weights[0]), g.linear(h, weights[1])
            down = g.linear(g.op("Mul", g.op("Mul", gate, g.op("Sigmoid", gate)), up), weights[2])
        x = g.op("Add", x, down)
        if head and i in FEATURE_LAYERS:
            features.append(x)

    if head:
        g.output(g.reshape(g.op("Concat", *features, axis=1), [1, rows, HIDDEN * len(FEATURE_LAYERS)]),
                 "features", [1, rows, HIDDEN * len(FEATURE_LAYERS)])
        h = g.rms_norm_fp32(x, get("norm.weight"))
        if fp8:
            codes, s_head = fp8_codes(get("embed_tokens.weight"))
            s_head_in = f32(scales["head_input"])
            logits = g.fp8_codes_linear(h, codes, s_head_in, s_head_in * f32(s_head))
        else:
            logits = g.linear(h, get("embed_tokens.weight"))
        g.output(g.op("ArgMax", g.cast(logits, T.FLOAT16), axis=1, keepdims=0), "posterior", [rows], T.INT64)
    for i, present in enumerate(presents):
        g.output(present, f"present_{i}", POOL_SHAPE)
    return g.save(f"alpamayo_verify_q{rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle, scales = open_bundle(args, "verify")
    for rows in (17, 1):
        print(export_verify(bundle.target, scales, args.output_dir / "verify" / f"q{rows}", rows))


if __name__ == "__main__":
    main()
