# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the Domino drafter round to ``<output-dir>/domino``.

One call projects the new target ``features`` rows into the drafter context (``fresh_context``,
kept by the runner and passed back as ``past``), runs the five drafter layers over 16 proposal
rows with non-causal attention over [context; proposal], and emits 16 draft ``tokens`` through
the GRU correction head over a 4096-token compact vocabulary.

Row counts are dynamic (``fresh``, ``past_rows``, ``context_rows``); the build step creates the
seed / sliding / incremental optimization profiles. With ``fp8``: FP16 Q/DQ scales on the context
projection and FFNs (drafter weights rounded to FP16, then weight scales amax / 448; activation
scales from the bundle's ``fp8-scales.json``). Everything else, and all of ``fp16``, is FP16; GRU and SiLU
math in FP32.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from onnx import TensorProto as T
from safetensors import safe_open

from alpamayo1_x_edgellm.export.graph import Graph, add_model_arguments, open_bundle

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, PROPOSALS, GRU = 2048, 5, 16, 8, 128, 16, 1024
# FP16: the FP8 scale layout with every activation scale absent (plain FP16 GEMMs).
NO_SCALES = {"target_input": None, "context_input": None, "ffn_input": [None] * LAYERS, "down_input": [None] * LAYERS}


def gru_step(g: Graph, state: str, inputs: str, hidden: str) -> str:
    """PyTorch GRU cell (gates r, z, n; reset applied after the hidden projection), FP32 math."""
    state, inputs, hidden = (g.cast(v, T.FLOAT) for v in (state, inputs, hidden))
    part = lambda v, i: g.slice(v, i * GRU, (i + 1) * GRU, 1)
    r = g.op("Sigmoid", g.op("Add", part(inputs, 0), part(hidden, 0)))
    z = g.op("Sigmoid", g.op("Add", part(inputs, 1), part(hidden, 1)))
    n = g.op("Tanh", g.op("Add", part(inputs, 2), g.op("Mul", r, part(hidden, 2))))
    blended = g.op("Add", g.op("Mul", g.op("Sub", g.const(1.0, "float32"), z), n), g.op("Mul", z, state))
    return g.cast(blended, T.FLOAT16)


def export_domino(drafter: Path, compact_to_global: list[int], scales: dict | None, output: Path) -> Path:
    """``scales`` None exports the FP16 graph."""
    ckpt = safe_open(str(drafter), "pt")
    get = ckpt.get_tensor
    g = Graph(output)
    linear = g.fp8_half_linear if scales is not None else (lambda x, w, _scale: g.linear(x, w))
    features = g.input("features", ["fresh", 5 * HIDDEN])
    past = g.input("past", ["past_rows", 5 * HIDDEN])
    noise = g.input("noise", [PROPOSALS, HIDDEN])
    context_rope = g.input("context_rope", ["context_rows", 128], T.FLOAT)
    proposal_rope = g.input("proposal_rope", [PROPOSALS, 128], T.FLOAT)
    hidden_gamma = get("draft.hidden_norm.weight")

    # Context projection: target features -> drafter hidden -> per-layer [K | V].
    s = scales or NO_SCALES
    target = linear(features, get("draft.fc.weight").half(), s["target_input"])
    context_weight = torch.cat([get(f"draft.layers.{i}.self_attn.{r}_proj.weight") for i in range(LAYERS) for r in "kv"])
    fresh = linear(g.rms_norm_fp32(target, hidden_gamma), context_weight.half(), s["context_input"])
    g.output(fresh, "fresh_context", ["fresh", 5 * HIDDEN])
    context = g.op("Concat", past, fresh, axis=0)
    rope = lambda r: tuple(g.reshape(g.slice(r, a, a + 64, 1), [-1, 1, 64]) for a in (0, 64))
    (cc, cs), (pc, ps) = rope(context_rope), rope(proposal_rope)

    x = noise
    for i in range(LAYERS):
        p = f"draft.layers.{i}."
        qkv = g.linear(g.rms_norm_fp32(x, get(p + "input_layernorm.weight")),
                       torch.cat([get(p + f"self_attn.{r}_proj.weight") for r in "qkv"]))
        k_gamma = get(p + "self_attn.k_norm.weight")
        q = g.rotate(g.rms_norm_fp32(g.reshape(g.slice(qkv, 0, HIDDEN, 1), [PROPOSALS, HEADS, HEAD_DIM]),
                                     get(p + "self_attn.q_norm.weight")), pc, ps)
        k = g.rotate(g.rms_norm_fp32(g.reshape(g.slice(qkv, HIDDEN, HIDDEN + 1024, 1), [PROPOSALS, KV_HEADS, HEAD_DIM]),
                                     k_gamma), pc, ps)
        v = g.reshape(g.slice(qkv, HIDDEN + 1024, 2 * HIDDEN, 1), [PROPOSALS, KV_HEADS, HEAD_DIM])
        base = i * HIDDEN
        ck = g.rotate(g.rms_norm_fp32(g.reshape(g.slice(context, base, base + 1024, 1), [-1, KV_HEADS, HEAD_DIM]),
                                      k_gamma), cc, cs)
        cv = g.reshape(g.slice(context, base + 1024, base + HIDDEN, 1), [-1, KV_HEADS, HEAD_DIM])
        k, v = g.op("Concat", ck, k, axis=0), g.op("Concat", cv, v, axis=0)
        q, k, v = (g.op("Transpose", g.reshape(t, [1, -1, n, HEAD_DIM]), perm=[0, 2, 1, 3])
                   for t, n in ((q, HEADS), (k, KV_HEADS), (v, KV_HEADS)))
        a = g.op("Attention", q, k, v, is_causal=0, scale=HEAD_DIM ** -0.5, TRT_decomposable=1)
        a = g.reshape(g.op("Transpose", a, perm=[0, 2, 1, 3]), [PROPOSALS, HIDDEN])
        x = g.op("Add", x, g.linear(a, get(p + "self_attn.o_proj.weight")))
        h = g.rms_norm_fp32(x, get(p + "post_attention_layernorm.weight"))
        s_in, s_down = s["ffn_input"][i], s["down_input"][i]
        gate = linear(h, get(p + "mlp.gate_proj.weight").half(), s_in)
        up = linear(h, get(p + "mlp.up_proj.weight").half(), s_in)
        swiglu = g.op("Mul", g.op("Mul", gate, g.op("Sigmoid", gate)), up)
        x = g.op("Add", x, linear(swiglu, get(p + "mlp.down_proj.weight").half(), s_down))

    # Head over the compact vocabulary plus the 16-step GRU correction.
    index = torch.tensor(compact_to_global)
    compact_embedding = get("embed_tokens.weight")[index]
    hidden = g.rms_norm_fp32(x, hidden_gamma)
    logits = g.linear(hidden, compact_embedding)
    embed_proj = get("draft.embed_proj.0.weight")  # [256, 2048 hidden | 1024 GRU state]
    static = g.linear(hidden, embed_proj[:, :HIDDEN])
    input_weight = get("draft.prefix_gru.weight_ih_l0").t().half()
    w_input, w_hidden, w_state, w_logit, gate_table = (g.weight(t.contiguous()) for t in (
        input_weight,
        get("draft.prefix_gru.weight_hh_l0").t().half(),
        embed_proj[:, HIDDEN:].t().half(),
        get("draft.embed_proj.2.weight")[index].t().half(),
        (compact_embedding.half().float() @ input_weight.float()).half()))
    mapping = g.const(compact_to_global, "int64")
    state = g.const([[0.0] * GRU], "float16")
    inputs = g.op("MatMul", g.slice(noise, 0, 1, 0), w_input)
    tokens = []
    for step in range(PROPOSALS):
        state = gru_step(g, state, inputs, g.op("MatMul", state, w_hidden))
        mixed = g.op("Add", g.cast(g.slice(static, step, step + 1, 0), T.FLOAT),
                     g.cast(g.op("MatMul", state, w_state), T.FLOAT))
        mixed = g.cast(g.op("Mul", mixed, g.op("Sigmoid", mixed)), T.FLOAT16)
        scores = g.op("Add", g.cast(g.slice(logits, step, step + 1, 0), T.FLOAT),
                      g.cast(g.op("MatMul", mixed, w_logit), T.FLOAT))
        selected = g.op("ArgMax", scores, axis=1, keepdims=0, select_last_index=0)
        tokens.append(g.op("Gather", mapping, selected, axis=0))
        if step < PROPOSALS - 1:
            inputs = g.op("Gather", gate_table, selected, axis=0)
    g.output(g.op("Concat", *tokens, axis=0), "tokens", [PROPOSALS], T.INT64)
    return g.save("alpamayo_domino_round")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle, scales = open_bundle(args, "domino")
    print(export_domino(bundle.path("drafter"), bundle.vocab(), scales, args.output_dir / "domino"))


if __name__ == "__main__":
    main()
