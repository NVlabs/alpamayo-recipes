# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the full-attention language prefill (one request, 3006 rows) to ``<output-dir>/prefill``.

Inputs: FP16 ``hidden`` [3006, 2048] and three ``deepstack`` adds, FP32 ``rope`` [3006, 128]
(cos | sin). Outputs: per-layer ``kv_i`` [2, 24, 128, 8, 128], the first 24 pages of the layer's paged
K/V pool (3006 rows, zero-padded to 3072; the runner copies them into the pool), five ``features``
taps [3006, 10240] for the drafter, and last-row ``logits``.

Precision: FP16 Q/K/V/O, native causal ``Attention`` and LM head. FFN with ``fp8``:
``AlpamayoFusedGateUpPlugin`` (TensorRT Edge-LLM) for gate/up + SwiGLU with FP8 output, Q/DQ FP8 down GEMM (weight scales
amax / 448 from the checkpoint, activation scales from the bundle's ``fp8-scales.json``); with ``fp16``:
FP16 gate/up/down GEMMs, no plugin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from onnx import TensorProto as T

from alpamayo1_x_edgellm.export.graph import (HEAD_DIM, HEADS, KV_HEADS, PAGE, Graph, add_model_arguments, fp8_codes,
                                                open_bundle, open_checkpoint)

ROWS, HIDDEN, LAYERS = 3006, 2048, 28
PAGES = -(-ROWS // PAGE)  # 24 pool pages hold the prompt
DEEPSTACK_LAYERS = 3
FEATURE_LAYERS = (1, 7, 13, 19, 25)
PREFIX = "vlm.model.language_model."


def pack_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """AlpamayoFusedGateUpPlugin weight layout: 24 blocks of [gate 128, up 128, gate 128, up 128] rows."""
    g, u = (t.view(torch.uint8).reshape(24, 2, 128, HIDDEN) for t in (gate, up))
    return torch.stack((g[:, 0], u[:, 0], g[:, 1], u[:, 1]), 1).reshape(-1, HIDDEN).view(torch.float8_e4m3fn)


def export_prefill(checkpoint: Path, scales: dict | None, output: Path) -> Path:
    """``scales`` None exports the FP16 graph."""
    fp8 = scales is not None
    if fp8 and (len(scales["ffn_input"]) != LAYERS or len(scales["down_input"]) != LAYERS):
        raise ValueError("prefill scales need 28 ffn_input and down_input values")
    ckpt = open_checkpoint(checkpoint)
    get = lambda key: ckpt.get_tensor(PREFIX + key)
    g = Graph(output)
    x = g.input("hidden", [ROWS, HIDDEN])
    deepstack = [g.input(f"deepstack_{i}", [ROWS, HIDDEN]) for i in range(DEEPSTACK_LAYERS)]
    rope = g.input("rope", [ROWS, 2 * 64], T.FLOAT)
    cos, sin = (g.reshape(g.slice(rope, a, a + 64, 1), [ROWS, 1, 64]) for a in (0, 64))
    features = []
    for i in range(LAYERS):
        p = f"layers.{i}."
        h = g.rms_norm(x, get(p + "input_layernorm.weight"))
        q = g.reshape(g.linear(h, get(p + "self_attn.q_proj.weight")), [ROWS, HEADS, HEAD_DIM])
        k = g.reshape(g.linear(h, get(p + "self_attn.k_proj.weight")), [ROWS, KV_HEADS, HEAD_DIM])
        v = g.reshape(g.linear(h, get(p + "self_attn.v_proj.weight")), [ROWS, KV_HEADS, HEAD_DIM])
        q = g.rotate(g.rms_norm(q, get(p + "self_attn.q_norm.weight")), cos, sin)
        k = g.rotate(g.rms_norm(k, get(p + "self_attn.k_norm.weight")), cos, sin)
        g.output(g.paged_kv(k, v, ROWS), f"kv_{i}", [2, PAGES, PAGE, KV_HEADS, HEAD_DIM])
        q, k, v = (g.op("Transpose", g.reshape(t, [1, ROWS, n, HEAD_DIM]), perm=[0, 2, 1, 3])
                   for t, n in ((q, HEADS), (k, KV_HEADS), (v, KV_HEADS)))
        a = g.op("Attention", q, k, v, is_causal=1, scale=HEAD_DIM ** -0.5, TRT_decomposable=1)
        a = g.reshape(g.op("Transpose", a, perm=[0, 2, 1, 3]), [ROWS, HIDDEN])
        x = g.op("Add", x, g.linear(a, get(p + "self_attn.o_proj.weight")))

        h = g.rms_norm(x, get(p + "post_attention_layernorm.weight"))
        if fp8:
            # codes = Q(norm, s_in) -> AlpamayoFusedGateUpPlugin (gate/up GEMM + SwiGLU, FP8 out) -> DQ -> down.
            s_in, s_down = np.float32(scales["ffn_input"][i]), np.float32(scales["down_input"][i])
            codes = g.op("QuantizeLinear", g.cast(h, T.FLOAT), g.const(s_in, "float32"), output_dtype=T.FLOAT8E4M3FN)
            gate, s_gate = fp8_codes(get(p + "mlp.gate_proj.weight"))
            up, s_up = fp8_codes(get(p + "mlp.up_proj.weight"))
            swiglu = g.op("AlpamayoFusedGateUpPlugin", codes, g.weight(pack_gate_up(gate, up)),
                          g.const([s_in * np.float32(s_gate)], "float32"),
                          g.const([s_in * np.float32(s_up) / s_down], "float32"),
                          domain="trt", plugin_version="1", plugin_namespace="")
            activation = g.op("DequantizeLinear", swiglu, g.const(s_down, "float32"))
            x = g.op("Add", x, g.fp8_linear(activation, get(p + "mlp.down_proj.weight")))
        else:
            gate, up = g.linear(h, get(p + "mlp.gate_proj.weight")), g.linear(h, get(p + "mlp.up_proj.weight"))
            swiglu = g.op("Mul", g.op("Mul", gate, g.op("Sigmoid", gate)), up)
            x = g.op("Add", x, g.linear(swiglu, get(p + "mlp.down_proj.weight")))
        if i < DEEPSTACK_LAYERS:
            x = g.op("Add", x, deepstack[i])
        if i in FEATURE_LAYERS:
            features.append(x)
    g.output(g.op("Concat", *features, axis=1), "features", [ROWS, HIDDEN * len(FEATURE_LAYERS)])
    last = g.rms_norm(g.slice(x, ROWS - 1, ROWS, 0), get("norm.weight"))
    vocab = get("embed_tokens.weight")
    g.output(g.linear(last, vocab), "logits", [1, vocab.shape[0]])
    return g.save("alpamayo_prefill_q3006")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle, scales = open_bundle(args, "prefill")
    print(export_prefill(bundle.target, scales, args.output_dir / "prefill"))


if __name__ == "__main__":
    main()
