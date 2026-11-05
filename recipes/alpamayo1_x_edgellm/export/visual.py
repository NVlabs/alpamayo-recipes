# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the vision encoder for the newest frame of the four cameras to ``<output-dir>/visual``.

Input: ``input`` [2880, 1536] FP16 patches (4 images x 720 patches of 2 x 16 x 16 x 3), the 2D
``rotary_pos_emb`` angles [2880, 32] and the bilinear position-embedding taps
(``fast_pos_embed_idx`` / ``fast_pos_embed_weight`` [4, 2880]). Outputs: merged ``output`` and
three ``deepstack_features_i`` [720, 2048] for the language prefill. Older frames are reused
by the runner, so one call covers one new tick.

Precision ``fp8``: FP8 Q/DQ on all 104 block/merger GEMM inputs and weights (weight scales
amax / 448, activation scales from the bundle's ``fp8-scales.json``), attention over FP8-rounded Q/K
(scale 1/32) with FP16 V. Precision ``fp16``: FP16 GEMMs and attention. Both: LayerNorm in FP32,
FP16 elsewhere, no plugins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from onnx import TensorProto as T

from alpamayo1_x_edgellm.export.graph import Graph, add_model_arguments, open_bundle, open_checkpoint, weight_scale

IMAGES, PATCHES, WIDTH, HEADS, HEAD_DIM, DEPTH, OUT = 4, 720, 1024, 16, 64, 24, 2048
DEEPSTACK = (5, 11, 17)
TOKENS = IMAGES * PATCHES
PREFIX = "vlm.model.visual."
QK_SCALE = 1.0 / 32
# FP16: the FP8 scale layout with every activation scale absent (plain FP16 GEMMs).
NO_SCALES = {"blocks": [[None] * 4] * DEPTH, "deepstack": [[None] * 2] * len(DEEPSTACK), "merger": [None] * 2}


def layer_norm(g: Graph, x: str, gamma: torch.Tensor, beta: torch.Tensor) -> str:
    f = g.cast(x, T.FLOAT)
    centered = g.op("Sub", f, g.op("ReduceMean", f, g.const([-1], "int64"), keepdims=1))
    variance = g.op("ReduceMean", g.op("Pow", centered, g.const(2.0, "float32")), g.const([-1], "int64"), keepdims=1)
    inverse = g.op("Reciprocal", g.op("Sqrt", g.op("Add", variance, g.const(1e-6, "float32"))))
    scaled = g.op("Mul", g.weight(gamma.half().float()), g.op("Mul", centered, inverse))
    return g.cast(g.op("Add", scaled, g.weight(beta.half().float())), T.FLOAT16)


def qdq(g: Graph, x: str, scale: float) -> str:
    s = g.const(scale, "float32")
    q = g.op("QuantizeLinear", g.cast(x, T.FLOAT), s, output_dtype=T.FLOAT8E4M3FN, saturate=1)
    return g.cast(g.op("DequantizeLinear", q, s), T.FLOAT16)


def gemm(g: Graph, x: str, get, name: str, input_scale: float | None) -> str:
    """x @ W^T + bias. With ``input_scale``: FP8 Q/DQ on input and FP16 weight (TensorRT folds the
    weight quantization); without: FP16."""
    weight = g.weight(get(name + ".weight").half())
    if input_scale is not None:
        weight, x = qdq(g, weight, weight_scale(get(name + ".weight").half())), qdq(g, x, input_scale)
    return g.op("Gemm", x, weight, g.weight(get(name + ".bias").half()), transB=1)


def rope(g: Graph, x: str, cos: str, sin: str) -> str:
    """Rotate-half RoPE in FP32 on [tokens, heads, 64]; FP16 result."""
    f = g.cast(x, T.FLOAT)
    rotated = g.op("Concat", g.op("Neg", g.slice(f, 32, 64, 2)), g.slice(f, 0, 32, 2), axis=-1)
    return g.cast(g.op("Add", g.op("Mul", f, cos), g.op("Mul", rotated, sin)), T.FLOAT16)


def merger(g: Graph, x: str, get, name: str, scales: list[float], *, post_shuffle: bool) -> str:
    """2x2 spatial merge (4 patches -> 4096) then fc1 -> GELU -> fc2; LayerNorm after the shuffle
    for deepstack mergers, before it for the final merger."""
    if post_shuffle:
        x = g.reshape(layer_norm(g, g.reshape(x, [-1, 4 * WIDTH]), get(name + ".norm.weight"), get(name + ".norm.bias")), [-1, 4 * WIDTH])
    else:
        x = g.reshape(layer_norm(g, x, get(name + ".norm.weight"), get(name + ".norm.bias")), [-1, 4 * WIDTH])
    x = g.op("Gelu", gemm(g, x, get, name + ".linear_fc1", scales[0]), approximate="tanh")
    return gemm(g, x, get, name + ".linear_fc2", scales[1])


def export_visual(checkpoint: Path, scales: dict | None, output: Path) -> Path:
    """``scales`` None exports the FP16 graph."""
    fp8 = scales is not None
    scales = scales or NO_SCALES
    ckpt = open_checkpoint(checkpoint)
    get = lambda key: ckpt.get_tensor(PREFIX + key)
    g = Graph(output)
    patches = g.input("input", [TOKENS, 1536])
    angles = g.input("rotary_pos_emb", [TOKENS, 32], T.FLOAT)
    pos_idx = g.input("fast_pos_embed_idx", [4, TOKENS], T.INT64)
    pos_weight = g.input("fast_pos_embed_weight", [4, TOKENS])

    patch_weight = get("patch_embed.proj.weight").reshape(WIDTH, -1).t().contiguous().half()  # [1536, 1024]
    x = g.op("Add", g.op("MatMul", patches, g.weight(patch_weight)), g.weight(get("patch_embed.proj.bias").half()))
    taps = g.op("Mul", g.op("Gather", g.weight(get("pos_embed.weight").half()), pos_idx, axis=0),
                g.op("Unsqueeze", pos_weight, g.const([2], "int64")))
    position = g.op("Squeeze", g.slice(taps, 0, 1, 0), g.const([0], "int64"))
    for i in range(1, 4):
        position = g.op("Add", position, g.op("Squeeze", g.slice(taps, i, i + 1, 0), g.const([0], "int64")))
    x = g.op("Add", x, position)
    angles = g.op("Concat", angles, angles, axis=-1)
    cos, sin = (g.op("Unsqueeze", g.op(op, angles), g.const([-2], "int64")) for op in ("Cos", "Sin"))

    outputs = {}
    for b in range(DEPTH):
        p, s = f"blocks.{b}.", scales["blocks"][b]
        h = layer_norm(g, x, get(p + "norm1.weight"), get(p + "norm1.bias"))
        qkv = g.op("Transpose", g.reshape(gemm(g, h, get, p + "attn.qkv", s[0]), [TOKENS, 3, HEADS, HEAD_DIM]), perm=[1, 0, 2, 3])
        q, k, v = (g.op("Squeeze", g.slice(qkv, j, j + 1, 0), g.const([0], "int64")) for j in range(3))
        q, k = (qdq(g, rope(g, t, cos, sin), QK_SCALE) if fp8 else rope(g, t, cos, sin) for t in (q, k))
        q, k, v = (g.op("Transpose", g.reshape(t, [IMAGES, PATCHES, HEADS, HEAD_DIM]), perm=[0, 2, 1, 3]) for t in (q, k, v))
        a = g.op("Attention", q, k, v, is_causal=0, scale=HEAD_DIM ** -0.5)
        a = g.reshape(g.op("Transpose", a, perm=[0, 2, 1, 3]), [TOKENS, WIDTH])
        x = g.op("Add", x, gemm(g, a, get, p + "attn.proj", s[1]))
        h = layer_norm(g, x, get(p + "norm2.weight"), get(p + "norm2.bias"))
        h = g.op("Gelu", gemm(g, h, get, p + "mlp.linear_fc1", s[2]), approximate="tanh")
        x = g.op("Add", x, gemm(g, h, get, p + "mlp.linear_fc2", s[3]))
        if b in DEEPSTACK:
            d = DEEPSTACK.index(b)
            outputs[f"deepstack_features_{d}"] = merger(g, x, get, f"deepstack_merger_list.{d}", scales["deepstack"][d], post_shuffle=True)
    outputs = {"output": merger(g, x, get, "merger", scales["merger"], post_shuffle=False), **outputs}
    for name, value in outputs.items():
        g.output(value, name, [PATCHES, OUT])
    return g.save("alpamayo_visual_newest_tick")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle, scales = open_bundle(args, "visual")
    print(export_visual(bundle.target, scales, args.output_dir / "visual"))


if __name__ == "__main__":
    main()
