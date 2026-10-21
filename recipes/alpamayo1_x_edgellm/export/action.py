# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the action expert (6 candidates x 64 tokens, 5 flow-matching steps) to ``<output-dir>/action``.

One engine call runs all five Euler steps: the action-input MLP, 28 expert layers per step and the
velocity head. ``noise`` [384, 2] is the initial trajectory noise; ``actions`` [384, 2] the result.

Each candidate attends, without a causal mask, to the language prefix plus its own 64 tokens.
Attention is TensorRT Edge-LLM's ``AttentionPlugin`` in tree mode over the paged FP16 K/V pool
``pool_i`` [2, 40, 128, 8, 128]: six sequences of 64 new rows, FP16 Q/K/V. The runner's ``page_table``
[6, 2, 25] shares the full prefix pages between the candidates and gives each candidate two private
pages after the language pages (the last, partly filled prefix page is copied there once per request,
then the candidate's 64 rows follow it); every step rewrites the same 64 rows. Every step's pool
output ``present_{step}_i`` aliases ``pool_i``.

``fp8``: FP8 FFN (tanh-form SiLU, down input scale folded into the up output; weight and activation
scales from the bundle's ``fp8-scales.json``); the rest FP16. ``fp16``: FP16 FFN (SiLU in FP32).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from onnx import TensorProto as T
from safetensors import safe_open

from alpamayo1_x_edgellm.export.graph import (FP8_MAX, HEAD_DIM, HEADS, KV_HEADS, POOL_SHAPE, Graph, TreeAttention,
                                                add_model_arguments, full_mask, open_bundle)

ROWS, WIDTH, LAYERS, STEPS = 384, 384, 28, 5
CANDIDATES, TOKENS = 6, 64
MAX_PAGES = 25  # per candidate: up to 24 prefix pages (3072 rows) plus the page its 64 rows run into


def fp8_scaled(weight: torch.Tensor, scale: float) -> torch.Tensor:
    """E4M3 codes of FP16 weights with a given calibrated scale (QuantizeLinear semantics)."""
    return (weight.half().float() / np.float32(scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def inverse_rms(g: Graph, f: str, eps: float) -> str:
    mean = g.op("ReduceMean", g.op("Mul", f, f), g.const([-1], "int64"), keepdims=1)
    return g.op("Reciprocal", g.op("Sqrt", g.op("Add", mean, g.const(eps, "float32"))))


def rms(g: Graph, x: str, gamma: torch.Tensor, eps: float = 1e-6) -> str:
    f = g.cast(x, T.FLOAT)
    return g.cast(g.op("Mul", g.op("Mul", f, inverse_rms(g, f, eps)), g.weight(gamma.half().float())), T.FLOAT16)


def residual_norm(g: Graph, x: str, delta: str, gamma: torch.Tensor) -> tuple[str, str]:
    """FP16 residual; RMS statistics of the unrounded FP32 sum applied to the rounded residual."""
    total = g.op("Add", g.cast(x, T.FLOAT), g.cast(delta, T.FLOAT))
    residual = g.cast(total, T.FLOAT16)
    normalized = g.op("Mul", g.cast(residual, T.FLOAT), inverse_rms(g, total, 1e-6))
    return residual, g.cast(g.op("Mul", normalized, g.weight(gamma.half().float())), T.FLOAT16)


def scaled_linear(g: Graph, x: str, codes: torch.Tensor, input_scale: float, decode: float) -> str:
    """Q(x, input_scale) -> DQ(decode) @ raw FP8 codes^T, FP32 result."""
    q = g.op("QuantizeLinear", g.cast(x, T.FLOAT), g.const(input_scale, "float32"), output_dtype=T.FLOAT8E4M3FN)
    w = g.op("DequantizeLinear", g.weight(codes), g.const(1.0, "float32"))
    return g.op("MatMul", g.op("DequantizeLinear", q, g.const(decode, "float32")), g.op("Transpose", w, perm=[1, 0]))


def input_projection(g: Graph, noise: str, step: int, get) -> tuple[str, str]:
    """Sinusoidal (noise x, noise y, time) features -> MLP -> LayerNorm; returns residual and normed."""
    features = g.op("Concat", g.cast(noise, T.FLOAT16), g.const([[step / STEPS]] * ROWS, "float16"), axis=1)
    phase = g.op("Mul", g.reshape(g.cast(features, T.FLOAT), [ROWS, 3, 1]),
                 g.weight(get("action_in_proj.sinus.0.freqs").half().float().reshape(1, 1, 10)))
    phase = g.op("Mul", g.op("Mul", phase, g.const(2.0, "float32")), g.const(3.14, "float32"))
    x = g.op("Mul", g.op("Concat", g.op("Sin", phase), g.op("Cos", phase), axis=2), g.const(1.414, "float32"))
    x = g.reshape(g.cast(x, T.FLOAT16), [ROWS, 60])
    for fc, norm in ((0, 2), (3, 5)):
        biased = g.op("Add", g.cast(g.linear(x, get(f"action_in_proj.encoder.trunk.{fc}.weight")), T.FLOAT),
                      g.weight(get(f"action_in_proj.encoder.trunk.{fc}.bias").half().float()))
        silu = g.cast(g.op("Mul", biased, g.op("Sigmoid", biased)), T.FLOAT16)
        x = rms(g, silu, get(f"action_in_proj.encoder.trunk.{norm}.weight"), 1e-5)
    out = g.op("Add", g.linear(x, get("action_in_proj.encoder.trunk.6.weight")),
               g.weight(get("action_in_proj.encoder.trunk.6.bias").half()))
    f = g.cast(out, T.FLOAT)
    centered = g.op("Sub", f, g.op("ReduceMean", f, g.const([-1], "int64"), keepdims=1))
    ln = g.op("Mul", g.op("Mul", centered, inverse_rms(g, centered, 1e-5)),
              g.weight(get("action_in_proj.norm.weight").half().float()))
    residual = g.cast(g.op("Add", ln, g.weight(get("action_in_proj.norm.bias").half().float())), T.FLOAT16)
    return residual, rms(g, residual, get("expert.layers.0.input_layernorm.weight"))


def fp8_ffn(g: Graph, get, scales: dict):
    """FP8 FFN: calibrated weight codes, FP8 activation Q/DQ, tanh-form SiLU in FP32."""
    f32 = np.float32
    layers = []
    for i in range(LAYERS):
        p, s = f"expert.layers.{i}.", scales["layers"][i]
        x_s, g_s = f32(s["ffn_input"]), f32(s["down_input"])
        gw, uw, dw = f32(s["gate_weight"]), f32(s["up_weight"]), f32(s["down_weight"])
        layers.append(dict(
            gate=fp8_scaled(get(p + "mlp.gate_proj.weight"), gw), up=fp8_scaled(get(p + "mlp.up_proj.weight"), uw),
            down=fp8_scaled(get(p + "mlp.down_proj.weight"), dw),
            x_scale=float(x_s), gate_decode=float(f32(float(x_s) * float(gw))), down_decode=float(f32(float(g_s) * float(dw))),
            up_encode=float(f32(float(x_s) * float(uw) / float(g_s)))))

    def ffn(i: int, h: str) -> str:
        w = layers[i]
        gate = scaled_linear(g, h, w["gate"], w["x_scale"], w["gate_decode"])
        up = scaled_linear(g, h, w["up"], w["x_scale"], 1.0)
        half = g.op("Mul", gate, g.const(0.5, "float32"))
        silu = g.op("Add", g.op("Mul", half, g.op("Tanh", half)), half)
        encoded = g.op("Mul", g.op("Mul", silu, up), g.const(w["up_encode"], "float32"))
        return g.cast(scaled_linear(g, encoded, w["down"], 1.0, w["down_decode"]), T.FLOAT16)

    return ffn


def fp16_ffn(g: Graph, get):
    """FP16 FFN, SiLU in FP32."""

    def ffn(i: int, h: str) -> str:
        p = f"expert.layers.{i}."
        gate, up = (g.cast(g.linear(h, get(p + f"mlp.{r}_proj.weight")), T.FLOAT) for r in ("gate", "up"))
        return g.linear(g.cast(g.op("Mul", g.op("Mul", gate, g.op("Sigmoid", gate)), up), T.FLOAT16),
                        get(p + "mlp.down_proj.weight"))

    return ffn


def export_action(checkpoint: Path, scales: dict | None, output: Path) -> Path:
    """Reads the action expert from the target checkpoint; ``scales`` None exports the FP16 graph."""
    ckpt = safe_open(str(checkpoint), "pt")
    get = ckpt.get_tensor
    g = Graph(output)
    noise = g.input("noise", [ROWS, 2], T.FLOAT)
    rope = g.input("rope", [ROWS, 128], T.FLOAT)
    pools = [g.input(f"pool_{i}", POOL_SHAPE) for i in range(LAYERS)]
    attention = TreeAttention(g, CANDIDATES, TOKENS, MAX_PAGES, full_mask(CANDIDATES, TOKENS))
    cos, sin = (g.reshape(g.slice(rope, a, a + 64, 1), [ROWS, 1, 64]) for a in (0, 64))
    ffn = fp8_ffn(g, get, scales) if scales is not None else fp16_ffn(g, get)

    presents = []
    for step in range(STEPS):
        residual, x = input_projection(g, noise, step, get)
        for i in range(LAYERS):
            p = f"expert.layers.{i}."
            qkv = g.linear(x, torch.cat([get(p + f"self_attn.{r}_proj.weight") for r in "qkv"]))
            q = g.rotate(g.rms_norm(g.reshape(g.slice(qkv, 0, 2048, 1), [ROWS, HEADS, HEAD_DIM]),
                                    get(p + "self_attn.q_norm.weight"), divide=True), cos, sin)
            k = g.rotate(g.rms_norm(g.reshape(g.slice(qkv, 2048, 3072, 1), [ROWS, KV_HEADS, HEAD_DIM]),
                                    get(p + "self_attn.k_norm.weight"), divide=True), cos, sin)
            packed = g.op("Concat", g.reshape(q, [ROWS, HEADS * HEAD_DIM]), g.reshape(k, [ROWS, KV_HEADS * HEAD_DIM]),
                          g.slice(qkv, 3072, 4096, 1), axis=1)
            # Each step's pool output is the in-place updated pool_i of that layer (bound to the same buffer).
            out, pools[i] = attention(packed, pools[i], f"present_{step}_{i}")
            presents.append(pools[i])
            delta = g.linear(g.reshape(out, [ROWS, HEADS * HEAD_DIM]), get(p + "self_attn.o_proj.weight"))
            residual, h = residual_norm(g, residual, delta, get(p + "post_attention_layernorm.weight"))
            gamma = get(f"expert.layers.{i + 1}.input_layernorm.weight" if i + 1 < LAYERS else "expert.norm.weight")
            residual, x = residual_norm(g, residual, ffn(i, h), gamma)
        velocity = g.op("Add", g.linear(x, get("action_out_proj.weight")), g.weight(get("action_out_proj.bias").half()))
        dt = g.op("Sub", g.const((step + 1) / STEPS, "float16"), g.const(step / STEPS, "float16"))
        noise = g.cast(g.op("Add", g.cast(noise, T.FLOAT16), g.op("Mul", dt, velocity)), T.FLOAT)
    g.output(noise, "actions", [ROWS, 2], T.FLOAT)
    for present in presents:
        g.output(present, present, POOL_SHAPE)
    return g.save("alpamayo_action_six_candidates_five_steps")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle, scales = open_bundle(args, "action")
    print(export_action(bundle.target, scales, args.output_dir / "action"))


if __name__ == "__main__":
    main()
