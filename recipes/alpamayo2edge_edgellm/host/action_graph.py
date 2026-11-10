# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the Alpamayo2-Edge action expert as one ONNX graph: every Euler step in a single engine call, attention
through TensorRT Edge-LLM's ``AttentionPlugin`` (tree mode) over the language model's paged K/V pool.

    python action_graph.py SHIM OUT_DIR [STEPS] [CAPACITY] [--fp8-kv] [--fp8 SCALES.json]

The stock action exporter is Alpamayo-R1 only (SwiGLU, q/k norm) and runs one Euler step per enqueue over the full
K/V capacity. This graph takes the release expert's relu2 layers and config, unrolls ``STEPS`` steps and reads the
language model's pool in place: ``pool_i`` [2, pages, 128, kv_heads, 128] per layer, bound as input and as every
``present_<step>_<layer>`` output (one buffer; the runtime gives each candidate private pages for its 64 rows, see
thor/runtime/action.py). Candidates are packed two per plugin sequence (the tree kernel's 128-row block) with a
block-diagonal mask, so all of them see the prefix and none sees another. RoPE is applied by the plugin from
``rope_rotary_cos_sin``. ``--fp8-kv``: the pool is FP8 and every plugin carries the language model's per-layer
[q, k, v] scales from the quantized shim (``k_proj.k_scale`` ...), so the expert's rows join the prefix in the same
quantization. ``--fp8 SCALES.json``: the layers' q/k/v, o, up and down projections run as per-tensor FP8 GEMMs
(E4M3 weights, activations quantized in the graph) with the input ranges in SCALES (``host/expert_scales.json``:
max |input| per projection over ten validation clips, PyTorch reference, 5 Euler steps); the input projection, the
output projection and the norms stay FP16.

Inputs: ``noise`` [B*64, 2] FP32; ``rope`` [64, 128] FP32 ([cos 64 | sin 64] of the 64 action positions);
``past_lengths``, ``attention_sequence_lengths`` [sequences] INT32; ``kv_page_table`` [sequences, 2, max_pages] INT32;
``attention_pos_id`` [B*64] INT32; the shape-only ``execution_phase_marker`` (extent 5) and
``context_sequence_count_carrier`` (extent 0). Output: ``actions`` [B*64, 2] FP32. Numerics: FP16 GEMMs, FP32 norm
statistics and Euler update. Needs the Edge-LLM plugin library at build and run time.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from onnx import TensorProto as T
from onnx import helper
from safetensors.torch import load_file

from onnx_graph import Graph

CANDIDATES, TOKENS = 6, 64


def fourier(g: Graph, x: str, freqs: torch.Tensor) -> str:
    """``FourierEncoder``: FP32 [rows, n] -> FP32 [rows, n * len(freqs) * 2] = sqrt(2) [sin | cos](x * 2 pi f)."""
    arg = g.op(
        "Mul",
        g.op("Unsqueeze", x, g.const([-1], "int64")),
        g.weight((freqs * 2 * math.pi).reshape(1, 1, -1)),
    )
    feats = g.op(
        "Mul",
        g.op("Concat", g.op("Sin", arg), g.op("Cos", arg), axis=2),
        g.const(math.sqrt(2), "float32"),
    )
    return g.op("Flatten", feats, axis=1)


def input_projection(g: Graph, noise: str, t: float, cfg: dict, get, rows: int) -> str:
    """``ActionInProjection``: Fourier features of (accel, curvature, t) -> MLPEncoder -> LayerNorm; FP16 [rows, hidden]."""
    freqs = torch.logspace(
        0, math.log10(cfg["max_freq"]), steps=cfg["num_fourier_feats"] // 2, dtype=torch.float32
    )
    time = torch.tensor([[t]], dtype=torch.float32).expand(rows, 1).contiguous()
    h = g.cast(
        g.op("Concat", fourier(g, noise, freqs), fourier(g, g.weight(time), freqs), axis=1),
        T.FLOAT16,
    )
    n = cfg["num_enc_layers"]
    for j in range(n):  # trunk: Linear, SiLU, RMSNorm(1e-5) x n, then Linear
        h = g.linear(h, get(f"encoder.trunk.{3 * j}.weight"), get(f"encoder.trunk.{3 * j}.bias"))
        f = g.cast(h, T.FLOAT)
        h = g.rms_norm(
            g.cast(g.op("Mul", f, g.op("Sigmoid", f)), T.FLOAT16),
            get(f"encoder.trunk.{3 * j + 2}.weight"),
            1e-5,
        )
    h = g.linear(h, get(f"encoder.trunk.{3 * n}.weight"), get(f"encoder.trunk.{3 * n}.bias"))
    return g.layer_norm(h, get("norm.weight"), get("norm.bias"), 1e-5)


PLUGIN_DOMAIN = "trt_edgellm"
PHASE_TREE = 5  # execution_phase_marker extent: kSpecTargetVerify -> tree decoding
TREE_BLOCK = 128  # the tree kernel's largest block (kMAX_EAGLE_DECODING_TOKENS): candidates per plugin sequence = 128 // 64


def tree_attention(
    g: Graph,
    candidates: int,
    tokens: int,
    heads: int,
    kv_heads: int,
    d: int,
    kv_scales: list[list[float]] | None = None,
):
    """Graph inputs shared by every layer and step, and a per-layer ``attend(qkv, pool, present, layer)`` closure.

    ``kv_scales`` ([q, k, v] per layer, the language model's calibrated FP8 K/V scales) switches the plugin to the FP8
    pool (``enable_fp8_kv_cache``). The plugin reads every sequence's prefix pages once per call, so candidates are
    packed ``per_seq`` to a plugin sequence of ``per_seq * tokens`` rows (the tree kernel's block limit): candidate j
    of a sequence keeps its own 64 rows at cache positions past + 64 j .. and a block-diagonal mask keeps the
    candidates apart while all of them see the prefix."""
    per_seq = max(1, TREE_BLOCK // tokens)
    while candidates % per_seq:
        per_seq -= 1
    sequences, block = candidates // per_seq, per_seq * tokens
    rows = candidates * tokens
    rope = g.op(
        "Tile", g.input("rope", [tokens, d], T.FLOAT), g.const([candidates, 1], "int64")
    )  # [rows, 128]
    words = -(-block // 32)
    mask = np.zeros((rows, words), np.uint32)
    for r in range(rows):
        j = (r % block) // tokens  # candidate within its sequence
        for col in range(j * tokens, (j + 1) * tokens):
            mask[r, col // 32] |= np.uint32(1) << np.uint32(col % 32)
    inputs = dict(
        query_lengths=g.const([block] * sequences, "int32"),
        past_lengths=g.input("past_lengths", [sequences], T.INT32),
        kv_page_table=g.input("kv_page_table", [sequences, 2, "max_pages"], T.INT32),
        attention_mask=g.const(mask.view(np.int32), "int32"),
        attention_pos_id=g.input("attention_pos_id", [rows], T.INT32),
        query_start_offsets=g.const(np.arange(sequences + 1) * block, "int32"),
        attention_sequence_lengths=g.input("attention_sequence_lengths", [sequences], T.INT32),
        execution_phase_marker=g.input("execution_phase_marker", ["phase_extent"], T.INT32),
        context_sequence_count_carrier=g.input(
            "context_sequence_count_carrier", ["context_sequences"], T.INT32
        ),
    )

    def attend(qkv: str, pool: str, present: str, layer: int) -> str:
        name = g._name("AttentionPlugin")
        g.nodes.append(
            helper.make_node(
                "AttentionPlugin",
                [
                    qkv,
                    pool,
                    inputs["query_lengths"],
                    rope,
                    inputs["past_lengths"],
                    inputs["kv_page_table"],
                    inputs["attention_mask"],
                    inputs["attention_pos_id"],
                    inputs["query_start_offsets"],
                    inputs["attention_sequence_lengths"],
                    inputs["execution_phase_marker"],
                    inputs["context_sequence_count_carrier"],
                ],
                [name, present],
                name=name,
                domain=PLUGIN_DOMAIN,
                plugin_version="1",
                num_q_heads=heads,
                num_kv_heads=kv_heads,
                head_size=d,
                enable_tree_attention=1,
                enable_fp8_kv_cache=0 if kv_scales is None else 1,
                enable_context_mask_selector=0,
                enable_vision_block_attention=0,
                sliding_window_size=-1,
                qkv_scales=[1.0, 1.0, 1.0]
                if kv_scales is None
                else [float(s) for s in kv_scales[layer]],
                attention_scale=d**-0.5,
                skip_softmax_scale_factor=0.0,
                rms_norm_eps=1e-6,
                enable_qk_norm=0,
                qk_norm_post_rope=0,
                enable_kv_shared=0,
                enable_attention_sink=0,
                enable_contiguous_query_swa=0,
            )
        )
        return g.reshape(name, [rows, heads * d])

    return attend


def kv_scales_of(shim: Path, layers: int) -> list[list[float]]:
    """[q, k, v] FP8 scales per language-model layer, as the stock exporter reads them (``--kv-fp8`` checkpoints)."""
    model = load_file(str(shim / "model.safetensors"))
    scale = lambda i, p: float(model[f"layers.{i}.self_attn.{p}_proj.{p}_scale"].reshape(-1)[0])  # noqa: E731
    return [[scale(i, p) for p in "qkv"] for i in range(layers)]


def export(
    shim: Path,
    out: Path,
    steps: int,
    capacity: int,
    candidates: int = CANDIDATES,
    fp8_kv: bool = False,
    fp8_scales: Path | None = None,
) -> Path:
    release = json.load(open(shim / "alpamayo2_edge.config.json"))
    expert = release["expert_config"]
    size, in_cfg = expert["expert_update_cfg"], expert["action_in_proj_cfg"]
    eps = expert["llm_config"]["rms_norm_eps"]
    tokens = expert["action_space_cfg"]["n_waypoints"]
    heads, kv_heads, d, layers = (
        size["num_attention_heads"],
        size["num_key_value_heads"],
        size["head_dim"],
        size["num_hidden_layers"],
    )
    weights = load_file(str(shim / "expert.safetensors"))
    get_in = lambda k: weights["action_in_proj." + k]
    get = lambda k: weights["expert." + k]
    rows = candidates * tokens
    amax = json.load(open(fp8_scales))["amax"] if fp8_scales else None

    def linear(x: str, key: str, weight: torch.Tensor) -> str:  # key: "layers.3.mlp.up_proj"
        return g.fp8_linear(x, weight, amax[key]) if amax else g.linear(x, weight)

    g = Graph(out)
    x = g.input("noise", [rows, 2], T.FLOAT)
    pool_type = T.FLOAT8E4M3FN if fp8_kv else T.FLOAT16
    attend = tree_attention(
        g, candidates, tokens, heads, kv_heads, d, kv_scales_of(shim, layers) if fp8_kv else None
    )
    pools = [g.input(f"pool_{i}", [2, "pages", 128, kv_heads, d], pool_type) for i in range(layers)]
    presents = []

    times = torch.linspace(0.0, 1.0, steps + 1)
    for step in range(steps):
        h = input_projection(g, x, float(times[step]), in_cfg, get_in, rows)
        for i in range(layers):
            p = f"layers.{i}."
            n = g.rms_norm(h, get(p + "input_layernorm.weight"), eps)
            # packed Q | K | V, RoPE inside the plugin. Every plugin reads the graph input pool and every updated pool
            # is a graph output (the runtime binds them all to the same buffer): chaining the in-place pool through
            # intermediate tensors left TensorRT scratch without the prefix rows.
            qkv = linear(
                n,
                p + "self_attn.qkv",
                torch.cat([get(p + f"self_attn.{r}_proj.weight") for r in "qkv"]),
            )
            present = f"present_{step}_{i}"
            a = attend(qkv, pools[i], present, i)
            presents.append(present)
            h = g.op(
                "Add", h, linear(a, p + "self_attn.o_proj", get(p + "self_attn.o_proj.weight"))
            )
            n = g.rms_norm(h, get(p + "post_attention_layernorm.weight"), eps)
            r = g.op("Relu", linear(n, p + "mlp.up_proj", get(p + "mlp.up_proj.weight")))
            h = g.op(
                "Add",
                h,
                linear(g.op("Mul", r, r), p + "mlp.down_proj", get(p + "mlp.down_proj.weight")),
            )
        n = g.rms_norm(h, get("norm.weight"), eps)
        velocity = g.cast(
            g.linear(n, weights["action_out_proj.weight"], weights["action_out_proj.bias"]), T.FLOAT
        )
        x = g.op(
            "Add",
            x,
            g.op("Mul", g.const(float(times[step + 1] - times[step]), "float32"), velocity),
        )
    g.output(x, "actions", [rows, 2], T.FLOAT)
    for present in presents:  # the pool, updated in place (every output aliases input pool_i)
        g.outputs.append(
            helper.make_tensor_value_info(present, pool_type, [2, "pages", 128, kv_heads, d])
        )
    path = g.save("alpamayo2_edge_action_tree")
    (out / "config.json").write_text(
        json.dumps(
            {
                "graph": "host/action_graph.py",
                "steps": steps,
                "candidates": candidates,
                "tokens": tokens,
                "capacity": capacity,
                "num_hidden_layers": layers,
                "head_dim": d,
                "num_key_value_heads": kv_heads,
                "attention": "tree_plugin",
                "kv_cache": "fp8" if fp8_kv else "fp16",
                "gemm": "fp8" if fp8_scales else "fp16",
                "candidates_per_sequence": max(1, TREE_BLOCK // tokens),
                "rope_theta": release["vlm_config"]["text_config"]["rope_parameters"]["rope_theta"],
                "mrope_section": release["vlm_config"]["text_config"]["rope_parameters"][
                    "mrope_section"
                ],
            },
            indent=1,
        )
    )
    return path


if __name__ == "__main__":
    argv = sys.argv[1:]
    scales = Path(argv.pop(argv.index("--fp8") + 1)) if "--fp8" in argv else None
    args = [a for a in argv if a not in ("--fp8-kv", "--fp8")]
    if len(args) not in (2, 3, 4):
        sys.exit(__doc__)
    print(
        export(
            Path(args[0]),
            Path(args[1]),
            int(args[2]) if len(args) > 2 else 5,
            int(args[3]) if len(args) > 3 else 3584,
            fp8_kv="--fp8-kv" in sys.argv,
            fp8_scales=scales,
        )
    )
