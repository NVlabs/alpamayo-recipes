# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal ONNX builder shared by the stage exporters.

Weights go to one external ``weights.bin`` beside ``model.onnx``; small scalars are
embedded. Opset 24 (native ``Attention``) plus one opset import per plugin domain used.

The language K/V cache is TensorRT Edge-LLM's paged FP16 pool, one per layer:
``[2, NUM_PAGES, PAGE, 8, 128]`` (K pages, then V pages; token-major inside a page). The prefill
publishes into it (``paged_kv``); the verifier and the action expert read and extend it through
Edge-LLM's ``AttentionPlugin`` in tree mode (``TreeAttention``).
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto as T
from onnx import helper, numpy_helper

from alpamayo1_x_edgellm.core import Bundle

FP8_MAX = 448.0
_TYPES = {torch.float16: T.FLOAT16, torch.float32: T.FLOAT, torch.float8_e4m3fn: T.FLOAT8E4M3FN}

HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
PAGE = 128  # AttentionPlugin kTOKENS_PER_PAGE
# Pool pages per layer: 25 for the language rows (3006 prompt + up to 128 generated, written by the
# verifier) and 2 private pages per action candidate (6 x 2) after them; 37 rounded up.
NUM_PAGES = 40
POOL_SHAPE = [2, NUM_PAGES, PAGE, KV_HEADS, HEAD_DIM]
PHASE_SPEC_TARGET_VERIFY = 5  # execution_phase_marker extent of the plugin's tree-decode phase


def open_checkpoint(path: Path):
    """Open a safetensors file (the bundle's target, expert or drafter weights)."""
    from safetensors import safe_open

    return safe_open(str(path), "pt")


def add_model_arguments(parser) -> None:
    """``--model`` (the bundle) and ``--precision fp8`` (default) or ``fp16``."""
    parser.add_argument("--model", type=Path, required=True, help="Model bundle directory (README).")
    parser.add_argument("--precision", choices=("fp8", "fp16"), default="fp8")


def open_bundle(args, section: str) -> tuple[Bundle, dict | None]:
    """The bundle and, for ``--precision fp8``, its activation scales for ``section``; None for fp16."""
    bundle = Bundle(args.model)
    return bundle, None if args.precision == "fp16" else bundle.scales(section)


def weight_scale(weight: torch.Tensor) -> float:
    """Per-tensor E4M3 weight scale amax / 448 (FP32)."""
    return float(np.float32(weight.float().abs().max().item() / FP8_MAX))


def fp8_codes(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Pre-quantized per-tensor E4M3 codes for plugin weights, with ONNX QuantizeLinear
    semantics: saturate(RNE(fp32(W) / fp32(amax / 448)))."""
    scale = weight_scale(weight)
    codes = (weight.float() / np.float32(scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale


class Graph:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.payload = (self.directory / "weights.bin").open("xb")
        self.nodes, self.initializers, self.inputs, self.outputs = [], [], [], []
        self.serial = 0
        self.stored: dict = {}

    def _name(self, stem: str) -> str:
        self.serial += 1
        return f"{stem}_{self.serial}"

    def op(self, kind: str, *inputs: str, domain: str = "", **attrs) -> str:
        name = self._name(kind)
        self.nodes.append(helper.make_node(kind, list(inputs), [name], name=name, domain=domain, **attrs))
        return name

    def op_multi(self, kind: str, inputs: list[str], outputs: int | list[str], *, domain: str = "", **attrs) -> list[str]:
        name = self._name(kind)
        names = outputs if isinstance(outputs, list) else [f"{name}_out{i}" for i in range(outputs)]
        self.nodes.append(helper.make_node(kind, inputs, names, name=name, domain=domain, **attrs))
        return names

    def const(self, value, dtype: str) -> str:
        name = self._name("const")
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def weight(self, tensor: torch.Tensor) -> str:
        """External initializer; identical tensors (e.g. across unrolled steps) are stored once."""
        value = tensor.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        key = (value.dtype, tuple(value.shape), hashlib.sha256(raw).digest())
        if key in self.stored:
            return self.stored[key]
        name = self.stored[key] = self._name("weight")
        item = T(name=name, data_type=_TYPES[value.dtype], dims=list(value.shape), data_location=T.EXTERNAL)
        for field, val in (("location", "weights.bin"), ("offset", str(self.payload.tell())), ("length", str(len(raw)))):
            item.external_data.add(key=field, value=val)
        self.payload.write(raw)
        self.initializers.append(item)
        return name

    def input(self, name: str, shape: list[int], dtype: int = T.FLOAT16) -> str:
        self.inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, value: str, name: str, shape: list[int], dtype: int = T.FLOAT16) -> None:
        if value != name:
            self.nodes.append(helper.make_node("Identity", [value], [name], name=name))
        self.outputs.append(helper.make_tensor_value_info(name, dtype, shape))

    # ---- common blocks -------------------------------------------------------------------------
    def cast(self, x: str, to: int) -> str:
        return self.op("Cast", x, to=to)

    def reshape(self, x: str, shape: list[int]) -> str:
        return self.op("Reshape", x, self.const(shape, "int64"))

    def slice(self, x: str, begin: int, end: int, axis: int) -> str:
        return self.op("Slice", x, self.const([begin], "int64"), self.const([end], "int64"), self.const([axis], "int64"))

    def rms_norm(self, x: str, gamma: torch.Tensor, *, divide: bool = False) -> str:
        """FP32 statistics, FP16 normalized value times FP16 gamma (Qwen3 RMSNorm, eps 1e-6).
        ``divide`` normalizes as x / rms instead of x * (1 / rms)."""
        f = self.cast(x, T.FLOAT)
        mean = self.op("ReduceMean", self.op("Mul", f, f), self.const([-1], "int64"), keepdims=1)
        root = self.op("Sqrt", self.op("Add", mean, self.const(1e-6, "float32")))
        normed = self.op("Div", f, root) if divide else self.op("Mul", f, self.op("Reciprocal", root))
        return self.op("Mul", self.cast(normed, T.FLOAT16), self.weight(gamma.half().reshape(-1)))

    def rms_norm_fp32(self, x: str, gamma: torch.Tensor) -> str:
        """Verifier/drafter variant: FP32 statistics, FP16-representable gamma widened to FP32,
        one FP16 rounding at the end."""
        f = self.cast(x, T.FLOAT)
        mean = self.op("ReduceMean", self.op("Mul", f, f), self.const([-1], "int64"), keepdims=1)
        normed = self.op("Div", f, self.op("Sqrt", self.op("Add", mean, self.const(1e-6, "float32"))))
        return self.cast(self.op("Mul", normed, self.weight(gamma.half().float().reshape(-1))), T.FLOAT16)

    def fp8_codes_linear(self, x: str, codes: torch.Tensor, input_scale: float, combined: float) -> str:
        """Q(x, input_scale) -> DQ(codes x combined decode) @ raw FP8 codes^T; FP32 result.
        ``combined`` = input_scale * weight_scale (optionally with a folded output encode)."""
        q = self.op("QuantizeLinear", self.cast(x, T.FLOAT), self.const(input_scale, "float32"),
                    output_dtype=T.FLOAT8E4M3FN)
        a = self.op("DequantizeLinear", q, self.const(combined, "float32"))
        w = self.op("DequantizeLinear", self.weight(codes), self.const(1.0, "float32"))
        return self.op("MatMul", a, self.op("Transpose", w, perm=[1, 0]))

    def fp8_half_linear(self, x: str, weight: torch.Tensor, input_scale: float) -> str:
        """FP16-scale Q/DQ on an FP16 activation and pre-quantized FP8 weight; FP16 GEMM result."""
        codes, weight_scale_value = fp8_codes(weight)
        a = self.const(input_scale, "float16")
        x = self.op("DequantizeLinear", self.op("QuantizeLinear", x, a, output_dtype=T.FLOAT8E4M3FN), a)
        w = self.op("DequantizeLinear", self.weight(codes), self.const(weight_scale_value, "float16"))
        return self.op("MatMul", x, self.op("Transpose", w, perm=[1, 0]))

    def linear(self, x: str, weight: torch.Tensor) -> str:
        """FP16 x @ W^T with W [out, in] stored FP16."""
        return self.op("MatMul", x, self.op("Transpose", self.weight(weight.half()), perm=[1, 0]))

    def fp8_linear(self, x: str, weight: torch.Tensor) -> str:
        """FP32 activation (already Q/DQ'd) @ Q/DQ(FP16 W, amax/448)^T; TensorRT folds the
        weight quantization and runs an FP8 GEMM. FP16 result."""
        scale = self.const(weight_scale(weight), "float32")
        codes = self.op("QuantizeLinear", self.cast(self.weight(weight.half()), T.FLOAT), scale,
                        output_dtype=T.FLOAT8E4M3FN)
        w = self.op("DequantizeLinear", codes, scale)
        return self.cast(self.op("MatMul", x, self.op("Transpose", w, perm=[1, 0])), T.FLOAT16)

    def rotate(self, x: str, cos: str, sin: str) -> str:
        """Half-split RoPE in FP32 on an FP16 [rows, heads, 128] tensor; cos/sin [rows, 1, 64]."""
        f = self.cast(x, T.FLOAT)
        a, b = self.slice(f, 0, 64, 2), self.slice(f, 64, 128, 2)
        first = self.op("Sub", self.op("Mul", a, cos), self.op("Mul", b, sin))
        second = self.op("Add", self.op("Mul", b, cos), self.op("Mul", a, sin))
        return self.cast(self.op("Concat", first, second, axis=2), T.FLOAT16)

    def paged_kv(self, k: str, v: str, rows: int) -> str:
        """Token-major FP16 K and V [rows, 8, 128] -> ``[2, pages, PAGE, 8, 128]`` (zero-padded to whole pages),
        the layout of the pool's first ``pages`` pages."""
        pages, pad = -(-rows // PAGE), -rows % PAGE
        planes = []
        for t in (k, v):
            if pad:
                t = self.op("Concat", t, self.const(np.zeros((pad, KV_HEADS, HEAD_DIM), np.float16), "float16"), axis=0)
            planes.append(self.reshape(t, [1, pages, PAGE, KV_HEADS, HEAD_DIM]))
        return self.op("Concat", *planes, axis=0)

    def save(self, name: str) -> Path:
        self.payload.close()
        opsets = [helper.make_opsetid("", 24)] + [helper.make_opsetid(d, 1) for d in sorted({n.domain for n in self.nodes} - {""})]
        model = helper.make_model(helper.make_graph(self.nodes, name, self.inputs, self.outputs, self.initializers),
                                  opset_imports=opsets, ir_version=10)
        path = self.directory / "model.onnx"
        onnx.save(model, path)
        # Validate a copy without TensorRT-only attributes (e.g. Attention TRT_decomposable).
        checked = onnx.load(path, load_external_data=False)
        for node in checked.graph.node:
            kept = [a for a in node.attribute if not a.name.startswith("TRT_")]
            del node.attribute[:]
            node.attribute.extend(kept)
        with contextlib.chdir(self.directory):  # resolve weights.bin like a path-based check
            onnx.checker.check_model(checked)
        return path


class TreeAttention:
    """TensorRT Edge-LLM ``AttentionPlugin`` (version 1) in tree mode over the paged pool, for ``sequences``
    sequences of ``rows`` new rows each (``rows`` <= 128). Q and K arrive rotated (the exporters apply
    RoPE in-graph), so the plugin's per-token cos/sin table is the identity (cos 1, sin 0). ``mask``
    [sequences * rows, ceil(rows / 32)] INT32: bit j of row i set lets new row i attend new row j; the
    prefix is always attended. The plugin writes the new K/V rows at [past, past + rows) of each sequence's
    pages and returns the pool in place (output 1 aliases input 1).

    Graph inputs, filled by the runner per call: ``past_lengths`` [sequences] (prefix rows),
    ``sequence_lengths`` [sequences] (prefix + rows), ``positions`` [sequences * rows] (absolute row ids),
    ``page_table`` [sequences, 2, max_pages] (K page ids; V ids = K ids + NUM_PAGES) unless given here as
    a constant, ``phase_marker`` [5] and ``sequence_count_carrier`` [0] (shape-only carriers)."""

    def __init__(self, g: Graph, sequences: int, rows: int, max_pages: int, mask: np.ndarray,
                 page_table: np.ndarray | None = None):
        if not 1 <= rows <= PAGE:
            raise ValueError(f"tree attention takes 1..{PAGE} new rows per sequence, got {rows}")
        total = sequences * rows
        identity = np.zeros((total, HEAD_DIM), np.float32)
        identity[:, :HEAD_DIM // 2] = 1.0
        self.g = g
        self.query_lengths = g.const([rows] * sequences, "int32")
        self.rope = g.weight(torch.from_numpy(identity))
        self.past_lengths = g.input("past_lengths", [sequences], T.INT32)
        self.page_table = (g.input("page_table", [sequences, 2, max_pages], T.INT32) if page_table is None
                           else g.const(np.asarray(page_table, np.int32).reshape(sequences, 2, max_pages), "int32"))
        self.mask = g.const(np.asarray(mask, np.int32).reshape(total, -(-rows // 32)), "int32")
        self.positions = g.input("positions", [total], T.INT32)
        self.offsets = g.const(np.arange(sequences + 1) * rows, "int32")
        self.lengths = g.input("sequence_lengths", [sequences], T.INT32)
        self.phase = g.input("phase_marker", [PHASE_SPEC_TARGET_VERIFY], T.INT32)
        self.carrier = g.input("sequence_count_carrier", [0], T.INT32)

    def __call__(self, qkv: str, pool: str, present: str) -> tuple[str, str]:
        """``qkv`` [sequences * rows, 4096] FP16 packed Q | K | V; returns (attention [sequences * rows, 16, 128],
        the updated pool). The pool output is named ``present`` so it can be a graph output without a copy."""
        return tuple(self.g.op_multi(
            "AttentionPlugin",
            [qkv, pool, self.query_lengths, self.rope, self.past_lengths, self.page_table, self.mask,
             self.positions, self.offsets, self.lengths, self.phase, self.carrier],
            [f"{present}_attention", present], domain="trt", plugin_version="1", plugin_namespace="",
            num_q_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_DIM, enable_tree_attention=1,
            enable_fp8_kv_cache=0, enable_vision_block_attention=0, enable_context_mask_selector=0,
            attention_scale=HEAD_DIM ** -0.5, qkv_scales=[1.0, 1.0, 1.0], rms_norm_eps=1e-6,
            skip_softmax_scale_factor=0.0, tp_size=1))


def causal_mask(rows: int) -> np.ndarray:
    """Packed tree mask: new row i attends new rows j <= i."""
    words = -(-rows // 32)
    mask = np.zeros((rows, words), np.int64)
    for i in range(rows):
        for j in range(i + 1):
            mask[i, j // 32] |= 1 << (j % 32)
    return mask.astype(np.uint32).view(np.int32)


def full_mask(sequences: int, rows: int) -> np.ndarray:
    """Packed tree mask: every new row of a sequence attends all of that sequence's new rows."""
    words = -(-rows // 32)
    row = np.zeros(words, np.int64)
    for j in range(rows):
        row[j // 32] |= 1 << (j % 32)
    return np.tile(row.astype(np.uint32).view(np.int32), (sequences * rows, 1))
