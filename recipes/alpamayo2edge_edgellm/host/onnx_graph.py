# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal ONNX graph builder for the recipe's hand-built engines (host/action_graph.py).

External weights go to ``weights.bin`` (deduplicated), dynamic dims are strings, custom-domain nodes
(TensorRT plugins) pass the checker unchecked.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto as T
from onnx import helper, numpy_helper

_TYPES = {torch.float16: T.FLOAT16, torch.float32: T.FLOAT, torch.float8_e4m3fn: T.FLOAT8E4M3FN}
FP8_MAX = 448.0


class Graph:
    """Minimal ONNX builder: weights go to one external ``weights.bin``, identical tensors stored once."""

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

    def op(self, kind: str, *inputs: str, **attrs) -> str:
        name = self._name(kind)
        self.nodes.append(helper.make_node(kind, list(inputs), [name], name=name, **attrs))
        return name

    def const(self, value, dtype: str) -> str:
        name = self._name("const")
        self.initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def weight(self, tensor: torch.Tensor) -> str:
        value = tensor.detach().cpu().contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        key = (value.dtype, tuple(value.shape), hash(raw))
        if key in self.stored:
            return self.stored[key]
        name = self.stored[key] = self._name("weight")
        item = T(
            name=name,
            data_type=_TYPES[value.dtype],
            dims=list(value.shape),
            data_location=T.EXTERNAL,
        )
        for field, val in (
            ("location", "weights.bin"),
            ("offset", str(self.payload.tell())),
            ("length", str(len(raw))),
        ):
            item.external_data.add(key=field, value=val)
        self.payload.write(raw)
        self.initializers.append(item)
        return name

    def input(self, name: str, shape: list, dtype: int) -> str:
        self.inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def output(self, value: str, name: str, shape: list[int], dtype: int) -> None:
        self.nodes.append(helper.make_node("Identity", [value], [name], name=name))
        self.outputs.append(helper.make_tensor_value_info(name, dtype, shape))

    def cast(self, x: str, to: int) -> str:
        return self.op("Cast", x, to=to)

    def reshape(self, x: str, shape: list[int]) -> str:
        return self.op("Reshape", x, self.const(shape, "int64"))

    def slice(self, x: str, begin: int, end: int, axis: int) -> str:
        return self.op(
            "Slice",
            x,
            self.const([begin], "int64"),
            self.const([end], "int64"),
            self.const([axis], "int64"),
        )

    def linear(self, x: str, weight: torch.Tensor, bias: torch.Tensor | None = None) -> str:
        y = self.op("MatMul", x, self.op("Transpose", self.weight(weight.half()), perm=[1, 0]))
        return self.op("Add", y, self.weight(bias.half())) if bias is not None else y

    def fp8_linear(self, x: str, weight: torch.Tensor, input_amax: float) -> str:
        """Per-tensor FP8 (E4M3) GEMM in the form Edge-LLM's FP8Linear exports: QuantizeLinear / DequantizeLinear
        on the FP16 activation with scale ``input_amax / 448``, the weight stored FP8 with its own scale and
        dequantized in the graph; TensorRT fuses the pairs into an FP8 MatMul."""
        w = weight.float()
        w_scale = float(w.abs().amax()) / FP8_MAX
        q = (w / w_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        x_scale = self.const(input_amax / FP8_MAX, "float16")
        xq = self.op("QuantizeLinear", x, x_scale, output_dtype=T.FLOAT8E4M3FN)
        wq = self.op("DequantizeLinear", self.weight(q), self.const(w_scale, "float16"))
        return self.op(
            "MatMul",
            self.op("DequantizeLinear", xq, x_scale),
            self.op("Transpose", wq, perm=[1, 0]),
        )

    def rms_norm(self, x: str, gamma: torch.Tensor, eps: float) -> str:
        """FP32 statistics, FP16 normalized value times FP16 gamma (the stock RMSNorm)."""
        f = self.cast(x, T.FLOAT)
        mean = self.op("ReduceMean", self.op("Mul", f, f), self.const([-1], "int64"), keepdims=1)
        normed = self.op(
            "Mul",
            f,
            self.op(
                "Reciprocal", self.op("Sqrt", self.op("Add", mean, self.const(eps, "float32")))
            ),
        )
        return self.op("Mul", self.cast(normed, T.FLOAT16), self.weight(gamma.half().reshape(-1)))

    def layer_norm(self, x: str, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> str:
        f = self.cast(x, T.FLOAT)
        centered = self.op(
            "Sub", f, self.op("ReduceMean", f, self.const([-1], "int64"), keepdims=1)
        )
        var = self.op(
            "ReduceMean", self.op("Mul", centered, centered), self.const([-1], "int64"), keepdims=1
        )
        normed = self.op(
            "Div", centered, self.op("Sqrt", self.op("Add", var, self.const(eps, "float32")))
        )
        return self.cast(
            self.op(
                "Add", self.op("Mul", normed, self.weight(gamma.float())), self.weight(beta.float())
            ),
            T.FLOAT16,
        )

    def rotate(self, x: str, cos: str, sin: str, head_dim: int) -> str:
        """Rotate-half RoPE in FP32 on FP16 [rows, heads, head_dim]; cos/sin [rows, 1, head_dim / 2]."""
        half = head_dim // 2
        f = self.cast(x, T.FLOAT)
        a, b = self.slice(f, 0, half, 2), self.slice(f, half, head_dim, 2)
        first = self.op("Sub", self.op("Mul", a, cos), self.op("Mul", b, sin))
        second = self.op("Add", self.op("Mul", b, cos), self.op("Mul", a, sin))
        return self.cast(self.op("Concat", first, second, axis=2), T.FLOAT16)

    def save(self, name: str) -> Path:
        self.payload.close()
        domains = sorted({n.domain for n in self.nodes} - {""})
        model = helper.make_model(
            helper.make_graph(self.nodes, name, self.inputs, self.outputs, self.initializers),
            opset_imports=[helper.make_opsetid("", 24)]
            + [helper.make_opsetid(d, 1) for d in domains],
            ir_version=10,
        )
        path = self.directory / "model.onnx"
        onnx.save(model, path)
        checked = onnx.load(path, load_external_data=False)
        for node in checked.graph.node:
            kept = [a for a in node.attribute if not a.name.startswith("TRT_")]
            del node.attribute[:]
            node.attribute.extend(kept)
        with contextlib.chdir(self.directory):  # resolve weights.bin like a path-based check
            onnx.checker.check_model(
                checked
            )  # custom-domain nodes (the plugin) are accepted unchecked
        return path
