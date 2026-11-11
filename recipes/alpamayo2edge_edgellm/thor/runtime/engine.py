# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One TensorRT engine driven from torch tensors: one execution context per optimization profile.

Bindings are set only where they changed since the previous call on that context (the LLM engine has 60 tensors,
the action engine 118; most addresses are persistent buffers). Outputs the caller does not pass are allocated once
per output shape and reused; aliased in-place tensors (K/V pools) are passed under both names by the caller.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

DTYPES = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.bfloat16: torch.bfloat16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
    trt.uint8: torch.uint8,
    trt.fp8: torch.float8_e4m3fn,
}


class Engine:
    def __init__(self, runtime: trt.Runtime, path: Path):
        self.path = Path(path)
        self.engine = runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {path}")
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [
            n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.dtype = {n: DTYPES[self.engine.get_tensor_dtype(n)] for n in self.names}
        self.profiles = self.engine.num_optimization_profiles
        stream = torch.cuda.current_stream().cuda_stream
        self.contexts = []
        for profile in range(self.profiles):
            context = self.engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    f"{self.path.name}: cannot create an execution context (device memory?)"
                )
            context.set_optimization_profile_async(profile, stream)
            self.contexts.append(context)
        torch.cuda.synchronize()
        self.bound = [
            dict() for _ in range(self.profiles)
        ]  # name -> (ptr, shape) last set on that context
        self.scratch = {}  # (profile, name, shape) -> allocated output
        self.placeholder = torch.zeros(256, dtype=torch.uint8, device="cuda")

    def shape(self, name: str) -> tuple:
        return tuple(self.engine.get_tensor_shape(name))

    def profile_shape(self, name: str, profile: int = 0) -> tuple:
        """(min, opt, max) dims of an input under one optimization profile."""
        return tuple(tuple(d) for d in self.engine.get_tensor_profile_shape(name, profile))

    def __call__(
        self, tensors: dict[str, torch.Tensor], profile: int = 0
    ) -> dict[str, torch.Tensor]:
        """Enqueue on the current torch stream. ``tensors`` holds every input (shape-only inputs as zero-size
        tensors of the right length) and optionally outputs; returns all outputs."""
        context, bound = self.contexts[profile], self.bound[profile]
        for name in self.inputs:
            tensor = tensors[name]
            if tensor.dtype != self.dtype[name]:
                raise TypeError(
                    f"{self.path.name}: {name} is {tensor.dtype}, engine wants {self.dtype[name]}"
                )
            self._bind(context, bound, name, tensor, is_input=True)
        outputs = {}
        for name in self.outputs:
            tensor = tensors.get(name)
            if tensor is None:
                shape = tuple(context.get_tensor_shape(name))
                if any(d < 0 for d in shape):
                    inputs = {
                        n: tuple(tensors[n].shape)
                        for n in self.inputs
                        if tensors[n].numel() < 64 or n in ("inputs_embeds",)
                    }
                    raise ValueError(
                        f"{self.path.name}: output {name} has no shape for input shapes {inputs} (profile {profile})"
                    )
                key = (profile, name, shape)
                tensor = self.scratch.get(key)
                if tensor is None:
                    tensor = self.scratch[key] = torch.empty(
                        shape, dtype=self.dtype[name], device="cuda"
                    )
            self._bind(context, bound, name, tensor, is_input=False)
            outputs[name] = tensor
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(f"{self.path.name}: enqueue failed")
        return outputs

    def _bind(self, context, bound, name, tensor, *, is_input):
        host = self.engine.is_shape_inference_io(name)  # shape inputs are read from host memory
        if tensor.is_cuda == host and tensor.numel():
            raise ValueError(f"{name} must live on the {'host' if host else 'GPU'}")
        ptr = tensor.data_ptr() if tensor.numel() else self.placeholder.data_ptr()
        shape = tuple(tensor.shape)
        if host or bound.get(name) != (ptr, shape):
            if is_input and not context.set_input_shape(name, shape):
                raise ValueError(f"{self.path.name}: shape {shape} rejected for {name}")
            context.set_tensor_address(name, ptr)
            bound[name] = (ptr, shape)
