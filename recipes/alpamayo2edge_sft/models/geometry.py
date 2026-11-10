# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consistent small-matrix solves for supervised trajectory targets."""

import torch
from torch.overrides import TorchFunctionMode


class CPUCholesky(TorchFunctionMode):
    """Fit trajectory targets with CPU Cholesky while preserving tensor devices.

    Only the factorization and solve move to CPU; matrix construction and all
    other operations retain their original device. This keeps target labels
    stable across CUDA solver backends. Use around target fitting/tokenization,
    outside the neural-network forward pass.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func not in (torch.linalg.cholesky, torch.cholesky_solve):
            return func(*args, **kwargs)
        device = next(
            value.device for value in (*args, *kwargs.values())
            if isinstance(value, torch.Tensor)
        )
        if device.type == "cpu":
            return func(*args, **kwargs)
        cpu_args = tuple(value.cpu() if isinstance(value, torch.Tensor) else value for value in args)
        cpu_kwargs = {
            key: value.cpu() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        return func(*cpu_args, **cpu_kwargs).to(device)
