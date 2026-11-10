# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.overrides import TorchFunctionMode

from alpamayo2edge_sft.models.geometry import CPUCholesky


class RecordSolvers(TorchFunctionMode):
    def __init__(self):
        self.devices = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (torch.linalg.cholesky, torch.cholesky_solve):
            self.devices.append((func, tuple(value.device.type for value in args if isinstance(value, torch.Tensor))))
        return func(*args, **({} if kwargs is None else kwargs))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA target fitting requires a GPU"))])
def test_cpu_cholesky_matches_cpu_reference_and_preserves_output_device(device):
    lhs = torch.tensor([[4., .5], [.5, 3.]], device=device)
    rhs = torch.tensor([[1.], [2.]], device=device)
    reference = torch.cholesky_solve(rhs.cpu(), torch.linalg.cholesky(lhs.cpu()))
    with RecordSolvers() as record, CPUCholesky():
        factor = torch.linalg.cholesky(lhs)
        actual = torch.cholesky_solve(rhs, factor)
        untouched = lhs @ rhs
    assert factor.device.type == actual.device.type == untouched.device.type == device
    assert record.devices == [
        (torch.linalg.cholesky, ("cpu",)),
        (torch.cholesky_solve, ("cpu", "cpu")),
    ]
    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    torch.testing.assert_close(untouched, lhs @ rhs, rtol=0, atol=0)


def test_target_solver_context_propagates_errors_and_restores_dispatch():
    original = torch.linalg.cholesky
    with pytest.raises(torch.linalg.LinAlgError), CPUCholesky():
        torch.linalg.cholesky(torch.zeros(2, 2))
    assert torch.linalg.cholesky is original
    with RecordSolvers() as record:
        torch.linalg.cholesky(torch.eye(2))
    assert record.devices == [(torch.linalg.cholesky, ("cpu",))]
