"""Lightweight tensor and model helpers required by the core training path."""

import collections
from collections import defaultdict
from typing import Any

import torch


def to(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Any:
    """Recursively cast tensors to a device, dtype, and memory format.

    The implementation is kept byte-for-byte equivalent in behavior to the
    training helper that previously lived in ``commons.misc``.  Keeping it in a
    focused module prevents the model import path from requiring optional
    reporting dependencies such as Pydantic and Rich.
    """

    assert device is not None or dtype is not None or memory_format is not None, (
        "at least one of device, dtype, memory_format should be specified"
    )
    if isinstance(data, torch.Tensor):
        if (
            memory_format == torch.channels_last
            and data.dim() != 4
            or memory_format == torch.channels_last_3d
            and data.dim() != 5
        ):
            memory_format = torch.preserve_format
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        return data.to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
            non_blocking=(not is_cpu),
        )
    if isinstance(data, collections.abc.Mapping):
        return type(data)(
            {
                key: to(data[key], device=device, dtype=dtype, memory_format=memory_format)
                for key in data
            }
        )
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)(
            [to(item, device=device, dtype=dtype, memory_format=memory_format) for item in data]
        )
    return data


def get_param_count(nn_model: torch.nn.Module, depth: int = 2) -> dict[str, int]:
    """Return parameter counts for each module prefix and for the whole model."""

    if depth < 1:
        raise ValueError("Provided depth must be greater than 0 (got %d)" % depth)
    param_counts = defaultdict(int, {"total_params": 0, "trainable_params": 0})
    for name, parameter in nn_model.named_parameters():
        prefix = ".".join(name.split(".")[:depth])
        param_counts[prefix] += parameter.numel()
        if parameter.requires_grad:
            param_counts["trainable_params"] += parameter.numel()
        param_counts["total_params"] += parameter.numel()

    return param_counts
