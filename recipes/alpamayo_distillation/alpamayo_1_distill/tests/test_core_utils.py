from __future__ import annotations

import torch

from alpamayo1_distill.data.collation import basic_collation_fn
from alpamayo1_distill.utils.model import get_param_count, to


def test_basic_collation_preserves_original_behavior() -> None:
    batch = [
        {"tensor": torch.tensor([1, 2]), "metadata": "a", "image_frames": torch.tensor([3])},
        {"tensor": torch.tensor([4, 5]), "metadata": "b", "image_frames": torch.tensor([6])},
    ]

    result = basic_collation_fn(batch, unstackable_keys=["image_frames"])

    assert torch.equal(result["tensor"], torch.tensor([[1, 2], [4, 5]]))
    assert result["metadata"] == ["a", "b"]
    assert len(result["image_frames"]) == 2
    assert torch.equal(result["image_frames"][0], torch.tensor([3]))


def test_to_recursively_converts_tensors_without_changing_structure() -> None:
    source = {"tensor": torch.ones(2), "nested": [torch.zeros(1), "metadata"]}

    result = to(source, device="cpu", dtype=torch.float64)

    assert isinstance(result, dict)
    assert isinstance(result["nested"], list)
    assert result["tensor"].dtype == torch.float64
    assert result["nested"][0].dtype == torch.float64
    assert result["nested"][1] == "metadata"


def test_get_param_count_matches_trainable_state() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[1].weight.requires_grad_(False)

    counts = get_param_count(model, depth=1)

    assert counts["total_params"] == sum(parameter.numel() for parameter in model.parameters())
    assert counts["trainable_params"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert counts["0"] == sum(parameter.numel() for parameter in model[0].parameters())
    assert counts["1"] == sum(parameter.numel() for parameter in model[1].parameters())
