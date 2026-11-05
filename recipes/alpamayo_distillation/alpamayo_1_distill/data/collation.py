"""Small, dependency-free collation helpers used by the training path."""

from typing import Any

import torch


def basic_collation_fn(
    batch: list[dict[str, Any]], unstackable_keys: list[str] = []
) -> dict[str, Any]:
    """Stack tensor fields and retain non-tensor fields as per-sample lists.

    This preserves the recipe's original lightweight collation behavior while
    keeping the active model processor independent of the retired WebDataset
    stack.
    """

    stackable = {key: isinstance(value, torch.Tensor) for key, value in batch[0].items()}

    for key in unstackable_keys:
        if key in stackable:
            stackable[key] = False

    return {
        key: (
            torch.stack([row[key] for row in batch])
            if stackable[key]
            else [row[key] for row in batch]
        )
        for key in stackable
    }
