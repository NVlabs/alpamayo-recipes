# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from alpamayo1_5_distill.diffusion.flow_matching import FlowMatching
from alpamayo1_5_distill.models.configuration_distill import SMALL_ACTION_EXPERT_CFG
from alpamayo1_5_distill.models.modeling_distill import (
    DistillAlpamayo1_5,
    _remove_synthesized_collation_masks,
)
from transformers import DynamicCache


def test_real_attention_mask_survives_collation_mask_filter() -> None:
    tokenized_data = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "input_ids_mask": torch.ones(1, 3, dtype=torch.bool),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "attention_mask_mask": torch.ones(1, 3, dtype=torch.bool),
        "pixel_attention_mask": torch.ones(1, 3, dtype=torch.bool),
    }

    filtered = _remove_synthesized_collation_masks(tokenized_data)

    assert set(filtered) == {
        "input_ids",
        "attention_mask",
        "pixel_attention_mask",
    }
    assert filtered["attention_mask"] is tokenized_data["attention_mask"]


def test_compact_expert_matches_reference_configuration() -> None:
    assert SMALL_ACTION_EXPERT_CFG == {
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_attention_heads": 16,
        "head_dim": 128,
    }


def test_flow_training_pair_follows_straight_path() -> None:
    target = torch.tensor([[[2.0, 4.0]], [[3.0, 5.0]]])
    noise = torch.tensor([[[0.0, 2.0]], [[1.0, 1.0]]])
    timesteps = torch.tensor([0.25, 0.75])
    flow = FlowMatching(x_dims=(1, 2))

    noisy, sampled_t, vector = flow.sample_training_pair(target, noise=noise, timesteps=timesteps)

    expected = (1 - timesteps[:, None, None]) * noise + timesteps[:, None, None] * target
    torch.testing.assert_close(noisy, expected)
    torch.testing.assert_close(sampled_t, timesteps)
    torch.testing.assert_close(vector, target - noise)


def test_flow_training_uses_reference_beta_timestep_distribution() -> None:
    torch.manual_seed(0)
    flow = FlowMatching(x_dims=(1, 2))
    target = torch.zeros(20_000, 1, 2)

    _, timesteps, _ = flow.sample_training_pair(target)

    assert flow.train_timestep_sampler == "beta"
    assert 0.38 < timesteps.mean().item() < 0.42
    assert timesteps.min() >= 0
    assert timesteps.max() <= 0.999


def test_compact_expert_is_cache_compatible_with_cosmos_2b() -> None:
    vlm = SimpleNamespace(num_hidden_layers=28, num_key_value_heads=8, head_dim=128)
    expert = SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=16,
    )

    DistillAlpamayo1_5._validate_expert_cache_compatibility(vlm, expert)


def test_expert_may_use_a_prefix_of_teacher_cache_layers() -> None:
    vlm = SimpleNamespace(num_hidden_layers=36, num_key_value_heads=8, head_dim=128)
    expert = SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        hidden_size=384,
        intermediate_size=1536,
        num_attention_heads=16,
    )

    DistillAlpamayo1_5._validate_expert_cache_compatibility(vlm, expert)


def test_detach_kv_cache_stops_action_gradient_path() -> None:
    cache = DynamicCache()
    keys = torch.zeros(1, 2, 3, 4, requires_grad=True)
    values = torch.ones(1, 2, 3, 4, requires_grad=True)
    cache.update(keys, values, 0)

    detached = DistillAlpamayo1_5._detach_kv_cache(cache)

    assert detached.layers[0].keys.requires_grad is False
    assert detached.layers[0].values.requires_grad is False


def test_cache_incompatible_expert_is_rejected() -> None:
    vlm = SimpleNamespace(num_hidden_layers=28, num_key_value_heads=8, head_dim=128)
    expert = SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=4,
        head_dim=128,
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=8,
    )

    with pytest.raises(ValueError, match="num_key_value_heads"):
        DistillAlpamayo1_5._validate_expert_cache_compatibility(vlm, expert)
