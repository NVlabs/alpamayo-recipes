# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
from alpamayo1_5_sft.models.sft_base_model import load_alpamayo1_vlm
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


def _write_vlm_checkpoint(checkpoint_dir: Path, state_dict: dict[str, torch.Tensor]) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    save_safetensors_file(state_dict, checkpoint_dir / shard_name)
    index = {"weight_map": {key: shard_name for key in state_dict}}
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def _tiny_qwen3_vl() -> Qwen3VLForConditionalGeneration:
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "rope_scaling": {"mrope_section": [2, 2, 4], "rope_type": "default"},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 8,
        },
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
    )
    return Qwen3VLForConditionalGeneration(config)


def test_load_alpamayo1_vlm_strips_prefix_for_nested_vlm(tmp_path: Path) -> None:
    model = _tiny_qwen3_vl()
    key = "model.language_model.embed_tokens.weight"
    replacement = torch.full_like(model.state_dict()[key], 2)
    _write_vlm_checkpoint(tmp_path, {f"vlm.{key}": replacement})

    load_alpamayo1_vlm(str(tmp_path), model)

    torch.testing.assert_close(model.state_dict()[key], replacement)


def test_load_alpamayo1_vlm_accepts_unprefixed_nested_vlm(tmp_path: Path) -> None:
    model = _tiny_qwen3_vl()
    key = "model.language_model.embed_tokens.weight"
    replacement = torch.full_like(model.state_dict()[key], 3)
    _write_vlm_checkpoint(tmp_path, {key: replacement})

    load_alpamayo1_vlm(str(tmp_path), model)

    torch.testing.assert_close(model.state_dict()[key], replacement)


def test_load_alpamayo1_vlm_keeps_prefix_for_full_model(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.vlm = torch.nn.Linear(1, 1, bias=False)
    replacement = torch.full_like(model.vlm.weight, 3)
    _write_vlm_checkpoint(tmp_path, {"vlm.weight": replacement})

    load_alpamayo1_vlm(str(tmp_path), model)

    torch.testing.assert_close(model.vlm.weight, replacement)


def test_load_alpamayo1_vlm_rejects_unmatched_keys(tmp_path: Path) -> None:
    _write_vlm_checkpoint(tmp_path, {"vlm.unknown": torch.ones(1)})

    with pytest.raises(ValueError, match="do not match the target model"):
        load_alpamayo1_vlm(str(tmp_path), torch.nn.Linear(1, 1))


def test_load_alpamayo1_vlm_rejects_indexed_tensor_missing_from_shard(
    tmp_path: Path,
) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    _write_vlm_checkpoint(tmp_path, {"vlm.weight": torch.ones(1, 1)})
    index = {
        "weight_map": {
            "vlm.weight": shard_name,
            "vlm.bias": shard_name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    model = torch.nn.Module()
    model.vlm = torch.nn.Linear(1, 1)

    with pytest.raises(ValueError, match="missing.*vlm.bias"):
        load_alpamayo1_vlm(str(tmp_path), model)


def test_load_alpamayo1_vlm_rejects_meta_target_when_preserving_placement(
    tmp_path: Path,
) -> None:
    _write_vlm_checkpoint(tmp_path, {"weight": torch.ones(1, 1)})
    model = torch.nn.Linear(1, 1, bias=False, device="meta")

    with pytest.raises(ValueError, match="meta.*cannot preserve"):
        load_alpamayo1_vlm(
            str(tmp_path),
            model,
            preserve_model_device_and_dtype=True,
        )


def test_from_pretrained_applies_stage1_after_parent_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = torch.full((1, 1), 2.0)
    _write_vlm_checkpoint(tmp_path, {"vlm.weight": replacement})

    def fake_parent_from_pretrained(
        cls: type[TrainableAlpamayoR1],
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> TrainableAlpamayoR1:
        assert pretrained_model_name_or_path == "base-checkpoint"
        assert "stage1_vlm_checkpoint_path" not in kwargs
        model = object.__new__(cls)
        torch.nn.Module.__init__(model)
        model.vlm = torch.nn.Linear(1, 1, bias=False, dtype=torch.float16)
        model.vlm.weight.data.fill_(1)
        model.cotrain_vlm = kwargs["cotrain_vlm"]
        return model

    monkeypatch.setattr(AlpamayoR1, "from_pretrained", classmethod(fake_parent_from_pretrained))

    model = TrainableAlpamayoR1.from_pretrained(
        "base-checkpoint",
        stage1_vlm_checkpoint_path=str(tmp_path),
        cotrain_vlm=False,
    )

    torch.testing.assert_close(model.vlm.weight, replacement.to(model.vlm.weight))
    assert model.vlm.weight.dtype == torch.float16
    assert not model.vlm.weight.requires_grad


def test_constructor_rejects_stage1_checkpoint() -> None:
    with pytest.raises(ValueError, match="only supported by.*from_pretrained"):
        TrainableAlpamayoR1(None, stage1_vlm_checkpoint_path="stage1-checkpoint")  # type: ignore[arg-type]
