from __future__ import annotations

import copy
import json
from pathlib import Path

import hydra.utils as hyu
import pytest
from accelerate import init_empty_weights
from transformers import AutoModel, Qwen3VLConfig, Qwen3VLTextConfig

from alpamayo1_distill.config import (
    AlpamayoR1Config,
    ExpertModelConfig,
    canonicalize_legacy_targets,
    validate_supported_student_vlm_config,
)
from alpamayo1_distill.defaults import (
    DEFAULT_COSMOS_STUDENT_MODEL,
    SUPPORTED_ACTION_EXPERT_ARCHITECTURE,
    SUPPORTED_ACTION_STACK_PARAMETER_COUNT,
    SUPPORTED_STUDENT_BACKEND,
    SUPPORTED_STUDENT_TEXT_ARCHITECTURE,
    SUPPORTED_STUDENT_VISION_ARCHITECTURE,
)
from alpamayo1_distill.models.base_model import ReasoningVLAConfig


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _packaged_student_config() -> dict:
    return json.loads((PACKAGE_ROOT / "config" / "config_cosmos_2b.json").read_text())


def _legacy_expert_config() -> ExpertModelConfig:
    legacy_config = _packaged_student_config()
    legacy_config["vlm_name_or_path"] = None
    for component in ("action_space_cfg", "action_in_proj_cfg", "diffusion_cfg"):
        legacy_config[component]["_target_"] = legacy_config[component]["_target_"].replace(
            "alpamayo1_distill.", "alpamayo_r1."
        )
    return ExpertModelConfig.from_dict(legacy_config)


def test_legacy_hydra_targets_are_canonicalized_recursively() -> None:
    legacy = {
        "_target_": "alpamayo_r1.models.action_in_proj.PerWaypointActionInProjV2",
        "nested": [
            {
                "_target_": (
                    "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace"
                )
            }
        ],
        "external": "torch.nn.Linear",
        "metadata": "alpamayo_r1.this_is_not_a_hydra_target",
    }

    canonical = canonicalize_legacy_targets(legacy)

    assert canonical["_target_"] == (
        "alpamayo1_distill.models.action_in_proj.PerWaypointActionInProjV2"
    )
    assert canonical["nested"][0]["_target_"] == (
        "alpamayo1_distill.action_space.UnicycleAccelCurvatureActionSpace"
    )
    assert canonical["external"] == "torch.nn.Linear"
    assert canonical["metadata"] == "alpamayo_r1.this_is_not_a_hydra_target"
    assert legacy["_target_"].startswith("alpamayo_r1.")


def test_released_teacher_model_type_remains_compatible() -> None:
    assert AlpamayoR1Config.model_type == "alpamayo_r1"
    assert ExpertModelConfig.model_type == "alpamayo_reasoning_vla_expert_distill"


def test_base_config_has_no_unshipped_tokenizer_checkpoint_fallback() -> None:
    config = ReasoningVLAConfig(vlm_name_or_path=None)
    assert config.traj_tokenizer_cfg is None


def test_packaged_student_uses_canonical_cosmos_2b_architecture() -> None:
    packaged = _packaged_student_config()
    expert = packaged["expert_cfg"]

    assert DEFAULT_COSMOS_STUDENT_MODEL == "nvidia/Cosmos-Reason2-2B"
    assert SUPPORTED_STUDENT_TEXT_ARCHITECTURE == {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    assert SUPPORTED_ACTION_EXPERT_ARCHITECTURE == {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    assert SUPPORTED_STUDENT_VISION_ARCHITECTURE == {
        "model_type": "qwen3_vl",
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [5, 11, 17],
    }
    assert packaged["vlm_name_or_path"] == DEFAULT_COSMOS_STUDENT_MODEL
    assert packaged["vlm_backend"] == SUPPORTED_STUDENT_BACKEND
    assert {
        field: expert[field] for field in SUPPORTED_ACTION_EXPERT_ARCHITECTURE
    } == SUPPORTED_ACTION_EXPERT_ARCHITECTURE

    # The expert consumes Cosmos's layer-wise KV cache. These dimensions must
    # agree even though the expert itself has a much smaller hidden width.
    for field in ("num_hidden_layers", "num_key_value_heads", "head_dim"):
        assert expert[field] == SUPPORTED_STUDENT_TEXT_ARCHITECTURE[field]
    assert expert["num_attention_heads"] % expert["num_key_value_heads"] == 0

    vlm_config = Qwen3VLConfig(
        text_config=SUPPORTED_STUDENT_TEXT_ARCHITECTURE,
        vision_config=SUPPORTED_STUDENT_VISION_ARCHITECTURE,
    )
    validate_supported_student_vlm_config(vlm_config)


def test_non_cosmos_backend_is_rejected() -> None:
    alternate = _packaged_student_config()
    alternate["vlm_backend"] = "qwen"

    with pytest.raises(ValueError, match="only supports Cosmos-Reason2-2B"):
        ExpertModelConfig.from_dict(alternate)


def test_alternate_action_expert_is_rejected() -> None:
    alternate = _packaged_student_config()
    alternate["expert_cfg"]["intermediate_size"] = 8256

    with pytest.raises(ValueError, match="action expert architecture"):
        ExpertModelConfig.from_dict(alternate)


def test_alternate_qwen3_vl_dimensions_are_rejected() -> None:
    alternate_text = copy.deepcopy(SUPPORTED_STUDENT_TEXT_ARCHITECTURE)
    alternate_text["hidden_size"] = 4096
    alternate_vlm = Qwen3VLConfig(
        text_config=alternate_text,
        vision_config=SUPPORTED_STUDENT_VISION_ARCHITECTURE,
    )

    with pytest.raises(ValueError, match="hidden_size"):
        validate_supported_student_vlm_config(alternate_vlm)


def test_alternate_vision_tower_is_rejected() -> None:
    alternate_vision = copy.deepcopy(SUPPORTED_STUDENT_VISION_ARCHITECTURE)
    alternate_vision["depth"] = 32
    alternate_vlm = Qwen3VLConfig(
        text_config=SUPPORTED_STUDENT_TEXT_ARCHITECTURE,
        vision_config=alternate_vision,
    )

    with pytest.raises(ValueError, match="vision architecture.*depth"):
        validate_supported_student_vlm_config(alternate_vlm)


def test_legacy_expert_inherits_missing_cache_dimensions() -> None:
    legacy = _packaged_student_config()
    legacy["expert_cfg"].pop("num_hidden_layers")
    legacy["expert_cfg"].pop("num_key_value_heads")

    config = ExpertModelConfig.from_dict(legacy)

    assert config.expert_cfg["num_hidden_layers"] == 28
    assert config.expert_cfg["num_key_value_heads"] == 8
    assert {
        field: config.expert_cfg[field]
        for field in SUPPORTED_ACTION_EXPERT_ARCHITECTURE
    } == SUPPORTED_ACTION_EXPERT_ARCHITECTURE


def test_supported_action_stack_has_exact_parameter_count() -> None:
    packaged = _packaged_student_config()
    expert_values = {
        field: packaged["expert_cfg"][field]
        for field in SUPPORTED_ACTION_EXPERT_ARCHITECTURE
    }
    text_config = Qwen3VLTextConfig(
        vocab_size=packaged["vocab_size"],
        dtype=packaged["expert_cfg"]["dtype"],
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        **expert_values,
    )
    action_dims = (packaged["action_space_cfg"]["n_waypoints"], 2)

    # Construct on meta so this exact architecture guard does not allocate the
    # 116M parameters it is intended to count.
    with init_empty_weights(include_buffers=True):
        expert = AutoModel.from_config(text_config)
        action_in = hyu.instantiate(
            packaged["action_in_proj_cfg"],
            in_dims=action_dims,
            out_dim=expert_values["hidden_size"],
        )
        action_out = hyu.instantiate(
            packaged["action_out_proj_cfg"],
            in_features=expert_values["hidden_size"],
            out_features=action_dims[-1],
        )

    # The action expert consumes projected actions, never token IDs.
    del expert.embed_tokens
    expert_count = sum(parameter.numel() for parameter in expert.parameters())
    action_in_count = sum(parameter.numel() for parameter in action_in.parameters())
    action_out_count = sum(parameter.numel() for parameter in action_out.parameters())

    assert next(expert.parameters()).device.type == "meta"
    assert expert_count == 115_634_560
    assert action_in_count == 492_672
    assert action_out_count == 770
    assert expert_count + action_in_count + action_out_count == (
        SUPPORTED_ACTION_STACK_PARAMETER_COUNT
    )


def test_legacy_expert_config_loads_and_instantiates_hydra_components() -> None:
    config = _legacy_expert_config()

    assert config.model_type == "alpamayo_reasoning_vla_expert_distill"
    assert config.action_space_cfg["_target_"].startswith("alpamayo1_distill.")
    action_space = hyu.instantiate(config.action_space_cfg)
    action_in_proj = hyu.instantiate(
        config.action_in_proj_cfg,
        in_dims=action_space.get_action_space_dims(),
        out_dim=64,
    )
    assert action_in_proj is not None


def test_legacy_diffusion_target_instantiates_with_runtime_dependencies() -> None:
    pytest.importorskip("diffusers")
    config = _legacy_expert_config()
    action_space = hyu.instantiate(config.action_space_cfg)

    diffusion = hyu.instantiate(
        config.diffusion_cfg,
        x_dims=action_space.get_action_space_dims(),
    )
    assert diffusion is not None


def test_packaged_configs_use_canonical_hydra_targets() -> None:
    for config_path in sorted((PACKAGE_ROOT / "config").glob("*.json")):
        config = json.loads(config_path.read_text())
        serialized = json.dumps(config, sort_keys=True)
        assert "alpamayo_r1." not in serialized, config_path
