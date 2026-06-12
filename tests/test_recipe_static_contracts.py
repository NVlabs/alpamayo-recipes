from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = ROOT / "recipes"


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict), f"{path} must contain a YAML mapping"
    return loaded


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assert_text_contains(path: Path, expected_snippets: list[str]) -> None:
    text = _read(path)
    for snippet in expected_snippets:
        assert snippet in text, f"{path} must document {snippet!r}"


def _defaults_include_override(config: dict, key: str, value: str) -> bool:
    return any(isinstance(item, dict) and item.get(key) == value for item in config["defaults"])


def test_recipe_structured_files_parse() -> None:
    """Every recipe config file should be syntactically parseable."""
    structured_files = [
        *RECIPES_DIR.glob("*/*.toml"),
        *RECIPES_DIR.glob("*/*.lock"),
        *RECIPES_DIR.glob("*/*/**/*.toml"),
        *RECIPES_DIR.glob("*/*/**/*.yaml"),
        *RECIPES_DIR.glob("*/*/**/*.json"),
    ]
    assert structured_files

    for path in structured_files:
        if path.suffix in {".toml", ".lock"}:
            _load_toml(path)
        elif path.suffix in {".yaml", ".yml"}:
            _load_yaml(path)
        elif path.suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            assert loaded is not None


def test_alpamayo1_sft_stage_configs_preserve_training_contract() -> None:
    recipe_dir = RECIPES_DIR / "alpamayo1_sft"
    readme = recipe_dir / "README.md"
    base = _load_yaml(recipe_dir / "configs" / "sft_base.yaml")
    stage1 = _load_yaml(recipe_dir / "configs" / "sft_stage1.yaml")
    stage2 = _load_yaml(recipe_dir / "configs" / "sft_stage2.yaml")
    processor = _load_yaml(recipe_dir / "configs" / "vla_processor" / "default.yaml")
    base_model = _load_yaml(recipe_dir / "configs" / "models" / "ar1_base.yaml")
    expert_model = _load_yaml(recipe_dir / "configs" / "models" / "ar1_expert.yaml")

    _assert_text_contains(
        readme,
        [
            "-m alpamayo1_sft.train_hf",
            "-m alpamayo1_sft.evaluate_hf",
            "--config-name sft_stage1",
            "--config-name sft_stage2",
            "+data.train_dataset.reasoning_metadata",
            "+data.train_dataset.clip_index_metadata",
            "use_default_keyframe=false",
        ],
    )

    assert "/models/ar1_base@model" in stage1["defaults"]
    assert "/models/ar1_expert@model" in stage2["defaults"]
    assert base["data"]["collate_fn"]["chat_template_version"] == "r1"
    assert base["data"]["train_dataset"]["_target_"] == "alpamayo.data.pai.PAIDataset"
    assert base["data"]["train_dataset"]["vla_preprocess_args"]["generation_mode"] is False
    assert base["data"]["val_dataset"]["vla_preprocess_args"]["generation_mode"] is True
    assert base["trainer"]["deepspeed"] == "configs/deepspeed/zero2.json"
    assert base["trainer"]["gradient_checkpointing"] is True
    assert stage2["trainer"]["deepspeed"] is None
    assert stage2["trainer"]["gradient_checkpointing"] is False

    assert processor["chat_template_version"] == "r1"
    assert processor["components_order"] == ["image", "traj_history", "prompt", "traj_future"]
    assert processor["label_components"] == ["traj_future"]
    assert base_model["_target_"].endswith("TrainableReasoningVLA.from_alpamayo_checkpoint")
    assert expert_model["pretrained_model_name_or_path"] == "nvidia/Alpamayo-R1-10B"
    assert expert_model["cotrain_vlm"] is False


def test_alpamayo1_5_sft_configs_preserve_nav_and_lingoqa_contracts() -> None:
    recipe_dir = RECIPES_DIR / "alpamayo1_5_sft"
    readme = recipe_dir / "README.md"
    base = _load_yaml(recipe_dir / "configs" / "sft_base.yaml")
    nav_stage1 = _load_yaml(recipe_dir / "configs" / "sft_stage1_nav.yaml")
    nav_stage2 = _load_yaml(recipe_dir / "configs" / "sft_stage2_nav.yaml")
    lingoqa_stage1 = _load_yaml(recipe_dir / "configs" / "sft_stage1_lingoqa.yaml")
    processors = {
        path.stem: _load_yaml(path)
        for path in sorted((recipe_dir / "configs" / "vla_processor").glob("*.yaml"))
    }
    base_model = _load_yaml(recipe_dir / "configs" / "models" / "ar1_5_base.yaml")
    expert_model = _load_yaml(recipe_dir / "configs" / "models" / "ar1_5_expert.yaml")

    _assert_text_contains(
        readme,
        [
            "-m alpamayo1_5_sft.train_hf",
            "-m alpamayo1_5_sft.evaluate_hf",
            "--config-name sft_stage1_nav",
            "--config-name sft_stage2_nav",
            "--config-name sft_stage1_lingoqa",
            "[vla_processor](./configs/vla_processor/)",
            "`nav`",
            "`vqa`",
        ],
    )

    assert base["data"]["collate_fn"]["chat_template_version"] == "r1_5"
    assert "/models/ar1_5_base@model" in nav_stage1["defaults"]
    assert "/models/ar1_5_expert@model" in nav_stage2["defaults"]
    assert "/models/ar1_5_base@model" in lingoqa_stage1["defaults"]

    for config in (nav_stage1, nav_stage2):
        assert config["data"]["train_dataset"]["_target_"] == "alpamayo.data.pai_nav.PAIDatasetWithNav"
        assert config["data"]["val_dataset"]["_target_"] == "alpamayo.data.pai_nav.PAIDatasetWithNav"
        assert config["data"]["train_dataset"]["vla_preprocess_args"]["generation_mode"] is False
        assert config["data"]["val_dataset"]["vla_preprocess_args"]["generation_mode"] is True
        assert _defaults_include_override(
            config,
            "override /vla_processor@data.train_dataset.vla_preprocess_args",
            "nav",
        )
        assert _defaults_include_override(
            config,
            "override /vla_processor@data.val_dataset.vla_preprocess_args",
            "nav",
        )

    assert lingoqa_stage1["data"]["train_dataset"]["_target_"] == "alpamayo.data.lingoqa.LingoQADataset"
    assert lingoqa_stage1["data"]["val_dataset"]["_target_"] == "alpamayo.data.lingoqa.LingoQADataset"
    assert _defaults_include_override(
        lingoqa_stage1,
        "override /vla_processor@data.train_dataset.vla_preprocess_args",
        "vqa",
    )
    assert _defaults_include_override(
        lingoqa_stage1,
        "override /vla_processor@data.val_dataset.vla_preprocess_args",
        "vqa",
    )
    assert nav_stage2["trainer"]["deepspeed"] is None
    assert nav_stage2["trainer"]["gradient_checkpointing"] is False

    assert processors["default"]["components_order"] == ["image", "traj_history", "prompt", "traj_future"]
    assert processors["nav"]["components_order"] == ["image", "traj_history", "route", "prompt", "traj_future"]
    assert processors["vqa"]["components_order"] == ["image", "question", "answer"]
    assert processors["nav"]["include_camera_ids"] is True
    assert processors["nav"]["include_frame_nums"] is True
    assert processors["vqa"]["label_components"] == ["answer"]
    assert base_model["_target_"].endswith("TrainableReasoningVLA.from_alpamayo_checkpoint")
    assert expert_model["pretrained_model_name_or_path"] == "nvidia/Alpamayo-1.5-10B"


def test_alpamayo1_x_rl_toml_configs_preserve_local_launch_contract() -> None:
    recipe_dir = RECIPES_DIR / "alpamayo1_x_rl"
    motion = _load_toml(recipe_dir / "toml" / "alpamayo_rvla_rl_local_test.toml")
    reasoning = _load_toml(recipe_dir / "toml" / "alpamayo_rvla_rl_local_test_with_reasoning.toml")

    for config in (motion, reasoning):
        assert config["rollout"]["backend"] == "reasoning_vla_vllm_rollout"
        assert config["train"]["train_policy"]["type"] == "grpo"
        assert config["train"]["train_policy"]["trainer_type"] == "reasoning_vla_grpo"
        assert config["policy"]["model_name_or_path"]
        assert config["policy"]["model_gradient_checkpointing"] is True
        assert config["rollout"]["parallelism"]["n_init_replicas"] == 1
        assert config["policy"]["parallelism"]["dp_shard_size"] == 4
        assert config["custom"]["alpamayo"]["prefetch"]["capacity"] > 0
        assert config["custom"]["alpamayo"]["prefetch"]["num_workers"] > 0

    assert motion["custom"]["alpamayo"]["reward"] == {
        "traj_l2_weight": 0.4,
        "comfort_weight": 0.1,
    }
    assert reasoning["custom"]["alpamayo"]["reward"] == {
        "traj_l2_weight": 0.2,
        "comfort_weight": 0.0,
        "reasoning_weight": 0.3,
    }
    assert reasoning["custom"]["alpamayo"]["reasoning_grader_type"] == "lingo_judge"
    assert reasoning["custom"]["alpamayo"]["reasoning_grading_device"] == "auto"
    assert reasoning["custom"]["alpamayo"]["reasoning_grading_model_path"]


def test_alpamayo1_x_rl_hydra_configs_preserve_dataset_contract() -> None:
    recipe_dir = RECIPES_DIR / "alpamayo1_x_rl"
    hydra_configs = {
        path.stem: _load_yaml(path)
        for path in sorted((recipe_dir / "hydra_configs").glob("*.yaml"))
    }
    assert set(hydra_configs) == {"alpamayo1_5_rvla_rl_pai", "alpamayo1_rvla_rl_pai"}

    for name, config in hydra_configs.items():
        dataset = config["data"]["train"]["dataset"]
        preprocess = dataset["vla_preprocess_args"]
        assert dataset["_target_"] == "alpamayo.data.pai.PAIDataset"
        assert dataset["reshape_tensors_for_rl"] is True
        assert dataset["use_default_keyframe"] is False
        assert dataset["clip_index_metadata"] == "clip_index.parquet"
        assert dataset["features_metadata"] == "features.csv"
        assert dataset["model_config"]["vlm_name_or_path"] == "nvidia/Cosmos-Reason2-8B"
        assert preprocess["_target_"] == "alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config"
        assert preprocess["components_order"] == ["image", "traj_history", "prompt", "cot"]
        assert preprocess["components_prompt"] == ["cot", "traj_future"]
        assert preprocess["generation_mode"] is True
        assert config["data"]["train"]["_target_"] == "torch.utils.data.DataLoader"

    assert hydra_configs["alpamayo1_rvla_rl_pai"]["data"]["train"]["dataset"]["vla_preprocess_args"][
        "include_camera_ids"
    ] is False
    assert hydra_configs["alpamayo1_rvla_rl_pai"]["data"]["train"]["dataset"]["vla_preprocess_args"][
        "include_frame_nums"
    ] is False
    assert hydra_configs["alpamayo1_5_rvla_rl_pai"]["data"]["train"]["dataset"]["vla_preprocess_args"][
        "include_camera_ids"
    ] is True
    assert hydra_configs["alpamayo1_5_rvla_rl_pai"]["data"]["train"]["dataset"]["vla_preprocess_args"][
        "include_frame_nums"
    ] is True


def test_alpamayo1_x_rl_entrypoints_preserve_reward_and_dataset_wiring() -> None:
    recipe_dir = RECIPES_DIR / "alpamayo1_x_rl"
    readme = recipe_dir / "README.md"
    motion_entry = recipe_dir / "models" / "reasoning_vla" / "alpamayo_cosmos_rl_post_training_entry.py"
    reasoning_entry = (
        recipe_dir / "models" / "reasoning_vla" / "alpamayo_cosmos_rl_post_training_reasoning_entry.py"
    )

    _assert_text_contains(
        readme,
        [
            "cosmos-rl",
            "alpamayo_rvla_rl_local_test.toml",
            "alpamayo_rvla_rl_local_test_with_reasoning.toml",
            "ALPAMAYO_PAI_LOCAL_DIR",
            "ALPAMAYO_PAI_REASONING_LOCAL_DIR",
            "convert_cosmos_rl_checkpoint.py",
        ],
    )

    _assert_text_contains(
        motion_entry,
        [
            'os.getenv("ALPAMAYO_PAI_LOCAL_DIR")',
            "aggregated_reward import compute_reward",
            'hydra_config_name="alpamayo1_5_rvla_rl_pai"',
            "clip_index_mini.parquet",
            "reasoning_metadata=null",
        ],
    )
    _assert_text_contains(
        reasoning_entry,
        [
            'os.getenv("ALPAMAYO_PAI_REASONING_LOCAL_DIR")',
            "aggregated_reward_with_reasoning import compute_reward",
            'hydra_config_name="alpamayo1_5_rvla_rl_pai"',
            "clip_index_reasoning_mini.parquet",
            "reasoning/ood_reasoning.parquet",
        ],
    )


def test_alpamayo1_x_rl_launcher_reads_policy_checkpoint_path_without_heavy_imports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "cosmos.toml"
    config_path.write_text('[policy]\nmodel_name_or_path = "/tmp/model"\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["cosmos-rl", "--config", str(config_path), "entry.py"])

    launcher_path = RECIPES_DIR / "alpamayo1_x_rl" / "launcher.py"
    spec = importlib.util.spec_from_file_location("alpamayo1_x_rl_launcher_test", launcher_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._read_ckpt_path_from_toml() == "/tmp/model"
