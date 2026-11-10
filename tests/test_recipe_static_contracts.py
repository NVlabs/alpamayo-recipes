# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
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
    # Check shipped configurations, not ignored environments or experiment outputs.
    structured_files = [
        path for path in TRACKED
        if RECIPES_DIR in path.parents
        and path.suffix in {".toml", ".lock", ".yaml", ".yml", ".json"}
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


def test_alpamayo2_sft_uses_release_native_model_and_held_out_validation_chunk() -> None:
    recipe_dir = RECIPES_DIR / "alpamayo2_sft"
    readme = recipe_dir / "README.md"
    base = _load_yaml(recipe_dir / "configs" / "sft_base.yaml")
    stage1 = _load_yaml(recipe_dir / "configs" / "sft_stage1.yaml")
    stage2 = _load_yaml(recipe_dir / "configs" / "sft_stage2.yaml")
    model = _load_yaml(recipe_dir / "configs" / "model" / "release.yaml")
    pyproject = _load_toml(recipe_dir / "pyproject.toml")

    _assert_text_contains(
        readme,
        [
            "nvidia/Alpamayo2-Super",
            "https://github.com/NVlabs/alpamayo2",
            "-m alpamayo2_sft.train",
            "-m alpamayo2_sft.evaluate",
            "--config-name sft_eval_loss",
            "--config-name sft_eval_trajectory",
            'chunk_ids="0-99"',
            'chunk_ids="99-100"',
        ],
    )

    assert base["data"]["train_dataset"]["chunk_ids"] == "0-99"
    assert base["data"]["val_dataset"]["chunk_ids"] == "99-100"
    assert len(base["camera_features"]) == 7
    assert base["num_frames"] == 4
    assert base["data"]["collator"]["max_length"] == 6144
    assert base["data"]["collator"]["pad_to_fixed_length"] is True
    assert stage1["model"]["training_stage"] == "vlm"
    assert stage2["model"]["training_stage"] == "expert"
    assert "data" not in stage2
    assert model["_target_"].endswith("TrainableAlpamayo2Super.from_pretrained")
    assert model["pretrained_model_name_or_path"] == "nvidia/Alpamayo2-Super"
    assert pyproject["project"]["dependencies"][0] == "alpamayo2_super"
    assert pyproject["tool"]["uv"]["sources"]["alpamayo2_super"] == {
        "git": "https://github.com/NVlabs/alpamayo2.git"
    }


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


# Edge-specific package, workflow and release contracts.
SFT_ROOT = ROOT
RECIPE = SFT_ROOT / "recipes/alpamayo2edge_sft"
SOURCE_ROOTS = (SFT_ROOT / "recipes", SFT_ROOT / "src")


def _tracked_files():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    return tuple(ROOT / name for name in sorted(set(names)) if name and (ROOT / name).is_file())


TRACKED = _tracked_files()
RECIPE_FILES = tuple(p for p in TRACKED if RECIPE in p.parents)
DOCS = tuple(p for p in RECIPE_FILES if p.suffix == ".md") + (ROOT / "README.md",)



def _module_exists(name: str) -> bool:
    parts = name.split(".")
    return any(
        root.joinpath(*parts).with_suffix(".py").is_file()
        or (root.joinpath(*parts) / "__init__.py").is_file()
        for root in SOURCE_ROOTS
    )


def _walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_structured_configs_parse_and_targets_are_shipped() -> None:
    for path in RECIPE_FILES:
        if path.suffix in {".toml", ".lock"}:
            tomllib.loads(path.read_text())
        elif path.suffix == ".json":
            json.loads(path.read_text())
        elif path.suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(path.read_text())
            assert isinstance(config, dict)
            for key, value in _walk(config):
                if key == "_target_":
                    assert value.startswith("alpamayo2edge_sft."), (path, value)
                    # A target can be a class, function, or class method.
                    parts = value.split(".")
                    assert any(_module_exists(".".join(parts[:end]))
                               for end in range(2, len(parts))), (path, value)


def test_runtime_imports_do_not_require_missing_recipe_backends() -> None:
    files = [p for p in RECIPE_FILES if p.suffix == ".py" and "tests" not in p.relative_to(RECIPE).parts]
    files += [ROOT / "src/alpamayo/data/pai_utils.py", ROOT / "scripts/prepare_pai_annotations.py"]
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                head = name.split(".")[0]
                assert head not in {"alpax", "alpamayo_r1", "alpamayo1_5", "alpamayo2_super"}, (
                    path, name
                )
                if head in {"alpamayo", "alpamayo2edge_sft"}:
                    assert _module_exists(name), f"{path} imports missing {name}"


@pytest.mark.parametrize("path", DOCS, ids=lambda path: str(path.relative_to(ROOT)))
def test_readme_relative_links_resolve(path: Path) -> None:
    text = re.sub(r"```.*?```", "", path.read_text(), flags=re.DOTALL)
    for raw in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        target = raw.split()[0].strip("<>")
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        # The upstream repository README links to GitHub's issue-template route.
        # This repository-relative web route does not name a checkout-local file.
        if path == ROOT / "README.md" and target == "../../issues/new/choose":
            continue
        linked = path.parent / unquote(url.path)
        assert linked.exists(), f"Broken link in {path.relative_to(ROOT)}: {target}"


@pytest.mark.parametrize("path", DOCS, ids=lambda path: str(path.relative_to(ROOT)))
def test_readme_shell_examples_have_valid_syntax_and_checkout_paths(path: Path) -> None:
    text = path.read_text()
    assert "$REPO_DIR/sft/" not in text, "Use the multi-recipe checkout layout"
    for script in re.findall(r"```(?:bash|sh|shell)\n(.*?)```", text, flags=re.DOTALL):
        result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        assert result.returncode == 0, f"Invalid shell in {path}: {result.stderr}"
        for relative in re.findall(r"\bpython\s+((?:scripts|recipes)/[^\s]+\.py)", script):
            assert (SFT_ROOT / relative).is_file(), f"Missing script in {path}: {relative}"
        for module in re.findall(r"\s-m\s+(alpamayo2edge_sft\.[\w.]+)", script):
            assert _module_exists(module), f"Missing entrypoint in {path}: {module}"
