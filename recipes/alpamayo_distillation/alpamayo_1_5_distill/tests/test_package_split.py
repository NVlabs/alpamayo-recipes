# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import fnmatch
import tomllib
from importlib.resources import files
from pathlib import Path


def test_distill_config_has_unique_registry_identity() -> None:
    recipe_root = Path(__file__).parents[1]
    module = ast.parse(
        (recipe_root / "models/configuration_distill.py").read_text(encoding="utf-8")
    )
    model_types = [
        node.value.value
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "model_type" for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    ]
    assert model_types == ["alpamayo1_5_distill"]


def test_package_import_registers_distill_config() -> None:
    import alpamayo1_5_distill
    from transformers import AutoConfig

    config = AutoConfig.for_model("alpamayo1_5_distill")
    assert isinstance(config, alpamayo1_5_distill.DistillAlpamayo1_5Config)


def test_packaged_validation_assets_exist() -> None:
    package = files("alpamayo1_5_distill")
    assert package.joinpath("assets/clip_ids_eval_gold.txt").is_file()


def test_gold_parquet_is_not_in_package_data() -> None:
    recipe_root = Path(__file__).parents[1]
    project = tomllib.loads((recipe_root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["package-data"]["alpamayo1_5_distill"]
    metadata = "assets/1005_7cam_gold_eval_metadb_public.parquet"
    assert not any(fnmatch.fnmatchcase(metadata, pattern) for pattern in patterns)


def test_upstream_base_model_was_not_copied() -> None:
    recipe_root = Path(__file__).parents[1]
    assert not (recipe_root / "models/base_model.py").exists()


def test_public_recipe_does_not_ship_cluster_launchers() -> None:
    recipe_root = Path(__file__).parents[1]
    assert list(recipe_root.rglob("*.sbatch")) == []


def test_public_configs_do_not_contain_internal_paths() -> None:
    recipe_root = Path(__file__).parents[1]
    forbidden = ("/lustre/", "/data1/", "/home/chefu", "/home/jinhu", "/home/zewenl")
    for config_path in (recipe_root / "configs").glob("*.yaml"):
        contents = config_path.read_text(encoding="utf-8")
        assert all(value not in contents for value in forbidden), config_path
