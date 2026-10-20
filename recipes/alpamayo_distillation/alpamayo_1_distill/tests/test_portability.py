from __future__ import annotations

import ast
import json
from pathlib import Path

from alpamayo1_distill import defaults


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_INFRA_PREFIXES = ("/alpamayo/", "/data/", "/data1/", "/lustre/")


def _absolute_infrastructure_literals(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(_INFRA_PREFIXES):
                matches.append((node.lineno, node.value))
    return matches


def test_runtime_python_has_no_fixed_infrastructure_paths() -> None:
    offenders = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        matches = _absolute_infrastructure_literals(path)
        if matches:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = matches
    assert offenders == {}


def test_runtime_python_has_no_bundled_checkpoint_fallbacks() -> None:
    offenders = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        matches = [
            (node.lineno, node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.lower().endswith((".ckpt", ".pth"))
        ]
        if matches:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = matches
    assert offenders == {}


def test_packaged_configs_have_no_fixed_infrastructure_paths() -> None:
    offenders = {}
    for path in sorted(defaults.CONFIG_DIR.glob("*.json")):
        serialized = json.dumps(json.loads(path.read_text()), sort_keys=True)
        matches = [prefix for prefix in _INFRA_PREFIXES if f'"{prefix}' in serialized]
        if matches:
            offenders[path.name] = matches
    assert offenders == {}


def test_packaged_default_resources_exist() -> None:
    for path in (
        defaults.DEFAULT_CONFIG_PATH,
        defaults.DEFAULT_GOLD_CLIP_IDS_PATH,
    ):
        assert path.is_file(), path

    assert defaults.DEFAULT_CONFIG_PATH.name == "config_cosmos_2b.json"
    assert sorted(path.name for path in defaults.CONFIG_DIR.glob("config*.json")) == [
        "config_cosmos_2b.json"
    ]
    assert not (defaults.CONFIG_DIR / "clip_ids_small.txt").exists()
    assert defaults.PAI_GOLD_EVAL_SIZE == 644
    assert defaults.PAI_TRAIN_SPLIT_SIZE == 153_625
    assert defaults.PAI_GOLD_TRAIN_OVERLAP == 325
    assert defaults.PAI_TRAIN_SIZE == 153_300
    assert defaults.PAI_FULL_VAL_SIZE == 90_928


def test_saved_rollout_dataset_modules_are_not_packaged() -> None:
    assert not (PACKAGE_ROOT / "data_processing.py").exists()
    assert not (PACKAGE_ROOT / "dataloader" / "distilled_dataset.py").exists()


def test_public_recipe_does_not_ship_cluster_launchers() -> None:
    assert list(PACKAGE_ROOT.rglob("*.sbatch")) == []


def test_student_model_module_uses_2b_namespace() -> None:
    assert (PACKAGE_ROOT / "models" / "alpamayo_r1_distill_2b.py").is_file()
    assert not (PACKAGE_ROOT / "models" / "alpamayo_r1_distill_5b.py").exists()
