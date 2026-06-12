from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = ROOT / "recipes"
RECIPE_DIRS = tuple(sorted(path.parent for path in RECIPES_DIR.glob("*/pyproject.toml")))
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _workflow_job_block(workflow: str, job_name: str) -> str:
    lines = workflow.splitlines()
    start = lines.index(f"  {job_name}:")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            end = index
            break
    return "\n".join(lines[start:end])


def test_cpu_ci_workflow_runs_required_baseline_checks() -> None:
    workflow = CI_WORKFLOW.read_text()
    cpu_job = _workflow_job_block(workflow, "cpu")

    required_commands = [
        "uv build --project src --sdist --wheel",
        'git archive --format=tar HEAD | tar -x -C "$build_root"',
        "python -m compileall -q src scripts recipes",
        "for pyproject in recipes/*/pyproject.toml; do",
        'recipe_dir="$(dirname "$pyproject")"',
        'uv lock --check --project "$recipe_dir"',
        'uv build --project "$recipe_dir" --sdist --wheel',
        "uv run --with pytest --with pyyaml pytest tests -q",
    ]
    for command in required_commands:
        assert command in cpu_job

    heavy_runtime_terms = ("flash-attn", "vllm", "cuda", "gpu", "nvidia-smi")
    cpu_job_lower = cpu_job.lower()
    for term in heavy_runtime_terms:
        assert term not in cpu_job_lower

    assert "timeout-minutes:" in cpu_job
    assert "persist-credentials: false" in cpu_job


def test_shared_package_build_metadata_stays_lightweight() -> None:
    pyproject = _load_toml(ROOT / "src" / "pyproject.toml")

    assert pyproject["project"]["name"] == "alpamayo-recipes"
    assert pyproject["project"]["requires-python"] == "==3.12.*"
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["alpamayo"]


def test_recipe_packages_keep_recipe_local_metadata() -> None:
    assert RECIPE_DIRS

    for recipe_dir in RECIPE_DIRS:
        pyproject_path = recipe_dir / "pyproject.toml"
        readme_path = recipe_dir / "README.md"
        lockfile_path = recipe_dir / "uv.lock"

        assert pyproject_path.is_file(), f"{recipe_dir} must define recipe-local metadata"
        assert readme_path.is_file(), f"{recipe_dir} must document installation and usage"
        assert lockfile_path.is_file(), f"{recipe_dir} must pin a recipe-local uv lockfile"

        pyproject = _load_toml(pyproject_path)
        dependencies = pyproject["project"]["dependencies"]
        package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]
        uv_sources = pyproject["tool"]["uv"]["sources"]

        assert pyproject["project"]["requires-python"] == "==3.12.*"
        assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
        assert "alpamayo-recipes" in dependencies
        assert package_finder["where"] == [".."]
        assert package_finder["include"] == [f"{recipe_dir.name}*"]
        assert uv_sources["alpamayo-recipes"] == {"path": "../../src", "editable": True}
