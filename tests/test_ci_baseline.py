from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = ROOT / "recipes"
RECIPE_DIRS = tuple(sorted(path.parent for path in RECIPES_DIR.glob("*/pyproject.toml")))
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
STAGE1_VLM_WORKFLOW = ROOT / ".github" / "workflows" / "stage1-vlm-cpu.yml"
STAGE1_VLM_REQUIREMENTS = ROOT / ".github" / "requirements" / "stage1-vlm-cpu.in"
STAGE1_VLM_PYLOCK = ROOT / ".github" / "requirements" / "pylock.stage1-vlm-cpu.toml"
STAGE1_VLM_RECIPE_LOCKS = (
    RECIPES_DIR / "alpamayo1_sft" / "uv.lock",
    RECIPES_DIR / "alpamayo1_5_sft" / "uv.lock",
)


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _packages_by_name(lock: dict) -> dict[str, dict]:
    return {package["name"]: package for package in lock["package"]}


def _contains_sha256(value: object) -> bool:
    if isinstance(value, dict):
        return "sha256" in value or any(
            _contains_sha256(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_sha256(item) for item in value)
    return False


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


def test_stage1_vlm_cpu_workflow_runs_targeted_loader_tests() -> None:
    assert STAGE1_VLM_WORKFLOW.is_file(), "Stage-1 VLM CPU workflow is missing"
    workflow = STAGE1_VLM_WORKFLOW.read_text()

    required_snippets = [
        "pull_request:",
        "push:",
        "branches:\n      - main",
        "workflow_dispatch:",
        ".github/workflows/**",
        ".github/requirements/stage1-vlm-cpu.in",
        ".github/requirements/pylock.stage1-vlm-cpu.toml",
        "recipes/alpamayo1_sft/**",
        "recipes/alpamayo1_5_sft/**",
        "src/alpamayo/**",
        "tests/test_ci_baseline.py",
        "contents: read",
        "group: stage1-vlm-cpu-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress: true",
        "timeout-minutes: 10",
        "uses: actions/checkout@v4",
        "uses: actions/setup-python@v5",
        "uses: astral-sh/setup-uv@v6",
        'version: "0.11.19"',
        "cache-suffix: stage1-vlm-cpu",
        "repository: NVlabs/alpamayo",
        "ref: ${{ steps.alpamayo_source.outputs.ref }}",
        "src/alpamayo_r1",
        'HF_HUB_OFFLINE: "1"',
        'TRANSFORMERS_OFFLINE: "1"',
        "uv pip sync",
        "--torch-backend cpu",
        "--only-binary :all:",
        "--require-hashes",
        "recipes/alpamayo1_sft/tests/test_stage1_vlm_loading.py",
        "recipes/alpamayo1_5_sft/tests/test_stage1_vlm_loading.py",
    ]
    for snippet in required_snippets:
        assert snippet in workflow

    assert workflow.count("uses: actions/checkout@v4") == 2
    assert workflow.count("python -m pytest") == 2
    assert "pull_request_target:" not in workflow
    assert "${{ secrets." not in workflow

    heavy_runtime_terms = ("flash-attn", "vllm", "cuda", "deepspeed", "cosmos-rl")
    workflow_lower = workflow.lower()
    for term in heavy_runtime_terms:
        assert term not in workflow_lower

    forbidden_commands = (
        "uv sync",
        "uv build",
        "uv pip install",
        "pip install",
        "python -m build",
    )
    for command in forbidden_commands:
        assert command not in workflow


def test_stage1_vlm_cpu_dependencies_match_recipe_locks() -> None:
    assert STAGE1_VLM_REQUIREMENTS.is_file(), (
        "Stage-1 VLM CPU requirements input is missing"
    )
    assert STAGE1_VLM_PYLOCK.is_file(), "Stage-1 VLM CPU pylock is missing"

    direct_requirements = {}
    for line in STAGE1_VLM_REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        assert separator, f"CPU requirement must use an exact pin: {line}"
        direct_requirements[name] = version

    expected_names = {
        "colorlog",
        "einops",
        "hydra-core",
        "numpy",
        "pytest",
        "rich",
        "safetensors",
        "scipy",
        "torch",
        "torchvision",
        "transformers",
    }
    assert set(direct_requirements) == expected_names

    alpamayo_sources = set()
    for lock_path in STAGE1_VLM_RECIPE_LOCKS:
        packages = _packages_by_name(_load_toml(lock_path))
        for name, version in direct_requirements.items():
            assert packages[name]["version"] == version
        alpamayo_sources.add(packages["alpamayo-r1"]["source"]["git"])

    assert len(alpamayo_sources) == 1
    alpamayo_source = alpamayo_sources.pop()
    repository_url, separator, revision = alpamayo_source.rpartition("#")
    assert separator and repository_url == "https://github.com/NVlabs/alpamayo.git"
    assert len(revision) == 40

    pylock = _load_toml(STAGE1_VLM_PYLOCK)
    locked_packages = {package["name"]: package for package in pylock["packages"]}
    for name, version in direct_requirements.items():
        assert locked_packages[name]["version"].partition("+")[0] == version

    forbidden_fragments = (
        "alpamayo",
        "cosmos",
        "cuda",
        "deepspeed",
        "flash-attn",
        "nvidia",
        "vllm",
    )
    assert not any(
        fragment in package_name
        for package_name in locked_packages
        for fragment in forbidden_fragments
    )

    sdist_packages = {
        package["name"] for package in pylock["packages"] if "sdist" in package
    }
    assert sdist_packages == {"antlr4-python3-runtime"}
    for package in pylock["packages"]:
        artifacts = [*package.get("wheels", [])]
        if "sdist" in package:
            artifacts.append(package["sdist"])
        assert artifacts
        assert all(_contains_sha256(artifact) for artifact in artifacts)

    for package_name in ("torch", "torchvision"):
        wheel_urls = [wheel["url"] for wheel in locked_packages[package_name]["wheels"]]
        assert wheel_urls and all("/whl/cpu/" in url for url in wheel_urls)


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
