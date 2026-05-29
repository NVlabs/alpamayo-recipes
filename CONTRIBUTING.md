# Contributing to Alpamayo Recipes

Thank you for your interest in contributing to Alpamayo Recipes. This repository is a
collection of end-to-end workflows around released Alpamayo models, including fine-tuning,
RL post-training, evaluation (open-loop, close-loop, etc), data curation, and related utilities.

## Repository Structure

Alpamayo Recipes is organized as a set of recipe-specific Python packages, not as one
monolithic Python environment.

- `recipes/<recipe_name>/` contains one self-contained recipe.
- Each recipe owns its own `pyproject.toml`, `uv.lock`, README, configs, and importable Python
  module.
- Users install only the recipe they want to run by changing into that recipe directory and
  running `uv sync --active`.
- `src/alpamayo/` contains lightweight utilities shared across recipes, such as chat templates,
  data loaders, metrics, checkpoint helpers, visualization, and common helpers.
- `scripts/` contains repo-level utility scripts that are useful across multiple recipes.

This layout lets each recipe choose the dependencies, lockfile, model path assumptions, and
runtime setup that fit that workflow.

## Installing a Recipe

Each recipe README should tell users to install from inside the recipe directory:

```bash
cd alpamayo-recipes/recipes/<recipe_name>
uv venv <venv_name>
source <venv_name>/bin/activate
uv sync --active
```

The existing recipes depend on:

- `alpamayo_r1`, fetched from `https://github.com/NVlabs/alpamayo.git`, for released model code,
  processors, geometry, and inference-time components.
- `alpamayo-recipes`, installed editable from `../../src`, for shared recipe-side utilities.

New recipes should follow this pattern unless there is a concrete reason to do something
different.

## Adding a Recipe for Released Alpamayo Models

Put a new recipe under `recipes/<recipe_slug>/`, where `<recipe_slug>` is short, lowercase, and
descriptive. A typical recipe includes:

- `README.md` with the workflow, setup, inputs, commands, expected outputs, and limitations.
- `pyproject.toml` with only the dependencies needed by this recipe.
- `uv.lock` generated from that recipe directory.
- An importable Python package with a unique module name, usually using underscores, such as
  `alpamayo1_5_eval`.
- Config files under the recipe directory, such as Hydra YAML files or TOML files.
- Optional notebooks, images, and small example outputs when they make the recipe easier to
  validate.

Avoid adding a dependency to the repository root. Recipe-specific dependencies should stay in
the recipe's own `pyproject.toml` so users can opt into only the workflow they need.

For a packaged recipe, follow the existing `pyproject.toml` pattern:

```toml
[project]
name = "alpamayo1-5-eval"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
  "alpamayo_r1",
  "alpamayo-recipes",
]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = [".."]
include = ["alpamayo1_5_eval*"]

[tool.uv.sources]
alpamayo_r1 = { git = "https://github.com/NVlabs/alpamayo.git" }
alpamayo-recipes = { path = "../../src", editable = true }
```

If a recipe is specific to a model version, keep that version-specific choice in the recipe
package, README, configs, or `[tool.alpamayo]` settings rather than adding global behavior at the
repository root.

## Shared Code vs Recipe Code

Prefer recipe-local code when the behavior is specific to one workflow, model version, dataset
format, or runtime system.

Use `src/alpamayo/` when the code is broadly reusable across recipes. Good candidates include:

- dataset wrappers shared by multiple recipes
- metrics and metric runners
- chat template components
- checkpoint conversion helpers
- visualization helpers
- distributed, logging, and configuration utilities

Keep `src/alpamayo/` lightweight and avoid adding heavy dependencies there. If a shared utility
needs an optional heavy package, keep that dependency in the consuming recipe and import it only
where needed.

## README Expectations

Every recipe README should make the workflow reproducible for someone starting from a clean
checkout. Include:

- what the recipe does and which Alpamayo model versions it supports
- hardware assumptions, especially GPU count and memory
- installation commands from inside the recipe directory
- required model, dataset, and credential inputs
- exact commands to run the workflow
- expected outputs, metrics, logs, or files
- known limitations and what the recipe does not cover

Use local paths and environment variables consistently, and avoid hardcoding user-specific paths.
Do not include secrets, tokens, private dataset paths, generated checkpoints, or large artifacts in
the repository.

## Validation Before Opening a Pull Request

Before opening a PR, run the strongest practical validation for the files you changed.

For documentation-only changes:

```bash
git diff --check
```

For a new or changed recipe, also validate the recipe from its own directory:

```bash
cd recipes/<recipe_name>
uv sync --active
python -m compileall .
```

If the recipe includes tests, run them from the recipe environment. If full training or evaluation
requires gated datasets, large checkpoints, or multi-GPU hardware, include the smaller smoke test
you ran and document any heavyweight validation that maintainers would need to run separately.

## Pull Request Scope

Keep PRs focused. A new recipe PR should usually avoid unrelated refactors, broad dependency
updates, or changes to existing recipe behavior. If you need shared utilities under `src/alpamayo/`,
explain which recipes use them and why they belong in shared code.
