## Alpamayo Recipes OSS Contribution Rules

#### Issue Tracking

- All enhancement, bugfix, or change requests must begin with the creation of an [Alpamayo Recipes Issue Request](https://github.com/NVlabs/alpamayo-recipes/issues).
  - The issue request must be reviewed by Alpamayo researchers and approved prior to code review.

#### Coding Guidelines

- Please follow the existing conventions in the relevant file, submodule, module, and project when you add new code or when you extend/fix existing functionality.

- To maintain consistency in code formatting and style, you should also run `pre-commit format` on the modified sources with the provided configuration file. This applies Alpamayo code formatting rules to:

  - class, function/method, and variable/field naming
  - comment style
  - indentation
  - line length

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Try to keep pull requests (PRs) as concise as possible:

  - Avoid committing commented-out code.
  - Wherever possible, each PR should address a single concern. If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate the dependencies in the description. The more complex the changes are in a single PR, the more time it will take to review those changes.

- Write commit titles using imperative mood and [these rules](https://chris.beams.io/posts/git-commit/), and reference the Issue number corresponding to the PR. Following is the recommended format for commit texts:

```
#<Issue Number> - <Commit Title>

<Commit Body>
```

- Ensure that the build log is clean, meaning no warnings or errors should be present.

- Ensure that all tests pass prior to submitting your code.

- All OSS components must contain accompanying documentation (READMEs) describing the functionality, dependencies, and known issues.

  - See `README.md` for existing samples and plugins for reference.

- All OSS components must have an accompanying test.

  - If introducing a new component, provide a test sample to verify the functionality.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.

- Thanks in advance for your patience as we review your contributions; we do appreciate them!

#### Pull Requests

Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/NVlabs/alpamayo-recipes) Alpamayo Recipes OSS repository.

2. Git clone the forked repository and push changes to the personal fork.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git Alpamayo-Recipes
# Checkout the targeted branch and commit changes
# Push the commits to a branch on the fork (remote).
git push -u origin <local-branch>:<remote-branch>
```

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.

- Exercise caution when selecting the source and target branches for the PR.
- Creation of a PR creation kicks off the code review process.
- At least one Alpamayo researcher will be assigned for the review.
- While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].

4. Basic CPU-only CI runs on pull requests and pushes to `main`. The required baseline checks
   validate whitespace in changed files, shared-package buildability, recipe lockfiles, Python
   syntax, and lightweight repository tests. PRs that touch heavy training, inference, dataset, or
   model-loading paths still need appropriate manual validation by the developer and/or Alpamayo
   researcher reviewing the code.

#### Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```

## Recipe Contribution Guidelines

Alpamayo Recipes is organized as a set of recipe-specific Python packages, not as one monolithic
Python environment.

### Repository Structure

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

### Installing a Recipe

Each recipe README should tell users to install from inside the recipe directory:

```bash
cd alpamayo-recipes/recipes/<recipe_name>
uv venv <venv_name>
source <venv_name>/bin/activate
uv sync --active
```

For recipes that need released model code or shared helpers, keep dependencies recipe-local.
Existing packaged recipes commonly use:

- `alpamayo_r1`, fetched from `https://github.com/NVlabs/alpamayo.git`, for released model code,
  processors, geometry, and inference-time components.
- `alpamayo-recipes`, installed editable from `../../src`, for shared recipe-side utilities.

New recipes should follow this repository convention unless there is a concrete reason to do
something different.

### Adding a Recipe for Released Alpamayo Models

Put a new recipe under `recipes/<recipe_name>/`, where `<recipe_name>` is short, lowercase, and
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

### Shared Code vs Recipe Code

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

### README Expectations

Every recipe README should make the workflow reproducible for someone starting from a clean
checkout. Include as applicable:

- what the recipe does and which Alpamayo model versions it supports
- hardware assumptions, especially GPU count and memory
- installation commands from inside the recipe directory
- required model, dataset, and credential inputs
- exact commands to run the workflow
- expected outputs, metrics, logs, or files
- known limitations and what the recipe does not cover
- etc

Use local paths and environment variables consistently, and avoid hardcoding user-specific paths.
Do not include secrets, tokens, private dataset paths, generated checkpoints, or large artifacts in
the repository.

### Recipe Validation Before Opening a Pull Request

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

The repository CI runs a CPU-only baseline on pull requests and pushes to `main`:

```bash
uv build --project src --sdist --wheel
for pyproject in recipes/*/pyproject.toml; do
  recipe_dir="$(dirname "$pyproject")"
  uv lock --check --project "$recipe_dir"
done
python -m compileall -q src scripts recipes
uv run --with pytest pytest tests -q
```

If the recipe includes tests, run them from the recipe environment. If full training, evaluation,
or recipe runtime coverage requires gated datasets, large checkpoints, Torch runtime setup,
specialized kernels, or multi-GPU hardware, include the smaller smoke test you ran and document any
heavyweight validation that maintainers would need to run separately.

### Recipe Pull Request Scope

Keep recipe PRs focused. A new recipe PR should usually avoid unrelated refactors, broad
dependency updates, or changes to existing recipe behavior. If you need shared utilities under
`src/alpamayo/`, explain which recipes use them and why they belong in shared code.
