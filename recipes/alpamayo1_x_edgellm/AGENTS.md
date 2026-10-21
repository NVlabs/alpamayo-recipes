# Alpamayo 2B TensorRT recipe for DRIVE Thor

Scope: `recipes/alpamayo1_x_edgellm/`. Inside `tensorrt-edge-llm/`, its own
`AGENTS.md` takes precedence.

## Purpose

- One model: Alpamayo 2B distill (`alpamayo_r1`), on DRIVE AGX Thor.
- One pipeline (export ONNX, build engines, run inference) with a Python runtime (`run.py`) and
  two precisions: FP8 (default) and FP16 (`--precision fp16`). Six engines, Domino drafter +
  speculative verifier, one Alpamayo plugin in TensorRT Edge-LLM (`AlpamayoFusedGateUpPlugin`, FP8
  only); attention in the verify and action engines is Edge-LLM's own `AttentionPlugin` in tree mode
  over its paged K/V pool. A C++ runtime is planned.
- No model registry, 10B path, or experiment framework.

## Vocabulary

- Product: **TensorRT Edge-LLM**; checkout: `tensorrt-edge-llm` (its plugin library and XQA kernel source).
- Model: **Alpamayo 2B distill**; reserve `alpamayo_r1` for identifiers.
- Pipeline stages: **export**, **build**, **run**.
- Outputs: **ONNX directory**, **engine directory**, **result JSON**.
- Per-run requests: **jobs** (`prepare.py` output).
- Customizable component boundary: **plugin**.

Avoid: seam, ladder, rung, stock, proof stack, golden path, deadly simple,
just works, trusted.

## Tone

- Short, literal, operational.
- Prefer commands and constraints over rationale.
- Document commands that run after the stated setup.
- Define every shell variable and placeholder used in a command.
- Separate status claims:
  - **implemented**: code exists;
  - **structurally checked**: static or CPU checks passed;
  - **measured**: correctness or latency ran on the named target and stack.
- Do not call an engine validated unless it was measured on that target.

## Code structure

- `core.py`: paths, JSON output, the model bundle (`Bundle`, `write_manifest`), checkpoint identity,
  request prompt, clip list reader. `bundle.py`: writes or checks `MANIFEST.json`.
- `export/graph.py`: the shared ONNX builder, `--precision` handling and the FP8 weight rule
  (codes = saturate(RNE(W / (amax/448)))); `export/{visual,prefill,verify,domino,action}.py`:
  one exporter per engine; `export/metadata.py`: the pinned Cosmos-Reason2 files.
- `build.py`: builds TensorRT Edge-LLM's plugin library (CMake, `auto-thor`, CuTe DSL group
  `fmha_decode_ctm` for the AttentionPlugin's tree-decode kernel, cuRAND headers from the
  `nvidia-curand` wheel when the toolkit lacks them) and the six engines with a `build.json` receipt.
- `pyproject.toml` + `uv.lock`: the tested Python environment; README step 0 installs it with
  `uv sync --frozen`. Change versions through the lock, not the README.
- `prepare.py`: the 20-clip x 13-window request stream from the public clip list.
- `run.py`: the request pipeline for either precision, summary and optional comparison with a
  reference run (guideline limits, not a hard gate).
- FP8 scales, the Domino vocabulary and the action noise are part of the model bundle (README),
  not of the recipe; the scales reach the engines through ONNX Q/DQ nodes and plugin inputs. Do
  not recalibrate inside the recipe. An exporter change must keep the FP8 graphs byte-identical
  unless the FP8 path is meant to change.

Use the pinned submodule (it carries the Alpamayo plugin and the CuTe DSL prebuilt tarball). Do not modify
a sibling TensorRT Edge-LLM checkout. The KV cache is a paged pool of 40 x 128 rows per layer
(`export/graph.py`): pages 0..24 language rows, pages 25..36 the action candidates' private pages.

## Artifacts

Track source and documentation. The clip list is the repository's shared
`dataset/1005_7cam_gold_eval_metadb_public.parquet`; do not copy it into the recipe. The model bundle is private: keep it, ONNX, engines, build trees, caches, wheels, and
run results outside the Git checkout. Never track `workspace.env`.

## Validation

Before handoff:

1. `python -m py_compile core.py prepare.py build.py run.py export/*.py`
2. Import affected CLIs and run `--help`.
3. `git diff --check`
4. A change to an exporter, plugin, `build.py` or `run.py` needs a DRIVE Thor Gold20 run of each
   affected precision before its numbers go into `README.md`.
