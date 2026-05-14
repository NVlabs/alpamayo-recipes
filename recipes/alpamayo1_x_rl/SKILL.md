---
name: alpamayo1-x-rl
description: >-
  Run end-to-end open-loop RL post-training of the Alpamayo VLM action model
  (Alpamayo 1 or 1.5) on the Physical AI AV (PAI) dataset using Cosmos-RL +
  GRPO. The agent collects a small set of choices from the user up front
  (Alpamayo version, reward mode — motion-only vs joint reasoning+motion,
  W&B preference, single-node vs multi-node, dataset/checkpoint paths),
  then drives the whole pipeline. Use when an agent must convert a released
  Alpamayo checkpoint into a training-ready one, curate a PAI mini subset,
  launch policy + rollout replicas via `cosmos-rl`, and export the resulting
  policy checkpoint back to a HuggingFace directory; when setting up the
  `a1x_rl` uv venv from scratch; when overriding TOML config
  (parallelism, replicas, reward weights, prefetch, optimizer); when
  diagnosing common RL failures (rollout-too-fast buffer growth, weight
  sync lag, prefetch misconfiguration, vLLM OOM, GRPO group collapse).
  Trigger keywords: alpamayo, alpamayo-1, alpamayo-1.5, alpamayo1, alpamayo1_5,
  rl, post-train, post-training, RL post-training, grpo, group relative policy
  optimization, cosmos-rl, cosmos rl, reasoning vla, rvla, action token,
  trajectory diffusion frozen, pai, physical_ai_av,
  physicalai-autonomous-vehicles, reward, ade, comfort, lingo-judge,
  reasoning grader, chain-of-thought reward, chain of causation,
  ood_reasoning, hydra, omegaconf, toml, vllm, flash-attn, transformers,
  huggingface, Alpamayo-R1-10B, Alpamayo-1.5-10B, Cosmos-Reason2-8B,
  wandb, fsdp, dp_shard_size, n_init_replicas, train_batch_per_replica,
  rollout.n_generation, sync_weight_interval, pending rollouts,
  weight_version, prefetch, prefetch.capacity, convert_release_config_to_training,
  convert_cosmos_rl_checkpoint, curate_pai_samples, RLWrapperReasoningVLA.
license: Apache-2.0
metadata:
  author: nvidia
  version: "2026.05"
---

# Alpamayo 1.x RL Post-training (Cosmos-RL + GRPO)

This skill teaches an agent to RL-post-train the **VLM backbone** of
released Alpamayo models (the autoregressive-trajectory-token pathway) using
**Cosmos-RL** with **GRPO**, on the **PAI** (Physical AI Autonomous
Vehicles) dataset. Both **Alpamayo 1** (`nvidia/Alpamayo-R1-10B`) and
**Alpamayo 1.5** (`nvidia/Alpamayo-1.5-10B`) are supported. Validated
single-node on **≥5× 80 GB GPUs**; large-scale recipe targets **80 nodes
(640 GPUs)**.

The recipe lives at [recipes/alpamayo1_x_rl/](.) and is built on
[Cosmos-RL](https://github.com/NVIDIA/Cosmos-RL): one or more **policy
replicas** train the model (FSDP), one or more **rollout replicas** run
vLLM to generate completions, and a **cosmos-controller** dispatches
rollouts, scores them, and syncs policy weights back to the rollouts.

## Table of Contents

1. [When to use this skill](#when-to-use-this-skill)
2. [Inputs to collect from the user (ask once, up front)](#inputs-to-collect-from-the-user-ask-once-up-front)
3. [Mental model — what gets trained, in what order](#mental-model--what-gets-trained-in-what-order)
4. [Install — `a1x_rl` venv](#install--a1x_rl-venv)
5. [Environment variables](#environment-variables)
6. [Model conversion — release ckpt → training ckpt](#model-conversion--release-ckpt--training-ckpt)
7. [Dataset — PAI mini subset](#dataset--pai-mini-subset)
8. [Reasoning grader (only if joint reward)](#reasoning-grader-only-if-joint-reward)
9. [Launch RL training](#launch-rl-training)
10. [Monitor — controller log, healthy vs sick patterns](#monitor--controller-log-healthy-vs-sick-patterns)
11. [Export — Cosmos-RL ckpt → HuggingFace ckpt](#export--cosmos-rl-ckpt--huggingface-ckpt)
12. [Logging — W&B, console, offline](#logging--wb-console-offline)
13. [Multi-node / cluster scale-up](#multi-node--cluster-scale-up)
14. [TOML override cheatsheet](#toml-override-cheatsheet)
15. [Common failure modes (and the fix)](#common-failure-modes-and-the-fix)
16. [Additional resources](#additional-resources)

---

## When to use this skill

| You want to… | Use |
|--------------|-----|
| Run the full local RL pipeline (convert → curate → train → export) | The whole skill in order |
| Run RL on Alpamayo 1.5 with motion-only reward | [Inputs](#inputs-to-collect-from-the-user-ask-once-up-front) → version=`1.5`, reward=`motion`; default TOML |
| Run RL with joint reasoning + motion reward | [Inputs](#inputs-to-collect-from-the-user-ask-once-up-front) → reward=`joint`; the `…_with_reasoning.toml` config + Lingo-Judge grader |
| Just convert a release checkpoint into the training-ready format | [Model conversion](#model-conversion--release-ckpt--training-ckpt) |
| Just export a trained Cosmos-RL policy ckpt → HF | [Export](#export--cosmos-rl-ckpt--huggingface-ckpt) |
| Scale a working local run to cluster | [Multi-node / cluster scale-up](#multi-node--cluster-scale-up) |
| Diagnose a stuck or drifting RL run | [Monitor](#monitor--controller-log-healthy-vs-sick-patterns) + [Common failure modes](#common-failure-modes-and-the-fix) |

If you want to **SFT** the VLM (no RL, no reward, no rollout), this is the
wrong skill — see [`recipes/alpamayo1_sft/SKILL.md`](../alpamayo1_sft/SKILL.md).
If you want to train the **action expert** (continuous flow-matching head),
that pathway is **not covered** by this RL pipeline — RL here only updates
the VLM backbone.

---

## Inputs to collect from the user (ask once, up front)

Before any download, install, or training step, ask the user for the
following. Confirm all answers before proceeding. If running in a context
where you cannot ask the user, halt and report rather than guess.

| Input | Why you need it | Default if user has no preference |
|-------|-----------------|-----------------------------------|
| **Alpamayo version** (`1` / `1.5`) | Picks the converted base model (`nvidia/Alpamayo-R1-10B` vs `nvidia/Alpamayo-1.5-10B`) and the entry-script `hydra_config_name` (`alpamayo1_rvla_rl_pai` vs `alpamayo1_5_rvla_rl_pai`) | `1.5` |
| **Reward mode** (`motion` / `joint`) | `motion` → `alpamayo_rvla_rl_local_test.toml` + `…_entry.py` (ADE + comfort only). `joint` → `…_local_test_with_reasoning.toml` + `…_reasoning_entry.py` (also grades CoT via Lingo-Judge). | `motion` |
| **PAI chunks to download** | Sets `download_pai.py --chunk-ids` and the curation chunk. Motion-only example uses `3116`; joint uses `--only-reasoning-chunks --num-reasoning-clips N`. | `3116` for motion, `--only-reasoning-chunks --num-reasoning-clips 16` for joint |
| **Number of curated clips** (motion-only path) | Sets `curate_pai_samples.py --num-samples`. Small for smoke tests (`16`), larger for real runs. | `16` |
| **W&B logging?** (`yes` / `no`) | If `yes`, also collect: `WANDB_API_KEY`, `project_name`, `experiment_name`. If `no`, set `[logging].logger = ["console"]` in TOML and leave `WANDB_API_KEY` unset. | `no` (smoke) / `yes` (real run) |
| **Compute shape** (`local` / `cluster`) | `local` → 1 node, `dp_shard_size=4`, `--policy 1 --rollout 1`. `cluster` → adjust `dp_shard_size=8`, `n_init_replicas`, enable `data_dispatch_as_rank_in_mesh`. | `local` for first run; `cluster` only after local works |
| **Conditional: Lingo-Judge directory** | Only if `reward = joint`. Path where `wayveai/Lingo-Judge` is (or should be) cached. The reward loads it with `local_files_only=True`, so it must be on disk. | n/a |
| **Conditional: existing trained policy ckpt** | Only if the user is skipping training and just exporting. Path to `<output_dir>/checkpoints/step_<N>/policy/`. | n/a |
| **Output / log directories** | `ALPAMAYO_MODEL_DIR` (converted ckpt), `ALPAMAYO_PAI_LOCAL_DIR` (dataset root), `ALPAMAYO_LOG_DIR` (Cosmos-RL logs), `[train].output_dir` in TOML (training outputs). | Ask — no safe default |

### Suggested question flow

Ask in a single round if your interface supports it; otherwise sequentially:

1. "Alpamayo version: `1` or `1.5`?"
2. "Reward mode: `motion` (trajectory ADE + comfort only) or `joint`
   (also grades chain-of-thought reasoning via Lingo-Judge)?"
3. "PAI chunks to download? (e.g. `3116` for motion-only; for `joint`,
   I'll use `--only-reasoning-chunks --num-reasoning-clips N` — how many
   clips, N?)"
4. *(motion only)* "How many clips to curate from that chunk? (`16` for a
   fast smoke test)"
5. "Log to W&B? If yes, paste `WANDB_API_KEY`, project name, experiment
   name. If no, I'll set logger to console only."
6. "Compute shape: `local` (1 node, ≥5× 80 GB GPUs) or `cluster`
   (multi-node, will need SLURM/launcher integration)?"
7. *(joint reward only)* "Path where Lingo-Judge should be cached?
   (default: `$YOUR_HOME/lingo_judge_model`)"
8. "Where should artifacts live? (`ALPAMAYO_MODEL_DIR`,
   `ALPAMAYO_PAI_LOCAL_DIR`, `ALPAMAYO_LOG_DIR`, `[train].output_dir`)"

### Capture the answers verbatim

Save the answers as shell variables at the top of your run log — every
later command references them:

```bash
export YOUR_HOME="/path/to/workspace"
export ALPAMAYO_WORKSPACE="$YOUR_HOME/alpamayo-recipes"

# from user answers
export ALPAMAYO_VERSION="1.5"                 # 1 | 1.5
export REWARD_MODE="motion"                   # motion | joint
export CHUNK_IDS="3116"                       # for motion path
export NUM_REASONING_CLIPS="16"               # for joint path
export NUM_CURATED_CLIPS="16"                 # for motion path
export USE_WANDB="no"                         # yes | no
export WANDB_API_KEY=""                       # only if USE_WANDB=yes
export WANDB_PROJECT="Alpamayo_RL"
export WANDB_EXPERIMENT="ReasoningVLA_Post_Training"
export COMPUTE_SHAPE="local"                  # local | cluster

# paths (also exported for the recipe's own env-var contract)
export ALPAMAYO_MODEL_DIR="$YOUR_HOME/alpamayo_model_converted_from_hf"
export ALPAMAYO_PAI_LOCAL_DIR="$YOUR_HOME/PAI_mini"
export ALPAMAYO_PAI_REASONING_LOCAL_DIR="$YOUR_HOME/PAI_Reasoning_mini"  # only if joint
export ALPAMAYO_LOG_DIR="$YOUR_HOME/alpamayo_cosmos_rl_job/logs"
export LINGO_JUDGE_DIR="$YOUR_HOME/lingo_judge_model"                    # only if joint
export TRAIN_OUTPUT_DIR="$YOUR_HOME/alpamayo_cosmos_rl_job/outputs"
```

After this, do not ask further questions — run the rest of the pipeline
end-to-end. The only acceptable mid-run halt conditions are hard-stop
errors (CUDA OOM, missing files, vLLM/Cosmos-RL crashes, GRPO collapse)
or the user explicitly interrupting.

---

## Mental model — what gets trained, in what order

```
   Release HF ckpt (nvidia/Alpamayo-1.5-10B or -R1-10B)
        │
        │  convert_release_config_to_training.py
        ▼
   $ALPAMAYO_MODEL_DIR   (training-ready, target paths remapped)
        │
        │  download_pai.py + curate_pai_samples.py   (or --only-reasoning-chunks)
        ▼
   $ALPAMAYO_PAI_LOCAL_DIR   (mini PAI subset + clip_index_mini.parquet
                              or clip_index_reasoning_mini.parquet)
        │
        ▼
   cosmos-rl --config <TOML> --policy P --rollout R \
       <entry_script.py>
        │
        ├── controller   (rollout dispatch + reward + buffer + weight sync)
        ├── policy x P   (FSDP training of VLM backbone)   ← writes ckpts every N steps
        └── rollout x R  (vLLM serving + generation + reward scoring)
        │
        │  convert_cosmos_rl_checkpoint.py
        ▼
   exported_model/   (standard HF dir, loadable via
                      RLWrapperReasoningVLA.from_pretrained)
```

Key facts an agent must internalise before running anything:

- **VLM backbone only.** RL here trains the autoregressive-trajectory-token
  pathway of the VLM. The action expert (flow-matching head) is **not**
  trained — its weights pass through unchanged and end up unused in the
  exported VLM-only ckpt.
- **GRPO + groups.** Each prompt produces `rollout.n_generation` completions
  ranked by reward (default `n_generation=12`). Reward variance within a
  group is the training signal — if all completions get the same reward,
  GRPO can't learn anything from that group.
- **Two processes, one job.** `cosmos-rl ... --policy P --rollout R` spawns
  P FSDP policy replicas + R vLLM rollout replicas. They communicate
  through the controller. `policy 1 / rollout 1` is the local-test default.
- **Required overrides.** The user must (a) edit the TOML's
  `[train].output_dir`, `[policy].model_name_or_path`, and
  `[policy.parallelism].dp_shard_size`, and (b) either set `WANDB_API_KEY`
  + uncomment `wandb` logger, or drop wandb from the logger list.
- **Conversion is mandatory.** You cannot point `model_name_or_path`
  directly at the HF release dir; it has to be run through
  `convert_release_config_to_training.py` first. That script remaps
  `_target_` paths in `config.json` to the training-time classes and
  pulls the tokenizer from `nvidia/Cosmos-Reason2-8B`.
- **Export is mandatory** if you want to *use* the trained model — Cosmos-RL
  saves DTensor-sharded per-rank `.pth` files that `from_pretrained` can't
  read directly. Run `convert_cosmos_rl_checkpoint.py` to assemble them
  into a standard HF directory.

---

## Install — `a1x_rl` venv

Single uv venv at `recipes/alpamayo1_x_rl/a1x_rl/`. `flash-attn` must build
against `torch`, so install in two `uv sync` passes:

```bash
# 1) Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2) Provision the venv
export UV_CACHE_DIR="$YOUR_HOME/.cache/uv"
cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_x_rl"

uv venv a1x_rl            # if `a1x_rl/` already exists, skip this line
source a1x_rl/bin/activate
uv sync --active --no-install-package flash-attn   # all deps minus flash-attn
uv sync --active                                   # then build flash-attn against the installed torch
```

> **Non-interactive harnesses (agents, CI, `bash -c "..."` per step):**
> the `export PATH=...` above only persists for the current shell. Either
> append to `~/.bashrc` once, or re-export it in every shell. Same caveat
> as the SFT recipe.

Verify the import contract after `uv sync` (this is the single most common
cause of opaque entry-script errors):

```bash
python -c "import alpamayo_r1, alpamayo, cosmos_rl, vllm; print('ok')"
python -c "from alpamayo1_x_rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA; print('ok')"
python -c "from alpamayo.utils.checkpoint_utils import collect_targets, remap_targets; print('ok')"
```

---

## Environment variables

Set these once per session (or `~/.bashrc`). Treat them as the contract
the recipe scripts read:

```bash
# ── Paths ────────────────────────────────────────────────────────
export ALPAMAYO_WORKSPACE="$YOUR_HOME/alpamayo-recipes"
export ALPAMAYO_MODEL_DIR="$YOUR_HOME/alpamayo_model_converted_from_hf"
export ALPAMAYO_PAI_LOCAL_DIR="$YOUR_HOME/PAI_mini"
export ALPAMAYO_LOG_DIR="$YOUR_HOME/alpamayo_cosmos_rl_job/logs"

# ── Cache ────────────────────────────────────────────────────────
export HF_HOME="$YOUR_HOME/.cache/huggingface"

# ── Runtime ──────────────────────────────────────────────────────
export WANDB_API_KEY="<your_wandb_api_key>"   # only if $USE_WANDB = yes
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `ALPAMAYO_WORKSPACE` | yes | Root of `alpamayo-recipes` |
| `ALPAMAYO_MODEL_DIR` | yes | Pre-trained converted Alpamayo model directory |
| `ALPAMAYO_PAI_LOCAL_DIR` | yes | PAI dataset root |
| `ALPAMAYO_LOG_DIR` | yes | Directory for Cosmos-RL logs |
| `UV_CACHE_DIR` | recommended | uv cache (set **before** `uv venv`) |
| `HF_HOME` | recommended | HuggingFace cache location |
| `HF_HUB_OFFLINE` | optional | `1` to skip Hub calls (air-gapped clusters / rate-limit fallbacks) |
| `TRANSFORMERS_OFFLINE` | optional | `1` alongside `HF_HUB_OFFLINE` |
| `WANDB_API_KEY` | conditional | Only if `$USE_WANDB = yes` |
| `ALPAMAYO_PAI_REASONING_LOCAL_DIR` | conditional | **Required when `$REWARD_MODE = joint`** — read by the reasoning entry script (`…_reasoning_entry.py`) **instead of** `ALPAMAYO_PAI_LOCAL_DIR`. The two are mutually exclusive: motion entry reads `ALPAMAYO_PAI_LOCAL_DIR`, joint entry reads `ALPAMAYO_PAI_REASONING_LOCAL_DIR`. Missing it → `RuntimeError: Missing required env var ALPAMAYO_PAI_REASONING_LOCAL_DIR` |
| `LINGO_JUDGE_DIR` | conditional | Only if `$REWARD_MODE = joint` |

### HuggingFace auth

The Alpamayo model and PAI dataset are gated. The user must have accepted
the licenses on HF web UI for whichever Alpamayo version they chose, plus
PAI. Then:

```bash
hf auth login
```

In non-interactive shells, the cached token at
`~/.cache/huggingface/token` is read automatically. Source it into
`$HF_TOKEN` explicitly if a downstream tool expects the env var:
`export HF_TOKEN=$(<~/.cache/huggingface/token)`.

---

## Model conversion — release ckpt → training ckpt

Run this **once** per Alpamayo version. The script downloads the HF
release model + `nvidia/Cosmos-Reason2-8B` tokenizer/processor, remaps
`_target_` paths in `config.json` to the training classes, and writes a
training-ready directory.

```bash
cd "$ALPAMAYO_WORKSPACE"

# motion or joint, version = 1.5 (default)
python scripts/convert_release_config_to_training.py \
  --output-dir "$ALPAMAYO_MODEL_DIR"

# version = 1
python scripts/convert_release_config_to_training.py \
  --alpamayo-model nvidia/Alpamayo-R1-10B \
  --output-dir "$ALPAMAYO_MODEL_DIR"
```

Sanity-check the result:

```bash
ls "$ALPAMAYO_MODEL_DIR"
# expect: config.json (remapped), model.safetensors.index.json,
#         model-0000{1..5}-of-00005.safetensors, tokenizer files,
#         preprocessor_config.json, generation_config.json
```

---

## Dataset — PAI mini subset

Two flows depending on `$REWARD_MODE`. Both write to
`$ALPAMAYO_PAI_LOCAL_DIR` (or `$ALPAMAYO_PAI_REASONING_LOCAL_DIR` for joint).

### `$REWARD_MODE = motion` — ego-motion only

```bash
cd "$ALPAMAYO_WORKSPACE"

# 1) Download one chunk with four cameras + egomotion
python scripts/download_pai.py \
  --chunk-ids "$CHUNK_IDS" \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --output-dir "$ALPAMAYO_PAI_LOCAL_DIR"

# 2) Curate a mini sample set (RL training reads clip_index_mini.parquet)
python scripts/curate_pai_samples.py \
  --clip-index-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index.parquet" \
  --chunk "$CHUNK_IDS" \
  --num-samples "$NUM_CURATED_CLIPS" \
  --output-path "$ALPAMAYO_PAI_LOCAL_DIR/clip_index_mini.parquet"
```

### `$REWARD_MODE = joint` — reasoning-bearing clips

```bash
cd "$ALPAMAYO_WORKSPACE"

python scripts/download_pai.py --only-reasoning-chunks \
  --num-reasoning-clips "$NUM_REASONING_CLIPS" \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion egomotion.offline obstacle.offline \
  --reasoning ood_reasoning.parquet \
  --output-dir "$ALPAMAYO_PAI_REASONING_LOCAL_DIR"
```

After success, `$ALPAMAYO_PAI_REASONING_LOCAL_DIR` should contain:

- `clip_index.parquet` — full PAI index (internal mapping).
- `reasoning/ood_reasoning.parquet` — full OOD reasoning table.
- `clip_index_reasoning_mini.parquet` — **the mini index the RL config
  consumes**; exactly `--num-reasoning-clips` rows.
- `camera/<subpart>/<subpart>.chunk_XXXX.zip`,
  `labels/<subpart>/<subpart>.chunk_XXXX.zip`,
  `calibration/<subpart>/...` — only the chunks containing the sampled clips.

> **Disk budget heads-up.** `--only-reasoning-chunks` pulls **every chunk
> that contains a sampled reasoning clip** — not just the clips' own
> footage. For `--num-reasoning-clips 16` this typically lands around
> **~85–90 GB** (vs ~5–10 GB for the motion-only path on a single chunk).
> Make sure `$ALPAMAYO_PAI_REASONING_LOCAL_DIR`'s mount has the headroom
> before starting the download.

Verify:
```bash
ls "$ALPAMAYO_PAI_REASONING_LOCAL_DIR/clip_index_reasoning_mini.parquet" \
   "$ALPAMAYO_PAI_REASONING_LOCAL_DIR/reasoning/ood_reasoning.parquet"
```

> **Heads-up:** if you set `HF_HUB_OFFLINE=1` during a prior step, set it
> back to `0` before `download_pai.py` — the script needs Hub access.

---

## Reasoning grader (only if joint reward)

Skip this section entirely when `$REWARD_MODE = motion`.

The default joint reward uses [Lingo-Judge](https://huggingface.co/wayveai/Lingo-Judge)
as a learned reasoning grader. It must be cached locally because the reward
function loads it with `local_files_only=True`:

```bash
hf download wayveai/Lingo-Judge --local-dir "$LINGO_JUDGE_DIR"
```

Then in [toml/alpamayo_rvla_rl_local_test_with_reasoning.toml](toml/alpamayo_rvla_rl_local_test_with_reasoning.toml),
set under `[custom.alpamayo]`:

| Field | Value |
|-------|-------|
| `reasoning_grader_type` | `"lingo_judge"` (default) |
| `reasoning_grading_model_path` | `"$LINGO_JUDGE_DIR"` (concrete path, not a shell var — TOML doesn't expand) |
| `reasoning_grading_device` | `"auto"`, `"cpu"`, or `"cuda:0"` depending on free GPU |
| `reward.reasoning_weight` (under `[custom.alpamayo.reward]`) | Tune to balance against trajectory reward |

If you want a custom grader, subclass `BaseReasoningGrader` in
[utils/light_weight_reasoning_grading_model.py](utils/light_weight_reasoning_grading_model.py).

---

## Launch RL training

The launcher is `cosmos-rl`, **not** `torchrun`. Two TOML configs + two
entry scripts pick the reward mode:

| `$REWARD_MODE` | TOML | Entry script |
|----------------|------|--------------|
| `motion` | `toml/alpamayo_rvla_rl_local_test.toml` | `models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py` |
| `joint` | `toml/alpamayo_rvla_rl_local_test_with_reasoning.toml` | `models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py` |

### Required TOML edits (before launching)

In whichever TOML you picked, set:

1. `[train].output_dir` → `$TRAIN_OUTPUT_DIR`
2. `[policy].model_name_or_path` → `$ALPAMAYO_MODEL_DIR`
3. `[policy.parallelism].dp_shard_size` → `4` for local (1 node, ≥5 GPUs),
   `8` for cluster
4. `[logging].logger`:
   - If `$USE_WANDB = yes`: `["console", "wandb"]`, plus set
     `project_name = "$WANDB_PROJECT"`, `experiment_name = "$WANDB_EXPERIMENT"`
   - If `$USE_WANDB = no`: `["console"]`

### Hydra config name (Alpamayo version selection)

Inside the entry script, the `hydra_config_name` argument selects the
PAI Hydra config:

- `"alpamayo1_5_rvla_rl_pai"` for Alpamayo 1.5
- `"alpamayo1_rvla_rl_pai"` for Alpamayo 1

The shipped entry scripts default to the 1.5 config. If `$ALPAMAYO_VERSION
= 1`, **edit the entry script** (or override via a sed step) to use
`"alpamayo1_rvla_rl_pai"` before launching.

### Canonical launch (motion-only, local)

```bash
cd "$ALPAMAYO_WORKSPACE"
cosmos-rl \
  --config recipes/alpamayo1_x_rl/toml/alpamayo_rvla_rl_local_test.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  recipes/alpamayo1_x_rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py
```

### Canonical launch (joint reward, local)

```bash
cd "$ALPAMAYO_WORKSPACE"
cosmos-rl \
  --config recipes/alpamayo1_x_rl/toml/alpamayo_rvla_rl_local_test_with_reasoning.toml \
  --policy 1 \
  --rollout 1 \
  --log-dir "$ALPAMAYO_LOG_DIR" \
  recipes/alpamayo1_x_rl/models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py
```

`--policy P --rollout R` override `n_init_replicas` in the TOML. Use `1`/`1`
for local smoke; scale on cluster ([Multi-node](#multi-node--cluster-scale-up)).

Logs land at `$ALPAMAYO_LOG_DIR/logs_<YYYYMMDD-HHMMSS>/`:

| File | Process | Contains |
|------|---------|----------|
| `controller.log` | cosmos-controller | Rollout dispatch, reward stats, buffer (`pending rollouts`), weight sync events |
| `policy_<i>.log` | Policy replica *i* | Model loading, loss, grad norms, ckpt saving, per-rank dataset distribution |
| `rollout_<i>.log` | Rollout replica *i* | vLLM startup, generation throughput, weight receives, reward per sample |

### Healthy progression (motion-only local)

Expect reward to climb (≈ `-0.28 → -0.21`) and trajectory L2 to drop
(≈ `1.66 → 1.34`) over ~10 min on a single 8× H100 node. The joint
reward local run takes ~1.1 hr on a single 8× A100 node and should show
reasoning score climbing alongside trajectory L2 dropping.

---

## Monitor — controller log, healthy vs sick patterns

The controller log is the canonical place to detect the most common RL
pathology: **rollout/policy throughput imbalance**. `pending rollouts`
is the buffer of completed-but-unconsumed rollouts.

### Healthy

```text
# controller.log — buffer stays low and oscillates
Stat: samples=  48  pending=  24
Stat: samples= 600  pending=  48
Stat: samples=1200  pending=  72
Stat: samples=2400  pending=  48
```

### Sick: rollout too fast (most common)

```text
Stat: samples=  24  pending=  24
Stat: samples= 600  pending= 264
Stat: samples=1200  pending= 552
Stat: samples=2400  pending= 984
[Controller] All rollouts have ended … 1104 remaining rollouts
# training at step 37/60 — remaining 23 steps consume stale, off-policy rollouts
```

When rollout outpaces policy, `weight_version` gap grows, training becomes
off-policy, quality degrades. Fix in this order:

1. Enable prefetch (`[custom.alpamayo].prefetch.capacity > 0`) — biggest
   single win, often 44 s → 5 s per step.
2. Reduce `[rollout].batch_size` or `[rollout].n_generation`.
3. Add policy replicas (`[policy.parallelism].n_init_replicas`).
4. Increase `[policy.parallelism].dp_shard_size`.
5. Lower `epoch` or set `max_num_steps` to bound total rollouts.

### Sick: rollout too slow

```text
# pending repeatedly drops to 0 — policy idles waiting for data
Stat: samples= 600  pending=   0
Stat: samples= 624  pending=   0
Stat: samples= 648  pending=   0
```

Fix: add rollout replicas, or reduce policy throughput (e.g. smaller
`train_batch_per_replica`).

### Target state

`pending rollouts` stays roughly stable. At cluster scale (64 policy /
128 rollout replicas, global batch 2560), buffer typically holds 4× global
batch. `weight_version` gap within a training batch should stay within a
few multiples of `sync_weight_interval`.

---

## Export — Cosmos-RL ckpt → HuggingFace ckpt

Cosmos-RL writes per-rank DTensor shards at
`$TRAIN_OUTPUT_DIR/checkpoints/step_<N>/policy/model_rank_*.pth`.
These **cannot** be loaded by `ReasoningVLA.from_pretrained`. Convert:

```bash
cd "$ALPAMAYO_WORKSPACE"

# pick the step (latest by default)
STEP_DIR=$(ls -d "$TRAIN_OUTPUT_DIR/checkpoints"/step_* | sort -V | tail -1)
echo "exporting: $STEP_DIR"

python scripts/convert_cosmos_rl_checkpoint.py \
  --cosmos-policy-ckpt "$STEP_DIR/policy" \
  --base-hf-ckpt "$ALPAMAYO_MODEL_DIR" \
  --output-dir "$YOUR_HOME/alpamayo_cosmos_rl_job/exported_model"
```

- `--cosmos-policy-ckpt` — directory with `model_rank_*.pth` files.
- `--base-hf-ckpt` — the training-ready dir from
  [Model conversion](#model-conversion--release-ckpt--training-ckpt).
  Config, tokenizer, processor are copied from here.
- `--output-dir` — where the HF-style export lands.

Load the exported model for inference:

```python
from alpamayo1_x_rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA

model = RLWrapperReasoningVLA.from_pretrained(
    "<output-dir>"
)
```

End-to-end inference / visualisation example:
[notebooks/rl_checkpoint_inference.ipynb](notebooks/rl_checkpoint_inference.ipynb).

> **The exported ckpt contains only the VLM backbone.** Loading it back
> into a full Alpamayo (1 or 1.5) directory works with non-strict weight
> loading — the action-expert weights remain randomly initialised.

---

## Logging — W&B, console, offline

Logger selection lives in `[logging].logger` in the TOML.

### `$USE_WANDB = no`

```toml
[logging]
logger = ["console"]
```

No `WANDB_API_KEY` needed. For belt-and-braces (e.g. machines where
`wandb login` has been run previously), prefix the launch:

```bash
WANDB_MODE=disabled cosmos-rl --config ... <entry_script.py>
```

### `$USE_WANDB = yes`

```toml
[logging]
logger = ["console", "wandb"]
project_name = "Alpamayo_RL"               # = $WANDB_PROJECT
experiment_name = "ReasoningVLA_Post_Training"  # = $WANDB_EXPERIMENT
```

Plus `export WANDB_API_KEY="..."` before launch. The token must have
write access to `project_name`. A 403 (`upsertBucket … permission denied`)
means the account isn't in that team/project — fix the project name or
the key. Fail fast and re-prompt the user for a working key; don't retry
blindly.

---

## Multi-node / cluster scale-up

After the local run works, two TOML changes are **mandatory** for cluster:

1. **`[policy.parallelism].dp_shard_size = 8`** — FSDP shards the model
   across 8 GPUs per replica. `n_policy_replicas × dp_shard_size` = total
   policy GPUs.
2. **`[train.train_policy].data_dispatch_as_rank_in_mesh = true`** —
   rank-based dataset dispatch so each policy replica consumes a stable,
   non-overlapping shard. Without this, replicas can train on duplicate
   samples. Also required for the prefetch path.

Recommended cluster-scale settings (these are the values used to RL
post-train Alpamayo 1.5):

| Parameter | Local | Cluster |
|-----------|-------|---------|
| `policy.parallelism.n_init_replicas` | 1 | 64 |
| `rollout.parallelism.n_init_replicas` | 1 | 128 |
| `train.train_batch_per_replica` | 48 | 40 |
| `train.optm_lr` | 2e-6 | 2e-6 |
| `train.sync_weight_interval` | 2 | 5 |
| `rollout.batch_size` | 2 | 6 |
| `rollout.n_generation` | 12 | 12 |
| `custom.alpamayo.prefetch.capacity` | 16 | 128 |

This gives a global batch of `64 × 40 = 2560` samples/step on
**512 policy GPUs + 128 rollout GPUs = 640 GPUs (80 nodes)**. Start
moderate (e.g. 4 policy / 8 rollout) and watch `pending rollouts` before
scaling up. SLURM launch instructions:
<https://nvidia-cosmos.github.io/cosmos-rl/multinodes/overview.html>.

### Recommended workflow when iterating on a new reward / data / model

1. **Overfit on 1 sample.** Confirms reward / data / model are wired
   correctly. Reward should climb.
2. **Overfit on 16–32 samples on one node.** Tune reward weights, LR,
   `n_generation`. Watch rollout/policy balance.
3. **Scale to multi-node.** Monitor `pending rollouts` and
   `weight_version` gap. Start at moderate global batch (~320) and
   scale up if reward variance is too high.
4. **Iterate on the reward.** RL optimises whatever you measure. If
   behaviour isn't improving, revisit reward design before scaling.

---

## TOML override cheatsheet

TOML edits happen in the file directly — Cosmos-RL doesn't accept
Hydra-style dotted CLI overrides for these. The `--policy N` / `--rollout N`
CLI flags override `n_init_replicas` for both replica types.

| TOML path | What it does |
|-----------|--------------|
| `[train].output_dir` | Where training outputs / checkpoints land |
| `[train].epoch` | Total epochs over the curated samples |
| `[train].max_num_steps` *(optional)* | Hard cap on training steps |
| `[train].train_batch_per_replica` | Samples consumed per replica per step. Global batch = `n_init_replicas × this` |
| `[train].sync_weight_interval` | Sync policy weights to rollouts every N steps |
| `[train].optm_lr` | Optimizer LR (default `2e-6` — RL fine-tune territory) |
| `[policy].model_name_or_path` | `$ALPAMAYO_MODEL_DIR` |
| `[policy].model_max_length` | Token cap for prompt + completion |
| `[policy].model_gradient_checkpointing` | `true` for memory savings |
| `[policy.parallelism].dp_shard_size` | FSDP shard degree for the **policy** (`4` local / `8` cluster). Distinct from `[rollout.parallelism].dp_shard_size`, which stays `1` for local — the policy + rollout values are summed when the launcher places replicas on GPUs |
| `[rollout.parallelism].dp_shard_size` | GPUs per rollout replica (default `1`). Increase only if a single rollout replica needs to shard the vLLM engine across multiple GPUs |
| `[policy.parallelism].n_init_replicas` | Number of policy replicas (overridden by `--policy`) |
| `[rollout].n_generation` | Completions per prompt (the GRPO group size). Lower if rollout too slow / OOM |
| `[rollout].batch_size` | Prompts per rollout batch |
| `[rollout].gpu_memory_utilization` | vLLM memory fraction (default `0.8`) |
| `[rollout.sampling_config].temperature` / `.top_p` / `.repetition_penalty` | Generation sampling |
| `[custom.alpamayo].prefetch.capacity` | Prefetch cache size. Set to `train_batch_per_replica × replicas_per_node`. `<= 0` disables |
| `[custom.alpamayo].prefetch.num_workers` | Prefetch worker threads (default `5`) |
| `[custom.alpamayo.reward].traj_l2_weight` | Weight on ADE penalty in the trajectory reward |
| `[custom.alpamayo.reward].comfort_weight` | Weight on comfort score |
| `[custom.alpamayo.reward].reasoning_weight` | Weight on reasoning grader (joint reward only) |
| `[custom.alpamayo].reasoning_grading_model_path` | Lingo-Judge dir (joint reward only) |
| `[custom.alpamayo].reasoning_grading_device` | `"auto"` / `"cpu"` / `"cuda:0"` |
| `[logging].logger` | `["console"]` or `["console", "wandb"]` |
| `[logging].project_name` / `.experiment_name` | W&B project / run name |

**One-sample overfit smoke**: set `num_samples=1` in the curate step,
`epoch=200`, `n_generation=8`, `train_batch_per_replica=1`. Watch reward
climb on the same single clip — that's the wiring check before scaling.

---

## Common failure modes (and the fix)

| Symptom | Root cause | Fix |
|---------|------------|-----|
| Hydra CLI override `<key>=None` is silently parsed as the **string** `"None"` (not Python `None`) — common symptom: a path-typed config field gets `"/<dir>/None"` joined onto it and the dataset complains the file doesn't exist | YAML / Hydra grammar: bare `None` is a string token | Use the YAML null literal: `<key>=null`. To drop the key entirely, use the delete prefix: `~<key>`. Same applies to any Hydra override you customize on the CLI or inside an entry script's `hydra_overrides=[…]` |
| `cosmos-rl` dies during launch with `RuntimeError: Replica with GPUs larger than 8 occurs but not on Lepton job, please specify --node-ip-list ...` | **Misleading message — actual cause is insufficient GPUs on the host.** The launcher hits this when `nvidia-smi -L \| wc -l` is less than `(policy.n_init_replicas × policy.dp_shard_size) + (rollout.n_init_replicas × rollout.dp_shard_size)`. For the shipped local-test TOML that's `1×4 + 1×1 = 5`. Don't chase `--node-ip-list`; that path is for genuine multi-node setups | `nvidia-smi -L \| wc -l` to confirm the count, then either run on a host with enough GPUs or temporarily lower `policy.dp_shard_size` (note: dropping below 4 may not produce a working model — this is purely for smoke-testing the wiring, not training) |
| `convert_release_config_to_training.py` fails on HF download | Gated model — license not accepted, or `HF_TOKEN` unset / wrong | Accept license on HF web UI for the chosen Alpamayo version; `hf auth login`; confirm with `huggingface-cli whoami` |
| `download_pai.py` errors with HF rate limit | Hub throttling | Re-run with backoff, or set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` once the data is cached locally |
| Joint reward run errors `Lingo-Judge not found` (`local_files_only=True`) | Grader not cached | `hf download wayveai/Lingo-Judge --local-dir "$LINGO_JUDGE_DIR"`; set `reasoning_grading_model_path` in the TOML |
| `clip_index_mini.parquet` / `clip_index_reasoning_mini.parquet` not found | Skipped the curation step, or used wrong dataset path | Re-run `curate_pai_samples.py` (motion) or re-download with `--only-reasoning-chunks --num-reasoning-clips N` (joint) |
| Cosmos-RL crashes with `Could not load model_rank_*.pth` from a trained ckpt | Trying to load policy ckpt directly | Use `convert_cosmos_rl_checkpoint.py` to assemble it into a HF dir first |
| vLLM rollout OOM on startup | `gpu_memory_utilization` too high, or model larger than rollout GPU | Drop `[rollout].gpu_memory_utilization` (e.g. `0.7`), or reduce `[rollout].n_generation` / `model_max_length` |
| Policy OOM | Insufficient FSDP sharding or grad-ckpt off | Increase `dp_shard_size`, ensure `model_gradient_checkpointing = true`, lower `train_batch_per_replica` |
| `pending rollouts` grows monotonically | Rollout too fast vs policy | Enable prefetch, drop `rollout.batch_size`/`n_generation`, add policy replicas. See [Monitor](#monitor--controller-log-healthy-vs-sick-patterns) |
| `pending rollouts` repeatedly hits 0 | Rollout too slow | Add rollout replicas, or reduce policy throughput |
| Reward variance ≈ 0 per group | GRPO group collapse (all `n_generation` completions get the same reward) | Raise sampling temperature, lower `repetition_penalty`, or design a denser reward |
| W&B 403 `upsertBucket … permission denied` | API key not authorized for `project_name`/team | Update `project_name` to one you own, or use a different `WANDB_API_KEY`; or set `WANDB_MODE=disabled` |
| Multi-node run trains on duplicate samples | `data_dispatch_as_rank_in_mesh` left at `false` | Set `[train.train_policy].data_dispatch_as_rank_in_mesh = true` for any multi-node run |
| Wrong Alpamayo version loaded (1 vs 1.5 mismatch) | Entry script's `hydra_config_name` doesn't match `$ALPAMAYO_VERSION` | Edit entry script: `"alpamayo1_5_rvla_rl_pai"` for 1.5, `"alpamayo1_rvla_rl_pai"` for 1 |

---

## Additional resources

- Recipe README (human-facing): [README.md](README.md)
- Entry scripts: [models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py](models/reasoning_vla/alpamayo_cosmos_rl_post_training_entry.py),
  [models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py](models/reasoning_vla/alpamayo_cosmos_rl_post_training_reasoning_entry.py)
- TOML configs: [toml/alpamayo_rvla_rl_local_test.toml](toml/alpamayo_rvla_rl_local_test.toml),
  [toml/alpamayo_rvla_rl_local_test_with_reasoning.toml](toml/alpamayo_rvla_rl_local_test_with_reasoning.toml)
- Hydra (PAI) configs: [hydra_configs/alpamayo1_5_rvla_rl_pai.yaml](hydra_configs/alpamayo1_5_rvla_rl_pai.yaml),
  [hydra_configs/alpamayo1_rvla_rl_pai.yaml](hydra_configs/alpamayo1_rvla_rl_pai.yaml)
- Reward implementations:
  [rewards/aggregated_reward.py](rewards/aggregated_reward.py) (motion),
  [rewards/aggregated_reward_with_reasoning.py](rewards/aggregated_reward_with_reasoning.py) (joint),
  building blocks: [rewards/traj_reward.py](rewards/traj_reward.py),
  [rewards/comfort_reward.py](rewards/comfort_reward.py)
- Shared launcher: [launcher.py](launcher.py)
- Prefetch / shared-memory dataset: [prefetch/server.py](prefetch/server.py),
  [prefetch/dataset.py](prefetch/dataset.py)
- Reasoning grader (Lingo-Judge wrapper): [utils/light_weight_reasoning_grading_model.py](utils/light_weight_reasoning_grading_model.py)
- Conversion scripts: `../../scripts/convert_release_config_to_training.py`,
  `../../scripts/convert_cosmos_rl_checkpoint.py`,
  `../../scripts/curate_pai_samples.py`,
  `../../scripts/download_pai.py`
- Inference notebook: [notebooks/rl_checkpoint_inference.ipynb](notebooks/rl_checkpoint_inference.ipynb)
- Cosmos-RL framework: <https://github.com/NVIDIA/Cosmos-RL>
- Cosmos-RL multi-node docs: <https://nvidia-cosmos.github.io/cosmos-rl/multinodes/overview.html>
- GRPO paper: <https://arxiv.org/abs/2402.03300>
- PAI dataset: <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>
- Pretrained Alpamayo-R1-10B (v1): <https://huggingface.co/nvidia/Alpamayo-R1-10B>
- Pretrained Alpamayo-1.5-10B: <https://huggingface.co/nvidia/Alpamayo-1.5-10B>
- Lingo-Judge reasoning grader: <https://huggingface.co/wayveai/Lingo-Judge>
- Companion SFT skill: [`recipes/alpamayo1_sft/SKILL.md`](../alpamayo1_sft/SKILL.md)
