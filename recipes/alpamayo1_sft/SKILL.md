---
name: alpamayo1-sft
description: >-
  Run end-to-end supervised fine-tuning (SFT) of the Alpamayo-1 VLM action
  model on the Physical AI AV (PAI) dataset. The agent collects a small set
  of choices from the user up front (PAI chunk range, CoC reasoning toggle,
  W&B preference, which stage(s) to run, dataset/checkpoint paths), then
  drives the whole pipeline. Use when an agent must run Stage 1 VLM SFT,
  Stage 2 trajectory-diffusion expert, or both on a multi-GPU host or
  cluster; when setting up the `a1_sft` uv venv from scratch; when overriding
  Hydra config (cameras, chunk IDs, learning rate, DeepSpeed); when
  evaluating a Stage-2 checkpoint against `val/metric/min_ade`; when
  diagnosing common SFT failures (flash-attn dtype warnings, wandb 403,
  DeepSpeed grad-accum mismatch, `use_cache` checkpointing warnings).
  Trigger keywords: alpamayo, alpamayo-1, alpamayo1, sft, post-train,
  post-training, fine-tune, finetune, vlm, action expert, trajectory diffusion,
  reasoning vla, qwen3-vl, qwen3, pai, physical_ai_av,
  physicalai-autonomous-vehicles, hydra, omegaconf, deepspeed, zero2, torchrun,
  accelerate, flash-attn, flash attention, transformers, huggingface,
  Alpamayo-R1-10B, min_ade, ade, corner_distance, CoC, chain of causation,
  reasoning labels, ood_reasoning, wandb, hydra override, hydra full error,
  use_cache, gradient checkpointing, ddp_find_unused_parameters,
  pretrained_model_name_or_path, stage1_vlm_checkpoint_path, uv sync.
license: Apache-2.0
metadata:
  author: nvidia
  version: "2026.05"
---

# Alpamayo-1 SFT (Post-training)

This skill teaches an agent to fine-tune **Alpamayo-1** end-to-end on the
**PAI** (Physical AI Autonomous Vehicles) dataset using the recipe at
[recipes/alpamayo1_sft/](.). The recipe is a two-stage Hugging Face / Hydra /
DeepSpeed pipeline:

1. **Stage 1** fine-tunes the full Qwen3-VL base VLM to emit discrete
   trajectory tokens.
2. **Stage 2** freezes the Stage-1 VLM and trains the **trajectory-diffusion
   action expert** on top.

Both stages share one entry point ([train_hf.py](train_hf.py)) and one base
config ([configs/sft_base.yaml](configs/sft_base.yaml)). Evaluation runs
through [evaluate_hf.py](evaluate_hf.py). Validated on **8× H100 80 GB**.

## Table of Contents

1. [When to use this skill](#when-to-use-this-skill)
2. [Inputs to collect from the user (ask once, up front)](#inputs-to-collect-from-the-user-ask-once-up-front)
3. [Mental model — what gets trained, in what order](#mental-model--what-gets-trained-in-what-order)
4. [Install — `a1_sft` venv](#install--a1_sft-venv)
5. [Dataset & checkpoint](#dataset--checkpoint)
6. [Stage 1 — VLM SFT](#stage-1--vlm-sft)
7. [Stage 2 — Action expert (trajectory diffusion)](#stage-2--action-expert-trajectory-diffusion)
8. [Evaluation](#evaluation)
9. [Logging — W&B, TensorBoard, offline](#logging--wb-tensorboard-offline)
10. [Multi-GPU, multi-node, and cluster runs](#multi-gpu-multi-node-and-cluster-runs)
11. [Hydra override cheatsheet](#hydra-override-cheatsheet)
12. [Common failure modes (and the fix)](#common-failure-modes-and-the-fix)
13. [Additional resources](#additional-resources)

---

## When to use this skill

| You want to… | Use |
|--------------|-----|
| Run a complete SFT pipeline on PAI from a pretrained `Alpamayo-R1-10B` | The whole skill — Stage 1 → Stage 2 → eval |
| Fine-tune only the VLM | [Stage 1](#stage-1--vlm-sft) |
| Train only the action expert against an existing Stage-1 checkpoint | [Stage 2](#stage-2--action-expert-trajectory-diffusion) |
| Evaluate a checkpoint without further training | [Evaluation](#evaluation) |
| Add CoC reasoning labels to Stage 1 | [Stage 1 — CoC reasoning](#enable-coc-reasoning-stage-1) |
| Run on a different camera / chunk subset | [Hydra override cheatsheet](#hydra-override-cheatsheet) |
| Disable wandb (e.g. on a node with no internet or wrong team) | [Logging](#logging--wb-tensorboard-offline) |

If you're driving the model from the **inference / serving** side instead of
training it, this is the wrong skill — use the `alpamayo_r1` inference docs in
the upstream repo.

---

## Inputs to collect from the user (ask once, up front)

Before any download, install, or training step, ask the user for the
following. Confirm all answers before proceeding. If running in a context
where you cannot ask the user, halt and report rather than guess.

| Input | Why you need it | Default if user has no preference |
|-------|-----------------|-----------------------------------|
| **PAI chunk range** (e.g. `0-10`) | Sets both the `download_pai.py --chunk-ids` flag and the Hydra `data.{train,val}_dataset.chunk_ids` overrides. For full-scale runs use `0-99`; for smoke tests `0-1` (single chunk). | Ask explicitly; do not assume |
| **Train / val split inside that range** (e.g. train `0-9`, val `9-10`) | The recipe wants distinct train/val chunk ranges within what you've downloaded. For a single-chunk smoke test, both sides use the same range. | train = all but last, val = last |
| **CoC reasoning labels enabled?** (`yes` / `no`) | Toggles the `components_order` / `components_prompt` / `label_components` lists in [configs/vla_processor/default.yaml](configs/vla_processor/default.yaml) and triggers the reasoning-label download path. | `no` |
| **W&B logging?** (`yes` / `no`) | If `yes`, also collect: `WANDB_API_KEY`, `team`, `project`. If `no`, agent sets `WANDB_MODE=disabled` and leaves the shipped config (`report_to: none`) untouched. | `no` |
| **Stage(s) to run** (`stage1` / `stage2` / `both` / `eval`) | Determines which `torchrun -m alpamayo1_sft.train_hf --config-name ...` (and/or `evaluate_hf`) invocations the agent issues. | `both` then `eval` |
| **PAI dataset directory** | Used as `--output-dir` for `download_pai.py` and as `data.{train,val}_dataset.local_dir`. | Ask — no safe default |
| **Pretrained checkpoint directory** | Used as `huggingface-cli download --local-dir` target and as `model.checkpoint_path` (Stage 1) / `model.pretrained_model_name_or_path` (Stage 2). | Ask — no safe default |
| **Conditional: Stage-1 checkpoint** | Only if Stage(s) = `stage2` or `eval`. Path to an existing `output_stage1/checkpoint-XXXX/` dir. | n/a |
| **Conditional: Stage-2 checkpoint** | Only if Stage(s) = `eval`. Path to an existing `output_stage2/checkpoint-XXXX/` dir. | n/a |

### Suggested question flow

Ask all of these in a single round if your interface supports multi-question
prompts; otherwise ask sequentially in the order above. Example phrasing:

1. "Which PAI chunks should I download and train on? (e.g. `0-10` for a
   representative subset, `0-1` for a fast smoke test, `0-99` for full)"
2. "Inside that range, how should I split train vs val? (e.g. for `0-10`:
   train `0-9`, val `9-10`)"
3. "Enable CoC (chain-of-causation) reasoning labels in Stage 1? (yes/no)"
4. "Log to W&B? If yes, paste `WANDB_API_KEY`, and tell me the team and
   project. If no, I'll disable W&B."
5. "Which stage(s) should I run? (`stage1`, `stage2`, `both`, or `eval`)"
6. "Where should PAI live (or where is it already)?"
7. "Where should `Alpamayo-R1-10B/` live (or where is it already)?"
8. *(If Stage 2 only / eval only:)* "Path to your existing Stage-1
   checkpoint?"
9. *(If eval only:)* "Path to your existing Stage-2 checkpoint?"

### Capture the answers verbatim

Save the answers as shell variables (or equivalent) at the top of your run
log — every later command references them:

```bash
export CHUNK_RANGE="0-10"               # full download span
export TRAIN_CHUNKS="0-9"               # data.train_dataset.chunk_ids
export VAL_CHUNKS="9-10"                # data.val_dataset.chunk_ids
export ENABLE_COC="no"                  # yes | no
export USE_WANDB="no"                   # yes | no
export WANDB_API_KEY="..."              # only if USE_WANDB=yes
export WANDB_TEAM="..."                 # only if USE_WANDB=yes
export WANDB_PROJECT="..."              # only if USE_WANDB=yes
export STAGES="both"                    # stage1 | stage2 | both | eval
export PAI_DIR="/path/to/pai_dataset"
export CKPT_DIR="/path/to/Alpamayo-R1-10B"
export STAGE1_CKPT=""                   # set if STAGES in {stage2, eval}
export STAGE2_CKPT=""                   # set if STAGES = eval
```

After this, do not ask the user further questions — run the rest of the
pipeline end-to-end. The only acceptable mid-run halt conditions are:
hard-stop errors (CUDA OOM, missing files, hydra errors), or the user
explicitly interrupting.

---

## Mental model — what gets trained, in what order

```
   PAI clip ─►  PAIDataset  ─►  vla_processor  ─►  collate_fn
                                                       │
                                                       ▼
   Alpamayo-R1-10B ckpt  ─►  Stage 1: TrainableReasoningVLA      ──► output_stage1/checkpoint-XXXX
        (VLM + trajectory tokenizer; full VLM trainable,
         visual subnet at 0.1× LR, ZeRO-2, grad-ckpt)
                                                       │
                                                       ▼
                              Stage 2: TrainableAlpamayoR1       ──► output_stage2/checkpoint-XXXX
        (Stage-1 VLM **frozen**, action expert trained;
         no DeepSpeed, no grad-ckpt, DDP with
         `find_unused_parameters=True`)
                                                       │
                                                       ▼
                              evaluate_hf.py → val/metric/min_ade, ade, corner_distance
```

Key facts an agent must internalise before running anything:

- **Same entry point both stages.** `alpamayo1_sft.train_hf` reads
  `--config-name sft_stage{1,2}`. The stage difference is **purely in the
  Hydra config** ([configs/sft_stage1.yaml](configs/sft_stage1.yaml),
  [configs/sft_stage2.yaml](configs/sft_stage2.yaml)), which compose on top of
  [configs/sft_base.yaml](configs/sft_base.yaml).
- **Stage 1 ≠ Stage 2 model class.** Stage 1 instantiates
  `TrainableReasoningVLA.from_alpamayo_checkpoint` (loads VLM weights from a
  **local directory** — the HF-downloaded `Alpamayo-R1-10B/` folder). Stage 2
  instantiates `TrainableAlpamayoR1.from_pretrained` and also needs the same
  local model directory **plus** the Stage-1 output as
  `stage1_vlm_checkpoint_path`.
- **DeepSpeed only in Stage 1.** Stage 2 turns DeepSpeed off
  (`trainer.deepspeed: null`) and enables
  `ddp_find_unused_parameters=true` (the diffusion expert leaves parts of the
  graph unused per step).
- **Required overrides (no defaults provided).** `model.checkpoint_path` (S1)
  / `model.pretrained_model_name_or_path` (S2) and both
  `data.{train,val}_dataset.local_dir` use Hydra's `???` mandatory sentinel —
  the run will error out at Hydra time if you forget them.
- **PAI chunk split** is by default `train: "0-99"`, `val: "99-100"` (see
  [sft_base.yaml:12-22](configs/sft_base.yaml#L12-L22)) — i.e. one held-out
  chunk. Override on the CLI for smoke tests.

---

## Install — `a1_sft` venv

This recipe builds a single `uv` venv at `recipes/alpamayo1_sft/a1_sft/`. uv
handles everything; **do not** create a separate conda env.

```bash
# 1) Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2) Clone (or `cd` into an existing clone) and provision the venv
export YOUR_HOME="/path/to/your/workspace"
git clone https://github.com/NVlabs/alpamayo-recipes.git "$YOUR_HOME/alpamayo-recipes"
cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_sft"

uv venv a1_sft           # if `a1_sft/` already exists, skip this line
source a1_sft/bin/activate
uv sync --active
```

> **Non-interactive harnesses (agents, CI, `bash -c "..."` per step):** the
> `export PATH=...` above only persists for the **current** shell. Either
> append the line to `~/.bashrc` once, or re-export it at the top of every
> non-interactive shell that calls `uv`. Forgetting this is the most common
> reason a subsequent step reports `uv: command not found`.

What `uv sync` resolves:

- `alpamayo_r1` from
  [`github.com/NVlabs/alpamayo`](https://github.com/NVlabs/alpamayo.git) (see
  [pyproject.toml:51-53](pyproject.toml#L51-L53)).
- `alpamayo-recipes` editable from `../../src/` (the recipe-side utilities).
- `torch==2.8.0`, `transformers==4.57.1`, `deepspeed==0.18.2`,
  `flash-attn>=2.8.3` (built without isolation per
  [pyproject.toml:48-49](pyproject.toml#L48-L49)).

After every `uv sync`, **verify the import contract** the recipe assumes —
this is the single most common cause of opaque Hydra `InstantiationException`s
(see [pitfalls](#common-failure-modes-and-the-fix)):

```bash
python -c "from alpamayo.processor.qwen_processor import \
  get_preprocess_data_fn_from_model_config, collate_fn_from_model_config; print('ok')"
python -c "from alpamayo.data.pai import PAIDataset; print('ok')"
```

---

## Dataset & checkpoint

### PAI dataset

Use **`$CHUNK_RANGE`** from the user's answers (see [Inputs](#inputs-to-collect-from-the-user-ask-once-up-front)).

```bash
export HF_TOKEN=<your HuggingFace token>
cd "$YOUR_HOME/alpamayo-recipes"

python scripts/download_pai.py \
  --chunk-ids "$CHUNK_RANGE" \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir "$PAI_DIR"
```

- `--chunk-ids` accepts `0` / `0 1 5` / `0-3`. Omit to download the **full
  ~97 TB** dataset — don't do this on shared scratch without checking quota.
- The PAI HF dataset license must be accepted on the HuggingFace web UI before
  `HF_TOKEN` will work.
- **Token discovery in non-interactive shells:** if `$HF_TOKEN` is not set
  in the agent's environment but you've run `huggingface-cli login` on the
  machine before, the cached token at `~/.cache/huggingface/token` is picked
  up automatically by `huggingface-cli download` and `datasets`. Source it
  into the env explicitly when a downstream tool (or `download_pai.py`)
  expects the `HF_TOKEN` variable:
  `export HF_TOKEN=$(<~/.cache/huggingface/token)`.

### Optional CoC reasoning labels

Run this **only if** `$ENABLE_COC` is `yes`. Otherwise skip — the standard
download above is sufficient.

```bash
python scripts/download_pai.py --only-reasoning-chunks \
  --num-reasoning-clips 16 \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --reasoning ood_reasoning.parquet \
  --output-dir "$PAI_DIR"
```

### Pretrained checkpoint

```bash
huggingface-cli download nvidia/Alpamayo-R1-10B --local-dir "$CKPT_DIR"
```

**This directory** is what both stages refer to as
`model.checkpoint_path` (S1) and `model.pretrained_model_name_or_path` (S2).
It must contain `config.json`, `model.safetensors.index.json`, shard files,
and the processor / tokenizer.

---

## Stage 1 — VLM SFT

Stage 1 ([configs/sft_stage1.yaml](configs/sft_stage1.yaml)) fine-tunes the
full Qwen3-VL VLM via `TrainableReasoningVLA.from_alpamayo_checkpoint`
([models/sft_base_model.py](models/sft_base_model.py)). DeepSpeed ZeRO-2 +
gradient checkpointing are on; visual encoder LR is 0.1×.

### Canonical launch

Run this **only if** `$STAGES` is `stage1` or `both`. Otherwise skip to the
next stage / eval.

```bash
cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_sft"
torchrun --nproc_per_node 8 \
  -m alpamayo1_sft.train_hf \
  --config-path pkg://alpamayo1_sft/configs \
  --config-name sft_stage1 \
  model.checkpoint_path="$CKPT_DIR" \
  data.train_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.train_dataset.chunk_ids="$TRAIN_CHUNKS" \
  data.val_dataset.chunk_ids="$VAL_CHUNKS"
```

Outputs land in `output_stage1/` (see [sft_stage1.yaml:11](configs/sft_stage1.yaml#L11)):

```
output_stage1/
├── config.yaml                          # fully resolved config
├── checkpoint-500/                      # save_steps=500, save_total_limit=2
├── checkpoint-1000/
└── ...
```

Each `checkpoint-XXXX/` contains `model.safetensors.index.json` + shards (HF
Trainer convention). The **highest-numbered** checkpoint is what feeds Stage 2.

### What "healthy" Stage-1 logs look like

```
{'loss': 1.668, 'grad_norm': 0.9368, 'learning_rate': 1.25e-08, 'epoch': 0.02}
{'loss': 1.708, 'grad_norm': 1.1734, 'learning_rate': 2.50e-08, 'epoch': 0.03}
{'loss': 1.641, 'grad_norm': 0.8667, 'learning_rate': 3.75e-08, 'epoch': 0.05}
```

The release model has already been trained on this data — **do not expect a
dramatic loss drop**. The numbers above confirm the run is wired correctly
(magnitudes, grad norms, warmup), not that fine-tuning is "doing something".

### Enable CoC reasoning (Stage 1)

Apply this **only if** `$ENABLE_COC = yes`. Three pieces must all be in place
— miss any one and the run aborts with
`AssertionError: cot not found in data but 'cot' in components_order`.

**1. Processor side** — edit
[configs/vla_processor/default.yaml](configs/vla_processor/default.yaml)
**before** launching Stage 1:

```yaml
components_order:   ["image", "traj_history", "prompt", "cot", "traj_future"]
components_prompt:  ["cot", "traj_future"]
label_components:   ["cot", "traj_future"]
```

**2. Dataset on disk** — the reasoning subset must be downloaded with both
`--reasoning ood_reasoning.parquet` and `--num-reasoning-clips N` (so the
filtered index `clip_index_reasoning_mini.parquet` is produced):

```bash
python scripts/download_pai.py --only-reasoning-chunks \
  --num-reasoning-clips 16 \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics vehicle_dimensions \
  --labels egomotion \
  --reasoning ood_reasoning.parquet \
  --output-dir "$PAI_DIR"
```

After this, `$PAI_DIR` must contain both `reasoning/ood_reasoning.parquet`
and `clip_index_reasoning_mini.parquet`. Verify with
`ls "$PAI_DIR/reasoning/" "$PAI_DIR/clip_index_reasoning_mini.parquet"`.

**3. Dataset config side** — the train/val datasets must (a) point at the
reasoning parquet, (b) point at the filtered clip index, AND (c) disable
the single-keyframe shortcut so each sample's `t0_us` comes from the
per-clip reasoning event timestamps. Either edit
[configs/sft_base.yaml](configs/sft_base.yaml) (add `reasoning_metadata` and
`clip_index_metadata`, set `use_default_keyframe: false` under both
`train_dataset` and `val_dataset`), or override on the CLI. Note the `+`
prefix on the **new** keys (Hydra struct mode rejects non-existent keys
otherwise — error: `Key 'reasoning_metadata' is not in struct ... To append
to your config use +data.train_dataset...`); `use_default_keyframe` already
exists in the YAML so it takes no `+`:

```bash
torchrun ... \
  +data.train_dataset.reasoning_metadata=reasoning/ood_reasoning.parquet \
  +data.val_dataset.reasoning_metadata=reasoning/ood_reasoning.parquet \
  +data.train_dataset.clip_index_metadata=clip_index_reasoning_mini.parquet \
  +data.val_dataset.clip_index_metadata=clip_index_reasoning_mini.parquet \
  data.train_dataset.use_default_keyframe=false \
  data.val_dataset.use_default_keyframe=false
```

> The reasoning / clip-index paths are **relative to `data.*.local_dir`**
> (the dataset joins them onto `local_dir` internally). Pass
> `reasoning/ood_reasoning.parquet`, **not** the absolute path.

Why all four are required:

- `reasoning_metadata` sets `self.avdi.reasoning_db` so the dataset can
  attach a `cot` field (see [src/alpamayo/data/pai.py:124](../../src/alpamayo/data/pai.py#L124)).
- `clip_index_metadata` points at the filtered index so the dataset
  enumerates only reasoning-bearing clips — without it, the dataset still
  iterates every clip in the chunk, including ones with no reasoning.
- `use_default_keyframe=false` makes `t0_us` come from
  `get_clip_key_frame(clip_id)` (a real event timestamp) instead of the
  global `DEFAULT_T0_US = 5_100_000`. Otherwise the lookup
  `get_reasoning_data(clip_id, 5_100_000)` raises
  `ValueError: Event timestamp 5100000 not found for <clip_id> in reasoning_db`,
  since the reasoning events live at per-clip times, not at the fixed
  5.1 s default.

### Warnings that are noise (do not treat as errors)

```
Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes,
but the current dype in Qwen3VLModel is bfloat16.
```

`bfloat16` *is* a supported dtype; the warning misreads the current dtype.
Transformers prints it once per submodule. Ignore.

```
You are attempting to use Flash Attention 2 without specifying a torch dtype.
```

Same family. Ignore (the trainer sets `bf16=True` and DeepSpeed handles the
cast).

```
`use_cache=True` is incompatible with gradient checkpointing.
Setting `use_cache=False`.
```

Expected: training has grad-ckpt on, so `use_cache` gets auto-flipped. Ignore.

```
Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1,
DeepSpeed config has 4. Using DeepSpeed's value.
```

Stage 1 sets `gradient_accumulation_steps: 4` and DeepSpeed wins; the
Accelerate plugin default of 1 is moot. Ignore.

---

## Stage 2 — Action expert (trajectory diffusion)

Stage 2 ([configs/sft_stage2.yaml](configs/sft_stage2.yaml)) loads the full
`AlpamayoR1` model via `TrainableAlpamayoR1.from_pretrained`
([models/sft_alpamayo_r1.py](models/sft_alpamayo_r1.py)), then **overrides** the
VLM weights with your Stage-1 output and trains only the trajectory expert.

DeepSpeed is **off**; gradient checkpointing is **off**;
`ddp_find_unused_parameters=true` (diffusion expert leaves parts of the graph
unused per step — without this DDP raises an error).

### Canonical launch

Run this **only if** `$STAGES` is `stage2` or `both`. If `$STAGES = both`,
this fires after Stage 1 finishes; if `$STAGES = stage2`, the user has
provided `$STAGE1_CKPT` as input.

```bash
# If $STAGES = both, pick the highest-numbered Stage-1 output automatically:
if [ "$STAGES" = "both" ]; then
  STAGE1_CKPT=$(ls -d output_stage1/checkpoint-* | sort -V | tail -1)
fi

cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_sft"
torchrun --nproc_per_node 8 \
  -m alpamayo1_sft.train_hf \
  --config-path pkg://alpamayo1_sft/configs \
  --config-name sft_stage2 \
  model.pretrained_model_name_or_path="$CKPT_DIR" \
  model.stage1_vlm_checkpoint_path="$STAGE1_CKPT" \
  data.train_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.train_dataset.chunk_ids="$TRAIN_CHUNKS" \
  data.val_dataset.chunk_ids="$VAL_CHUNKS"
```

Required parameter contract:

- **`model.pretrained_model_name_or_path`** — the **same local folder** used
  for Stage 1 (`Alpamayo-R1-10B/`). Do **not** point at the Stage-1 output
  here; this loads the action-expert structure + processor, not the VLM
  weights.
- **`model.stage1_vlm_checkpoint_path`** — the Stage-1 Trainer output dir
  (must contain `model.safetensors.index.json` + shards).
  This is the directory whose VLM weights overwrite the
  `pretrained_model_name_or_path` VLM weights at load time.

Outputs land in `output_stage2/`.

> **Smoke testing Stage 2 without a real Stage-1 run.** On memory-constrained
> hardware where Stage 1 OOMs, you can still exercise the Stage 2 wiring
> (model load, dataloader, optimizer init, first training steps) by pointing
> `stage1_vlm_checkpoint_path` at the **same directory** as
> `pretrained_model_name_or_path` (the base `Alpamayo-R1-10B/`). The base
> checkpoint already satisfies the `model.safetensors.index.json` + shards
> contract, so the load path runs identically — you just won't get the
> Stage-1 fine-tuning gains. Use this only for pipeline validation, never
> for results.

### Picking the right Stage-1 checkpoint

`save_total_limit: 2` means only the **last two** Stage-1 checkpoints stay on
disk. After Stage 1 finishes, `ls output_stage1/checkpoint-*` and use the
highest-numbered directory. If you need an earlier one for ablations, bump
`save_total_limit` **before** launching Stage 1.

---

## Evaluation

Run this **only if** `$STAGES` is `eval` or `both`. If `$STAGES = both`,
this fires after Stage 2 finishes; if `$STAGES = eval`, the user has
provided `$STAGE2_CKPT` as input.

```bash
# If $STAGES = both, pick the highest-numbered Stage-2 output automatically:
if [ "$STAGES" = "both" ]; then
  STAGE2_CKPT=$(ls -d output_stage2/checkpoint-* | sort -V | tail -1)
fi

cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_sft"
torchrun --nproc_per_node 8 \
  -m alpamayo1_sft.evaluate_hf \
  --config-path pkg://alpamayo1_sft/configs \
  --config-name sft_stage2 \
  evaluate.eval_ckpt="$STAGE2_CKPT" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.chunk_ids="$VAL_CHUNKS"
```

[evaluate_hf.py](evaluate_hf.py) auto-detects the model class
(`TrainableReasoningVLA` vs `TrainableAlpamayoR1`) and routes `eval_ckpt` into
the correct field (`checkpoint_path` vs `pretrained_model_name_or_path`).

### What "healthy" eval looks like

With the default chunk split (`train: "0-99"`, `val: "99-100"`) and the
release checkpoint as base, `val/metric/min_ade` should fall **below 1**.
Reference numbers:

```
val/metric/ade              2.0072
val/metric/ade/by_t=3.0     0.3970
val/metric/corner_distance  0.6632
val/metric/min_ade          0.6270
val/metric/min_ade/by_t=0.5 0.0079
val/metric/min_ade/by_t=1.0 0.0261
val/metric/min_ade/by_t=3.0 0.2008
val/metric/min_ade/by_t=5.0 0.4351
```

The release model is already trained on PAI, so absolute metric **gain** over
the base checkpoint will be small. Use these numbers as a **shape check**, not
as a target to beat. The metric definition lives in
`alpamayo.metrics.metric_api.DistanceMetrics` (see
[configs/sft_base.yaml:32-39](configs/sft_base.yaml#L32-L39)).

---

## Logging — W&B, TensorBoard, offline

The branch the agent does here is determined by `$USE_WANDB` (collected in
[Inputs](#inputs-to-collect-from-the-user-ask-once-up-front)).

### If `$USE_WANDB = no` (the default)

Do nothing — `sft_base.yaml` ships with `trainer.report_to: none` and the
wandb defaults **commented out** ([configs/sft_base.yaml:1-3](configs/sft_base.yaml#L1-L3)).
For belt-and-braces (e.g. on a machine where someone has previously run
`wandb login`), prefix every `torchrun` with:

```bash
WANDB_MODE=disabled torchrun ...
```

### If `$USE_WANDB = yes`

The user has provided `$WANDB_API_KEY`, `$WANDB_TEAM`, `$WANDB_PROJECT`.
Two equivalent ways to wire them in:

**Option A — CLI overrides (recommended, no file edits):**

```bash
export WANDB_API_KEY="$WANDB_API_KEY"   # authenticates wandb
torchrun ... \
  trainer.report_to=wandb \
  +wandb.team="$WANDB_TEAM" \
  +wandb.project="$WANDB_PROJECT"
```

**Option B — file edits (if you also want logging defaults persisted):**

1. Uncomment `- /wandb: default` in [configs/sft_base.yaml:2](configs/sft_base.yaml#L2).
2. Set `trainer.report_to: wandb` in the same file.
3. Fill in `team: $WANDB_TEAM` and `project: $WANDB_PROJECT` in
   [configs/wandb/default.yaml](configs/wandb/default.yaml) (both ship as `???`).
4. `export WANDB_API_KEY="$WANDB_API_KEY"` before launching `torchrun`.

In either option, the API key must belong to an account that has **write
access** to the team/project — otherwise wandb fails at `wandb.init()` with
a 403 (see [pitfalls](#common-failure-modes-and-the-fix)). The agent should
fail fast and re-prompt the user for a working key rather than retry blindly.

### W&B 403 = wrong team

```
wandb.errors.errors.CommError: Error uploading run: returned error 403:
{"errors":[{"message":"permission denied","path":["upsertBucket"],"extensions":
{"code":"PERMISSION_ERROR"}}]}
```

Your account isn't a member of the entity/project configured in
`configs/wandb/default.yaml`. Either change `team` / `project` to an entity
you own, or pass `wandb.team=<your-entity>` on the CLI, or disable wandb for
this run.

---

## Multi-GPU, multi-node, and cluster runs

The defaults assume **single-node, 8× H100 80 GB**. For other shapes:

### Single-node, fewer GPUs

```bash
torchrun --nproc_per_node 4 -m alpamayo1_sft.train_hf ...
```

For Stage 1 with <8 GPUs, also raise `trainer.gradient_accumulation_steps`
proportionally to preserve effective batch size (Stage 1 default is 4).

### Multi-node

```bash
torchrun --nnodes $NNODES --nproc_per_node 8 \
         --node-rank $NODE_RANK \
         --rdzv-id alpamayo-sft --rdzv-backend c10d \
         --rdzv-endpoint $RDZV_HOST:$RDZV_PORT \
         -m alpamayo1_sft.train_hf ...
```

On NVIDIA OSMO / SLURM clusters, wire these env vars from the scheduler's
`SLURM_*` / OSMO equivalents. The recipe doesn't ship a SLURM template;
adapt the pattern in [`scripts/`](../../scripts) at the repo root.

### Cluster gotchas

- **Same paths on every node.** PAI dataset, model checkpoint, and the venv
  must be visible at identical absolute paths on every rank (or mount the
  same network filesystem). The script does not stage anything.
- **Flash-attn is built from source per node.** First run on a new node
  takes 5–10 min before training starts; this is normal, not a hang.

---

## Hydra override cheatsheet

All overrides go on the `torchrun` command line **without** a leading `--`.

| Override | What it does |
|----------|--------------|
| `data.train_dataset.local_dir=<path>` | PAI dataset root (train split) |
| `data.val_dataset.local_dir=<path>` | PAI dataset root (val split) |
| `data.train_dataset.chunk_ids="0-50"` | Train chunk range (default `0-99`) |
| `data.val_dataset.chunk_ids="99-100"` | Val chunk range |
| `model.checkpoint_path=<path>` | **Stage 1** base VLM checkpoint dir |
| `model.pretrained_model_name_or_path=<path>` | **Stage 2** base model dir (same folder as S1) |
| `model.stage1_vlm_checkpoint_path=<path>` | **Stage 2** Stage-1 trainer output |
| `trainer.learning_rate=1e-5` | LR override (Stage 1 default is 1e-5; Stage 2 inherits 1e-4 from base) |
| `trainer.num_train_epochs=1` | Epochs (default 3) |
| `trainer.per_device_train_batch_size=1` | Per-rank batch size |
| `trainer.gradient_accumulation_steps=8` | Grad-accum (Stage 1 ships 4) |
| `trainer.save_steps=200 trainer.save_total_limit=5` | Checkpoint cadence |
| `trainer.report_to=wandb` | Switch on W&B |
| `paths.output_dir=output_smoke` | Re-route outputs (smoke tests) |
| `evaluate.eval_ckpt=<path>` | Checkpoint for [evaluate_hf.py](evaluate_hf.py) |
| `evaluate.max_eval_steps=10` | Cap eval (smoke test); default `-1` = full val |

For a one-shot **smoke test** (very small chunk range, 1 epoch, low save):

```bash
torchrun --nproc_per_node 8 -m alpamayo1_sft.train_hf \
  --config-path pkg://alpamayo1_sft/configs --config-name sft_stage1 \
  model.checkpoint_path=/path/to/Alpamayo-R1-10B \
  data.train_dataset.local_dir=/path/to/pai_dataset \
  data.val_dataset.local_dir=/path/to/pai_dataset \
  data.train_dataset.chunk_ids="0-1" \
  data.val_dataset.chunk_ids="0-1" \
  trainer.num_train_epochs=1 \
  trainer.save_steps=50 trainer.save_total_limit=1 \
  paths.output_dir=output_smoke
```

> If you only downloaded **one** chunk (e.g. `download_pai.py --chunk-ids 0-1`
> yields just chunk 0), set both `train_dataset.chunk_ids` and
> `val_dataset.chunk_ids` to the **same** range — train/val overlap is
> acceptable for a smoke test, and any non-overlapping val range would
> reference data that isn't on disk. For real runs, download more chunks
> (`--chunk-ids 0-10`) and use distinct train/val ranges (e.g. train `"0-9"`,
> val `"9-10"`).

If this finishes a few optimizer steps without a Hydra `InstantiationException`
and emits non-NaN loss, the wiring is good.

---

## Common failure modes (and the fix)

Run **once** with `HYDRA_FULL_ERROR=1` to expose the chained traceback —
without it, `InstantiationException` hides the real cause behind a one-liner.

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `InstantiationException("Error locating target 'alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config'")` | The installed `alpamayo-recipes` doesn't export that symbol — the venv is stale or wasn't fully synced | Verify with `python -c "from alpamayo.processor.qwen_processor import get_preprocess_data_fn_from_model_config"`; re-run `uv sync --active` |
| `InstantiationException` on `alpamayo.data.pai.PAIDataset` | The `alpamayo-recipes` editable install (`../../src`) didn't take | `uv pip show alpamayo-recipes` → re-run `uv sync --active` |
| `AssertionError: cot not found in data but 'cot' in components_order` | You enabled CoC on the processor (`vla_processor/default.yaml`) but didn't wire the dataset side (`reasoning_metadata` + `clip_index_metadata` + `use_default_keyframe=false`) | See [Enable CoC reasoning (Stage 1)](#enable-coc-reasoning-stage-1) — all **four** pieces (processor config, dataset on disk with `--num-reasoning-clips`, dataset config pointing at the filtered index, default-keyframe disabled) must be set together |
| `Key 'reasoning_metadata' is not in struct ... Could not override 'data.train_dataset.reasoning_metadata'` | Hydra struct mode rejects setting keys that aren't in the shipped YAML | Prefix the override with `+` to **append** rather than override: `+data.train_dataset.reasoning_metadata=...`. Same for `clip_index_metadata` and any other key not already in `sft_base.yaml` |
| `ValueError: Event timestamp 5100000 not found for <clip_id> in reasoning_db` | CoC is enabled and `reasoning_metadata` is wired, but `use_default_keyframe: true` is still in effect — `t0_us` is forced to the global 5.1 s keyframe, which doesn't match the per-clip reasoning event timestamps | Override `data.train_dataset.use_default_keyframe=false` and `data.val_dataset.use_default_keyframe=false` (no `+` — the key is already in `sft_base.yaml`). See [Enable CoC reasoning (Stage 1)](#enable-coc-reasoning-stage-1) |
| `wandb.errors.CommError: 403 ... permission denied ... upsertBucket` | Account not in the wandb entity/project from `configs/wandb/default.yaml` | Change `team`/`project`, override on CLI, or `WANDB_MODE=disabled` |
| `Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes ...` repeated per submodule | Spurious transformers warning misreading the current dtype | **Ignore** — bfloat16 is supported; trainer is configured correctly |
| `You are attempting to use Flash Attention 2 without specifying a torch dtype` | Same as above | Ignore |
| `use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False.` | grad-ckpt is on in Stage 1; transformers auto-disables KV cache during training | Ignore |
| `Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 4` | Accelerate plugin default vs. Stage-1 DeepSpeed setting; DeepSpeed wins | Cosmetic; ignore. (Or set `trainer.gradient_accumulation_steps=4` to silence.) |
| `RuntimeError: Expected to mark a variable ready only once` / DDP unused-param errors **in Stage 2** | `ddp_find_unused_parameters` not true; diffusion expert leaves unused subnets | Inherited from `sft_stage2.yaml`; if you override `trainer` make sure `ddp_find_unused_parameters: true` survives |
| OOM during Stage 1 | Effective batch too large for 80 GB | Lower `trainer.per_device_train_batch_size` (already 1 by default) → raise `gradient_accumulation_steps`. Don't disable grad-ckpt in Stage 1 |
| OOM during Stage 2 | Stage 2 doesn't use DeepSpeed or grad-ckpt by design | Either reduce `per_device_train_batch_size`, freeze more submodules (`cotrain_vlm: false` is already the default in `ar1_expert.yaml`), or move to >8 GPUs |
| `Hydra` complains `MISSING ???` for `model.checkpoint_path` or `data.*.local_dir` | Forgot to pass the override | Re-launch with the missing override(s); these are `???` sentinels in [sft_base.yaml](configs/sft_base.yaml) / [ar1_base.yaml](configs/models/ar1_base.yaml) |
| Stage 2 launches but loss is NaN immediately | Stage-1 checkpoint path points at the **base** model dir instead of the `output_stage1/checkpoint-XXXX` dir | `model.pretrained_model_name_or_path` = base; `model.stage1_vlm_checkpoint_path` = trainer output. Don't swap them |
| `flash-attn` build hangs / fails at install time | Compiling against a CUDA/torch version it doesn't support | Make sure `torch==2.8.0` is resolved; check `nvcc --version` against flash-attn's compat matrix; rebuild with `uv pip install flash-attn --no-build-isolation --force-reinstall` |
| All 8 ranks segfault on first batch | Dataset `local_dir` doesn't exist on this node (cluster path mismatch) | Echo the path, `ls` it from every node; stage data if needed |

---

## Additional resources

- Recipe README (concise human-facing version): [README.md](README.md)
- Stage 1 model code: [models/sft_base_model.py](models/sft_base_model.py)
- Stage 2 model code: [models/sft_alpamayo_r1.py](models/sft_alpamayo_r1.py)
- Custom HF trainer subclass: [trainer.py](trainer.py)
- Training entry point: [train_hf.py](train_hf.py)
- Evaluation entry point: [evaluate_hf.py](evaluate_hf.py)
- Base config (shared): [configs/sft_base.yaml](configs/sft_base.yaml)
- Stage 1 / Stage 2 overlays:
  [configs/sft_stage1.yaml](configs/sft_stage1.yaml),
  [configs/sft_stage2.yaml](configs/sft_stage2.yaml)
- VLA processor (CoC toggle lives here):
  [configs/vla_processor/default.yaml](configs/vla_processor/default.yaml)
- W&B defaults: [configs/wandb/default.yaml](configs/wandb/default.yaml)
- DeepSpeed ZeRO-2 (Stage 1 only): [configs/deepspeed/zero2.json](configs/deepspeed/zero2.json)
- PAI download helper (run from repo root): `../../scripts/download_pai.py`
- Upstream `alpamayo_r1`: <https://github.com/NVlabs/alpamayo>
- PAI dataset: <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles>
- Pretrained Alpamayo-R1-10B: <https://huggingface.co/nvidia/Alpamayo-R1-10B>
