---
name: alpamayo1-5-sft
description: >-
  Run end-to-end supervised fine-tuning (SFT) of the Alpamayo-1.5 VLM action
  model on PAI (with navigation conditioning) or LingoQA (for visual question
  answering). The agent collects a small set of choices from the user up front
  (Stage-1 task — nav / vqa / default trajectory, W&B preference, dataset
  and checkpoint paths, which stage(s) to run), then drives the whole
  pipeline. Use when an agent must convert the released `nvidia/Alpamayo-1.5-10B`
  checkpoint into A1-format with `convert_checkpoint.py to-a1`, prepare PAI +
  nav annotations or LingoQA Scenery, run Stage-1 VLM SFT on the chosen task,
  optionally continue into Stage-2 trajectory-diffusion-expert training (for
  trajectory tasks), and evaluate; when setting up the `a1_5_sft` uv venv from
  scratch; when overriding Hydra config (`vla_processor` variant,
  `chunk_ids`, learning rate, DeepSpeed); when diagnosing common SFT failures
  (flash-attn dtype warnings, wandb 403, DeepSpeed grad-accum mismatch,
  `use_cache` checkpointing warnings, hardcoded-path slip-ups in the shipped
  config).
  Trigger keywords: alpamayo, alpamayo-1.5, alpamayo1_5, alpamayo1.5, sft,
  post-train, post-training, fine-tune, finetune, vlm, action expert,
  trajectory diffusion, reasoning vla, qwen3-vl, qwen3, pai, physical_ai_av,
  physicalai-autonomous-vehicles, lingoqa, lingo qa, scenery, vqa,
  visual question answering, navigation, nav, route, hydra, omegaconf,
  deepspeed, zero2, torchrun, accelerate, flash-attn, flash attention,
  transformers, huggingface, Alpamayo-1.5-10B, Alpamayo-R1-10B,
  convert_checkpoint, to-a1, A1-format, min_ade, ade, corner_distance,
  wandb, hydra override, hydra full error, use_cache,
  gradient checkpointing, ddp_find_unused_parameters,
  pretrained_model_name_or_path, stage1_vlm_checkpoint_path, uv sync,
  PAIDatasetWithNav, LingoQADataset, vla_processor, gdown.
license: Apache-2.0
metadata:
  author: nvidia
  version: "2026.05"
---

# Alpamayo-1.5 SFT (Post-training)

This skill teaches an agent to fine-tune **Alpamayo-1.5** end-to-end. The
recipe at [recipes/alpamayo1_5_sft/](.) extends the Alpamayo-1 SFT pipeline
with two new Stage-1 task flavours:

1. **Navigation-conditioned trajectory** — trajectory prediction guided by
   route instructions (e.g. *"Turn left in 40m"*), trained on PAI with a
   separate nav-annotation JSON.
2. **Visual Question Answering** — driving-scene Q&A, demonstrated on
   the **LingoQA Scenery** split.

A *default-trajectory* Stage-1 variant (same pattern as Alpamayo-1 SFT) is
also available. Stage-2 trajectory-diffusion-expert training is supported
on top of the trajectory-style Stage-1 outputs (nav or default). Both
stages share one entry point ([train_hf.py](train_hf.py)) and one base
config ([configs/sft_base.yaml](configs/sft_base.yaml)). Evaluation runs
through [evaluate_hf.py](evaluate_hf.py). Validated on **8× H100 80 GB**.

## Table of Contents

1. [When to use this skill](#when-to-use-this-skill)
2. [Inputs to collect from the user (ask once, up front)](#inputs-to-collect-from-the-user-ask-once-up-front)
3. [Mental model — what gets trained, in what order](#mental-model--what-gets-trained-in-what-order)
4. [Install — `a1_5_sft` venv](#install--a1_5_sft-venv)
5. [Checkpoint — download Alpamayo-1.5 and convert to A1 format](#checkpoint--download-alpamayo-15-and-convert-to-a1-format)
6. [Dataset — PAI (nav / trajectory) or LingoQA (VQA)](#dataset--pai-nav--trajectory-or-lingoqa-vqa)
7. [VLA processor variants](#vla-processor-variants)
8. [Stage 1 — Nav-conditioned trajectory](#stage-1--nav-conditioned-trajectory)
9. [Stage 1 — LingoQA VQA](#stage-1--lingoqa-vqa)
10. [Stage 2 — Action expert (trajectory diffusion)](#stage-2--action-expert-trajectory-diffusion)
11. [Evaluation](#evaluation)
12. [Logging — W&B, TensorBoard, offline](#logging--wb-tensorboard-offline)
13. [Multi-GPU, multi-node, and cluster runs](#multi-gpu-multi-node-and-cluster-runs)
14. [Hydra override cheatsheet](#hydra-override-cheatsheet)
15. [Common failure modes (and the fix)](#common-failure-modes-and-the-fix)
16. [Additional resources](#additional-resources)

---

## When to use this skill

| You want to… | Use |
|--------------|-----|
| Run a complete Alpamayo-1.5 SFT pipeline on PAI with nav conditioning | [Inputs](#inputs-to-collect-from-the-user-ask-once-up-front) → task = `nav`, full Stage 1 + Stage 2 + eval |
| Fine-tune Alpamayo-1.5 on LingoQA VQA only | task = `vqa`, Stage 1 only (Stage 2 doesn't apply) |
| Just convert the released Alpamayo-1.5-10B to A1 format | [Checkpoint conversion](#checkpoint--download-alpamayo-15-and-convert-to-a1-format) |
| Train only the action expert against an existing Stage-1 trajectory checkpoint | [Stage 2](#stage-2--action-expert-trajectory-diffusion) |
| Evaluate a Stage-2 checkpoint | [Evaluation](#evaluation) |

If you're fine-tuning Alpamayo-1 (not 1.5), use the companion skill at
[`recipes/alpamayo1_sft/SKILL.md`](../alpamayo1_sft/SKILL.md). For RL
post-training, use [`recipes/alpamayo1_x_rl/SKILL.md`](../alpamayo1_x_rl/SKILL.md).

---

## Inputs to collect from the user (ask once, up front)

Before any download, install, or training step, ask the user for the
following. Confirm all answers before proceeding. If running in a context
where you cannot ask the user, halt and report rather than guess.

| Input | Why you need it | Default if user has no preference |
|-------|-----------------|-----------------------------------|
| **Stage-1 task** (`nav` / `vqa`) | Picks the Hydra `--config-name` (`sft_stage1_nav` / `sft_stage1_lingoqa`), the `vla_processor` variant (`nav` / `vqa`), and the dataset class (`PAIDatasetWithNav` / `LingoQADataset`) | Ask explicitly — there's no safe default; each task pulls a different dataset |
| **Stage(s) to run** (`stage1` / `stage2` / `both` / `eval`) | Determines which `torchrun -m alpamayo1_5_sft.train_hf …` / `evaluate_hf` invocations to issue. Stage 2 only applies to the trajectory task (`nav`), not `vqa` | `both` then `eval` for `nav`; `stage1` only for `vqa` |
| **PAI dataset directory** | Required for `nav`. Used as `--output-dir` for `download_pai.py` and as `data.{train,val}_dataset.local_dir`. | Ask — no safe default |
| **PAI chunk IDs** | For `nav`: use the specific list bundled by the nav annotations (`214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868`). Not used for `vqa`. | The 19-chunk nav list for `nav` |
| **Nav annotations JSON** | Path to the `nav_demo_samples.json` from the [Alpamayo-1.5 repo](https://github.com/NVlabs/alpamayo1.5/blob/main/notebooks/nav_demo_samples.json). Each entry has `clip_id`, `t0_relative`, `nav_text`, optionally `cot`. Required for `nav`. | Ask — must be downloaded out-of-band |
| **LingoQA data root** | Required for `vqa`. Directory containing `train.parquet` + `images/train/` from the [LingoQA Scenery split](https://drive.google.com/drive/folders/1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7) (downloaded via `gdown`). | Ask — must be downloaded out-of-band (Google Drive) |
| **Alpamayo-1.5 checkpoint dir** | Where `huggingface-cli download nvidia/Alpamayo-1.5-10B` lands. | Ask — no safe default |
| **A1-format converted dir** | Output of `convert_checkpoint.py to-a1`. Used as `model.checkpoint_path` (Stage 1) and `model.pretrained_model_name_or_path` (Stage 2). | `<ckpt-dir>-A1-format` next to the raw HF dir |
| **W&B logging?** (`yes` / `no`) | `configs/sft_base.yaml` ships with the `wandb` default commented out and `trainer.report_to: none`. If `yes`, uncomment `- /wandb: default` in `sft_base.yaml`, set `trainer.report_to: wandb`, fill in `team` + `project` in `configs/wandb/default.yaml`, and export `WANDB_API_KEY`. If `no`, leave the shipped config as-is. | `no` |
| **Conditional: Stage-1 checkpoint** | Only if Stage(s) = `stage2` or `eval`. Path to an existing `output_stage1_<task>/checkpoint-XXXX/` dir. | n/a |
| **Conditional: Stage-2 checkpoint** | Only if Stage(s) = `eval`. Path to an existing `output_stage2/checkpoint-XXXX/` dir. | n/a |

### Suggested question flow

Ask in a single round if your interface supports it; otherwise sequentially:

1. "Stage-1 task: `nav` (PAI + nav annotations) or `vqa` (LingoQA)?"
2. "Which stage(s) should I run? (`stage1`, `stage2`, `both`, `eval`).
   Note: Stage 2 doesn't apply to `vqa`."
3. *(if nav)* "Where should the PAI dataset live (or where is it
   already)?"
4. *(if nav)* "Path to the nav annotations JSON (`nav_demo_samples.json`)?"
5. *(if vqa)* "Path to the LingoQA root (contains `train.parquet` +
   `images/train/`)?"
6. "Where should `Alpamayo-1.5-10B/` live (or where is it already)?"
7. "Where should the converted A1-format checkpoint be written?
   (default: `<above>-A1-format`)"
8. "Log to W&B? If yes, paste `WANDB_API_KEY`, project, team. If no,
   I'll disable W&B (the shipped config has it on by default)."
9. *(stage2/eval)* "Path to existing Stage-1 / Stage-2 checkpoint?"

### Capture the answers verbatim

```bash
export YOUR_HOME="/path/to/workspace"

export TASK="nav"                                # nav | vqa
export STAGES="both"                             # stage1 | stage2 | both | eval
export PAI_DIR="/path/to/pai_dataset"            # nav only
export PAI_CHUNK_IDS="214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868"
export NAV_ANNOTATIONS="/path/to/nav_demo_samples.json"   # nav only
export LINGOQA_DIR="/path/to/LingoQA"            # vqa only
export CKPT_DIR_RAW="/path/to/Alpamayo-1.5-10B"  # raw HF download target
export CKPT_DIR_A1="${CKPT_DIR_RAW}-A1-format"   # converted (consumed by training)
export USE_WANDB="no"                            # yes | no
export WANDB_API_KEY=""                          # only if USE_WANDB=yes
export WANDB_TEAM="..."                          # only if USE_WANDB=yes
export WANDB_PROJECT="..."                       # only if USE_WANDB=yes
export STAGE1_CKPT=""                            # set if STAGES in {stage2, eval}
export STAGE2_CKPT=""                            # set if STAGES = eval
```

After this, do not ask the user further questions — run the rest of the
pipeline end-to-end. Mid-run halt conditions: hard-stop errors (CUDA OOM,
missing files, hydra errors), or the user explicitly interrupting.

---

## Mental model — what gets trained, in what order

```
   Released HF ckpt (nvidia/Alpamayo-1.5-10B)
        │
        │  huggingface-cli download
        ▼
   $CKPT_DIR_RAW
        │
        │  scripts/convert_checkpoint.py to-a1
        ▼
   $CKPT_DIR_A1  (consumed by training; A1-format `_target_` paths)
        │
        ▼
                  ┌──────────────────────────────────────────────┐
                  │              Stage 1 — VLM SFT               │
                  │  one entry script, two Hydra configs:        │
                  │   • sft_stage1_nav      (PAIDatasetWithNav)  │
                  │   • sft_stage1_lingoqa  (LingoQADataset)     │
                  │  ZeRO-2 + grad-ckpt, visual @ 0.1× LR        │
                  └──────────────────────────────────────────────┘
                                      │
                                      ▼
                  output_stage1_<task>/checkpoint-XXXX/
                                      │
                                      ▼  (trajectory tasks only)
                  ┌──────────────────────────────────────────────┐
                  │      Stage 2 — Action expert (frozen VLM)    │
                  │  config: sft_stage2_nav.yaml                 │
                  │  no DeepSpeed, no grad-ckpt,                 │
                  │  ddp_find_unused_parameters=true             │
                  └──────────────────────────────────────────────┘
                                      │
                                      ▼
                  output_stage2/checkpoint-XXXX/
                                      │
                                      ▼
                  evaluate_hf.py → val/metric/min_ade, ade, corner_distance
```

Key facts an agent must internalise before running anything:

- **Same entry point for every Stage-1 variant.** `alpamayo1_5_sft.train_hf`
  reads `--config-name sft_stage1_{nav,lingoqa}` and the differences are
  purely in the YAML (dataset class, vla_processor variant, output dir,
  hyperparams).
- **Convert before training.** You cannot point Stage 1 at the raw HF
  `Alpamayo-1.5-10B` dir — its `_target_` paths use `alpamayo1_5.*`. Run
  `scripts/convert_checkpoint.py to-a1` once to remap them to
  `alpamayo_r1.*`. Output of that script is what training reads.
- **Stage 1 ≠ Stage 2 model class.** Stage 1 instantiates
  `TrainableReasoningVLA.from_alpamayo_checkpoint` (loads VLM weights from
  the converted A1-format directory). Stage 2 instantiates
  `TrainableAlpamayoR1.from_pretrained` and additionally takes the
  Stage-1 trainer output as `stage1_vlm_checkpoint_path`.
- **Stage 2 only applies to the trajectory task** (`nav`). It does not
  run on a VQA Stage-1 checkpoint — the action expert is
  trajectory-specific.
- **DeepSpeed only in Stage 1.** Stage 2 turns DeepSpeed off
  (`trainer.deepspeed: null`) and enables `ddp_find_unused_parameters=true`.
- **W&B is OFF by default.** Same as Alpamayo-1 SFT, the shipped
  [configs/sft_base.yaml](configs/sft_base.yaml) has the `wandb` default
  commented out and `trainer.report_to: none`. To enable, uncomment
  `- /wandb: default` in `sft_base.yaml`, set `trainer.report_to: wandb`,
  fill in `team` + `project` in
  [configs/wandb/default.yaml](configs/wandb/default.yaml), and export
  `WANDB_API_KEY`.
- **The shipped task configs have absolute hardcoded paths** for the
  author's workstation (e.g. `local_dir: /home/yesfandiari/sample_data_1_5/`
  in [sft_stage1_nav.yaml](configs/sft_stage1_nav.yaml),
  `data_root: /home/yesfandiari/lingo` in
  [sft_stage1_lingoqa.yaml](configs/sft_stage1_lingoqa.yaml),
  `checkpoint_path: /home/yesfandiari/model_1/` in
  [configs/models/ar1_5_base.yaml](configs/models/ar1_5_base.yaml)).
  **You must override** all three on the CLI (or edit the YAMLs before
  launching).

---

## Install — `a1_5_sft` venv

Single uv venv at `recipes/alpamayo1_5_sft/a1_5_sft/`. uv handles
everything; **do not** create a separate conda env.

```bash
# 1) Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2) Provision the venv
cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft"

uv venv a1_5_sft         # if `a1_5_sft/` already exists, skip this line
source a1_5_sft/bin/activate
uv sync --active
```

> **Non-interactive harnesses (agents, CI, `bash -c "..."` per step):**
> the `export PATH=...` above only persists for the current shell. Either
> append the line to `~/.bashrc` once, or re-export it at the top of every
> non-interactive shell that calls `uv`. Forgetting this is the most
> common reason a subsequent step reports `uv: command not found`.

After every `uv sync`, verify the import contract the recipe assumes:

```bash
python -c "from alpamayo.processor.qwen_processor import \
  get_preprocess_data_fn_from_model_config, collate_fn_from_model_config; print('ok')"
python -c "from alpamayo.data.pai_nav import PAIDatasetWithNav; print('ok')"   # nav
python -c "from alpamayo.data.lingoqa import LingoQADataset; print('ok')"      # vqa
```

---

## Checkpoint — download Alpamayo-1.5 and convert to A1 format

Run this **once** per fresh setup.

```bash
# 1) Download the release ckpt (~21 GB)
export HF_TOKEN=<your HuggingFace token>
# (Or, in non-interactive shells: export HF_TOKEN=$(<~/.cache/huggingface/token))
huggingface-cli download nvidia/Alpamayo-1.5-10B --local-dir "$CKPT_DIR_RAW"

# 2) Remap _target_ paths from alpamayo1_5.* → alpamayo_r1.* so the
#    training entry can load it
cd "$YOUR_HOME/alpamayo-recipes"
python scripts/convert_checkpoint.py to-a1 \
  --input  "$CKPT_DIR_RAW" \
  --output "$CKPT_DIR_A1"
```

Sanity check:

```bash
ls "$CKPT_DIR_A1"
# expect: config.json (with alpamayo_r1.* targets),
#         model.safetensors.index.json, model-0000{1..5}-of-00005.safetensors,
#         tokenizer files, preprocessor_config.json
```

Then either edit
[configs/models/ar1_5_base.yaml](configs/models/ar1_5_base.yaml) to set
`checkpoint_path: "$CKPT_DIR_A1"`, or pass
`model.checkpoint_path="$CKPT_DIR_A1"` on every Stage-1 launch.

---

## Dataset — PAI (nav / trajectory) or LingoQA (VQA)

### PAI — `nav` task

```bash
export HF_TOKEN=<your HuggingFace token>
cd "$YOUR_HOME/alpamayo-recipes"

# Download exactly the chunks the bundled nav annotations reference.
# IMPORTANT: --chunk-ids takes ONE string argument (the script splits it
# internally); always wrap $PAI_CHUNK_IDS in double quotes so bash doesn't
# word-split the space-separated list into multiple argparse tokens.
python scripts/download_pai.py \
  --chunk-ids "$PAI_CHUNK_IDS" \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir "$PAI_DIR"
```

### Nav annotations — `nav` only

The bundled `nav_demo_samples.json` lives in the Alpamayo-1.5 repo at:
<https://github.com/NVlabs/alpamayo1.5/blob/main/notebooks/nav_demo_samples.json>

Each entry has `clip_id`, `t0_relative`, `nav_text`, and optionally `cot`.
Save the file locally and remember the path as `$NAV_ANNOTATIONS` — it
goes into `data.{train,val}_dataset.annotations_path`.

There are 20 samples in the bundled JSON. They're a smoke-test set, **not**
training data for the released Alpamayo 1.5. Loss should drop to near 0
within hundreds of steps (overfit) — that's the wiring check.

### LingoQA — `vqa` only

Wayve distributes LingoQA via Google Drive. Use `gdown`:

```bash
source "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft/a1_5_sft/bin/activate"
uv pip install gdown

mkdir -p "$LINGOQA_DIR"
gdown --folder https://drive.google.com/drive/folders/1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7 \
      -O "$LINGOQA_DIR"
cd "$LINGOQA_DIR" && unzip images.zip
```

Expected layout the loader assumes:

```
$LINGOQA_DIR/
├── train.parquet      # 148k QA pairs, ~3.5k video segments
└── images/train/      # front-camera JPEGs (5 frames per segment)
```

The loader uses **Scenery** only. The evaluation split lives separately;
the shipped `sft_stage1_lingoqa.yaml` references `val.parquet` for
validation — if you only downloaded train, set
`data.val_dataset.parquet_name=train.parquet` (overlapping val is fine
for the loss-shape smoke).

---

## VLA processor variants

[configs/vla_processor/](configs/vla_processor/) ships three variants; each
Stage-1 config selects one via a Hydra `defaults` `override`.

| Variant   | Components                                      | Supervised on | Selected by Stage-1 config |
|-----------|-------------------------------------------------|---------------|-----------------------------|
| `default` | image, traj_history, prompt, traj_future        | traj_future   | *(not selected by any shipped Stage-1 or Stage-2 config — `sft_stage2_nav.yaml` overrides to the `nav` variant; this row is reference-only)* |
| `nav`     | image, traj_history, **route**, prompt, traj_future | traj_future | `sft_stage1_nav.yaml`       |
| `vqa`     | image, **question, answer**                     | answer        | `sft_stage1_lingoqa.yaml`   |

The Stage-1 task you chose in [Inputs](#inputs-to-collect-from-the-user-ask-once-up-front)
determines which one is used; an agent normally doesn't edit these
directly. If you want to combine route conditioning with VQA, you'd
write a third Stage-1 YAML and the matching `vla_processor` variant.

---

## Stage 1 — Nav-conditioned trajectory

Run this when `$TASK = nav` and `$STAGES` is `stage1` or `both`.

### Canonical launch

```bash
cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft"
torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_nav \
  model.checkpoint_path="$CKPT_DIR_A1" \
  data.train_dataset.local_dir="$PAI_DIR" \
  data.train_dataset.annotations_path="$NAV_ANNOTATIONS" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.annotations_path="$NAV_ANNOTATIONS"
```

Outputs land in `output_stage1_nav/`. The shipped config sets
`num_train_epochs: 100000` so the 20-sample overfit can be run as long as
you like — kill it once loss is near zero.

### What "healthy" Stage-1 nav logs look like

Loss should drop sharply toward zero within hundreds of steps (overfit on
20 samples). Reference plot: [loss_A1-5_nav.png](loss_A1-5_nav.png).
This 20-sample set is **not** released-model training data — the curve is
a wiring confirmation, not a fine-tuning gain signal.

### Warnings that are noise

Same family as Alpamayo-1 SFT — flash-attn "only supports float16 and
bfloat16" (bfloat16 *is* supported; misreported warning), `use_cache=True
is incompatible with gradient checkpointing` (transformers auto-disables
KV cache), DeepSpeed-vs-Accelerate grad-accum mismatch (DeepSpeed wins).
All cosmetic; ignore.

---

## Stage 1 — LingoQA VQA

Run this when `$TASK = vqa` and `$STAGES = stage1` (Stage 2 doesn't apply
to VQA).

### Canonical launch

```bash
cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft"
torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage1_lingoqa \
  model.checkpoint_path="$CKPT_DIR_A1" \
  data.train_dataset.data_root="$LINGOQA_DIR" \
  data.val_dataset.data_root="$LINGOQA_DIR"
```

Outputs land in `output_stage1_lingoqa/`.

### What "healthy" Stage-1 LingoQA logs look like

LingoQA was included in the released Alpamayo-1.5 training mix, so loss
stays **low and stable** rather than dropping dramatically. That's the
expected shape; don't interpret flat loss as "broken". Reference plot:
[loss_A1-5_lingoqa.png](loss_A1-5_lingoqa.png).

### Stage 2 is not applicable

VQA Stage-1 outputs are not consumed by the action expert — there is no
trajectory target to diffuse. If you want a model that does both VQA and
trajectory, run VQA Stage 1 first, then run trajectory Stage 1 from that
checkpoint (set `model.checkpoint_path` to the VQA output), then Stage 2.

---

## Stage 2 — Action expert (trajectory diffusion)

Run this when `$STAGES` is `stage2` or `both` and `$TASK ≠ vqa`. If
`$STAGES = both`, this fires after Stage 1; if `$STAGES = stage2`, the
user has provided `$STAGE1_CKPT` as input.

```bash
# If $STAGES = both, pick the highest-numbered Stage-1 output:
if [ "$STAGES" = "both" ]; then
  if [ "$TASK" != "nav" ]; then
    echo "Stage 2 only applies to task=nav" >&2; exit 1
  fi
  STAGE1_CKPT=$(ls -d output_stage1_nav/checkpoint-* | sort -V | tail -1)
fi

cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft"
torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  model.pretrained_model_name_or_path="$CKPT_DIR_A1" \
  model.stage1_vlm_checkpoint_path="$STAGE1_CKPT" \
  data.train_dataset.local_dir="$PAI_DIR" \
  data.train_dataset.annotations_path="$NAV_ANNOTATIONS" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.annotations_path="$NAV_ANNOTATIONS"
```

> **`chunk_ids` is already baked into `sft_stage2_nav.yaml`** as the same
> 19-chunk nav list used in Stage 1 — no override needed. If you only have
> a subset on disk, narrow the list in the YAML or override on the CLI to
> match what you actually downloaded.
> annotations reference (e.g. 214, 224, 276, …) — Stage 2 will look for
> chunk 0 and crash with `FileNotFoundError: .../labels/egomotion/egomotion.chunk_0008.zip`.
> Pass `data.{train,val}_dataset.chunk_ids="<X>-<X+1>"` for any chunk you
> actually have on disk (e.g. `"214-215"` for chunk 214 only).

Required parameter contract:

- **`model.pretrained_model_name_or_path`** — the **same converted
  A1-format folder** used for Stage 1 (`$CKPT_DIR_A1`). Do **not** point
  at the Stage-1 output here; this loads the action-expert structure +
  processor, not the VLM weights.
- **`model.stage1_vlm_checkpoint_path`** — the Stage-1 Trainer output dir
  (must contain `model.safetensors.index.json` + shards). This is the
  directory whose VLM weights overwrite the
  `pretrained_model_name_or_path` VLM weights at load time.

Outputs land in `output_stage2/`.

> **Smoke testing Stage 2 without a real Stage-1 run.** On
> memory-constrained hardware where Stage 1 OOMs, you can still exercise
> Stage 2 wiring by pointing `stage1_vlm_checkpoint_path` at the **same
> directory** as `pretrained_model_name_or_path` (the A1-format base).
> The base checkpoint already satisfies the
> `model.safetensors.index.json` + shards contract, so the load path
> runs identically — you just won't get the Stage-1 fine-tuning gains.
> Use this only for pipeline validation.

---

## Evaluation

Run this when `$STAGES` is `eval` or `both`. If `$STAGES = both`, this
fires after Stage 2 finishes; if `$STAGES = eval`, the user has provided
`$STAGE2_CKPT`.

```bash
# If $STAGES = both, pick the highest-numbered Stage-2 output:
if [ "$STAGES" = "both" ]; then
  STAGE2_CKPT=$(ls -d output_stage2/checkpoint-* | sort -V | tail -1)
fi

cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_sft"
torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.evaluate_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_stage2_nav \
  evaluate.eval_ckpt="$STAGE2_CKPT" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.annotations_path="$NAV_ANNOTATIONS"
```

[evaluate_hf.py](evaluate_hf.py) auto-detects the model class
(`TrainableReasoningVLA` vs `TrainableAlpamayoR1`) and routes
`eval_ckpt` into the correct field. Metric set is the same as
Alpamayo-1: `val/metric/ade`, `val/metric/min_ade`,
`val/metric/corner_distance`, and `by_t=` breakdowns. Reference numbers
live in [`recipes/alpamayo1_sft/SKILL.md`](../alpamayo1_sft/SKILL.md);
expect Alpamayo-1.5 results to be in the same magnitude band on a
trajectory split.

---

## Logging — W&B, TensorBoard, offline

[configs/sft_base.yaml](configs/sft_base.yaml) ships with the `wandb`
default commented out and `trainer.report_to: none`, so unless the user
opts in nothing phones home.

### If `$USE_WANDB = no` (the default)

Do nothing — the shipped config is already safe. For belt-and-braces
(e.g. machines where someone has previously run `wandb login`), prefix
launches with:

```bash
WANDB_MODE=disabled torchrun ...
```

### If `$USE_WANDB = yes`

1. Uncomment `- /wandb: default` in
   [configs/sft_base.yaml:2](configs/sft_base.yaml#L2).
2. Set `trainer.report_to: wandb` in the same file.
3. Fill in `team` and `project` in
   [configs/wandb/default.yaml](configs/wandb/default.yaml).
4. `export WANDB_API_KEY="..."` before launch.

CLI-only alternative (no YAML edits):

```bash
export WANDB_API_KEY="$WANDB_API_KEY"
torchrun ... \
  trainer.report_to=wandb \
  +wandb.team="$WANDB_TEAM" \
  +wandb.project="$WANDB_PROJECT"
```

The API key must belong to an account with write access to
`team`/`project` — otherwise wandb fails at `wandb.init()` with a 403
(see [pitfalls](#common-failure-modes-and-the-fix)). Fail fast and
re-prompt the user; don't retry blindly.

---

## Multi-GPU, multi-node, and cluster runs

Defaults assume **single-node, 8× H100 80 GB**. The Alpamayo-1 SFT
SKILL's multi-GPU section applies verbatim here — same `torchrun`
patterns, same OSMO / SLURM caveats. See
[`recipes/alpamayo1_sft/SKILL.md` § Multi-GPU](../alpamayo1_sft/SKILL.md#multi-gpu-multi-node-and-cluster-runs).

Cluster gotchas worth repeating:

- **Same paths on every node.** PAI dataset, LingoQA dataset, nav
  annotations JSON, A1-format checkpoint, and the venv must all be
  visible at identical absolute paths on every rank.
- **Flash-attn builds from source per node.** First run on a new node
  takes 5–10 min before training starts.

---

## Hydra override cheatsheet

All overrides go on the `torchrun` command line **without** a leading `--`.

| Override | What it does |
|----------|--------------|
| `model.checkpoint_path=<path>` | **Stage 1** A1-format base checkpoint dir |
| `model.pretrained_model_name_or_path=<path>` | **Stage 2** base ckpt dir (same as `$CKPT_DIR_A1`) |
| `model.stage1_vlm_checkpoint_path=<path>` | **Stage 2** Stage-1 trainer output |
| `data.train_dataset.local_dir=<path>` | PAI dataset root (nav / default) |
| `data.val_dataset.local_dir=<path>` | PAI val root (nav / default) |
| `data.train_dataset.annotations_path=<path>` | Nav annotations JSON (nav only) |
| `data.train_dataset.data_root=<path>` | LingoQA root (vqa only) |
| `data.train_dataset.parquet_name=<file>` | LingoQA parquet filename (default `train.parquet`) |
| `data.train_dataset.chunk_ids="..."` | PAI chunk range / list |
| `trainer.learning_rate=1e-5` | LR override |
| `trainer.num_train_epochs=1` | Epoch cap (nav config defaults to 100000) |
| `trainer.per_device_train_batch_size=1` | Per-rank batch size |
| `trainer.gradient_accumulation_steps=8` | Grad-accum (Stage 1 configs default to 4) |
| `trainer.save_steps=200 trainer.save_total_limit=5` | Checkpoint cadence |
| `trainer.report_to=none` | Switch off W&B/HF Hub logging without editing files |
| `paths.output_dir=output_smoke` | Re-route outputs (smoke tests) |
| `evaluate.eval_ckpt=<path>` | Checkpoint for [evaluate_hf.py](evaluate_hf.py) |
| `evaluate.max_eval_steps=10` | Cap eval (smoke test) |
| `+wandb.team=<team>` / `+wandb.project=<project>` | Override wandb routing without editing `configs/wandb/default.yaml` (note the `+` — these keys can be appended on the CLI) |

For a one-shot **smoke test** (single epoch, low save), wire any of the
three Stage-1 configs with a stripped-down output dir. Example for nav:

```bash
torchrun --nproc_per_node 8 -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs --config-name sft_stage1_nav \
  model.checkpoint_path="$CKPT_DIR_A1" \
  data.train_dataset.local_dir="$PAI_DIR" \
  data.train_dataset.annotations_path="$NAV_ANNOTATIONS" \
  data.val_dataset.local_dir="$PAI_DIR" \
  data.val_dataset.annotations_path="$NAV_ANNOTATIONS" \
  trainer.num_train_epochs=1 trainer.save_steps=50 trainer.save_total_limit=1 \
  trainer.report_to=none paths.output_dir=output_smoke_nav
```

If this emits non-NaN loss in the first few optimizer steps without a
Hydra `InstantiationException`, the wiring is good.

---

## Common failure modes (and the fix)

Run **once** with `HYDRA_FULL_ERROR=1` to expose the chained traceback —
without it, `InstantiationException` hides the real cause behind a
one-liner.

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `Hydra` complains `MISSING ???` for `data.val_dataset.local_dir` / `annotations_path` | Forgot to override the val-side path (nav config has `???` placeholders on val) | Set `data.val_dataset.local_dir` and (for nav) `data.val_dataset.annotations_path` — easiest is to reuse the train paths |
| `InstantiationException` on `alpamayo.data.pai_nav.PAIDatasetWithNav` or `alpamayo.data.lingoqa.LingoQADataset` | The shared `alpamayo` editable install didn't take | `uv pip show alpamayo-recipes` → re-run `uv sync --active` |
| Hydra fails loading `model.checkpoint_path=…` with "target not found" / config-class errors | Pointed Stage 1 at the **raw** `Alpamayo-1.5-10B` HF dir instead of the converted A1-format one | Run `scripts/convert_checkpoint.py to-a1 --input <raw> --output <a1-format>` first; point `model.checkpoint_path` at the converted dir |
| Stage 1 launches but immediately `FileNotFoundError: /home/yesfandiari/...` | The shipped task configs have absolute hardcoded paths for the recipe author's machine | Override `model.checkpoint_path`, `data.{train,val}_dataset.local_dir`, `data.{train,val}_dataset.annotations_path` (nav) or `data_root` (vqa) on the CLI — see launch examples above |
| Stage 2 launches but `FileNotFoundError: .../labels/egomotion/egomotion.chunk_0008.zip` (or any chunk you didn't download) | Stage 2 inherits `chunk_ids: "0-99"` from `sft_base.yaml`. If you only downloaded the nav chunks (or any subset), the dataloader iterates `0..98` and hits the first missing one | Override `data.{train,val}_dataset.chunk_ids="<X>-<X+1>"` to a chunk you have on disk. For the nav recipe, any single chunk from the 19-chunk nav list (e.g. `"214-215"`) is enough |
| `wandb.errors.CommError: 403 … upsertBucket … permission denied` | The shipped wandb config points at a team/project you don't own, and `report_to: wandb` is on by default in this recipe | Set `+wandb.team=<yours> +wandb.project=<yours>` on the CLI, or pass `trainer.report_to=none WANDB_MODE=disabled` for tests, or edit `configs/wandb/default.yaml` |
| `Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes …` repeated per submodule | Spurious transformers warning misreading the current dtype | **Ignore** — bfloat16 is supported; trainer is configured correctly |
| `use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False.` | Stage 1 grad-ckpt is on; transformers auto-disables KV cache | Ignore |
| `Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 4` | Accelerate plugin default vs. Stage-1 DeepSpeed setting; DeepSpeed wins | Cosmetic; ignore |
| LingoQA loader complains about missing `images/train/...` files | Forgot to `unzip images.zip` after `gdown` | `cd "$LINGOQA_DIR" && unzip images.zip`; verify with `ls "$LINGOQA_DIR/images/train" \| head` |
| OOM during Stage 1 | Effective batch too large for 80 GB | Lower `trainer.per_device_train_batch_size` (already 1) → raise `gradient_accumulation_steps`. Don't disable grad-ckpt in Stage 1 |
| OOM during Stage 2 | Stage 2 doesn't use DeepSpeed or grad-ckpt by design | Reduce `per_device_train_batch_size`, freeze more submodules (`cotrain_vlm: false` is the default), or move to >8 GPUs |
| Stage 2 launches but loss is NaN immediately | `model.pretrained_model_name_or_path` pointed at the Stage-1 output instead of the A1-format base | `pretrained_model_name_or_path` = base (`$CKPT_DIR_A1`); `stage1_vlm_checkpoint_path` = trainer output. Don't swap them |
| Attempting Stage 2 after a VQA Stage 1 errors / produces nonsense | Stage 2 is trajectory-only; VQA Stage-1 has no trajectory targets in the expected shape | Skip Stage 2 for `vqa` — see [Stage 2 — Action expert](#stage-2--action-expert-trajectory-diffusion) |

---

## Additional resources

- Recipe README (concise human-facing version): [README.md](README.md)
- Stage 1 model code: [models/sft_base_model.py](models/sft_base_model.py)
- Stage 2 model code: [models/sft_alpamayo_r1.py](models/sft_alpamayo_r1.py)
- Custom HF trainer subclass: [trainer.py](trainer.py)
- Training entry point: [train_hf.py](train_hf.py)
- Evaluation entry point: [evaluate_hf.py](evaluate_hf.py)
- Base config (shared): [configs/sft_base.yaml](configs/sft_base.yaml)
- Stage 1 task configs:
  [configs/sft_stage1_nav.yaml](configs/sft_stage1_nav.yaml) (nav),
  [configs/sft_stage1_lingoqa.yaml](configs/sft_stage1_lingoqa.yaml) (vqa)
- Stage 2 config: [configs/sft_stage2_nav.yaml](configs/sft_stage2_nav.yaml)
- Model targets:
  [configs/models/ar1_5_base.yaml](configs/models/ar1_5_base.yaml),
  [configs/models/ar1_5_expert.yaml](configs/models/ar1_5_expert.yaml)
- VLA processor variants:
  [configs/vla_processor/default.yaml](configs/vla_processor/default.yaml),
  [configs/vla_processor/nav.yaml](configs/vla_processor/nav.yaml),
  [configs/vla_processor/vqa.yaml](configs/vla_processor/vqa.yaml)
- W&B defaults: [configs/wandb/default.yaml](configs/wandb/default.yaml)
- DeepSpeed ZeRO-2 (Stage 1 only): [configs/deepspeed/zero2.json](configs/deepspeed/zero2.json)
- PAI download / curation helpers (run from repo root):
  `../../scripts/download_pai.py`, `../../scripts/curate_pai_samples.py`
- Checkpoint conversion: `../../scripts/convert_checkpoint.py` (`to-a1` /
  `to-a15` subcommands)
- Shared dataset code:
  [`../../src/alpamayo/data/pai.py`](../../src/alpamayo/data/pai.py),
  [`../../src/alpamayo/data/pai_nav.py`](../../src/alpamayo/data/pai_nav.py),
  [`../../src/alpamayo/data/lingoqa.py`](../../src/alpamayo/data/lingoqa.py)
- Nav annotations (download out-of-band):
  <https://github.com/NVlabs/alpamayo1.5/blob/main/notebooks/nav_demo_samples.json>
- LingoQA Scenery split (Google Drive, via gdown):
  <https://drive.google.com/drive/folders/1GiwWGfrM8pO27CYLu_9Uwtdcz0JoqHr7>
- Pretrained Alpamayo-1.5-10B: <https://huggingface.co/nvidia/Alpamayo-1.5-10B>
- Upstream `alpamayo_r1`: <https://github.com/NVlabs/alpamayo>
- Alpamayo-1.5 repo (release-side code, nav-demo notebook):
  <https://github.com/NVlabs/alpamayo1.5>
- Companion skills:
  [`recipes/alpamayo1_sft/SKILL.md`](../alpamayo1_sft/SKILL.md) (Alpamayo-1 SFT),
  [`recipes/alpamayo1_x_rl/SKILL.md`](../alpamayo1_x_rl/SKILL.md) (RL post-training)
