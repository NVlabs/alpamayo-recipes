# Alpamayo 1.5 Single-Clip minADE Evaluation

This recipe evaluates trajectory prediction accuracy for [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) on a single [PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle) clip using **minimum Average Displacement Error (minADE)** in the BEV (XY) plane.

The evaluation script lives under [`recipes/alpamayo1_5_eval/`](./).

## What This Recipe Does

1. Loads ego-motion history and future ground-truth trajectory from a single PhysicalAI-AV clip.
2. Decodes four camera streams (front-left, front-wide, front-right, front-tele) for the requested timestamp.
3. Runs Alpamayo 1.5 VLM-rollout inference to produce K trajectory candidates.
4. Computes minADE against the ground-truth future ego trajectory using the shared [`alpamayo.metrics`](../../src/alpamayo/metrics/distance_metrics.py) module.
5. Prints per-sample ADE, the chain-of-thought reasoning, and optionally saves a JSON result file.

### Supported Models

| Model | HuggingFace ID |
|-------|----------------|
| Alpamayo 1.5 (10B) | `nvidia/Alpamayo-1.5-10B` |

## Hardware Requirements

- A single GPU with at least **40 GB** VRAM (e.g. A100 or H100) is recommended for BF16 inference.
- CPU inference is supported but will be slow.

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2. Clone and install

```bash
export YOUR_HOME="/path/to/your/workspace"

git clone https://github.com/NVlabs/alpamayo-recipes.git $YOUR_HOME/alpamayo-recipes
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_eval
uv venv a1_5_eval
source a1_5_eval/bin/activate
uv sync --active
```

`alpamayo_r1` (model code, action space, geometry) is fetched automatically from
[NVlabs/alpamayo](https://github.com/NVlabs/alpamayo.git) during `uv sync`.

### Optional: flash-attn

For faster attention (strongly recommended on A100/H100):

```bash
uv pip install flash-attn --no-build-isolation
```

Then pass `--attn_implementation flash_attention_2` when running the script.

## Prepare Dataset

The recipe uses `physical_ai_av` to stream data from the
[PhysicalAI-AV HuggingFace dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle).
Set your HuggingFace token before running:

```bash
export HF_TOKEN=<your HuggingFace token>
```

By default the script streams data on demand (`--no_stream` is not set).
For offline use, first download the relevant chunk with:

```bash
cd $YOUR_HOME/alpamayo-recipes
python scripts/download_pai.py \
  --chunk-ids "<chunk_id>" \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
            camera_cross_right_120fov camera_front_tele_30fov \
  --labels egomotion \
  --output-dir /path/to/pai_dataset
```

Then pass `--no_stream` to the evaluation script and point `physical_ai_av` at the local directory.

## Prepare Model Checkpoint

Download the released Alpamayo 1.5 weights:

```bash
huggingface-cli download nvidia/Alpamayo-1.5-10B --local-dir /path/to/Alpamayo-1.5-10B
```

### Optional: permanent checkpoint conversion (recommended for repeated use)

The released checkpoint uses `model_type: "alpamayo1_5"` in its `config.json`.
When you pass it directly, the recipe auto-converts the config to the
`alpamayo_r1` format in a temporary directory on every run (weights are
symlinked, so no data is copied).

For repeated evaluation we recommend performing a one-time permanent
conversion instead, which avoids the per-run overhead:

```bash
cd $YOUR_HOME/alpamayo-recipes
python scripts/convert_checkpoint.py to-a1 \
  --input /path/to/Alpamayo-1.5-10B \
  --output /path/to/Alpamayo-1.5-10B-A1-format
```

The converted directory only contains a new `config.json` plus symlinks to
the original weight files — no data is duplicated.  Pass the converted path
as `--model_name` to skip the auto-conversion step entirely.

## Run Evaluation

Clips are evaluated **sequentially** on a single GPU.  The model is loaded once and reused across all clips, so model-loading overhead is paid only once regardless of how many clips you evaluate.

### Single clip (default example)

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_eval
python -m alpamayo1_5_eval.evaluate_single_clip \
  --clip_id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
  --t0_us 5100000 \
  --model_name /path/to/Alpamayo-1.5-10B \
  --output outputs/minade_results.json
```

### Multiple clip IDs inline

Pass multiple UUIDs to `--clip_id` and either a single `--t0_us` (broadcast to all clips)
or one timestamp per clip:

```bash
python -m alpamayo1_5_eval.evaluate_single_clip \
  --clip_id CLIP_A CLIP_B CLIP_C \
  --t0_us 5100000 \
  --model_name /path/to/Alpamayo-1.5-10B \
  --num_traj_samples 4 \
  --output outputs/minade_results.json
```

### Annotations file

For larger evaluations, prepare a JSON annotations file and pass it via `--annotations`.
The format matches the nav annotations used by [alpamayo1_5_sft](../alpamayo1_5_sft/):

```json
[
  {"clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26", "t0_us": 5100000},
  {"clip_id": "1a2b3c4d-...", "t0_us": 6200000, "nav_text": "Turn left in 40m"}
]
```

Extra fields (e.g. `"nav_text"`, `"cot"`) are passed through to the output unchanged.

```bash
python -m alpamayo1_5_eval.evaluate_single_clip \
  --annotations /path/to/clips.json \
  --model_name /path/to/Alpamayo-1.5-10B \
  --attn_implementation flash_attention_2 \
  --num_traj_samples 4 \
  --output outputs/minade_results.json
```

### All CLI options

**Clip specification** (`--annotations` takes priority over `--clip_id`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--annotations` | `None` | Path to a JSON clip-list file |
| `--clip_id` | built-in example | One or more clip UUIDs (`nargs="+"`) |
| `--t0_us` | `5100000` | One timestamp or one per clip (`nargs="+"`) |

**Model**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `nvidia/Alpamayo-1.5-10B` | HuggingFace ID or local path |
| `--vlm_name_or_path` | `None` | HuggingFace ID or local path for the VLM backbone processor. When `None`, read from the model's `config.vlm_name_or_path`. Override when the config points to a HF ID but network is unavailable. |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--dtype` | `bfloat16` | `bfloat16`, `float16`, or `float32` |
| `--attn_implementation` | `None` | `flash_attention_2`, `sdpa`, or `eager` |

**Inference**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_traj_samples` | `1` | Trajectory candidates per clip (K) |
| `--top_p` | `0.98` | Nucleus sampling probability |
| `--temperature` | `0.6` | Sampling temperature |
| `--max_generation_length` | `256` | Max tokens to generate |
| `--seed` | `42` | CUDA random seed |

**Dataset / output**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_stream` | `False` | Disable streaming; requires local download |
| `--dataset_local_dir` | `None` | Path to local PhysicalAI-AV dataset copy |
| `--dataset_revision` | `None` | HF revision to skip `list_repo_refs` network call |
| `--output` | `outputs/minade_results.json` | JSON result output path |

### Using a local dataset copy

If you have a local copy of the PhysicalAI-AV dataset (e.g. the 5 % evaluation
subset), pass `--dataset_local_dir` to read files from disk instead of
downloading them.  Also pass `--dataset_revision main` to avoid the
`list_repo_refs` network round-trip that normally resolves the default branch
commit hash:

```bash
python -m alpamayo1_5_eval.evaluate_single_clip \
  --clip_id 82acb8e5-abcf-4ea0-aa1a-7f18f6c79ff2 \
  --t0_us 5100000 \
  --model_name /path/to/Alpamayo-1.5-10B \
  --dataset_local_dir /path/to/PhysicalAI-AV-5pct \
  --dataset_revision main \
  --num_traj_samples 4
```

Clip IDs and `t0_us` values for the 5 % evaluation subset are listed in
`eval_samples_ood_val_5pct.csv` inside the dataset directory.

### Fully offline evaluation

When running without internet access, set the following environment variables
to prevent `transformers` and `datasets` from attempting HuggingFace network
calls even when loading from local paths:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

The model's `config.json` also contains a `vlm_name_or_path` field pointing to
the VLM backbone (e.g. `Qwen/Qwen3-VL-8B-Instruct`).  Pass its local directory
via `--vlm_name_or_path` to avoid any remaining network requests.

One-time setup:

```bash
# Download only the processor / tokenizer files from the VLM backbone
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --include "*.json" "*.tiktoken" "*.model" \
  --local-dir /path/to/Qwen3-VL-8B-Instruct
```

Then evaluate fully offline:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m alpamayo1_5_eval.evaluate_single_clip \
  --clip_id 82acb8e5-abcf-4ea0-aa1a-7f18f6c79ff2 \
  --t0_us 5100000 \
  --model_name /path/to/Alpamayo-1.5-10B \
  --vlm_name_or_path /path/to/Qwen3-VL-8B-Instruct \
  --dataset_local_dir /path/to/PhysicalAI-AV-5pct \
  --dataset_revision main \
  --num_traj_samples 4
```

## Expected Output

Console (single clip):

```
[1/3] Loading model (shared across 1 clip(s))...
[2/3] Evaluating 1 clip(s)...

--- Clip 1/1 ---
  clip_id:  030c760c-ae38-49aa-9ad8-f5650a545d26
  t0_us:    5100000
  minADE:   0.373767 m
  all_ADE:  [0.373767]

  ========== Predicted Trajectory ==========
  ...

========== Aggregate Summary ==========
total_clips:         1
mean_minADE@6.4s:    0.373767 m
```

Result JSON (top-level has `"summary"` and per-clip `"results"` list):

```json
{
  "summary": {
    "total_clips": 1,
    "mean_minADE@6.4s": 0.373767
  },
  "results": [
    {
      "clip_id": "030c760c-ae38-49aa-9ad8-f5650a545d26",
      "t0_us": 5100000,
      "num_traj_samples": 4,
      "minADE": 0.373767,
      "all_ADE": [0.373767, 0.512345, 0.698765, 1.123456],
      "cot": [["Nudge to the left ..."]]
    }
  ]
}
```

## Validation

After installation, verify the recipe compiles cleanly:

```bash
cd $YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_eval
python -m compileall .
```

## Known Limitations

- Clips are evaluated **sequentially** on a single GPU; there is no cross-clip batch inference. For dataset-scale evaluation, use the `evaluate_hf.py` script in [alpamayo1_5_sft](../alpamayo1_5_sft/).
- Each clip is evaluated at a **single timestamp** (`t0_us`). Multi-timestamp evaluation per clip is not supported in this recipe.
- Streaming requires a valid `HF_TOKEN` and internet access at evaluation time.  For offline evaluation set `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` and use `--dataset_local_dir` / `--vlm_name_or_path`.
- The chain-of-thought output is not guaranteed to be deterministic across different hardware configurations even with a fixed seed.
