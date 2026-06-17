# FP8 / AutoQuant (FP8 + NVFP4) Quantization

This Recipe defines a reproducible post-training quantization (PTQ) procedure for quantizing Alpamayo 1.5 to FP8 or AutoQuant (FP8 + NVFP4)

## Prerequisites

This recipe is tested in the following settings.

- NVIDIA B300 GPU with CUDA 13
- Python 3.12
- Python Libraries: torch==2.8.0, torchvision==0.27.0, nvidia-modelopt==0.43.0

**NVIDIA Model Optimizer (ModelOpt)** is a library comprising state-of-the-art model optimization techniques including quantization and sparsity to compress models. In this recipe, we utilize ModelOpt to quantize Alpamayo 1.5.

## Table of contents

1. [Getting started](#getting-started)
    1. [Python environment](#1-python-environment)
    2. [Environment variables](#2-environment-variables)
    3. [Authenticate with HuggingFace](#3-authenticate-with-huggingface)
2. [Quantization](#quantization)
    1. [Settings](#settings)
    2. [FP8 Quantization](#fp8-quantization)
    3. [Autoquant (NVFP4 + FP8 Mixed Precision)](#autoquant)
3. [Evaluation](#evaluation)
    1. [Settings](#evaluation-settings)
    2. [Running evaluation](#running-evaluation)
    3. [Expected output](#expected-output)

<!-- 4. [FAQ](#faq) -->

## Getting started

```bash
export YOUR_HOME="/path/to/your/workspace"
```

### 1. Python environment

```bash
export UV_CACHE_DIR="$YOUR_HOME/.cache/uv"

cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_quant"
uv venv am15_quant
source am15_quant/bin/activate
uv sync --active --no-install-package flash-attn   # install all deps except flash-attn
uv sync --active                                   # then build flash-attn (needs torch)

# flash-attn can take a lot of resources to build, so MAX_JOBS can help restrict this
MAX_JOBS=4 uv sync --active

```

### 2. Environment variables

Set the following once per session (or add to `~/.bashrc`):

```bash
# ── Paths ────────────────────────────────────────────────────────
export ALPAMAYO_WORKSPACE="$YOUR_HOME/alpamayo-recipes"
export ALPAMAYO_MODEL_DIR="$YOUR_HOME/alpamayo_model_converted_from_hf"
export ALPAMAYO_PAI_LOCAL_DIR="$YOUR_HOME/PAI_mini"
export ALPAMAYO_LOG_DIR="$YOUR_HOME/alpamayo_logs"

# ── Cache ────────────────────────────────────────────────────────
export HF_HOME="$YOUR_HOME/.cache/huggingface"
```

> **Tip:** If you hit HuggingFace Hub rate limits, set `export HF_HUB_OFFLINE=1`
> and `export TRANSFORMERS_OFFLINE=1` to force all model/tokenizer loads from
> local cache.

| Variable                 | Required    | Purpose                                                                                  |
| ------------------------ | ----------- | ---------------------------------------------------------------------------------------- |
| `ALPAMAYO_WORKSPACE`     | yes         | Root of the `alpamayo-recipes` checkout                                                  |
| `ALPAMAYO_MODEL_DIR`     | yes         | Pre-trained Alpamayo model directory (output of step 4)                                  |
| `ALPAMAYO_PAI_LOCAL_DIR` | yes         | PAI dataset root (output of step 5); read by entry scripts at runtime                    |
| `ALPAMAYO_LOG_DIR`       | yes         | Directory for Cosmos-RL logs                                                             |
| `UV_CACHE_DIR`           | recommended | uv cache location (set in step 1, before `uv venv`)                                      |
| `HF_HOME`                | recommended | HuggingFace cache location                                                               |
| `HF_HUB_OFFLINE`         | optional    | Set to `1` to skip HuggingFace Hub calls (useful for rate limits or air-gapped clusters) |
| `TRANSFORMERS_OFFLINE`   | optional    | Set to `1` alongside `HF_HUB_OFFLINE`                                                    |

### 3. Authenticate with HuggingFace

The model and dataset require access to gated resources. Request access here: <br>
🤗 [PhysicalAI-Autonomous-Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) <br>
🤗 [Alpamayo-1.5-10B Model](https://huggingface.co/nvidia/Alpamayo-1.5-10B)

Get your token at: https://huggingface.co/settings/tokens. Then authenticate:

```bash
hf auth login
```

## Quantization

Quantization (`quantize.py`) calibrates and saves a quantized checkpoint. Evaluation (`eval.py`) is a separate step that loads any checkpoint — base or quantized — and reports minADE metrics.

### Settings

The quantization path is controlled by the following arguments:

- `--ckpt <path_or_hub_id>`: model to quantize. Defaults to `nvidia/Alpamayo-1.5-10B`.
- `--quant_format <fmt>`: quantization format — `fp8`, `nvfp4`, `w4a8_nvfp4_fp8`, or `auto`. **Required.**
- `--quant_algo <algo>`: calibration algorithm. Defaults to `max`; `smoothquant` is also supported.
- `--quant_weight_only` (optional): enables weight-only quantization.
- `--auto_quantize_bits <N>`: effective-bits budget for AutoQuant. Defaults to `4.8`.
- `--calib_parquet <path>`: calibration clip source. Defaults to `0417_5k_train_set_for_calibration_25.10.parquet`.
- `--num_of_calib_clips <N>`: number of calibration clips (1–5000). Defaults to `100`.
- `--save_model_dir <path>`: directory under which the quantized model is saved. **Required.** The checkpoint is written to a subdirectory named after the format and calibration settings (e.g. `alpamayo1.5_fp8_calib100`).

### FP8 quantization

Quantize and save a FP8 checkpoint:

```bash
uv run --active quantize.py --quant_format=fp8 --num_of_calib_clips=100 --save_model_dir=./outputs
```

To run in the background and capture logs:

```bash
nohup uv run --active quantize.py --quant_format=fp8 --num_of_calib_clips=100 --save_model_dir=./outputs > quantize_fp8.log 2>&1 &
```

### AutoQuant

AutoQuant searches per-layer across NVFP4 and FP8 under an effective-bits budget, enabling mixed-precision quantization with minimal accuracy loss.

Quantize and save an AutoQuant (FP8 + NVFP4) checkpoint at 6.5 effective bits:

```bash
uv run --active quantize.py --quant_format=auto --auto_quantize_bits=6.5 --num_of_calib_clips=100 --save_model_dir=./outputs
```

## Evaluation

### Evaluation settings

Evaluation (`eval.py`) loads any Alpamayo 1.5 checkpoint — base FP16 or a quantized checkpoint saved by `quantize.py` — and reports per-clip and average minADE metrics.

- `--ckpt <path_or_hub_id>`: checkpoint to evaluate. Defaults to `nvidia/Alpamayo-1.5-10B`. Pass the path saved by `quantize.py` to evaluate a quantized model.
- `--parquet <path>`: evaluation clip source. Defaults to `1005_7cam_gold_eval_metadb_public.parquet`.
- `--limit <N>`: number of clips to evaluate. Defaults to `644`.
- `--num_traj_samples <N>`: trajectory samples per clip. Defaults to `6`.
- `--print_every <N>`: log a progress line every N clips. Defaults to `25`.
- `--seed <N>`: random seed for reproducible sampling. Set to `-1` to disable. Defaults to `42`.

### Running evaluation

Evaluate the base FP16 model:

```bash
uv run --active eval.py
```

Evaluate a quantized checkpoint saved by `quantize.py`:

```bash
uv run --active eval.py --ckpt ./outputs/alpamayo1.5_fp8_calib100
```

To run in the background and capture logs:

```bash
nohup uv run --active eval.py --ckpt ./outputs/alpamayo1.5_fp8_calib100 > eval_fp8.log 2>&1 &
```

### Expected output

During a correct run, logs will show:

- Clip IDs loaded from `--parquet`.
- Evaluation progress (`Evaluating clips: ...%`) with per-clip minADE and timing every `--print_every` clips.
- Any failed clips and their errors.
- Final summary: average minADE and average evaluation time per clip.
