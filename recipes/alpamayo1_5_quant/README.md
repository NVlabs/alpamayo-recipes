# FP8 / AutoQuant (FP8 + NVFP4) Quantization

This Recipe defines a reproducible post-training quantization (PTQ) procedure for quantizing Alpamayo 1.5 to FP8 or AutoQuant (FP8 + NVFP4)

## Prerequisites

This recipe is tested in the following settings. Other settings may also work but not guaranteed.

- NVIDIA B300 GPU with CUDA 13
- Python 3.12
- Python Libraries: torch==2.12.0, torchvision==0.27.0, nvidia-modelopt==0.43.0

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
    4. [Expected Output](#expected-runtime-behaviors-and-outputs)

<!-- 3. [FAQ](#faq) -->

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

### Settings

The quantization path is controlled by the following arguments:

- `--parquet <path>`: evaluation clip source. `1005_7cam_gold_eval_metadb_public.parquet` is used by default.
- `--quant_format fp8`: enables FP8 PTQ.
- `--quant_algo <algo>`: keeps default `max` for FP8 runs.
- `--quant_weight_only` (optional): enables FP8 weight-only PTQ.
- `--calib_parquet <path>`: calibration clip source. `0417_5k_train_set_for_calibration_25.10.parquet` is used by default.
- `--num_of_calib_clips <N>`: number of calibration clips (1 to 5000). `100` is used by default.

Please refer to `quantize.py` to find the usage of more arguments.

### FP8 quantization

Run an example command below to quantize alpamayo1.5 in FP8 and save the quantized model:

```bash
uv run --active quantize.py --quant_format=fp8 --num_of_calib_clips=100 --save_model_dir=./outputs
```

### AutoQuant

Autoquant is a tool that allows for mixed NVFP4 + FP8 quantization while still remaining lossless.

Run an example command below to quantize alpamayo1.5 in AutoQuant (FP8 + NVFP4) with 6.5 effective bits and save the quantized model:

```bash
uv run --active quantize.py --quant_format=auto --auto_quantize_bits=6.5 --num_of_calib_clips=100 --save_model_dir=./outputs
```

### Expected runtime behaviors and outputs

During a correct run, logs would show:

- Calibration clips are loaded from `--calib_parquet`.
- Calibration loop progress (`calibration: ...%`) is executed.
- Quantization summary is printed.
- Evaluation starts and reports per-clip metrics and final averages.

After the evaluation, you will see the following outputs:

- Average minADE
- Average evaluation time per clip
