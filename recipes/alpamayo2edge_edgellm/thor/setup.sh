#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare a DRIVE AGX Thor for the recipe; run once per board, repeat after a reboot (idempotent).
# Inputs:  WORK           directory on a disk with >= 40 GB free (the DRIVE OS root partition is too small)
#          TENSORRT_ROOT  TensorRT 11.3 SDK for aarch64 (include/, lib/, bin/, python/); DRIVE OS ships only the runtime libraries
# Outputs: $WORK/venv (cmake, ninja, cuRAND headers, TensorRT Python, torch + tokenizers for thor/run.py), $WORK/env.sh (source it in
#          every later step), $WORK/board.json
set -euo pipefail
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # before cd: the script may be named by a relative path
: "${WORK:?set WORK to a directory with >= 40 GB free}"
: "${TENSORRT_ROOT:?set TENSORRT_ROOT to the TensorRT 11.3 aarch64 SDK}"
test -f "$TENSORRT_ROOT/include/NvInfer.h" || { echo "no NvInfer.h under $TENSORRT_ROOT/include"; exit 1; }
WHEEL=$(ls "$TENSORRT_ROOT"/python/tensorrt-11.*-cp312-*linux_aarch64.whl | head -1)
mkdir -p "$WORK/bin" "$WORK/cache/tmp" && cd "$WORK"
FREE_GB=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9); [ "$FREE_GB" -ge 40 ] || echo "warning: $FREE_GB GB free under $WORK, 40 GB recommended"
if [ ! -x bin/uv ]; then   # DRIVE OS has python3 but neither pip nor ensurepip; uv supplies both and creates the venv
  wget -qO- https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-unknown-linux-gnu.tar.gz | tar xz --strip-components=1 -C bin
fi
[ -x venv/bin/python ] || bin/uv venv --python /usr/bin/python3.12 venv >/dev/null
# the locked environment (../pyproject.toml + ../uv.lock): cmake, ninja, the CUDA 13 cuRAND headers (nvidia-curand 10.4.x,
# header under venv/.../nvidia/cu13/include), torch (aarch64 CUDA 13 wheel from the PyTorch index; the PyPI aarch64 wheel is
# CPU-only), tokenizers, safetensors, numpy, pillow; then the TensorRT wheel of the SDK, which no index carries
# The repository's shared package alpamayo-recipes (an editable path dependency every recipe declares) is skipped: the board code does not use it
UV_CACHE_DIR="$WORK/cache/uv" UV_PROJECT_ENVIRONMENT="$WORK/venv" bin/uv sync --project "$RECIPE_DIR" --frozen --inexact --no-install-project --no-install-package alpamayo-recipes --quiet
UV_CACHE_DIR="$WORK/cache/uv" bin/uv pip install --python venv/bin/python --quiet "$WHEEL"
cat > env.sh <<EOT
export WORK=$WORK TENSORRT_ROOT=$TENSORRT_ROOT CUDA_HOME=/usr/local/cuda
export PATH=$WORK/venv/bin:$WORK/bin:/usr/local/cuda/bin:$TENSORRT_ROOT/bin:\$PATH
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib
export CUDA_CACHE_PATH=$WORK/cache/cuda UV_CACHE_DIR=$WORK/cache/uv TMPDIR=$WORK/cache/tmp
EOT
# shellcheck disable=SC1091
source env.sh
python - <<'EOT' > board.json
import json, re, subprocess, tensorrt
v = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print(json.dumps({"platform": "DRIVE AGX Thor", "drive_os": v("cat /etc/nvidia/version-ubuntu-rootfs.txt"), "kernel": v("uname -r"),
  "cuda": re.search(r"release ([\d.]+)", v("nvcc --version")).group(1), "tensorrt": tensorrt.__version__,
  "cpus": int(v("nproc"))}, indent=1))
EOT
cat board.json
trtexec --help 2>&1 | head -1          # banner carries the version: "[TensorRT v110301] [b9]" = 11.3.1, build 9
