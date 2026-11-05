# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Frozen training environment for the Alpamayo expert SFT (stage 2) and
# expert RL (stage 3) recipes.
#
# WHY AN IMAGE AND NOT A REQUIREMENTS FILE: training trajectories are only
# reproducible WITHIN one kernel/library stack — a reprovisioned venv with
# drifted wheels produces a different (equally valid) trajectory from the
# same seed. docker/constraints.txt is the exact pip freeze of the
# environment the released checkpoints were trained in (verified
# byte-identical across all four training nodes); this image bakes it.
#
# Build (repo root as context):
#   docker build -f alpamayo1_x_distillation_expert_rl/docker/Dockerfile -t alpamayo-expert:frozen .
# Run:    mount the repo at /workspace/alpamayo1_5-distill and the
#         PhysicalAI-AV mirror somewhere stable, then source
#         /usr/local/bin/on_node_start.sh (idempotent; run it after EVERY
#         node (re)start — see the checks inside).

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        git git-lfs curl ca-certificates build-essential ninja-build \
        ffmpeg redis-server \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv

COPY alpamayo1_x_distillation_expert_rl/docker/constraints.txt /opt/constraints.txt

# Torch first (cu128 wheels), then the rest under the frozen constraint set.
RUN pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128
RUN pip install -c /opt/constraints.txt \
        transformers==4.57.1 accelerate==1.13.0 vllm==0.11.0 \
        safetensors numpy==2.2.6 physical_ai_av==0.2.2 \
        omegaconf hydra-core pandas pyarrow pillow decord av \
        webdataset wids nvitop
# flash-attn needs torch present at build time.
RUN pip install flash_attn==2.8.3 --no-build-isolation
# Public pinned-SHA installs: the exact revisions the recipes were validated with.
RUN pip install \
    "cosmos-rl @ git+https://github.com/nvidia-cosmos/cosmos-rl.git@747d1bd7cd46a266895fd1f98f28b10122c0b923" \
    "alpamayo_r1 @ git+https://github.com/NVlabs/alpamayo.git@f3198c2f6158e1158ae8cb4fc7d7716bd25085bb"

COPY alpamayo1_x_distillation_expert_rl/docker/on_node_start.sh /usr/local/bin/on_node_start.sh
RUN chmod +x /usr/local/bin/on_node_start.sh

# Environment fingerprint baked into the image for provenance.
RUN pip freeze > /opt/pip_freeze_baked.txt && \
    sha256sum /opt/pip_freeze_baked.txt | tee /opt/env_fingerprint.txt

WORKDIR /workspace
