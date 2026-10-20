#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Idempotent node-start checks for the expert SFT/RL recipes. Run this after
# EVERY node (re)start, before launching anything. Every check here encodes a
# failure mode that actually happened during recipe development.
#
# Expected env (set what applies):
#   ALPAMAYO_DATASET_DIR   local PhysicalAI-AV mirror
#   EXPECTED_TRAIN_CLIPS   expected clip count of your training list (e.g.
#                          3200) — guards against silent dataset filtering
#   EXPERT_CKPT            checkpoint whose prompt-layout config to verify
#   EXPECT_CAMERA          true|false — expected include_camera_ids of that ckpt
set -u

echo "==================== on_node_start $(date -u +%FT%TZ) ===================="

# 1) Environment fingerprint: FIRST thing to compare when a trajectory
#    refuses to reproduce.
python -V
python - <<'EOF'
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda} "
      f"cudnn {torch.backends.cudnn.version()}")
EOF
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
if [ -f /opt/env_fingerprint.txt ]; then
    echo "baked env fingerprint: $(cat /opt/env_fingerprint.txt)"
    pip freeze | sha256sum | awk '{print "live  env fingerprint: " $1}'
fi

# 2) Dataset reachability + silent-filter guard. A historical incident: a
#    reprovisioned data pipeline silently filtered a 3,200-clip training list
#    down to 737 (reasoning-coverage filter) and trained "successfully" on
#    the wrong data. If you know your expected clip count, assert it.
if [ -n "${ALPAMAYO_DATASET_DIR:-}" ]; then
    test -d "$ALPAMAYO_DATASET_DIR" || { echo "FATAL: ALPAMAYO_DATASET_DIR missing"; exit 1; }
    echo "dataset root OK: $ALPAMAYO_DATASET_DIR"
fi
if [ -n "${EXPECTED_TRAIN_CLIPS:-}" ] && [ -n "${EXPERT_CLIP_LIST:-}" ]; then
    N=$(grep -c . "$EXPERT_CLIP_LIST")
    [ "$N" -eq "$EXPECTED_TRAIN_CLIPS" ] \
        && echo "clip list OK: $N clips" \
        || { echo "FATAL: clip list has $N clips, expected $EXPECTED_TRAIN_CLIPS"; exit 1; }
fi

# 3) Prompt-layout guard: feeding a model the other family's prompt layout
#    silently degrades it. Verify the checkpoint config matches expectations.
if [ -n "${EXPERT_CKPT:-}" ] && [ -n "${EXPECT_CAMERA:-}" ]; then
    python - <<EOF
import json, sys
c = json.load(open("${EXPERT_CKPT}/config.json"))
got = bool(c.get("include_camera_ids", False))
want = "${EXPECT_CAMERA}" == "true"
assert got == want, f"include_camera_ids={got}, expected {want}"
print(f"prompt-layout OK: include_camera_ids={got}")
EOF
fi

# 4) HF offline-mode visibility: offline is correct for training from a local
#    mirror, but silently breaks anything that must download (benchmarks,
#    judge models). Make the state loud.
for v in HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE; do
    echo "$v=${!v:-<unset>}"
done

echo "==================== on_node_start done ===================="
