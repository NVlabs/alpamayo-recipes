# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overlay a stage-3 head snapshot (head_stepN/head.safetensors, the 321
trainable tensors saved by train_draft_ade.py) onto a full base checkpoint,
producing a loadable full model for evaluation. The base only contributes the
frozen VLM (bit-identical across the whole stage-2/3 family), so any family
checkpoint works as --base.

Usage:
  python snapshot_to_ckpt.py --snapshot run/head_step80 \
      --base /path/to/full_ckpt --out /path/to/eval_ckpt
"""
import argparse
import glob
import os
import shutil

from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full = {}
    for f in sorted(glob.glob(os.path.join(args.base, "*.safetensors"))):
        full.update(load_file(f))
    head = load_file(os.path.join(args.snapshot, "head.safetensors"))
    assert len(head) == 321, len(head)
    for k, v in head.items():
        tk = k if k in full else "model." + k
        full[tk] = v.to(full[tk].dtype)
    os.makedirs(args.out, exist_ok=True)
    save_file(full, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    for p in os.listdir(args.base):
        q = os.path.join(args.base, p)
        if os.path.isfile(q) and not p.endswith(".safetensors") \
                and not p.endswith(".safetensors.index.json"):
            shutil.copy2(q, os.path.join(args.out, p))
    print(f"[snapshot->ckpt] {args.snapshot} -> {args.out}")


if __name__ == "__main__":
    main()
