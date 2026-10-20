# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministically regenerate the stage-3 clip lists from the shared
s3_v2 subset (`alpamayo1_x_distillation_expert_sft/assets/clip_ids_s3_v2.txt`, 3,200
clips). Seeds are fixed so every list is byte-reproducible:

  heldout16   seed 42  - 16-clip running probe (too small to judge
                         generalization; large-panel eval is the judge)
  train64     seed 42  - first 64 of the same draw (learnability check)
  train128    seed 43  - the resident ADE pool used by all pool-mode arms
  train256    seed 44  - train128 + 128 extra (pool-size ablation)
  replay256   seed 77  - broad-coverage replay scenes for the velocity
                         anchor; disjoint from train256 and heldout16

Usage:
  python make_clip_lists.py --s3 ../../alpamayo1_x_distillation_expert_sft/assets/clip_ids_s3_v2.txt --out-dir .
"""
import argparse
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.s3) if l.strip()]
    pick = random.Random(42).sample(ids, 80)
    train64, heldout16 = pick[:64], pick[64:]
    held = set(heldout16)
    train128 = random.Random(43).sample([c for c in ids if c not in held], 128)
    t128 = set(train128)
    extra = random.Random(44).sample(
        [c for c in ids if c not in held and c not in t128], 128)
    train256 = sorted(t128) + extra
    used = set(train256) | held
    replay256 = random.Random(77).sample([c for c in ids if c not in used], 256)

    assert not set(train128) & held
    assert not set(replay256) & used and len(set(train256)) == 256
    os.makedirs(args.out_dir, exist_ok=True)
    for name, lst in (("train64", train64), ("heldout16", heldout16),
                      ("train128", train128), ("train256", train256),
                      ("replay256", replay256)):
        with open(os.path.join(args.out_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lst) + "\n")
        print(f"{name}: {len(lst)} clips")


if __name__ == "__main__":
    main()
