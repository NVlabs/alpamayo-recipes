# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare streaming requests for run.py from the public clip list.

    python -m alpamayo1_x_edgellm.prepare --output-dir <jobs> [--limit 20]

Per clip: 16 frames per camera at 10 Hz ending at t0 (ticks 0-15, JPEG quality 95), the t0 ego
history and the t0 ground truth, then 13 requests T04..T16; request Tk shows ticks k-4..k-1 of
each camera, so consecutive requests share three ticks per camera (the rolling vision cache in
run.py reuses them). Clips come from the repository's dataset/1005_7cam_gold_eval_metadb_public.parquet
in file order. Needs read access to nvidia/PhysicalAI-Autonomous-Vehicles on Hugging Face.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpamayo1_x_edgellm.core import PLAN, RECIPE_DIR, SYSTEM_PROMPT, USER_PROMPT, read_clip_ids_from_parquet, write_json

CLIPS = RECIPE_DIR.parents[1] / "dataset/1005_7cam_gold_eval_metadb_public.parquet"  # repo-root dataset/
TICKS, WINDOW = 16, 4


def request(images: list[str], history: list) -> dict:
    content = [{"type": "image", "image": image} for image in images]
    content += [{"type": "trajectory", "trajectory": history}, {"type": "text", "text": USER_PROMPT}]
    return {"batch_size": PLAN.action_batch, "temperature": PLAN.temperature, "top_p": PLAN.top_p, "top_k": PLAN.top_k,
            "max_generate_length": PLAN.max_new_tokens, "apply_chat_template": True, "add_generation_prompt": True,
            "requests": [{"messages": [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                                       {"role": "user", "content": content}]}]}


def prepare_clip(index: int, clip_id: str, t0_us: int, output: Path) -> list[dict]:
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from PIL import Image

    sample = load_physical_aiavdataset(clip_id, t0_us=t0_us, num_frames=TICKS)
    frames = sample["image_frames"]  # [4 cameras, 16 ticks, 3, H, W] uint8
    if tuple(frames.shape[:2]) != (4, TICKS):
        raise ValueError(f"{clip_id}: expected [4, {TICKS}] frames, got {tuple(frames.shape)}")
    name = f"{index:06d}_{clip_id}"
    (output / name / "images").mkdir(parents=True, exist_ok=True)
    for camera in range(4):
        for tick in range(TICKS):
            array = frames[camera, tick].permute(1, 2, 0).contiguous().numpy()
            Image.fromarray(array).save(output / name / f"images/camera_{camera}_tick_{tick:02d}.jpg", quality=95)
    write_json(output / name / "metric.json",
               {"schema_version": 1, "ground_truth_xy": sample["ego_future_xyz"][0, 0, :, :2].tolist()})
    history = sample["ego_history_xyz"][0, 0].tolist()
    jobs = []
    for horizon in range(WINDOW, TICKS + 1):
        first = horizon - WINDOW
        images = [f"images/camera_{c}_tick_{t:02d}.jpg" for c in range(4) for t in range(first, horizon)]
        write_json(output / name / f"requests/T{horizon:02d}.json", request(images, history))
        jobs.append({"eval_index": index * (TICKS - WINDOW + 1) + first, "source_eval_index": index,
                     "stream_ordinal": index, "horizon": horizon, "first_tick": first, "incoming_tick": horizon - 1,
                     "clip_id": clip_id, "t0_us": t0_us, "input": f"{name}/requests/T{horizon:02d}.json",
                     "metric": f"{name}/metric.json", "raw_output": f"{name}/outputs/T{horizon:02d}.json",
                     "log": f"{name}/logs/T{horizon:02d}.log"})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for images, requests and jobs.json (written last).")
    parser.add_argument("--limit", type=int, default=20, help="Number of clips (default 20).")
    parser.add_argument("--t0-us", type=int, default=5_100_000)
    args = parser.parse_args()
    if (args.output_dir / "jobs.json").exists():
        raise FileExistsError(f"{args.output_dir / 'jobs.json'} exists; use a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)  # a failed earlier run may be rerun in place
    jobs = []
    for index, clip_id in enumerate(read_clip_ids_from_parquet(str(CLIPS))[: args.limit]):
        jobs += prepare_clip(index, clip_id, args.t0_us, args.output_dir)
        print(f"{index + 1}/{args.limit} {clip_id}", flush=True)
    write_json(args.output_dir / "jobs.json",
               {"protocol_id": "alpamayo_r1_v1p1d1a6_stream_t4_t16_v1", "action_backend": None, "vision_mode": "rolling-v4x4",
                "window_ticks": WINDOW, "slides": TICKS - WINDOW, "ordering": "clip-major, then ascending horizon", "jobs": jobs})
    print(f"{len(jobs)} requests in {args.output_dir / 'jobs.json'}")


if __name__ == "__main__":
    main()
