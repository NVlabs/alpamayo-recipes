#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import math
import time
from pathlib import Path

import modelopt.torch.opt as mto
import numpy as np
import torch
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo_r1.common import logging
from alpamayo_r1.common.logging import setup_logging
from tqdm import tqdm

from alpamayo1_5_quant.utils import read_clip_ids_from_parquet

setup_logging()

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")

# Limit CPU threads to reduce memory bandwidth contention with GPU on unified memory (GB10)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)


@torch.inference_mode()
def compute_minade_for_clip_pytorch(
    model: Alpamayo1_5,
    processor,
    clip_id: str,
    t0_us: int,
    top_p: float,
    temperature: float,
    num_traj_samples: int,
    max_generation_length: int,
    device: str = "cuda",
    seed: int | None = 42,
) -> tuple[float, float]:
    """
    Returns minADE (meters) for one clip.
    """
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)

    messages = helper.create_message(
        data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()  # (T,2)
    # free image frames and raw data before inference — largest tensors in unified memory
    del data, messages, inputs
    gc.collect()
    # yield the memory bus to GPU before starting inference — critical on unified memory (GB10)
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    del model_inputs

    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]  # (S,T,2)
    del pred_xyz, pred_rot

    d = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)  # (S,T)
    ade = d.mean(axis=-1)  # (S,)
    min_ade = float(ade.min())
    return min_ade, elapsed_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default="1005_7cam_gold_eval_metadb_public.parquet")
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--ckpt", type=str, default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--num_traj_samples", type=int, default=6)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=644, help="How many unique clip_ids to evaluate.")
    ap.add_argument("--seed", type=int, default=42, help="Set -1 to disable reseeding per clip.")
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument(
        "--gc_every",
        type=int,
        default=1,
        help="Run Python garbage collection every N clips (0 disables).",
    )
    ap.add_argument(
        "--empty_cache_every",
        type=int,
        default=1,
        help="Call torch.cuda.empty_cache() every N clips (0 disables).",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    parquet_path = (script_dir / args.parquet).resolve()

    clip_ids = read_clip_ids_from_parquet(str(parquet_path))
    if args.limit is not None and args.limit > 0:
        clip_ids = clip_ids[: args.limit]
    logger.info(f"Loaded {len(clip_ids)} clip_ids from: {parquet_path}")

    device = "cuda"
    mto.enable_huggingface_checkpointing()
    model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.float16).to(
        device=device, dtype=torch.float16
    )
    if args.ckpt != "nvidia/Alpamayo-1.5-10B":
        import modelopt.torch.quantization as mtq

        mtq.print_quant_summary(model)
    model.eval()

    processor = helper.get_processor(model.tokenizer)

    seed = None if args.seed < 0 else args.seed

    it = tqdm(clip_ids, desc="Evaluating clips")

    per_clip = []
    per_clip_ms = []
    failed = []

    for i, clip_id in enumerate(it, start=1):
        try:
            minade, elapsed_ms = compute_minade_for_clip_pytorch(
                model=model,
                processor=processor,
                clip_id=clip_id,
                t0_us=args.t0_us,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=args.max_generation_length,
                device=device,
                seed=seed,
            )
            per_clip.append(minade)
            per_clip_ms.append(elapsed_ms)

            if args.print_every and (i % args.print_every == 0):
                avg_so_far = float(np.mean(per_clip)) if per_clip else math.nan
                logger.info(
                    f"[{i}/{len(clip_ids)}] clip_id={clip_id} "
                    f"minADE={minade:.4f}m time={elapsed_ms:.2f}ms | avg_so_far={avg_so_far:.4f}m"
                )

        except Exception as e:
            failed.append((clip_id, repr(e)))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.print_every:
                logger.info(f"[{i}/{len(clip_ids)}] FAILED clip_id={clip_id}: {e}")
        finally:
            if args.gc_every > 0 and (i % args.gc_every == 0):
                gc.collect()
            if (
                torch.cuda.is_available()
                and args.empty_cache_every > 0
                and (i % args.empty_cache_every == 0)
            ):
                torch.cuda.empty_cache()

    if per_clip:
        avg_minade = float(np.mean(per_clip))
        avg_time_ms = float(np.mean(per_clip_ms))
        logger.info("============================================================")
        logger.info(
            f"Average minADE over {len(per_clip)}/{len(clip_ids)} clips: {avg_minade:.6f} meters"
        )
        logger.info(f"Average eval time: {avg_time_ms:.2f} ms/clip")
    else:
        logger.info("No successful clips; average minADE not computed.")

    if failed:
        logger.info("============================================================")
        logger.info(f"Failed clips: {len(failed)}")
        for cid, err in failed[:10]:
            logger.info(f"  {cid}: {err}")
        if len(failed) > 10:
            logger.info("  ...")


if __name__ == "__main__":
    main()
