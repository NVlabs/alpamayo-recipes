#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import argparse
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

import torch
import numpy as np
import modelopt.torch.opt as mto

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5 import helper


def make_joint_calibration_forward_loop(
    *,
    clip_ids: list[str],
    processor,
    t0_us: int,
    top_p: float,
    temperature: float,
    max_generation_length: int,
    calibration_traj_samples: int,
    device: str,
):
    """
    Build a calibration loop that exercises both VLM generation and diffusion.

    This avoids text-only calibration and ensures quantizers in the rollout path
    (vlm/expert/diffusion-related modules) observe representative activations.
    """
    def _calibration_loop(runtime_model):
        runtime_model.eval()
        with torch.no_grad():
            for clip_id in tqdm(clip_ids, desc="calibration"):
                data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
                messages = helper.create_message(data["image_frames"].flatten(0, 1))
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

                with torch.autocast("cuda", dtype=torch.float16):
                    runtime_model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=top_p,
                        temperature=temperature,
                        num_traj_samples=calibration_traj_samples,
                        max_generation_length=max_generation_length,
                    )

    return _calibration_loop

def read_clip_ids_from_parquet(parquet_path: str) -> list[str]:
    """
    Reads clip_ids from parquet. Tries common column names; falls back to index if needed.
    Returns clip_ids as a list of strings (unique, preserving first occurrence order).
    """
    parquet_path = str(parquet_path)
    df = pd.read_parquet(parquet_path)
    cols_lower = {c.lower(): c for c in df.columns}
    clip_ids = df[cols_lower["key"]].astype(str).tolist()

    seen = set()
    uniq = []
    for cid in clip_ids:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq


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

    # Build chat message and tokenize
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
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
        # make sampling more stable/reproducible across clips
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
            max_generation_length=max_generation_length,
            return_extra=True,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # GT: (T,2)
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()  # (T,2)

    # pred_xyz: assume (B, num_traj_sets, num_traj_samples, T, 3)
    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]  # (S,T,2)

    # ADE per sample: mean over time of L2 in XY
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
    ap.add_argument(
        "--compile_trt",
        action="store_true",
        help="Compile TRT vision/language/diffusion path before running evaluation.",
    )
    ap.add_argument(
        "--offload_module_to_cpu",
        action="store_true",
        help="Enable Torch-TensorRT offload_module_to_cpu for TRT module compile (LM/diffusion; default: disabled).",
    )
    ap.add_argument(
        "--trt_max_seq_len",
        type=int,
        default=0,
        help="Override max_seq_len for TRT LM compile (0 = use observed prefix + max_generation_length).",
    )
    ap.add_argument(
        "--trt_max_prefix_len",
        type=int,
        default=0,
        help="Override max_prefix_len for TRT compile (0 = use observed prefix_seq_len).",
    )
    ap.add_argument(
        "--quant_format",
        type=str,
        default=None,
        choices=["fp8", "nvfp4", "w4a8_nvfp4_fp8", "auto"],
        help="Jointly quantize the entire pytorch model to the specified format before running evaluation.",
    )
    ap.add_argument("--auto_quantize_bits", type=float, default=4.8, help="Effective-bits budget for AutoQuantize (only used when --quant_format auto)")
    ap.add_argument("--quant_algo", type=str, default="max", choices=["max", "smoothquant"])
    ap.add_argument(
        "--quant_weight_only",
        action="store_true",
        help="Jointly quantize the entire pytorch model to weight-only before running evaluation.",
    )
    ap.add_argument("--calib_parquet", type=str, default="0417_5k_train_set_for_calibration_25.10.parquet")
    ap.add_argument("--num_of_calib_clips", type=int, default=100)
    ap.add_argument("--save_model_dir", type=str, default=None, help="Directory to save the quantized model.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    parquet_path = (script_dir / args.parquet).resolve()

    clip_ids = read_clip_ids_from_parquet(str(parquet_path))
    if args.limit is not None and args.limit > 0:
        clip_ids = clip_ids[: args.limit]

    print(f"Loaded {len(clip_ids)} clip_ids from: {parquet_path}")

    device = "cuda"
    # Enable automatic ModelOpt save/restore with huggingface checkpointing APIs
    # This needs to be done only once in the program
    mto.enable_huggingface_checkpointing()
    customized_model = False
    if args.ckpt == "nvidia/Alpamayo-1.5-10B":
        model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.float16).to(
            device=device, dtype=torch.float16
        )
    else:
        customized_model = True
        model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.float16).to(
            device=device, dtype=torch.float16
        )
        import modelopt.torch.quantization as mtq
        mtq.print_quant_summary(model)
    model.eval()

    # IMPORTANT: build processor once (do NOT rebuild per clip)
    processor = helper.get_processor(model.tokenizer)

    if args.quant_format is not None and not customized_model:
        assert args.calib_parquet is not None, "--calib_parquet is required when quant_format is not None"
        assert 0 < args.num_of_calib_clips <= 5000, "--num_of_calib_clips must be between 1 and 5000"
        calib_parquet_path = (script_dir / args.calib_parquet).resolve()
        calib_clip_ids = read_clip_ids_from_parquet(str(calib_parquet_path))
        calib_clip_ids = calib_clip_ids[: args.num_of_calib_clips]
        print(f"Loaded {len(calib_clip_ids)} calibration clip_ids from: {calib_parquet_path}")

        from alpamayo1_5.quantize_utils import quantize_model, auto_quantize_model

        print(f"Quantizing model ({args.quant_format}) ...")

        quantization_args = argparse.Namespace(
            quant_format=args.quant_format,
            quant_algo=args.quant_algo,
            weight_only=args.quant_weight_only,
            debug=True,
            auto_quantize_bits=args.auto_quantize_bits,
        )
        calibration_forward_loop = make_joint_calibration_forward_loop(
            clip_ids=calib_clip_ids,
            processor=processor,
            t0_us=args.t0_us,
            top_p=args.top_p,
            temperature=args.temperature,
            max_generation_length=args.max_generation_length,
            calibration_traj_samples=args.num_traj_samples,
            device=device,
        )

        if args.quant_format == "auto":
            with torch.enable_grad():
                model = auto_quantize_model(
                    model,
                    quantization_args,
                    clip_ids=calib_clip_ids,
                    processor=processor,
                    t0_us=args.t0_us,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    max_generation_length=args.max_generation_length,
                    calibration_traj_samples=args.num_traj_samples,
                    device=device,
                )
        else:
            model = quantize_model(
                model,
                quantization_args,
                calibration_forward_loop=calibration_forward_loop,
            )
        model.eval()

    if args.save_model_dir is not None:
        save_dir = os.path.join(args.save_model_dir, f"alpamayo1.5{'_' + str(args.quant_format) if args.quant_format is not None else '_fp16'}{'_' + str(args.auto_quantize_bits) + 'bits' if args.quant_format == 'auto' else ''}{'_weight_only' if args.quant_weight_only else ''}{'_calib' + str(args.num_of_calib_clips) if args.quant_format is not None else ''}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving quantized model to: {save_dir}")
        model.save_pretrained(save_dir)
        print(f"Quantized model saved to: {save_dir}")

    seed = None if args.seed < 0 else args.seed

    if args.compile_trt:
        raise NotImplementedError("TRT compilation is not yet supported.")

    it = tqdm(clip_ids, desc="Evaluating clips")

    per_clip = []
    per_clip_ms = []
    failed = []

    for i, clip_id in enumerate(it, start=1):
        try:
            if args.compile_trt:
                raise NotImplementedError("TRT compilation is not yet supported.")
            else:
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
                print(
                    f"[{i}/{len(clip_ids)}] clip_id={clip_id} "
                    f"minADE={minade:.4f}m time={elapsed_ms:.2f}ms | avg_so_far={avg_so_far:.4f}m"
                )

        except Exception as e:
            failed.append((clip_id, repr(e)))
            # try to recover GPU memory if something went wrong mid-inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.print_every:
                print(f"[{i}/{len(clip_ids)}] FAILED clip_id={clip_id}: {e}")
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
        print("============================================================")
        print(f"Average minADE over {len(per_clip)}/{len(clip_ids)} clips: {avg_minade:.6f} meters")
        print(f"Average eval time: {avg_time_ms:.2f} ms/clip")
    else:
        print("No successful clips; average minADE not computed.")

    if failed:
        print("============================================================")
        print(f"Failed clips: {len(failed)}")
        # print a few
        for cid, err in failed[:10]:
            print(f"  {cid}: {err}")
        if len(failed) > 10:
            print("  ...")


if __name__ == "__main__":
    with torch.no_grad():
        main()
