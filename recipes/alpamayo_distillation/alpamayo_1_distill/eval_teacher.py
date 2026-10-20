#!/usr/bin/env python3
"""
Standalone teacher model evaluation — computes minADE over the val set.
Reads raw clips through physicalAiAvDataset, matching the training data path.

Launch with torchrun (same as training):
  torchrun --standalone --nproc_per_node=8 \
      --module alpamayo1_distill.eval_teacher \
      --teacher_ckpt /path/to/Alpamayo-R1-10B \
      --local_dir /path/to/PhysicalAI-Autonomous-Vehicles
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import copy

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from alpamayo1_distill import helper
from alpamayo1_distill.defaults import (
    DEFAULT_TEACHER_MODEL,
    PAI_FULL_VAL_SIZE,
    PAI_GOLD_EVAL_SIZE,
)
from alpamayo1_distill.models.alpamayo_r1 import AlpamayoR1
from alpamayo1_distill.dataloader.physical_aiavdataset import physicalAiAvDataset
from alpamayo1_distill.dataloader.physical_ai_av.src.physical_ai_av.dataset_local import (
    PhysicalAIAVDatasetInterfaceLocal,
)
from alpamayo1_distill.training_resume import ExactDistributedEvalSampler


# ----------------- DDP helpers -----------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return local_rank, dist.get_rank(), dist.get_world_size()
    return 0, 0, 1


# ----------------- collate -----------------
def collate_fn(batch):
    out = {}
    out["image_frames"] = torch.stack([b["image_frames"] for b in batch], dim=0)
    out["camera_indices"] = torch.stack([b["camera_indices"] for b in batch], dim=0)
    out["relative_timestamps"] = torch.stack([b["relative_timestamps"] for b in batch], dim=0)

    def squeeze_11(x):
        return x.squeeze(0).squeeze(0)

    out["ego_history_xyz"] = torch.stack([squeeze_11(b["ego_history_xyz"]) for b in batch], dim=0).unsqueeze(1)
    out["ego_history_rot"] = torch.stack([squeeze_11(b["ego_history_rot"]) for b in batch], dim=0).unsqueeze(1)
    out["ego_future_xyz"]  = torch.stack([squeeze_11(b["ego_future_xyz"])  for b in batch], dim=0).unsqueeze(1)
    out["ego_future_rot"]  = torch.stack([squeeze_11(b["ego_future_rot"])  for b in batch], dim=0).unsqueeze(1)
    out["t0_us"] = [b["t0_us"] for b in batch]
    out["clip_id"] = [b["clip_id"] for b in batch]
    return out


# ----------------- minADE -----------------
def _minade_sum(pred_xyz: torch.Tensor, gt_xy: torch.Tensor) -> float:
    pred_xy = pred_xyz[..., :2]
    B, T = pred_xy.shape[0], pred_xy.shape[-2]
    pred_xy = pred_xy.reshape(B, -1, T, 2)
    d = torch.linalg.vector_norm(pred_xy - gt_xy[:, None, :, :], dim=-1)
    ade = d.mean(dim=-1)
    return float(ade.min(dim=1).values.sum().detach().to(torch.float32).item())


def _sample(model, model_inputs, *, greedy: bool):
    kwargs = dict(data=model_inputs, max_generation_length=256, return_extra=True)
    if greedy:
        kwargs.update(top_p=1.0, temperature=1.0, num_traj_samples=1, do_sample=False)
    else:
        kwargs.update(top_p=0.98, temperature=0.6, num_traj_samples=6)
    fn = getattr(model, "_sample_trajectories_from_data_with_vlm_rollout", None) or \
         getattr(model, "sample_trajectories_from_data_with_vlm_rollout")
    return fn(**kwargs)


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser("eval_teacher")
    p.add_argument("--teacher_ckpt", type=str, default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--local_dir", type=str, required=True,
                   help="Root directory of the PhysicalAI-AV dataset")
    p.add_argument("--local_files_only", action="store_true", default=False)
    p.add_argument("--t0_us", type=int, default=5_100_000)
    p.add_argument(
        "--eval_set",
        choices=("gold", "full"),
        default="gold",
        help=(
            "gold: packaged 644-clip set (default); full: full PAI evaluation set."
        ),
    )
    p.add_argument("--batch_size_per_gpu", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"[eval_teacher] loading model from {args.teacher_ckpt}")
    model = AlpamayoR1.from_pretrained(
        args.teacher_ckpt, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = copy.deepcopy(model.tokenizer)
    processor = helper.get_processor(tokenizer)

    # ----- dataset -----
    ds_kwargs = dict(t0_us=args.t0_us, tokenizer=tokenizer, local_dir=args.local_dir)
    if args.local_files_only:
        ds_kwargs["local_files_only"] = True
    if args.eval_set == "gold":
        ds_kwargs["gold_eval"] = True

    avdi = PhysicalAIAVDatasetInterfaceLocal(
        local_dir=args.local_dir, local_files_only=args.local_files_only
    )
    ds_val = physicalAiAvDataset(split="val", avdi=avdi, **ds_kwargs)
    expected_samples = (
        PAI_GOLD_EVAL_SIZE if args.eval_set == "gold" else PAI_FULL_VAL_SIZE
    )
    if len(ds_val) != expected_samples:
        raise RuntimeError(
            "The selected PAI evaluation set has the wrong cardinality: "
            f"expected={expected_samples}, constructed={len(ds_val)}. "
            "Verify that the local dataset is the supported snapshot and is complete."
        )

    sampler = ExactDistributedEvalSampler(
        ds_val,
        num_replicas=world_size,
        rank=rank,
    )

    loader = DataLoader(
        ds_val,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    if rank == 0:
        print(
            f"[eval_teacher] eval_set={args.eval_set} split=val "
            f"total_samples={len(ds_val)} batches_this_rank={len(loader)}"
        )

    total_samples = 0
    total_stoch_sum = 0.0
    total_greedy_sum = 0.0
    common_batches = len(loader)
    if world_size > 1:
        common_batches_tensor = torch.tensor(
            common_batches, device=device, dtype=torch.int64
        )
        dist.all_reduce(common_batches_tensor, op=dist.ReduceOp.MIN)
        common_batches = int(common_batches_tensor.item())

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            ego_future_xyz = batch["ego_future_xyz"].to(device, non_blocking=True)
            gt_xy = ego_future_xyz[:, 0, :, :2]
            bsz = int(ego_future_xyz.shape[0])

            tokenized = model._get_generation_mode_tokenized_data_online_distill(
                batch, helper, processor, device
            )
            model_inputs = helper.to_device({
                "tokenized_data": tokenized,
                "ego_history_xyz": batch["ego_history_xyz"],
                "ego_history_rot": batch["ego_history_rot"],
            }, device)

            model_inputs_stoch = copy.deepcopy(model_inputs)
            model_inputs_greedy = copy.deepcopy(model_inputs)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz_s, _, _, _ = _sample(model, model_inputs_stoch, greedy=False)
            total_stoch_sum += _minade_sum(pred_xyz_s, gt_xy)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz_g, _, _, _ = _sample(model, model_inputs_greedy, greedy=True)
            total_greedy_sum += _minade_sum(pred_xyz_g, gt_xy)

            total_samples += bsz

            if args.log_every and step % args.log_every == 0 and step <= common_batches:
                if world_size > 1:
                    packed = torch.tensor(
                        [total_stoch_sum, total_greedy_sum, float(total_samples)],
                        device=device, dtype=torch.float64,
                    )
                    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                    s, g, n = packed.tolist()
                else:
                    s, g, n = total_stoch_sum, total_greedy_sum, float(total_samples)
                if rank == 0:
                    print(
                        f"[eval_teacher] batch={step}/{len(loader)} "
                        f"avg_minADE(stoch)={s/max(1,n):.4f}m "
                        f"avg_minADE(greedy)={g/max(1,n):.4f}m "
                        f"samples={int(n)}"
                    )

    if world_size > 1:
        packed = torch.tensor(
            [total_stoch_sum, total_greedy_sum, float(total_samples)],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        s, g, n = packed.tolist()
    else:
        s, g, n = total_stoch_sum, total_greedy_sum, float(total_samples)

    evaluated_samples = int(round(n))
    if evaluated_samples != len(ds_val):
        raise RuntimeError(
            "Distributed teacher evaluation did not cover the selected dataset "
            f"exactly: expected={len(ds_val)}, evaluated={evaluated_samples}"
        )

    if rank == 0:
        print(
            f"[eval_teacher] DONE "
            f"final_minADE(stoch)={s/max(1,n):.4f}m "
            f"final_minADE(greedy)={g/max(1,n):.4f}m "
            f"total_samples={evaluated_samples}"
        )

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
