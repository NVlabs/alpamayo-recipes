#!/usr/bin/env python3
import os
import math
import time
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import copy
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoConfig
from alpamayo1_distill.config import ExpertModelConfig
from alpamayo1_distill.defaults import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_COSMOS_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    PAI_FULL_VAL_SIZE,
    PAI_GOLD_EVAL_SIZE,
)

from alpamayo1_distill.models.alpamayo_r1_distill_2b import ExpertModelDistill
from alpamayo1_distill.dataloader.physical_aiavdataset import physicalAiAvDataset
from alpamayo1_distill.dataloader.physical_ai_av.src.physical_ai_av.dataset_local import (
    PhysicalAIAVDatasetInterfaceLocal,
)
from alpamayo1_distill import helper
from alpamayo1_distill.losses import OnPolicyDistillConfig
from alpamayo1_distill.models.alpamayo_r1 import AlpamayoR1
from alpamayo1_distill.training_resume import (
    CHECKPOINT_PHASE,
    CHECKPOINT_VERSION,
    ExactDistributedEvalSampler,
    ResumeDistributedSampler,
    capture_rng_state,
    gather_rank_states,
    iter_dataloader_without_advancing_model_rng,
    restore_rng_state,
    restore_rank_state,
    resume_cursor_from_state,
    validate_resume_metadata,
)
from alpamayo1_distill.training import combine_training_loss

# ----------------- DDP helpers -----------------
def setup_ddp():
    """Initialize DDP if launched by torchrun; otherwise run single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return local_rank, rank, world_size
    else:
        return 0, 0, 1


def is_main(rank: int):
    return rank == 0


# ----------------- resume/checkpoint helpers -----------------
_STEP_RE = re.compile(r"(?:^|/)(?:step_)(\d+)(?:$|/)")


def infer_step_from_ckpt_dir(ckpt_dir: str) -> int:
    matches = _STEP_RE.findall(ckpt_dir.rstrip("/"))
    return int(matches[-1]) if matches else 0


def infer_run_dir_from_resume(resume_ckpt: str) -> str | None:
    """
    If resume_ckpt looks like .../<run_dir>/ckpts/step_XXXXX, return <run_dir>.
    Otherwise return None.
    """
    p = Path(resume_ckpt).resolve()
    # step dir -> parent should be ckpts
    if p.name.startswith("step_") and p.parent.name in ("ckpts", "checkpoints"):
        return str(p.parent.parent)
    return None


def resolve_student_config_source(
    resume_ckpt: str | None,
    init_ckpt: str | None,
) -> Path:
    """Select the structural config that owns the student checkpoint weights.

    Fresh runs use the recipe's canonical Cosmos configuration. Resume and
    weights-only initialization must reconstruct the architecture serialized
    alongside those weights; model/processor paths are relocated separately.
    """

    if resume_ckpt is not None and init_ckpt is not None:
        raise ValueError("--resume_ckpt and --init_ckpt are mutually exclusive")
    checkpoint_dir = resume_ckpt if resume_ckpt is not None else init_ckpt
    if checkpoint_dir is None:
        return DEFAULT_CONFIG_PATH
    source = Path(checkpoint_dir).expanduser().resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(
            f"Student checkpoint config not found: {source / 'config.json'}"
        )
    return source


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    def _move_states(state_values):
        for state in state_values:
            if state is None:
                continue
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device, non_blocking=True)

    _move_states(optimizer.state.values())
    # ZeRO-1: inner AdamW has its own state dict; move it too so fused Adam gets
    # step tensors on the correct device.
    inner = getattr(optimizer, "optim", None)
    if inner is not None:
        _move_states(inner.state.values())


def save_trainer_state(
    ckpt_dir: str,
    *,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    loss_keys,
    teacher_val: bool,
    rank_states: list[dict[str, Any]],
    resume_metadata: dict[str, Any],
):
    state = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_phase": CHECKPOINT_PHASE,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "loss_keys": loss_keys,
        "teacher_val": bool(teacher_val),
        "world_size": int(resume_metadata["world_size"]),
        "rank_states": rank_states,
        "resume_metadata": resume_metadata,
    }
    torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))


def try_load_trainer_state(
    ckpt_dir: str,
    *,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    device: torch.device,
    rank: int,
    world_size: int,
    resume_metadata: dict[str, Any],
):
    path = os.path.join(ckpt_dir, "trainer_state.pt")
    if not os.path.exists(path):
        return None, None

    st = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_metadata(st, resume_metadata)

    if st.get("optimizer", None) is not None:
        optimizer.load_state_dict(st["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)

    if lr_scheduler is not None and st.get("scheduler", None) is not None:
        lr_scheduler.load_state_dict(st["scheduler"])

    rank_state = restore_rank_state(
        st,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    return st, rank_state


def save_checkpoint(
    ckpt_dir: str,
    *,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    loss_keys,
    teacher_val: bool,
    rank_states: list[dict[str, Any]],
    resume_metadata: dict[str, Any],
    args: argparse.Namespace,
):
    # Write to a staging directory, then atomically rename to the final path.
    # This guarantees the final checkpoint is either fully written or absent —
    # a mid-save crash leaves only the .tmp dir and the previous checkpoint intact.
    tmp_dir = ckpt_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    base_model.save_pretrained(tmp_dir, safe_serialization=True)
    save_trainer_state(
        tmp_dir,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        global_step=global_step,
        epoch=epoch,
        step_in_epoch=step_in_epoch,
        loss_keys=loss_keys,
        teacher_val=teacher_val,
        rank_states=rank_states,
        resume_metadata=resume_metadata,
    )
    # Save args for convenience (json-serializable only)
    try:
        with open(os.path.join(tmp_dir, "train_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    except Exception:
        pass

    # A final checkpoint path must never be visible until every file is written.
    # Refuse to delete an existing checkpoint: periodic/final save coordination
    # should prevent duplicate writes, and failing closed protects known-good data.
    if os.path.exists(ckpt_dir):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {ckpt_dir}")
    os.rename(tmp_dir, ckpt_dir)


# ----------------- loss formatting helpers -----------------
def _sync_loss_keys(loss_dict: dict[str, torch.Tensor] | None, rank: int, world_size: int):
    """
    Build a stable loss key ordering across ranks.
    Prefer: total_loss first, then remaining keys sorted.
    """
    if loss_dict is None or len(loss_dict) == 0:
        keys = ["loss"]
    else:
        keys = list(loss_dict.keys())
        if "total_loss" in keys:
            keys = ["total_loss"] + sorted([k for k in keys if k != "total_loss"])
        else:
            keys = sorted(keys)

    if world_size > 1:
        obj = [keys if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        keys = obj[0]
    return keys


def _loss_dict_to_float_map(loss_dict: dict[str, torch.Tensor] | None, keys: list[str]):
    """Convert a tensor loss_dict to python floats for a fixed key list. Missing keys -> 0.0"""
    out: dict[str, float] = {}
    if loss_dict is None:
        for k in keys:
            out[k] = 0.0
        return out

    for k in keys:
        v = loss_dict.get(k, None)
        if v is None:
            out[k] = 0.0
        else:
            out[k] = float(v.detach().to(torch.float32).item())
    return out


def _format_loss_kv(loss_avgs: dict[str, float], keys: list[str], precision: int = 6):
    parts = []
    for k in keys:
        if k in loss_avgs:
            parts.append(f"{k}={loss_avgs[k]:.{precision}f}")
    return " ".join(parts)


# ----------------- data -----------------
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


# ----------------- scheduler -----------------
class CosineWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Cosine LR schedule with linear warmup and per-group min_lr.

    The last `num_vlm_groups` param groups use `vlm_min_lr` as their floor;
    all other groups use `min_lr`.
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 min_lr=1e-6, vlm_min_lr=1e-6, num_vlm_groups=2, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.vlm_min_lr = vlm_min_lr
        self.num_vlm_groups = num_vlm_groups
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        n = len(self.base_lrs)
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            floor = self.vlm_min_lr if i >= (n - self.num_vlm_groups) else self.min_lr
            lrs.append(floor + (base_lr - floor) * scale)
        return lrs


# ----------------- optimizer -----------------
def _unify_trainable_param_dtype(model: torch.nn.Module, target_dtype: torch.dtype) -> int:
    """In-place cast all trainable model params to a single dtype.

    Required for ZeroRedundancyOptimizer, which rejects parameter groups whose
    dense tensors don't share a dtype. The Alpamayo model has a bf16 VLM but the
    action expert and heads are constructed in fp32 by ``AutoModel.from_config``
    (it ignores the ``dtype`` field in the expert config). We cast trainable
    params only — frozen ones don't enter the optimizer.

    Returns the number of params that were re-cast.
    """
    n_cast = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.dtype != target_dtype:
            p.data = p.data.to(target_dtype)
            n_cast += 1
    return n_cast


def build_adamw(
    model: torch.nn.Module,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
    vlm_lr_scale: float = 1.0,
    use_zero1: bool = False,
    vlm_weight_decay: float | None = None,
):
    # Note: when use_zero1 is set, _unify_trainable_param_dtype must already have
    # been called BEFORE DDP wrap (in main()). Calling it here would be too late
    # because the DDP reducer is already pinned to the original dtype/bucket layout.
    vlm_wd = vlm_weight_decay if vlm_weight_decay is not None else weight_decay
    vlm_decay, vlm_no_decay = [], []
    other_decay, other_no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_no_decay = p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "layernorm" in n.lower()
        is_vlm = n.startswith("vlm.")
        if is_vlm:
            (vlm_no_decay if is_no_decay else vlm_decay).append(p)
        else:
            (other_no_decay if is_no_decay else other_decay).append(p)

    vlm_lr = lr * vlm_lr_scale
    param_groups = [
        {"params": other_decay,    "weight_decay": weight_decay, "lr": lr},
        {"params": other_no_decay, "weight_decay": 0.0,          "lr": lr},
        {"params": vlm_decay,      "weight_decay": vlm_wd,       "lr": vlm_lr},
        {"params": vlm_no_decay,   "weight_decay": 0.0,          "lr": vlm_lr},
    ]

    if use_zero1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        return ZeroRedundancyOptimizer(
            param_groups,
            optimizer_class=torch.optim.AdamW,
            parameters_as_bucket_view=True,
            lr=lr,
            betas=betas,
            fused=True,
        )
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, fused=True)


# ----------------- validation -----------------
@torch.no_grad()
def run_validation_fast_distributed(
    ddp_model, loader_val, device, rank, world_size, processor,
    compute_minade=True, compute_teacher_minade=True, prefix="[val]", log_every_val_batches=50,
    teacher_processor=None,
    expected_samples=None,
):
    if expected_samples is None or expected_samples <= 0:
        raise ValueError(
            f"expected_samples must be a positive integer, got {expected_samples!r}"
        )
    ddp_model.eval()
    base_model = ddp_model.module if world_size > 1 else ddp_model
    teacher_model = getattr(base_model, "teacher_model", None)
    compute_teacher_minade = compute_teacher_minade and (teacher_model is not None)
    if compute_minade and (teacher_model is None) and rank == 0:
        print(f"{prefix} NOTE: base_model has no teacher_model; teacher minADE will be skipped.")

    def _minade_sum_from_pred_xyz(pred_xyz: torch.Tensor, gt_xy: torch.Tensor) -> float:
        pred_xy = pred_xyz[..., :2]
        B = pred_xy.shape[0]
        T = pred_xy.shape[-2]
        pred_xy = pred_xy.reshape(B, -1, T, 2)
        d = torch.linalg.vector_norm(pred_xy - gt_xy[:, None, :, :], dim=-1)
        ade = d.mean(dim=-1)
        minade = ade.min(dim=1).values
        return float(minade.sum().detach().to(torch.float32).item())

    def _sample_with_rollout(model, model_inputs):
        if hasattr(model, "_sample_trajectories_from_data_with_vlm_rollout"):
            return model._sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=6,
                max_generation_length=256,
                return_extra=True,
            )
        if hasattr(model, "sample_trajectories_from_data_with_vlm_rollout"):
            return model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=6,
                max_generation_length=256,
                return_extra=True,
            )
        raise RuntimeError("No supported sampling method found on the given model.")

    def _greedy_with_rollout(model, model_inputs):
        if hasattr(model, "_sample_trajectories_from_data_with_vlm_rollout"):
            return model._sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=1.0,
                temperature=1.0,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
                do_sample=False,
            )
        if hasattr(model, "sample_trajectories_from_data_with_vlm_rollout"):
            return model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=1.0,
                temperature=1.0,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
                do_sample=False,
            )
        raise RuntimeError("No supported sampling method found on the given model.")

    total_samples = 0
    total_minade_sum = 0.0
    total_greedy_minade_sum = 0.0
    total_teacher_minade_sum = 0.0
    total_loss_sums: dict[str, float] = {}

    win_samples = 0
    win_batches = 0
    win_minade_sum = 0.0
    win_greedy_minade_sum = 0.0
    win_teacher_minade_sum = 0.0
    win_loss_sums: dict[str, float] = {}

    loss_keys: list[str] | None = None

    # Exact, unpadded rank shards can differ by one batch. Distributed progress
    # reductions are therefore safe only while every rank is still iterating.
    common_val_batches = len(loader_val)
    if world_size > 1:
        common_batches_tensor = torch.tensor(
            common_val_batches, device=device, dtype=torch.int64
        )
        dist.all_reduce(common_batches_tensor, op=dist.ReduceOp.MIN)
        common_val_batches = int(common_batches_tensor.item())

    for val_step, batch in enumerate(loader_val, start=1):
        inner_batch = batch
        ego_history_xyz =  inner_batch["ego_history_xyz"].to(device, non_blocking=True)
        ego_history_rot =  inner_batch["ego_history_rot"].to(device, non_blocking=True)
        ego_future_xyz  =  inner_batch["ego_future_xyz"].to(device, non_blocking=True)
        ego_future_rot  =  inner_batch["ego_future_rot"].to(device, non_blocking=True)

        tokenized = base_model._get_generation_mode_tokenized_data_online_distill(inner_batch, helper, processor, device)
        tokenized_minADE = dict(tokenized)

        teacher_tokenized = None
        if teacher_processor is not None:
            teacher_tokenized = base_model._get_generation_mode_tokenized_data_online_distill(inner_batch, helper, teacher_processor, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Validation shards are intentionally uneven, so bypass DDP's
            # forward-time buffer synchronization and call the underlying
            # module directly. Final metrics are reduced explicitly below.
            out = base_model(
                tokenized_data=tokenized,
                ego_history_xyz=ego_history_xyz,
                ego_history_rot=ego_history_rot,
                ego_future_xyz=ego_future_xyz,
                ego_future_rot=ego_future_rot,
                teacher_tokenized_data=teacher_tokenized,
                labels_mask=None,
            )

        if loss_keys is None:
            ld0 = out.loss_dict if hasattr(out, "loss_dict") else None
            if ld0 is None:
                ld0 = {"loss": out.loss}
            loss_keys = _sync_loss_keys(ld0, rank=rank, world_size=world_size)

        ld = out.loss_dict if hasattr(out, "loss_dict") else None
        if ld is None:
            ld = {"loss": out.loss}
        ld_f = _loss_dict_to_float_map(ld, loss_keys)

        bsz = int(ego_future_xyz.shape[0])

        for k in loss_keys:
            total_loss_sums[k] = total_loss_sums.get(k, 0.0) + ld_f[k] * bsz
            win_loss_sums[k]   = win_loss_sums.get(k, 0.0) + ld_f[k] * bsz

        if compute_minade:
            #messages = helper.create_message(batch["image_frames"].squeeze(0).flatten(0, 1))
            #inputs = processor.apply_chat_template(
            #    messages,
            #    tokenize=True,
            #    add_generation_prompt=False,
            #    continue_final_message=True,
            #    return_dict=True,
            #    return_tensors="pt",
            #)
            model_inputs = {
                #"tokenized_data": inputs,
                "tokenized_data": tokenized_minADE,
                "ego_history_xyz":  inner_batch["ego_history_xyz"],
                "ego_history_rot":  inner_batch["ego_history_rot"],
            }
            model_inputs_student = helper.to_device(model_inputs, device)
            if compute_teacher_minade:
                if teacher_processor is not None:
                    teacher_tokenized = base_model._get_generation_mode_tokenized_data_online_distill(inner_batch, helper, teacher_processor, device)
                    model_inputs_teacher = {
                        "tokenized_data": teacher_tokenized,
                        "ego_history_xyz": inner_batch["ego_history_xyz"],
                        "ego_history_rot": inner_batch["ego_history_rot"],
                    }
                    model_inputs_teacher = helper.to_device(model_inputs_teacher, device)
                else:
                    model_inputs_teacher = copy.deepcopy(model_inputs_student)
            model_inputs_greedy = copy.deepcopy(model_inputs_student)

            gt_xy = ego_future_xyz[:, 0, :, :2]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz_s, _, _, _ = _sample_with_rollout(base_model, model_inputs_student)
            minade_batch_sum = _minade_sum_from_pred_xyz(pred_xyz_s, gt_xy)
            total_minade_sum += minade_batch_sum
            win_minade_sum   += minade_batch_sum


            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz_g, _, _, _ = _greedy_with_rollout(base_model, model_inputs_greedy)
            greedy_minade_batch_sum = _minade_sum_from_pred_xyz(pred_xyz_g, gt_xy)
            total_greedy_minade_sum += greedy_minade_batch_sum
            win_greedy_minade_sum   += greedy_minade_batch_sum

            if compute_teacher_minade:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz_t, _, _, _ = _sample_with_rollout(teacher_model, model_inputs_teacher)
                t_minade_batch_sum = _minade_sum_from_pred_xyz(pred_xyz_t, gt_xy)
                total_teacher_minade_sum += t_minade_batch_sum
                win_teacher_minade_sum   += t_minade_batch_sum

        total_samples += bsz
        win_samples   += bsz
        win_batches   += 1

        if (
            log_every_val_batches
            and (val_step % log_every_val_batches == 0)
            and val_step <= common_val_batches
        ):
            if world_size > 1:
                packed = torch.tensor(
                    [win_loss_sums.get(k, 0.0) for k in loss_keys]
                    + [win_minade_sum, win_greedy_minade_sum, win_teacher_minade_sum,
                       float(win_samples), float(win_batches)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                packed_list = packed.tolist()
            else:
                packed_list = (
                    [win_loss_sums.get(k, 0.0) for k in loss_keys]
                    + [win_minade_sum, win_greedy_minade_sum, win_teacher_minade_sum,
                       float(win_samples), float(win_batches)]
                )

            g_loss_sums = packed_list[: len(loss_keys)]
            g_minade_sum, g_greedy_sum, g_tminade_sum, g_samples, g_batches = packed_list[len(loss_keys):]

            loss_avgs = {k: (g_loss_sums[i] / max(1.0, g_samples)) for i, k in enumerate(loss_keys)}
            msg_losses = _format_loss_kv(loss_avgs, loss_keys, precision=6)

            if rank == 0:
                if compute_minade:
                    avg_minade = g_minade_sum / max(1.0, g_samples)
                    avg_greedy = g_greedy_sum / max(1.0, g_samples)
                    if compute_teacher_minade:
                        avg_tminade = g_tminade_sum / max(1.0, g_samples)
                        print(
                            f"{prefix} batch={val_step}/{len(loader_val)} "
                            f"avg_losses[{int(g_batches)} batches]={msg_losses} "
                            f"avg_minADE(student)={avg_minade:.4f}m "
                            f"avg_greedy_minADE(student)={avg_greedy:.4f}m "
                            f"avg_minADE(teacher)={avg_tminade:.4f}m "
                            f"samples_in_window={int(g_samples)}"
                        )
                    else:
                        print(
                            f"{prefix} batch={val_step}/{len(loader_val)} "
                            f"avg_losses[{int(g_batches)} batches]={msg_losses} "
                            f"avg_minADE(student)={avg_minade:.4f}m "
                            f"avg_greedy_minADE(student)={avg_greedy:.4f}m "
                            f"samples_in_window={int(g_samples)}"
                        )
                else:
                    print(
                        f"{prefix} batch={val_step}/{len(loader_val)} "
                        f"avg_losses[{int(g_batches)} batches]={msg_losses} "
                        f"samples_in_window={int(g_samples)}"
                    )

            win_loss_sums.clear()
            win_minade_sum = 0.0
            win_greedy_minade_sum = 0.0
            win_teacher_minade_sum = 0.0
            win_samples = 0
            win_batches = 0

    if loss_keys is None:
        loss_keys = ["loss"]

    if world_size > 1:
        packed = torch.tensor(
            [total_loss_sums.get(k, 0.0) for k in loss_keys]
            + [total_minade_sum, total_greedy_minade_sum, total_teacher_minade_sum, float(total_samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed_list = packed.tolist()
    else:
        packed_list = (
            [total_loss_sums.get(k, 0.0) for k in loss_keys]
            + [total_minade_sum, total_greedy_minade_sum, total_teacher_minade_sum, float(total_samples)]
        )

    g_loss_sums = packed_list[: len(loss_keys)]
    g_minade_sum, g_greedy_sum, g_tminade_sum, g_samples = packed_list[len(loss_keys):]

    evaluated_samples = int(round(g_samples))
    if evaluated_samples != expected_samples:
        raise RuntimeError(
            "Distributed validation did not cover the selected dataset exactly: "
            f"expected={expected_samples}, evaluated={evaluated_samples}"
        )

    final_loss_dict = {k: (g_loss_sums[i] / max(1.0, g_samples)) for i, k in enumerate(loss_keys)}
    final_loss = final_loss_dict.get("total_loss", final_loss_dict[loss_keys[0]])

    final_minade = (g_minade_sum / max(1.0, g_samples)) if compute_minade else None
    final_greedy_minade = (g_greedy_sum / max(1.0, g_samples)) if compute_minade else None
    final_teacher_minade = (g_tminade_sum / max(1.0, g_samples)) if compute_teacher_minade else None

    if rank == 0 and compute_minade:
        if final_teacher_minade is not None:
            print(
                f"{prefix} DONE "
                f"final_avg_minADE(student)={final_minade:.4f}m "
                f"final_greedy_minADE(student)={final_greedy_minade:.4f}m "
                f"final_avg_minADE(teacher)={final_teacher_minade:.4f}m "
                f"total_samples={evaluated_samples}"
            )
        else:
            print(
                f"{prefix} DONE "
                f"final_avg_minADE(student)={final_minade:.4f}m "
                f"final_greedy_minADE(student)={final_greedy_minade:.4f}m "
                f"total_samples={evaluated_samples}"
            )

    ddp_model.train()
    return final_loss, final_minade, final_greedy_minade, final_teacher_minade, final_loss_dict, loss_keys


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser("alpamayo distill trainer")

    # model/config
    p.add_argument("--teacher_ckpt", type=str, default=DEFAULT_TEACHER_MODEL)
    p.add_argument("--teacher_vlm_path", type=str, default=None,
                   help="Override vlm_name_or_path in the teacher checkpoint config (needed when the saved path doesn't exist on the current cluster)")
    p.add_argument("--student_vlm_path", type=str, default=DEFAULT_COSMOS_STUDENT_MODEL,
                   help="Cosmos-Reason2-2B repository ID or a local mirror of the same architecture")
    p.add_argument("--student_processor_path", type=str, default=None,
                   help="Optional Cosmos processor path; defaults to --student_vlm_path")

    # hparams
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--vlm_lr_scale", type=float, default=1.0,
                   help="LR multiplier for VLM backbone params (e.g. 0.1 = 10x lower than expert head)")
    p.add_argument("--max_steps", type=int, default=150_000)
    p.add_argument("--save_steps", type=int, default=15_000)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--vlm_weight_decay", type=float, default=None,
                   help="Weight decay for VLM parameters. Defaults to --weight_decay if not set. "
                        "Set higher than --weight_decay to regularise the VLM more strongly.")
    p.add_argument("--grad_clip", type=float, default=0,
                   help="Max gradient norm for clipping (0 = disabled)")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps (1 = disabled; e.g. 4 for 4x effective batch)")
    p.add_argument("--future_traj_weight", type=float, default=1.0,
                   help="Loss weight for future_traj_loss (e.g. 2.0 to upweight trajectory learning)")
    p.add_argument("--future_traj_vlm_weight", type=float, default=1.0,
                   help="Loss weight for SFT-style VLM CE on GT traj_future tokens.")
    p.add_argument("--distill_weight", type=float, default=1.0,
                   help="Loss weight for distill_loss (pure KL, no T*T Hinton scaling).")
    p.add_argument("--vlm_loss_weight", type=float, default=1.0,
                   help="Loss weight for vlm_loss (hard CE on greedy teacher tokens).")
    p.add_argument("--distill_T", type=float, default=0.5,
                   help="Softmax temperature for KL distillation. Affects target softness only; "
                        "magnitude is controlled by --distill_weight (Hinton T*T scaling removed).")
    p.add_argument("--hidden_distill_weight", type=float, default=0.0,
                   help="Weight for VLM last-layer hidden state distillation loss "
                        "(squared L2 distance on normalized states; per token this is "
                        "2 - 2*cosine_similarity). 0 = disabled.")
    p.add_argument("--min_lr", type=float, default=1e-6,
                   help="Min LR for diffusion/expert head at end of cosine schedule")
    p.add_argument("--vlm_min_lr", type=float, default=None,
                   help="Min LR for VLM backbone (defaults to min_lr if not set)")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--batch_size_per_gpu", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--compile", action="store_true", default=False,
                   help="Apply torch.compile to student VLM backbone for ~15-30%% throughput improvement")
    p.add_argument("--val_num_workers", type=int, default=2,
                   help="Dataloader workers for validation. Lower than --num_workers to avoid fork ENOMEM.")
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Number of batches prefetched per dataloader worker (default 4, PyTorch default is 2)")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--val_every", type=int, default=15_000)
    p.add_argument("--full_eval", action="store_true",
                   help="Evaluate on the full PAI evaluation set from the clip index instead of the "
                        "default 644-clip gold set.")

    # output / resume
    p.add_argument("--output_dir", type=str, default=None,
                   help="Run output dir. Defaults to ./outputs/alpamayo1-distill-<timestamp> "
                        "(or inferred from --resume_ckpt).")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Checkpoint dir to resume from (e.g., .../ckpts/step_15000).")
    p.add_argument("--resume_step", type=int, default=None,
                   help="Override global_step when resuming (if ckpt dir name isn't step_XXXX).")

    # early stop
    p.add_argument("--early_stop_step", type=int, default=None,
                   help="Stop training after this many steps (overrides --max_steps if smaller).")

    # diffusion-head-only training
    p.add_argument("--diffusion_head_only", action="store_true",
                   help="Freeze VLM from step 0 and train only the diffusion head.")
    p.add_argument("--flow_grad_to_vlm", action="store_true",
                   help="Allow future_traj_loss gradient to flow back through the VLM "
                        "(sets config.stop_grad_from_vlm=False). Default: off, matching prior runs.")
    p.add_argument("--init_ckpt", type=str, default=None,
                   help="Load model weights from this checkpoint for a fresh training run (no optimizer/scheduler/step restore). "
                        "Use with --diffusion_head_only to start diffusion-only training from an existing checkpoint.")

    # misc
    p.add_argument("--compute_minade", action="store_true", help="Compute minADE in validation.")
    p.add_argument("--compute_minade_teacher", action="store_true", help="Compute minADE in validation.")

    p.add_argument("--t0_us", type=int, default=5_100_000,
                   help="Trajectory t0 timestamp in microseconds.")
    p.add_argument("--t0_us_list", type=str, default=None,
                   help="Comma-separated list of t0 timestamps in microseconds to sample per clip "
                        "(training dataset only). E.g. '3600000,5100000,7000000'. "
                        "When set, each clip yields one sample per t0 value, multiplying dataset size. "
                        "Validation always uses a single t0 (--t0_us). "
                        "Each value must satisfy t0 > num_history_steps * time_step * 1e6 (>1.6s) "
                        "and leave room for the 6.4s future window.")
    # ---- on-policy distillation ----
    # Off-policy distillation only ever scores the student on teacher-written
    # prefixes, so it never supervises the states the student actually reaches when
    # generating. These flags mix in steps that distil on the student's own rollouts.
    p.add_argument("--on_policy_ratio", type=float, default=0.0,
                   help="Fraction of optimizer steps that distil on a student-sampled "
                        "sequence instead of a teacher-generated one (0 = disabled). "
                        "Rounded to a 1-in-N stride so the two regimes interleave. "
                        "On-policy steps are supervised by the divergence alone — hard CE "
                        "against the student's own samples would just reinforce them.")
    p.add_argument("--on_policy_start_step", type=int, default=0,
                   help="Global step at which on-policy mixing begins. Warm up off-policy "
                        "first: the student must already emit well-formed sequences (with "
                        "<traj_future_start>) for its rollouts to be usable.")
    p.add_argument("--on_policy_divergence", type=str, default="forward_kl",
                   choices=["forward_kl", "reverse_kl", "jsd"],
                   help="Divergence for on-policy steps. forward_kl = KL(teacher||student), "
                        "mode-covering (matches the off-policy term). reverse_kl = "
                        "KL(student||teacher), mode-seeking — the choice when single-sample "
                        "quality matters more than best-of-N coverage. jsd = generalized "
                        "Jensen-Shannon. Cost rises in that order (2/4/6 vocab-sized buffers).")
    p.add_argument("--on_policy_beta", type=float, default=0.5,
                   help="Teacher mixture weight for --on_policy_divergence jsd.")
    p.add_argument("--on_policy_T", type=float, default=1.0,
                   help="Softmax temperature for the on-policy divergence. 1.0 compares the "
                        "distributions as actually sampled; unlike --distill_T there is no "
                        "reason to soften targets here.")
    p.add_argument("--on_policy_sample_temperature", type=float, default=0.6,
                   help="Sampling temperature for the student rollout (default matches eval).")
    p.add_argument("--on_policy_sample_top_p", type=float, default=0.98,
                   help="Nucleus cutoff for the student rollout (default matches eval).")
    p.add_argument("--on_policy_kl_weight", type=float, default=1.0,
                   help="Loss weight on the on-policy divergence term.")

    p.add_argument("--random_t0", action="store_true",
                   help="Train split only: draw t0 uniformly at random from each clip's valid "
                        "window instead of the fixed --t0_us / --t0_us_list timestamps. The window "
                        "is computed per clip from the egomotion and camera coverage, so every "
                        "sampled t0 has a complete history, future and image stack — nothing runs "
                        "past the end of the video. Validation always keeps the fixed --t0_us. "
                        "Note: data order is no longer bit-exact on resume (model/optimizer state is).")
    p.add_argument("--t0_samples_per_clip", type=int, default=None,
                   help="Random t0 draws per clip per epoch (only with --random_t0). Defaults to "
                        "len(--t0_us_list) if that is given, else 1 — so swapping a 5-entry "
                        "--t0_us_list for --random_t0 keeps the same epoch size.")
    p.add_argument("--t0_margin_us", type=int, default=0,
                   help="Extra safety margin in microseconds shaved off both ends of the valid t0 "
                        "window (only with --random_t0).")
    p.add_argument("--t0_seed", type=int, default=None,
                   help="Seed for random t0 sampling. Unset = fresh entropy per worker (different "
                        "t0s every epoch); set = reproducible for a fixed rank/worker layout.")
    p.add_argument("--local_dir", type=str, default=None,
                   help="Root directory for PhysicalAIAVDatasetInterfaceLocal (required).")
    p.add_argument("--local_files_only", action="store_true", default=False,
                   help="Pass local_files_only=True to PhysicalAIAVDatasetInterfaceLocal.")

    # ---- ZeRO-1 optimizer state sharding ----
    p.add_argument("--use_zero1", action="store_true",
                   help="Shard AdamW optimizer state across ranks via "
                        "torch.distributed.optim.ZeroRedundancyOptimizer (ZeRO-1). "
                        "Saves ~24 GB/GPU at 2.4B params. Weights and gradients are "
                        "still replicated (DDP-style); only optimizer state is sharded. "
                        "No-op in single-GPU runs.")
    return p.parse_args()



def main():
    launch_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args = parse_args()

    # Cosmos-Reason2-2B is the recipe's only student architecture. A separate
    # processor path is optional for local mirrors but must describe the same model.
    if args.student_processor_path is None:
        args.student_processor_path = args.student_vlm_path
    helper.BASE_PROCESSOR_NAME = args.student_processor_path
    helper._load_base_processor.cache_clear()

    if args.local_dir is None:
        raise ValueError("--local_dir is required to read the Physical AI AV dataset")

    warmup_steps = int(args.max_steps * args.warmup_ratio)

    # perf knobs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)

    # ----- output dir -----
    if args.output_dir is None:
        if args.resume_ckpt is not None:
            inferred = infer_run_dir_from_resume(args.resume_ckpt)
            args.output_dir = (
                inferred
                if inferred is not None
                else str(Path("outputs") / f"alpamayo1-distill-{launch_time}")
            )
        else:
            args.output_dir = str(Path("outputs") / f"alpamayo1-distill-{launch_time}")
    run_dir = Path(args.output_dir).resolve()
    ckpts_root = run_dir / "ckpts"
    if is_main(rank):
        os.makedirs(ckpts_root, exist_ok=True)

    # ----- load config/model -----
    config_source = resolve_student_config_source(args.resume_ckpt, args.init_ckpt)
    config_overrides = {"vlm_name_or_path": args.student_processor_path}
    config = ExpertModelConfig.from_pretrained(
        config_source,
        local_files_only=True,
        **config_overrides,
    )
    config.future_traj_vlm_loss_weight = args.future_traj_vlm_weight
    if args.hidden_distill_weight > 0:
        config.hidden_distill_weight = args.hidden_distill_weight
    if args.flow_grad_to_vlm:
        config.stop_grad_from_vlm = False
        if is_main(rank):
            print("[config] flow_grad_to_vlm=True -> stop_grad_from_vlm=False "
                  "(future_traj_loss will update the VLM)")

    teacher_config = AutoConfig.from_pretrained(args.teacher_ckpt, local_files_only=True)
    if args.teacher_vlm_path is not None:
        teacher_config.vlm_name_or_path = args.teacher_vlm_path
    teacher_model = AlpamayoR1.from_pretrained(
        args.teacher_ckpt, config=teacher_config, local_files_only=True, dtype=torch.bfloat16
    ).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    tokenizer = copy.deepcopy(teacher_model.tokenizer)
    processor = helper.get_processor(tokenizer)
    _teacher_proc_path = args.teacher_vlm_path or args.teacher_ckpt
    teacher_processor = helper.get_processor_from_path(tokenizer, _teacher_proc_path)

    # ----- build / resume student model -----
    _expert_kwargs: dict[str, Any] = {
        "teacher_model": teacher_model,
        "student_model_path": args.student_vlm_path,
        "processor_model_path": args.student_processor_path,
    }

    if args.resume_ckpt is not None:
        resume_dir = str(Path(args.resume_ckpt).resolve())
        try:
            model = ExpertModelDistill.from_pretrained(
                resume_dir,
                local_files_only=True,
                config=config,
                torch_dtype=torch.bfloat16,
                **_expert_kwargs,
            ).to(device)
        except TypeError:
            model = ExpertModelDistill.from_pretrained(
                resume_dir,
                local_files_only=True,
                config=config,
                **_expert_kwargs,
            ).to(device)
        # Ensure we use the teacher specified by args.teacher_ckpt
        setattr(model, "teacher_model", teacher_model)
        model.train()
        if is_main(rank):
            print(f"[resume] loaded student weights from {resume_dir}")
    elif args.init_ckpt is not None:
        init_dir = str(Path(args.init_ckpt).resolve())
        try:
            model = ExpertModelDistill.from_pretrained(
                init_dir,
                local_files_only=True,
                config=config,
                torch_dtype=torch.bfloat16,
                **_expert_kwargs,
            ).to(device)
        except TypeError:
            model = ExpertModelDistill.from_pretrained(
                init_dir,
                local_files_only=True,
                config=config,
                **_expert_kwargs,
            ).to(device)
        setattr(model, "teacher_model", teacher_model)
        model.train()
        if is_main(rank):
            print(f"[init_ckpt] loaded weights from {init_dir} (fresh training run, no trainer state restored)")
    else:
        model = ExpertModelDistill(
            config=config,
            distill_T=args.distill_T,
            **_expert_kwargs,
        ).to(device)
        model.train()

    # ----- on-policy distillation config -----
    on_policy_cfg = OnPolicyDistillConfig(
        ratio=args.on_policy_ratio,
        start_step=args.on_policy_start_step,
        divergence=args.on_policy_divergence,
        beta=args.on_policy_beta,
        temperature=args.on_policy_T,
        sample_temperature=args.on_policy_sample_temperature,
        sample_top_p=args.on_policy_sample_top_p,
        kl_weight=args.on_policy_kl_weight,
    )
    model.on_policy_cfg = on_policy_cfg
    if is_main(rank):
        print(f"[on_policy] {on_policy_cfg.describe()}")

    # always sync distill_T and hidden_distill_weight from CLI onto the model
    # (covers resume / init_ckpt / from_pretrained paths that may not pass through __init__).
    model.distill_T = args.distill_T
    model.config.future_traj_vlm_loss_weight = args.future_traj_vlm_weight
    if args.hidden_distill_weight > 0:
        config.hidden_distill_weight = args.hidden_distill_weight

    # ----- freeze VLM for diffusion-head-only training -----
    if args.diffusion_head_only:
        n_frozen = 0
        for name, p in model.named_parameters():
            if name.startswith("vlm."):
                p.requires_grad_(False)
                n_frozen += 1
        if is_main(rank):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"[diffusion_head_only] froze {n_frozen} VLM param tensors. "
                f"Trainable params: {trainable:,}"
            )

    # ----- ZeRO-1 dtype unification -----
    # ZeroRedundancyOptimizer requires all dense params share a dtype. The Alpamayo
    # model is mixed (VLM bf16, expert + heads fp32) because AutoModel.from_config
    # ignores the configured dtype. Cast trainable params to bf16 *before* DDP wrap
    # so the reducer is built with the final dtype and bucket layout.
    if args.use_zero1:
        n_cast = _unify_trainable_param_dtype(model, torch.bfloat16)
        if is_main(rank):
            print(f"[zero1] cast {n_cast} trainable param tensors to bf16 before DDP wrap")

    # torch.compile disabled — Qwen3VL visual encoder causes CUDA misaligned address errors
    # due to dynamic image patch shapes. Re-enable when transformers supports it properly.
    # if args.compile:
    #     model.vlm.model = torch.compile(model.vlm.model, mode="default")

    if world_size > 1:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=False,
        )
    else:
        ddp_model = model

    # ----- data -----
    # Online dataloader: read raw clips via PhysicalAIAVDatasetInterfaceLocal.
    # Build the raw PhysicalAI AV dataset for online teacher supervision.

    _t0_us_list = (
        [int(x.strip()) for x in args.t0_us_list.split(",")]
        if args.t0_us_list is not None
        else None
    )
    ds_kwargs: dict[str, Any] = dict(t0_us=args.t0_us, tokenizer=tokenizer)
    if _t0_us_list is not None:
        ds_kwargs["t0_us_list"] = _t0_us_list
    if args.random_t0:
        # Keep the epoch size the operator asked for: an explicit
        # --t0_samples_per_clip wins, otherwise match the length of --t0_us_list.
        if args.t0_samples_per_clip is not None:
            _samples_per_clip = args.t0_samples_per_clip
        elif _t0_us_list is not None:
            _samples_per_clip = len(_t0_us_list)
        else:
            _samples_per_clip = 1
        ds_kwargs["random_t0"] = True
        ds_kwargs["num_samples_per_clip"] = _samples_per_clip
        ds_kwargs["t0_margin_us"] = args.t0_margin_us
        ds_kwargs["t0_seed"] = args.t0_seed
        if rank == 0:
            print(f"[data] random t0 sampling: {_samples_per_clip} draw(s) per clip per epoch, "
                  f"margin={args.t0_margin_us}us, seed={args.t0_seed}")
    ds_kwargs["local_dir"] = args.local_dir
    if args.local_files_only:
        ds_kwargs["local_files_only"] = True
    # Eval always uses a single fixed t0 — drop both the t0 list and the random-t0
    # knobs so validation stays comparable across runs and checkpoints.
    _train_only_keys = {"t0_us_list", "random_t0", "num_samples_per_clip", "t0_margin_us", "t0_seed"}
    ds_kwargs_eval = {k: v for k, v in ds_kwargs.items() if k not in _train_only_keys}
    if args.full_eval:
        pass  # use the full val split from clip_index
    else:
        ds_kwargs_eval["gold_eval"] = True

    avdi = PhysicalAIAVDatasetInterfaceLocal(
        local_dir=args.local_dir, local_files_only=args.local_files_only
    )

    # Keep one training cohort regardless of the selected validation mode. The
    # packaged gold clips are always held out because some are indexed as train
    # clips in the source metadata and must not leak into the 644-clip benchmark.
    ds_train = physicalAiAvDataset(
        split="train",
        avdi=avdi,
        **ds_kwargs,
    )
    ds_val   = physicalAiAvDataset(split="val",   avdi=avdi, **ds_kwargs_eval)
    expected_eval_samples = (
        PAI_FULL_VAL_SIZE if args.full_eval else PAI_GOLD_EVAL_SIZE
    )
    if len(ds_val) != expected_eval_samples:
        raise RuntimeError(
            "The selected PAI evaluation set has the wrong cardinality: "
            f"expected={expected_eval_samples}, constructed={len(ds_val)}. "
            "Verify that the local dataset is the supported snapshot and is complete."
        )

    # Use a deterministic epoch-based sampler even for world_size=1 so a
    # resumed job can reconstruct the original shuffle and jump to its exact
    # per-rank batch cursor.
    sampler_train = ResumeDistributedSampler(
        ds_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    sampler_val = ExactDistributedEvalSampler(
        ds_val,
        num_replicas=world_size,
        rank=rank,
    )

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler_train,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler_val,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    steps_per_epoch = max(1, len(loader_train))
    train_dataset_size = len(ds_train)

    # ----- optimizer + scheduler -----
    base_model = ddp_model.module if world_size > 1 else ddp_model
    if args.diffusion_head_only:
        # VLM already frozen — build a simple 2-group optimizer over diffusion head only
        # Note: dtype unification for ZeRO-1 already happened before DDP wrap in main().
        _decay, _no_decay = [], []
        for name, p in base_model.named_parameters():
            if not p.requires_grad:
                continue
            is_no_decay = (
                p.ndim == 1 or name.endswith(".bias")
                or "norm" in name.lower() or "layernorm" in name.lower()
            )
            (_no_decay if is_no_decay else _decay).append(p)
        _diff_param_groups = [
            {"params": _decay,    "weight_decay": args.weight_decay, "lr": args.lr},
            {"params": _no_decay, "weight_decay": 0.0,               "lr": args.lr},
        ]
        if args.use_zero1:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            opt = ZeroRedundancyOptimizer(
                _diff_param_groups,
                optimizer_class=torch.optim.AdamW,
                parameters_as_bucket_view=True,
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                fused=True,
            )
        else:
            opt = torch.optim.AdamW(
                _diff_param_groups,
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                fused=True,
            )
        lr_scheduler = CosineWarmupScheduler(
            optimizer=opt,
            warmup_steps=warmup_steps,
            total_steps=args.max_steps,
            min_lr=args.min_lr,
            vlm_min_lr=args.min_lr,
            num_vlm_groups=0,
        )
    else:
        opt = build_adamw(
            base_model,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.weight_decay,
            vlm_lr_scale=args.vlm_lr_scale,
            use_zero1=args.use_zero1,
            vlm_weight_decay=args.vlm_weight_decay,
        )

        vlm_min_lr = args.vlm_min_lr if args.vlm_min_lr is not None else args.min_lr
        lr_scheduler = CosineWarmupScheduler(
            optimizer=opt,
            warmup_steps=warmup_steps,
            total_steps=args.max_steps,
            min_lr=args.min_lr,
            vlm_min_lr=vlm_min_lr,
            num_vlm_groups=2,  # last 2 param groups are always VLM (see build_adamw)
        )

    # Versioned checkpoints reject changes that would make the optimizer, data
    # cursor, or random stream diverge from uninterrupted training. To fine-tune
    # with changed settings, load weights through --init_ckpt instead.
    resume_metadata: dict[str, Any] = {
        "world_size": world_size,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "grad_accum_steps": args.grad_accum_steps,
        "steps_per_epoch": steps_per_epoch,
        "train_dataset_size": train_dataset_size,
        "max_steps": args.max_steps,
        "warmup_steps": warmup_steps,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "vlm_min_lr": args.vlm_min_lr,
        "vlm_lr_scale": args.vlm_lr_scale,
        "weight_decay": args.weight_decay,
        "vlm_weight_decay": args.vlm_weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "grad_clip": args.grad_clip,
        "future_traj_weight": args.future_traj_weight,
        "future_traj_vlm_weight": args.future_traj_vlm_weight,
        "distill_weight": args.distill_weight,
        "vlm_loss_weight": args.vlm_loss_weight,
        "hidden_distill_weight": args.hidden_distill_weight,
        "distill_T": args.distill_T,
        "val_every": args.val_every,
        "compute_minade": args.compute_minade,
        "compute_minade_teacher": args.compute_minade_teacher,
        "full_eval": args.full_eval,
        "use_zero1": args.use_zero1,
        "diffusion_head_only": args.diffusion_head_only,
        "flow_grad_to_vlm": args.flow_grad_to_vlm,
        "t0_us": args.t0_us,
        "t0_us_list": args.t0_us_list,
        "on_policy_ratio": args.on_policy_ratio,
        "on_policy_effective_ratio": on_policy_cfg.effective_ratio,
        "on_policy_start_step": args.on_policy_start_step,
        "on_policy_divergence": args.on_policy_divergence,
        "on_policy_beta": args.on_policy_beta,
        "on_policy_T": args.on_policy_T,
        "on_policy_sample_temperature": args.on_policy_sample_temperature,
        "on_policy_sample_top_p": args.on_policy_sample_top_p,
        "on_policy_kl_weight": args.on_policy_kl_weight,
        "random_t0": args.random_t0,
        "t0_samples_per_clip": args.t0_samples_per_clip,
        "t0_margin_us": args.t0_margin_us,
        "t0_seed": args.t0_seed,
        "teacher_ckpt": args.teacher_ckpt,
        "teacher_vlm_path": args.teacher_vlm_path,
        "student_vlm_path": args.student_vlm_path,
        "student_processor_path": args.student_processor_path,
        "local_dir": args.local_dir,
    }

    # ----- running window stats -----
    loss_keys: list[str] | None = None
    run_loss_sums: dict[str, float] = {}
    run_lr_sum = 0.0
    run_count = 0
    run_window_start = time.time()

    # ----- resume trainer state (opt/sched/epoch/step) -----
    global_step = 0
    epoch = 0
    step_in_epoch = 0
    resume_skip_batches = 0
    teacher_val = True

    if args.resume_ckpt is not None:
        resume_dir = str(Path(args.resume_ckpt).resolve())
        inferred_step = infer_step_from_ckpt_dir(resume_dir)
        global_step = args.resume_step if args.resume_step is not None else inferred_step

        st, rank_state = try_load_trainer_state(
            resume_dir,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
            device=device,
            rank=rank,
            world_size=world_size,
            resume_metadata=resume_metadata,
        )

        if st is not None:
            global_step = int(st.get("global_step", global_step))
            if inferred_step != 0 and inferred_step != global_step:
                raise ValueError(
                    "Checkpoint directory and trainer state disagree on global_step: "
                    f"directory={inferred_step}, trainer_state={global_step}."
                )
            if args.resume_step is not None and args.resume_step != global_step:
                raise ValueError(
                    "--resume_step conflicts with trainer_state.pt: "
                    f"argument={args.resume_step}, checkpoint={global_step}. "
                    "Overriding only the counter would desynchronize optimizer and scheduler state."
                )
            epoch, step_in_epoch, is_legacy = resume_cursor_from_state(
                st,
                grad_accum_steps=args.grad_accum_steps,
                steps_per_epoch=steps_per_epoch,
            )
            loss_keys = st.get("loss_keys", None)
            teacher_val = bool(st.get("teacher_val", True))
            if rank_state is not None:
                run_loss_sums = dict(rank_state.get("run_loss_sums", {}))
                run_lr_sum = float(rank_state.get("run_lr_sum", 0.0))
                run_count = int(rank_state.get("run_count", 0))
            if is_main(rank):
                print(
                    f"[resume] trainer_state loaded: global_step={global_step} "
                    f"epoch={epoch} step_in_epoch={step_in_epoch}"
                )
                if is_legacy:
                    print(
                        "[resume] WARNING: legacy pre-optimizer checkpoint detected; "
                        f"rewound {args.grad_accum_steps} micro-batch(es) so the unapplied "
                        "accumulation is replayed. Exact legacy RNG continuation is unavailable."
                    )
        else:
            raise FileNotFoundError(
                f"Resume checkpoint has no trainer_state.pt: {resume_dir}. "
                "Use --init_ckpt for an intentional weights-only fresh run."
            )

        # Jump directly to the right position in the sampler — no skip loop needed
        if sampler_train is not None and isinstance(sampler_train, ResumeDistributedSampler):
            sampler_train.set_start_batch(step_in_epoch, args.batch_size_per_gpu)
        else:
            resume_skip_batches = step_in_epoch  # fallback when sampler cursor restore is unavailable
    effective_max_steps = args.max_steps
    if args.early_stop_step is not None:
        effective_max_steps = min(args.max_steps, args.early_stop_step)

    def _save_training_checkpoint(ckpt_dir: str, checkpoint_epoch: int, cursor: int) -> None:
        """Save a clean post-step checkpoint and preserve every rank's RNG stream."""

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        # ZeRO-1 owns optimizer shards on every rank. Consolidation is collective
        # and must finish before rank 0 serializes the full optimizer state.
        if args.use_zero1 and hasattr(opt, "consolidate_state_dict"):
            opt.consolidate_state_dict(to=0)

        local_rank_state = capture_rng_state(device)
        local_rank_state.update(
            {
                "run_loss_sums": dict(run_loss_sums),
                "run_lr_sum": float(run_lr_sum),
                "run_count": int(run_count),
            }
        )
        rank_states = gather_rank_states(local_rank_state, world_size)

        if is_main(rank):
            print("=" * 80)
            print(
                f"saving checkpoint: epoch={checkpoint_epoch} step={global_step} "
                f"step_in_epoch={cursor} dir={ckpt_dir}"
            )
            base_model = ddp_model.module if world_size > 1 else ddp_model
            save_checkpoint(
                ckpt_dir,
                base_model=base_model,
                optimizer=opt,
                lr_scheduler=lr_scheduler,
                global_step=global_step,
                epoch=checkpoint_epoch,
                step_in_epoch=cursor,
                loss_keys=loss_keys,
                teacher_val=teacher_val,
                rank_states=rank_states,
                resume_metadata=resume_metadata,
                args=args,
            )
            print("Checkpoint saved")
            print("=" * 80)

        if world_size > 1:
            dist.barrier(device_ids=[local_rank])

        # Serialization and object collectives should not perturb the random
        # sequence relative to a run with checkpointing disabled.
        restore_rng_state(local_rank_state, device)

    # main loop
    resumed_target = ckpts_root / f"step_{global_step}"
    last_saved_position: tuple[int, int, int] | None = (
        (global_step, epoch, step_in_epoch)
        if args.resume_ckpt is not None and resumed_target.exists()
        else None
    )
    stop_training = False
    while global_step < effective_max_steps:
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)

        iter_in_epoch = step_in_epoch  # only meaningful when resuming mid-epoch; otherwise 0
        _accum_micro = 0
        opt.zero_grad(set_to_none=True)

        train_iterator = iter_dataloader_without_advancing_model_rng(loader_train)
        for it, batch in enumerate(train_iterator):
            if global_step >= effective_max_steps:
                stop_training = True
                break

            # Skip already-consumed batches only for the first resumed epoch
            if resume_skip_batches > 0:
                resume_skip_batches -= 1
                if is_main(rank) and resume_skip_batches % 100 == 0 and resume_skip_batches > 0:
                    print(f"[resume] skipping batches... {resume_skip_batches} remaining")
                continue

            # ---- Training code ----
            inner_batch = batch
            ego_history_xyz = inner_batch["ego_history_xyz"].to(device, non_blocking=True)
            ego_history_rot = inner_batch["ego_history_rot"].to(device, non_blocking=True)
            ego_future_xyz  = inner_batch["ego_future_xyz"].to(device, non_blocking=True)
            ego_future_rot  = inner_batch["ego_future_rot"].to(device, non_blocking=True)
            base_model = ddp_model.module if world_size > 1 else ddp_model
            batched_tokenized_data = base_model._get_generation_mode_tokenized_data_online_distill(inner_batch, helper, processor, device)
            teacher_batched_tokenized_data = base_model._get_generation_mode_tokenized_data_online_distill(inner_batch, helper, teacher_processor, device)
            # Derived from global_step alone, so every rank picks the same regime
            # without communicating and a resumed run replays the same schedule.
            step_on_policy = on_policy_cfg.is_on_policy_step(global_step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = ddp_model(
                    tokenized_data=batched_tokenized_data,
                    ego_history_xyz=ego_history_xyz,
                    ego_history_rot=ego_history_rot,
                    ego_future_xyz=ego_future_xyz,
                    ego_future_rot=ego_future_rot,
                    labels_mask=None,
                    teacher_tokenized_data=teacher_batched_tokenized_data,
                    on_policy=step_on_policy,
                )
            # set/sync loss keys once
            if loss_keys is None:
                ld0 = out.loss_dict if hasattr(out, "loss_dict") else None
                if ld0 is None:
                    ld0 = {"loss": out.loss}
                loss_keys = _sync_loss_keys(ld0, rank=rank, world_size=world_size)

            ld = out.loss_dict if hasattr(out, "loss_dict") else None
            if ld is None:
                ld = {"loss": out.loss}
            ld_f = _loss_dict_to_float_map(ld, loss_keys)

            # Exactly one of distill_loss / on_policy_kl is non-zero per step,
            # so composition adds the active regime's term without double-counting.
            weighted_loss = combine_training_loss(
                out.loss,
                out.loss_dict if hasattr(out, "loss_dict") else None,
                future_traj_weight=args.future_traj_weight,
                future_traj_vlm_weight=args.future_traj_vlm_weight,
                distill_weight=args.distill_weight,
                vlm_loss_weight=args.vlm_loss_weight,
                on_policy_kl_weight=args.on_policy_kl_weight,
                hidden_distill_weight=args.hidden_distill_weight,
            )
            (weighted_loss / args.grad_accum_steps).backward()
            _accum_micro += 1
            iter_in_epoch += 1

            # accumulate loss stats from every micro-batch (divide by accum steps for correct mean)
            for k in loss_keys:
                run_loss_sums[k] = run_loss_sums.get(k, 0.0) + ld_f[k] / args.grad_accum_steps

            # optimizer step only every grad_accum_steps micro-batches
            if _accum_micro % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip)
                opt.step()
                lr_scheduler.step()
                opt.zero_grad(set_to_none=True)

                global_step += 1

                # interval stats
                lr0 = float(opt.param_groups[0]["lr"])
                run_lr_sum += lr0
                run_count += 1

                # ---- log ----
                if global_step % args.log_every == 0:
                    if world_size > 1:
                        packed = torch.tensor(
                            [run_loss_sums.get(k, 0.0) for k in loss_keys] + [run_lr_sum, float(run_count)],
                            device=device,
                            dtype=torch.float64,
                        )
                        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                        packed_list = packed.tolist()
                    else:
                        packed_list = [run_loss_sums.get(k, 0.0) for k in loss_keys] + [run_lr_sum, float(run_count)]

                    g_loss_sums = packed_list[: len(loss_keys)]
                    lr_sum, count = packed_list[len(loss_keys):]

                    loss_avgs = {k: (g_loss_sums[i] / max(1.0, count)) for i, k in enumerate(loss_keys)}
                    avg_lr = lr_sum / max(1.0, count)

                    if is_main(rank):
                        window_secs = time.time() - run_window_start
                        secs_per_step = window_secs / max(1, args.log_every)
                        msg_losses = _format_loss_kv(loss_avgs, loss_keys, precision=6)
                        print(
                            f"epoch={epoch} step={global_step} "
                            f"avg_losses[{args.log_every} steps]={msg_losses} "
                            f"avg_lr[{args.log_every} steps]={avg_lr:.6e} "
                            f"global_samples_in_window={int(round(count))} "
                            f"secs_per_step={secs_per_step:.1f}"
                        )

                    run_window_start = time.time()
                    run_loss_sums.clear()
                    run_lr_sum = 0.0
                    run_count = 0

                # ---- validation ----
                if (args.val_every is not None) and (global_step % args.val_every == 0):
                    # Teacher minADE is intentionally computed once per run. Keep
                    # this state in the checkpoint so resume follows the same path.
                    do_teacher = args.compute_minade_teacher and teacher_val
                    if world_size > 1:
                        dist.barrier(device_ids=[local_rank])

                    val_start = time.time()
                    val_loss, val_minade, val_greedy_minade, val_teacher_minade, val_loss_dict, val_loss_keys = run_validation_fast_distributed(
                        ddp_model=ddp_model,
                        loader_val=loader_val,
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        processor=processor,
                        compute_minade=args.compute_minade,
                        compute_teacher_minade=do_teacher,
                        prefix=f"[val step={global_step}]",
                        log_every_val_batches=args.log_every,
                        teacher_processor=teacher_processor,
                        expected_samples=len(ds_val),
                    )

                    if is_main(rank):
                        val_secs = time.time() - val_start
                        msg_val_losses = _format_loss_kv(val_loss_dict, val_loss_keys, precision=6)
                        if val_minade is None:
                            print(f"[val] step={global_step} losses={msg_val_losses} val_secs={val_secs:.1f}")
                        else:
                            if val_teacher_minade is None:
                                print(
                                    f"[val] step={global_step} losses={msg_val_losses} "
                                    f"final_minADE(student)={val_minade:.4f}m "
                                    f"final_greedy_minADE(student)={val_greedy_minade:.4f}m "
                                    f"val_secs={val_secs:.1f}"
                                )
                            else:
                                print(
                                    f"[val] step={global_step} losses={msg_val_losses} "
                                    f"final_minADE(student)={val_minade:.4f}m "
                                    f"final_greedy_minADE(student)={val_greedy_minade:.4f}m "
                                    f"final_minADE(teacher)={val_teacher_minade:.4f}m "
                                    f"val_secs={val_secs:.1f}"
                                )

                    if world_size > 1:
                        dist.barrier(device_ids=[local_rank])
                    if do_teacher:
                        teacher_val = False

                # Checkpoint only after the optimizer/scheduler update, logging,
                # and scheduled validation have all completed for this step.
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = str(ckpts_root / f"step_{global_step}")
                    _save_training_checkpoint(ckpt_dir, epoch, iter_in_epoch)
                    last_saved_position = (global_step, epoch, iter_in_epoch)

                if global_step >= effective_max_steps:
                    stop_training = True
                    if (
                        args.early_stop_step is not None
                        and global_step >= args.early_stop_step
                        and is_main(rank)
                    ):
                        print(f"[early_stop] reached step {global_step}, stopping.")
                    break

        # Preserve a mid-epoch cursor when max_steps/early-stop ends the loop.
        # Only advance the epoch after the DataLoader is naturally exhausted.
        step_in_epoch = iter_in_epoch
        if stop_training or global_step >= effective_max_steps:
            break

        # epoch boundary
        epoch += 1
        step_in_epoch = 0
        resume_skip_batches = 0  # only skip for the first resumed epoch
        if sampler_train is not None and isinstance(sampler_train, ResumeDistributedSampler):
            sampler_train.set_start_batch(0, args.batch_size_per_gpu)

    # ----- final save -----
    final_position = (global_step, epoch, step_in_epoch)
    if last_saved_position != final_position:
        final_dir = str(ckpts_root / f"step_{global_step}")
        _save_training_checkpoint(final_dir, epoch, step_in_epoch)
    elif is_main(rank):
        print(
            f"[checkpoint] step {global_step} was already saved at the current cursor; "
            "skipping duplicate final save."
        )

    if world_size > 1:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
