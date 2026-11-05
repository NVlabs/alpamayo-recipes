#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRaFT-style direct ADE backprop through the product Euler sampler (stage 3).

Freeze the VLM, differentiate
    Euler-N sample -> action_to_traj -> soft-min ADE vs GT
directly w.r.t. the expert head (expert.* / action_in_proj.* /
action_out_proj.*). Scalar-reward RL channels discard the per-waypoint
error structure; direct ADE gradients carry it in full.

Parity strategy — reuse the validated inference path verbatim instead of
re-implementing:
  * model load + prompt flavor      : same AlpamayoR1.from_pretrained path
  * conditioning (vlm_stage)        : imported from conditioning.py (KV cache
                                      is computed by frozen VLM weights ->
                                      CONSTANT w.r.t. the head; precomputed
                                      once per task and reused all steps)
  * Euler loop                      : line-for-line copy of FlowMatching._euler
                                      minus @torch.no_grad (self-check:
                                      --parity-check bit-compares vs
                                      model.diffusion.sample on a real ctx)
  * step_fn / action_to_traj        : copy of eval head_stage, with the KV
                                      cache REBUILT per call from detached
                                      prompt tensors (the eval crop() trick
                                      would leak autograd graphs across steps)

Learnability-check protocol:
  * train pool: 32-64 TRAIN clips (s3_v2), fixed t0 5.1s, CoC fixed per task
  * metrics every --probe-every steps:
      train/fixed  : ADE under a FIXED probe noise  (improvement = can memorize)
      train/fresh  : ADE under FRESH noise          (improvement = real signal)
      heldout/fixed: ADE on held-out clips          (improvement = generalizes)
  * gate for a full arm: train/fresh AND heldout both improve.

Launch (8 GPUs; see the package README for the validated recipe):
  export ALPAMAYO_DATASET_DIR=/path/to/physical_ai_av_mirror
  torchrun --nproc_per_node 8 draft/train_draft_ade.py \
      --init-ckpt /path/to/stage2_expert_sft_ckpt \
      --train-clips ./clip_lists/train128.txt \
      --heldout-clips ./clip_lists/heldout16.txt \
      --out-dir ./outputs/draft_run
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import conditioning as EF   # ClipDataset, install_offline_avdi, create_message_r15
EHS = EF                    # vlm_stage, HEAD_PREFIXES (same module)

from alpamayo_r1 import helper             # noqa: E402

try:
    from transformers import DynamicCache
except ImportError:  # transformers moved it around across versions
    from transformers.cache_utils import DynamicCache



# --------------------------------------------------------------- distributed
def dist_init():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    return rank, world, local


def p0(rank, *a, **k):
    if rank == 0:
        print(*a, **k, flush=True)


# ------------------------------------------------------- grad-enabled sampler
def build_cache(legacy_kv):
    """Fresh DynamicCache over the (constant, detached) prompt KV tensors.

    Rebuilt per step_fn call: reusing one cache across calls (the eval crop()
    pattern) would thread each call's autograd graph into the next via the
    cropped views and blow up on backward."""
    try:
        return DynamicCache.from_legacy_cache(legacy_kv)
    except Exception:
        cache = DynamicCache()
        for i, (k, v) in enumerate(legacy_kv):
            cache.update(k, v, i)
        return cache


def make_step_fn(model, ctx):
    """head_stage step_fn from the eval harness, cache-rebuilding variant."""
    n_tok = ctx["n_diffusion_tokens"]

    def step_fn(x, t):
        b_star = x.shape[0]
        emb = model.action_in_proj(x, t)
        if emb.dim() == 2:
            emb = emb.view(b_star, n_tok, -1)
        cache = build_cache(ctx["legacy_kv"])
        out = model.expert(
            inputs_embeds=emb,
            position_ids=ctx["position_ids"],
            past_key_values=cache,
            attention_mask=ctx["attention_mask"],
            use_cache=True,
            **ctx["forward_kwargs"],
        )
        h = out.last_hidden_state[:, -n_tok:]
        return model.action_out_proj(h).view(
            -1, *model.action_space.get_action_space_dims())

    return step_fn


def seed_rollout(seed):
    """Product parity: the pkgs alpamayo_r1 FlowMatching._euler draws its
    initial noise with plain torch.randn from the GLOBAL RNG stream (it takes
    no generator arg at all) — eval controls it by capturing/restoring global
    RNG state, so this experiment seeds the global stream too."""
    seed = int(seed) % 2**31
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


USE_EULER_CKPT = False   # set in main() from --euler-checkpoint
BACKPROP_LAST_K = 0      # set in main() from --backprop-last-k (0 = full chain)


def euler_sample_grad(model, ctx, inference_step, device):
    """Line-for-line copy of the pkgs alpamayo_r1 FlowMatching._euler
    (verified source 2026-10-05: global-stream torch.randn, linspace 0->1,
    x += dt*v) WITHOUT @torch.no_grad. Call seed_rollout() first.

    With USE_EULER_CKPT, each euler step runs under gradient checkpointing:
    only the inter-step x is stored; step_fn activations are recomputed in
    backward. step_fn is eval-mode (no dropout, consumes no RNG), so the
    recomputation is bit-identical — same loss/gradients, ~1/10 peak VRAM,
    ~+30% step time."""
    bk = ctx["B"] * ctx["k"]
    x_dims = model.diffusion.x_dims
    x = torch.randn(bk, *x_dims, device=device)
    timesteps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
    n_dim = len(x_dims)
    step_fn = make_step_fn(model, ctx)
    use_ckpt = USE_EULER_CKPT and torch.is_grad_enabled()
    if use_ckpt:
        from torch.utils.checkpoint import checkpoint
    for i in range(inference_step):
        if (BACKPROP_LAST_K and torch.is_grad_enabled()
                and i == inference_step - BACKPROP_LAST_K):
            # truncated DRaFT (arXiv 2309.17400): full forward, grads only
            # through the last K euler steps
            x = x.detach()
        dt = (timesteps[i + 1] - timesteps[i]).view(1, *[1] * n_dim).expand(bk, *[1] * n_dim)
        t_start = timesteps[i].view(1, *[1] * n_dim).expand(bk, *[1] * n_dim)
        if use_ckpt:
            v = checkpoint(lambda xx, tt: step_fn(x=xx, t=tt), x, t_start,
                           use_reentrant=False)
        else:
            v = step_fn(x=x, t=t_start)
        x = x + dt * v
    return x


def get_live_ctx(ctx, device):
    """ctx with KV on the compute device. With --ctx-device cpu the resident
    copy lives in host RAM (~2.6GB/task for k=6 x ~4k tokens x 28 layers) and
    is moved here per rollout; the moved copy is freed when it goes out of
    scope."""
    if ctx["legacy_kv"][0][0].device.type == device.type:
        return ctx
    live = dict(ctx)
    live["legacy_kv"] = tuple(
        (k.to(device, non_blocking=True), v.to(device, non_blocking=True))
        for k, v in ctx["legacy_kv"])
    return live


def iter_prefetched(tasks, device):
    """Yield (ctx, live_kv): while task i computes, task i+1's pinned KV is
    copied H2D on a side stream (double buffering; +1 task KV of VRAM)."""
    if not tasks:
        return
    if tasks[0]["legacy_kv"][0][0].device.type == device.type:
        for ctx in tasks:
            yield ctx, ctx["legacy_kv"]
        return
    stream = torch.cuda.Stream(device=device)

    def start(ctx):
        with torch.cuda.stream(stream):
            kv = tuple((k.to(device, non_blocking=True),
                        v.to(device, non_blocking=True))
                       for k, v in ctx["legacy_kv"])
        ev = torch.cuda.Event()
        ev.record(stream)
        return kv, ev

    nxt = start(tasks[0])
    for i, ctx in enumerate(tasks):
        kv, ev = nxt
        if i + 1 < len(tasks):
            nxt = start(tasks[i + 1])
        torch.cuda.current_stream(device).wait_event(ev)
        for k, v in kv:
            k.record_stream(torch.cuda.current_stream(device))
            v.record_stream(torch.cuda.current_stream(device))
        yield ctx, kv


def rollout_pred_xy(model, ctx, inference_step, device, live_kv=None):
    """sample -> action_to_traj -> (k, T, 2) xy, fp32 (grad-capable).
    Caller must seed_rollout() first (global-stream noise, product parity).
    live_kv: pre-transferred GPU KV from iter_prefetched (skips sync H2D)."""
    if live_kv is not None:
        ctx = dict(ctx)
        ctx["legacy_kv"] = live_kv
    else:
        ctx = get_live_ctx(ctx, device)
    action = euler_sample_grad(model, ctx, inference_step, device)
    k = ctx["k"]
    hist_xyz = ctx["hist_xyz_rep"]
    hist_rot = ctx["hist_rot_rep"]
    pred_xyz, _ = model.action_space.action_to_traj(action, hist_xyz, hist_rot)
    return pred_xyz.float().view(k, -1, 3)[..., :2]


def ade_per_sample(pred_xy, gt_xy):
    """(k,T,2) vs (T,2) -> (k,) mean L2 over time (eval parity: xy only)."""
    return torch.linalg.norm(pred_xy - gt_xy[None], dim=-1).mean(dim=-1)


def softmin_loss(ades, tau):
    return -tau * torch.logsumexp(-ades / tau, dim=0) + tau * float(np.log(len(ades)))


def flow_anchor_loss(model, ctx, live_kv):
    """Stage-2 SFT objective on this task's GT action, conditioned through the
    SAME cached KV: draw (t, noise), noisy = t*x+(1-t)*noise, expert predicts
    v, MSE against (x - noise). One expert forward (vs ~60 for the ADE term).
    Expanded to k rows so every CoC condition anchors the field."""
    x = ctx["gt_action"].unsqueeze(0).expand(ctx["k"], *ctx["gt_action"].shape)
    x = x.contiguous().float()
    try:
        td = model.diffusion.construct_training_data(x)
        noisy, t, noise = td["noisy_x"], td["timesteps"], td["noise"]
    except AttributeError:
        tt = torch.distributions.Beta(
            torch.tensor(1.5, device=x.device),
            torch.tensor(1.0, device=x.device)).sample((x.shape[0],))
        tt = 0.999 - tt * 0.999
        while tt.dim() < x.dim():
            tt = tt.unsqueeze(-1)
        noise = torch.randn_like(x)
        noisy, t = tt * x + (1 - tt) * noise, tt
    live = dict(ctx)
    live["legacy_kv"] = live_kv
    step_fn = make_step_fn(model, live)
    pred = step_fn(x=noisy, t=t)
    return torch.nn.functional.mse_loss(pred.float(), (x - noise).float())


# ----------------------------------------------------------- ctx precompute
def build_anchor_targets(model, ctx, inference_step, device, seed, n_states):
    """Run the (still-initial) weights' own Euler10 on this replay task with a
    fixed noise, record (x_t, t, v_ref) at n_states sampled steps. MUST be
    called BEFORE the first optimizer step (reference = init function)."""
    import random as _rnd
    seed_rollout(seed)
    live = get_live_ctx(ctx, device)
    step_fn = make_step_fn(model, live)
    bk = ctx["B"] * ctx["k"]
    x_dims = model.diffusion.x_dims
    picks = set(_rnd.Random(seed).sample(range(inference_step), n_states))
    states = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.randn(bk, *x_dims, device=device)
        timesteps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(x_dims)
        for i in range(inference_step):
            dt = (timesteps[i + 1] - timesteps[i]).view(1, *[1] * n_dim).expand(bk, *[1] * n_dim)
            t_start = timesteps[i].view(1, *[1] * n_dim).expand(bk, *[1] * n_dim)
            v = step_fn(x=x, t=t_start)
            if i in picks:
                states.append((x.detach().float().cpu(),
                               t_start.detach().float().cpu(),
                               v.detach().float().cpu()))
            x = x + dt * v
    ctx["anchor_states"] = states


def anchor_loss_for(model, rctx, device):
    """MSE between the CURRENT expert's velocity and the stored init-reference
    velocity at the frozen (x_t, t) states, same conditioning KV."""
    live = get_live_ctx(rctx, device)
    step_fn = make_step_fn(model, live)
    losses = []
    for x_s, t_s, v_ref in rctx["anchor_states"]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = step_fn(x=x_s.to(device), t=t_s.to(device))
        losses.append(torch.nn.functional.mse_loss(
            pred.float(), v_ref.to(device)))
    return torch.stack(losses).mean()


def _conditioning_from_inputs(model, model_inputs, eff, args, k_override=None):
    """vlm_stage -> head-stage conditioning fields. Called at startup AND on
    every CoC refresh (--coc-refresh-every): new seed -> new sampled CoC ->
    new KV. model_inputs must be a fresh (deep-copied, on-device) dict —
    vlm_stage pops input_ids from tokenized_data."""
    torch.manual_seed(eff)
    torch.cuda.manual_seed_all(eff)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = EHS.vlm_stage(model, model_inputs, args.top_p, args.temperature,
                            k_override or args.num_traj_samples,
                            args.max_generation_length)

    cache = ctx["prompt_cache"]
    legacy = []
    kv_bytes = 0
    for layer in range(len(cache)):
        k, v = cache[layer]
        k = k.detach().to(args.ctx_device)
        v = v.detach().to(args.ctx_device)
        if args.ctx_device == "cpu":
            # page-locked host memory: required for truly async H2D (pageable
            # non_blocking copies silently degrade to synchronous)
            k = k.pin_memory()
            v = v.pin_memory()
        kv_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        legacy.append((k, v))

    import einops
    hist_xyz_rep = einops.repeat(
        ctx["ego_history_xyz"][:, -1], "b ... -> (b n) ...", n=ctx["k"]).detach()
    hist_rot_rep = einops.repeat(
        ctx["ego_history_rot"][:, -1], "b ... -> (b n) ...", n=ctx["k"]).detach()
    return {
        "legacy_kv": tuple(legacy),
        "position_ids": ctx["position_ids"].detach(),
        "attention_mask": ctx["attention_mask"].detach(),
        "forward_kwargs": ctx["forward_kwargs"],
        "n_diffusion_tokens": ctx["n_diffusion_tokens"],
        "B": ctx["B"],
        "k": ctx["k"],
        "hist_xyz_rep": hist_xyz_rep,
        "hist_rot_rep": hist_rot_rep,
        "n_miss_fs": ctx["n_miss_fs"],
        "kv_gb": kv_bytes / 1e9,
    }


def precompute_ctx(model, processor, clip_id, data, args, device, k_override=None,
                   seed_override=None):
    """One task's conditioning + GT fields. With --coc-refresh-every the raw
    (CPU) model inputs are kept on the ctx so conditioning can be re-sampled
    mid-training."""
    frames_flat = data["image_frames"].flatten(0, 1)
    messages = EF.create_message_r15(frames_flat, data["camera_indices"])
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    inputs_cpu = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"]}
    model_inputs = helper.to_device(copy.deepcopy(inputs_cpu), device)

    eff = (seed_override if seed_override is not None
           else (zlib.crc32(clip_id.encode()) + args.seed) % 2**31)
    cond = _conditioning_from_inputs(model, model_inputs, eff, args, k_override)
    out = {
        "clip_id": clip_id,
        "gt_xy": data["ego_future_xyz"][0, 0, :, :2].float().to(device),
        "gt_action": model.action_space.traj_to_action(
            data["ego_history_xyz"][:, -1].to(device),
            data["ego_history_rot"][:, -1].to(device),
            data["ego_future_xyz"][:, -1].to(device),
            data["ego_future_rot"][:, -1].to(device),
        ).squeeze(0).detach().float(),
        "probe_seed": eff ^ 0x5EED,
    }
    out.update(cond)
    if getattr(args, "coc_refresh_every", 0):
        out["model_inputs_cpu"] = inputs_cpu
    return out


class StreamFeeder:
    """Endless stream over a big clip list: per-epoch shuffle, rank-sharded,
    raw samples prefetched by DataLoader workers; ctx built transiently on
    GPU with an epoch-dependent CoC seed (fresh conditioning every visit)."""

    def __init__(self, model, processor, clip_ids, args, device, rank, world):
        self.model, self.processor = model, processor
        self.ids = list(clip_ids)
        self.args, self.device = args, device
        self.rank, self.world = rank, world
        self.sargs = copy.copy(args)
        self.sargs.ctx_device = "cuda"          # transient ctx: no pinning
        self.sargs.coc_refresh_every = 0        # no model_inputs retention
        self.epoch = -1
        self._iter = None
        self.n_err = 0
        self.avdi = EF.install_offline_avdi(args.pai_root)

    def _new_epoch(self):
        import random as _rnd
        self.epoch += 1
        order = list(self.ids)
        _rnd.Random(self.args.seed * 31 + self.epoch).shuffle(order)
        mine = order[self.rank::self.world]
        from torch.utils.data import DataLoader
        ds = EF.ClipDataset(mine, self.args.t0_us, self.avdi, t0_map={})
        self._iter = iter(DataLoader(
            ds, batch_size=1, shuffle=False,
            num_workers=self.args.stream_workers,
            prefetch_factor=2 if self.args.stream_workers > 0 else None,
            persistent_workers=False, collate_fn=lambda b: b[0]))
        if self.rank == 0:
            print(f"[stream] epoch {self.epoch} begins "
                  f"({len(mine)} clips/rank)", flush=True)

    def next_ctx(self):
        while True:
            if self._iter is None:
                self._new_epoch()
            try:
                item = next(self._iter)
            except StopIteration:
                self._iter = None
                continue
            if "_err" in item:
                self.n_err += 1
                continue
            cid = item["clip_id"]
            # epoch-dependent CoC seed: fresh conditioning every visit
            eff = (zlib.crc32(cid.encode()) + self.args.seed
                   + self.epoch * 7919) % 2**31
            ctx = precompute_ctx(self.model, self.processor, cid,
                                 item["data"], self.sargs, self.device,
                                 seed_override=eff)
            del item
            return ctx


def load_tasks(model, processor, clip_ids, args, device, rank, tag, k_override=None):
    avdi = EF.install_offline_avdi(args.pai_root)
    ds = EF.ClipDataset(clip_ids, args.t0_us, avdi, t0_map={})
    out = []
    for idx in range(len(clip_ids)):
        item = ds[idx]
        if "_err" in item:
            print(f"[r{rank}] {tag} clip {item['clip_id']} LOAD FAIL: {item['_err']}",
                  flush=True)
            continue
        ctx = precompute_ctx(model, processor, item["clip_id"], item["data"],
                             args, device, k_override)
        if ctx["n_miss_fs"]:
            print(f"[r{rank}] {tag} {item['clip_id']}: {ctx['n_miss_fs']} rollouts "
                  f"missed traj_future_start (kept; matches eval behavior)", flush=True)
        out.append(ctx)
        del item
    tot_kv = sum(c["kv_gb"] for c in out)
    print(f"[r{rank}] {tag}: {len(out)}/{len(clip_ids)} ctx ready "
          f"(KV {tot_kv:.1f}GB on {args.ctx_device}; "
          f"cuda alloc {torch.cuda.memory_allocated() / 1e9:.1f}GB)", flush=True)
    return out


# ------------------------------------------------------------------- probes
@torch.no_grad()
def probe(model, tasks, inference_step, device, fresh_seed=None):
    """Mean over tasks of (minADE, meanADE). fresh_seed=None -> per-task fixed
    probe generator (low-variance tracking); int -> fresh-noise draw."""
    mins, means = [], []
    for ctx, kv in iter_prefetched(tasks, device):
        seed = ctx["probe_seed"] if fresh_seed is None else (ctx["probe_seed"] + fresh_seed)
        seed_rollout(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = rollout_pred_xy(model, ctx, inference_step, device, live_kv=kv)
        ades = ade_per_sample(pred, ctx["gt_xy"])
        mins.append(ades.min().item())
        means.append(ades.mean().item())
    return float(np.mean(mins)), float(np.mean(means))


def reduce_mean(vals, world):
    t = torch.tensor(vals, dtype=torch.float64, device="cuda")
    if world > 1:
        dist.all_reduce(t)
        t /= world
    return [v.item() for v in t]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--train-clips", default=None,
                    help="resident pool mode (required unless --stream-clips)")
    ap.add_argument("--heldout-clips", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--pai-root",
                    default=os.environ.get("ALPAMAYO_DATASET_DIR",
                                           "/path/to/physical_ai_av_mirror"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--num_traj_samples", type=int, default=6)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.2,
                    help="soft-min temperature (meters). ->0 hard min (winner-"
                         "only, MAX6 failure mode); ->inf mean ADE (collapse)")
    ap.add_argument("--loss", choices=["softmin", "mean", "min"], default="softmin")
    ap.add_argument("--fp32-master", action="store_true", default=True)
    ap.add_argument("--no-fp32-master", dest="fp32_master", action="store_false")
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--save-head-every", type=int, default=50)
    ap.add_argument("--save-full-at-end", action="store_true", default=True)
    ap.add_argument("--parity-check", action="store_true",
                    help="bit-compare grad-euler vs model.diffusion.sample on "
                         "ONE task (clip lists truncated to 1), then exit")
    ap.add_argument("--stream-clips", default=None,
                    help="STREAMING mode: clip list (e.g. full s3_v2) sampled "
                         "batch-by-batch; per visit the CoC is re-generated "
                         "with an epoch-dependent seed (condition diversity "
                         "for free); ctx is transient on GPU. Replaces the "
                         "resident --train-clips pool.")
    ap.add_argument("--stream-batch", type=int, default=8,
                    help="streaming tasks per rank per optimizer step")
    ap.add_argument("--stream-workers", type=int, default=4,
                    help="dataloader workers prefetching raw clips")
    ap.add_argument("--stream-probe-tasks", type=int, default=2,
                    help="resident in-stream probe tasks per rank (train/fixed "
                         "and train/fresh probes need a fixed set)")
    ap.add_argument("--replay-clips", default=None,
                    help="clip list for the init-reference velocity anchor: "
                         "broad-coverage replay scenes (1 CoC x 1 noise each). "
                         "Anchor targets are the INIT weights' own velocities "
                         "along their frozen Euler10 paths, captured at "
                         "startup (init-reference regularizer).")
    ap.add_argument("--anchor-beta", type=float, default=0.0,
                    help="weight of the reference-velocity MSE term")
    ap.add_argument("--replay-per-step", type=int, default=8,
                    help="replay tasks sampled per rank per step (round-robin)")
    ap.add_argument("--anchor-states", type=int, default=2,
                    help="intermediate euler states sampled per replay task")
    ap.add_argument("--grad-probe-every", type=int, default=80,
                    help="every N steps: measure the two terms' separate grad "
                         "norms + cosine (beta calibration evidence)")
    ap.add_argument("--coc-refresh-every", type=int, default=0,
                    help="re-sample every train task's CoC conditioning (new "
                         "seed -> new reasoning texts + KV) every N steps. "
                         "Fights condition-memorization; ~2s/task per refresh. "
                         "heldout tasks are NOT refreshed (probe comparability)."
                         " NOTE: train/fixed probe changes meaning after each "
                         "refresh (conditioning moved); trust train/fresh + "
                         "heldout.")
    ap.add_argument("--flow-lambda", type=float, default=0.0,
                    help="hybrid anchor: total = ADE-loss + lambda * flow-"
                         "matching MSE on the GT action (the stage-2 SFT "
                         "objective, reusing each task's cached KV). Pins the "
                         "velocity field to the SFT manifold to fight the "
                         "diversity collapse on the full PAI evaluation set "
                         "while ADE polishes gold.")
    ap.add_argument("--backprop-last-k", type=int, default=0,
                    help="truncated DRaFT: full Euler10 forward, detach before "
                         "the last K steps, backprop only through them (0=full)")
    ap.add_argument("--euler-checkpoint", action="store_true",
                    help="gradient-checkpoint each euler step: bit-identical "
                         "loss/grads, ~1/10 peak VRAM, ~+30%% step time")
    ap.add_argument("--ctx-device", choices=["cuda", "cpu"], default="cuda",
                    help="where the per-task prompt-KV ctx lives (~2.6GB/task "
                         "at k=6). cpu = host-resident, moved to GPU per "
                         "rollout (slower, scales to big pools)")
    args = ap.parse_args()

    global USE_EULER_CKPT, BACKPROP_LAST_K
    USE_EULER_CKPT = bool(args.euler_checkpoint)
    BACKPROP_LAST_K = int(args.backprop_last_k)
    rank, world, local = dist_init()
    device = torch.device(f"cuda:{local}")
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(vars(args), open(out_dir / "train_args.json", "w"), indent=2)

    import alpamayo_r1.models.alpamayo_r1 as M
    model = M.AlpamayoR1.from_pretrained(
        args.init_ckpt, dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    processor = helper.get_processor(model.tokenizer)

    diff = model.diffusion
    assert not getattr(diff, "use_classifier_free_guidance", False), \
        "CFG model needs the guided branch ported into euler_sample_grad"
    inference_step = int(getattr(diff, "num_inference_steps", 10))
    int_method = getattr(diff, "int_method", "euler")
    assert int_method == "euler", f"product sampler expected euler, got {int_method}"
    p0(rank, f"[draft-ade] sampler: euler x {inference_step} (product config)")

    # trainable = head only; VLM stays frozen bf16
    head_params, n_frozen = [], 0
    for name, p in model.named_parameters():
        if name.startswith(EHS.HEAD_PREFIXES):
            p.requires_grad_(True)
            if args.fp32_master:
                p.data = p.data.float()
            head_params.append(p)
        else:
            p.requires_grad_(False)
            n_frozen += 1
    p0(rank, f"[draft-ade] trainable head tensors={len(head_params)} "
             f"(fp32_master={args.fp32_master}), frozen={n_frozen}")

    opt = torch.optim.AdamW(head_params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    # OOM-airbag grad snapshot buffers (~0.5GB fp32 on-device)
    grad_bufs = [torch.zeros_like(p) for p in head_params]

    def shard(path):
        ids = [l.strip() for l in open(path) if l.strip()]
        return ids[rank::world]

    assert args.train_clips or args.stream_clips, \
        "need --train-clips (resident pool) or --stream-clips (streaming)"
    feeder = None
    if args.stream_clips:
        all_stream = [l.strip() for l in open(args.stream_clips) if l.strip()]
        held_set = set(l.strip() for l in open(args.heldout_clips) if l.strip())
        stream_ids = [c for c in all_stream if c not in held_set]
        p0(rank, f"[stream] {len(stream_ids)} clips in stream "
                 f"(heldout excluded), batch={args.stream_batch}/rank, "
                 f"{len(stream_ids)//(world*args.stream_batch)} steps/epoch")
        import random as _rnd
        probe_ids = _rnd.Random(args.seed * 17).sample(
            stream_ids, args.stream_probe_tasks * world)
        train_ids = probe_ids[rank::world]   # resident probe set only
    else:
        train_ids = shard(args.train_clips)
    held_ids = shard(args.heldout_clips)
    if args.parity_check:
        # parity needs exactly one ctx — do NOT load the full pool (a full
        # single-GPU load is ~80 x 2.6GB KV and OOMs by design, not by leak)
        train_ids, held_ids = train_ids[:1], []
    train_tasks = load_tasks(model, processor, train_ids, args,
                             device, rank, "train")
    held_tasks = load_tasks(model, processor, held_ids, args,
                            device, rank, "heldout")
    assert train_tasks, f"rank {rank}: empty train shard"

    replay_tasks = []
    if args.replay_clips:
        assert args.anchor_beta > 0, "--replay-clips set but --anchor-beta 0"
        replay_tasks = load_tasks(model, processor, shard(args.replay_clips),
                                  args, device, rank, "replay", k_override=1)
        assert replay_tasks, f"rank {rank}: empty replay shard"
        t_a = time.time()
        for ri, rctx in enumerate(replay_tasks):
            build_anchor_targets(
                model, rctx, int(getattr(model.diffusion, "num_inference_steps", 10)),
                device, (rctx["probe_seed"] ^ 0xA17C) % 2**31, args.anchor_states)
        p0(rank, f"[anchor] {len(replay_tasks)} replay tasks x "
                 f"{args.anchor_states} states captured from INIT weights "
                 f"({time.time() - t_a:.0f}s) — reference frozen from here")

    if args.parity_check:
        ctx = get_live_ctx(train_tasks[0], device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            seed_rollout(123)
            a = euler_sample_grad(model, ctx, inference_step, device)
            seed_rollout(123)
            b = model.diffusion.sample(
                batch_size=ctx["B"] * ctx["k"], step_fn=make_step_fn(model, ctx),
                device=device, return_all_steps=False)
        same = torch.equal(a, b)
        amax = (a - b).abs().max().item()
        print(f"[r{rank}] PARITY grad-euler vs diffusion.sample: "
              f"bit-equal={same} max|diff|={amax:.3e}", flush=True)
        if world > 1:
            dist.barrier()
        return

    def run_probes(step):
        tf = probe(model, train_tasks, inference_step, device)
        tn = probe(model, train_tasks, inference_step, device, fresh_seed=1000 + step)
        hf = probe(model, held_tasks, inference_step, device) if held_tasks else (float("nan"),) * 2
        vals = reduce_mean([tf[0], tf[1], tn[0], tn[1], hf[0], hf[1]], world)
        p0(rank, f"[probe step={step}] train/fixed min={vals[0]:.4f} mean={vals[1]:.4f} | "
                 f"train/fresh min={vals[2]:.4f} mean={vals[3]:.4f} | "
                 f"heldout/fixed min={vals[4]:.4f} mean={vals[5]:.4f}")
        if rank == 0:
            with open(out_dir / "probes.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "train_fixed_min": vals[0],
                                    "train_fixed_mean": vals[1],
                                    "train_fresh_min": vals[2],
                                    "train_fresh_mean": vals[3],
                                    "heldout_fixed_min": vals[4],
                                    "heldout_fixed_mean": vals[5]}) + "\n")

    def save_head(step):
        if rank != 0:
            return
        from safetensors.torch import save_file
        sd = {n: p.detach().to(torch.bfloat16).cpu().contiguous()
              for n, p in model.named_parameters()
              if n.startswith(EHS.HEAD_PREFIXES)}
        d = out_dir / f"head_step{step}"
        d.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(d / "head.safetensors"))

    if args.stream_clips:
        feeder = StreamFeeder(model, processor, stream_ids, args, device,
                              rank, world)

    run_probes(0)
    save_head(0)
    p0(rank, f"[draft-ade] training: {args.steps} steps, loss={args.loss} "
             f"tau={args.tau}, lr={args.lr}, global tasks/step="
             f"{world}x{args.stream_batch if feeder is not None else len(train_tasks)}"
             + (f" STREAMING({len(stream_ids)} clips)" if feeder is not None else ""))

    for step in range(1, args.steps + 1):
        t0 = time.time()
        if (feeder is None and args.coc_refresh_every and step > 1
                and (step - 1) % args.coc_refresh_every == 0):
            t_r = time.time()
            for ctx in train_tasks:
                eff = (zlib.crc32(ctx["clip_id"].encode())
                       + args.seed + step * 7919) % 2**31
                mi = helper.to_device(copy.deepcopy(ctx["model_inputs_cpu"]), device)
                ctx.update(_conditioning_from_inputs(model, mi, eff, args))
                del mi
            torch.cuda.empty_cache()
            p0(rank, f"[refresh] step {step}: re-sampled CoC for "
                     f"{len(train_tasks)} tasks/rank in {time.time() - t_r:.1f}s")
        opt.zero_grad(set_to_none=True)
        loss_sum, min_sum, flow_sum = 0.0, 0.0, 0.0
        if feeder is not None:
            def _stream_iter():
                for _ in range(args.stream_batch):
                    c = feeder.next_ctx()
                    yield c, c["legacy_kv"]
            task_iter = _stream_iter()
            task_denom = args.stream_batch
        else:
            task_iter = iter_prefetched(train_tasks, device)
            task_denom = len(train_tasks)
        for ti, (ctx, kv) in enumerate(task_iter):
            task_seed = args.seed * 1_000_003 + step * 977 + rank * 131 + ti
            # grad snapshot: the OOM airbag below must be able to roll back a
            # partially-accumulated backward before retrying the task
            for buf, p in zip(grad_bufs, head_params):
                buf.copy_(p.grad) if p.grad is not None else buf.zero_()

            def run_task():
                seed_rollout(task_seed)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = rollout_pred_xy(model, ctx, inference_step, device,
                                           live_kv=kv)
                ades = ade_per_sample(pred, ctx["gt_xy"])
                if args.loss == "softmin":
                    loss = softmin_loss(ades, args.tau)
                elif args.loss == "mean":
                    loss = ades.mean()
                else:
                    loss = ades.min()
                fl = None
                if args.flow_lambda > 0:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        fl = flow_anchor_loss(model, ctx, kv)
                    total = loss + args.flow_lambda * fl
                else:
                    total = loss
                (total / task_denom).backward()
                return loss.item(), ades.min().item(), (fl.item() if fl is not None else 0.0)

            try:
                l_item, m_item, f_item = run_task()
            except torch.cuda.OutOfMemoryError:
                # airbag: roll back grads, free, retry THIS task checkpointed
                # (same seed -> same noise; recompute is bit-identical, only
                # slower). Training trajectory is unchanged.
                for buf, p in zip(grad_bufs, head_params):
                    if p.grad is not None:
                        p.grad.copy_(buf)
                torch.cuda.empty_cache()
                print(f"[r{rank}] OOM at step {step} task {ti} — retrying "
                      f"with euler checkpointing", flush=True)
                USE_EULER_CKPT = True
                try:
                    l_item, m_item, f_item = run_task()
                finally:
                    USE_EULER_CKPT = bool(args.euler_checkpoint)
            loss_sum += l_item
            min_sum += m_item
            flow_sum += f_item
        anchor_val = 0.0
        if replay_tasks and args.anchor_beta > 0:
            idxs = [(step * args.replay_per_step + j) % len(replay_tasks)
                    for j in range(args.replay_per_step)]
            for j in idxs:
                al = anchor_loss_for(model, replay_tasks[j], device)
                (args.anchor_beta * al / len(idxs)).backward()
                anchor_val += al.item() / len(idxs)
        if (replay_tasks and args.grad_probe_every
                and step % args.grad_probe_every == 0 and rank == 0):
            # separate-term gradient norms + cosine (beta calibration evidence)
            gprobe = [torch.zeros_like(pp) for pp in head_params]
            for buf, pp in zip(gprobe, head_params):
                if pp.grad is not None:
                    buf.copy_(pp.grad)
                    pp.grad.zero_()
            al = anchor_loss_for(model, replay_tasks[0], device)
            al.backward()
            ga = torch.cat([pp.grad.flatten() for pp in head_params
                            if pp.grad is not None])
            na = ga.norm().item()
            gm = torch.cat([buf.flatten() for buf in gprobe])
            cos = torch.nn.functional.cosine_similarity(
                gm, ga, dim=0).item() if na > 0 else float("nan")
            print(f"[probe step={step}] |g_main+anchor|={gm.norm().item():.3f} "
                  f"|g_anchor_1task|={na:.3f} cos={cos:+.3f} "
                  f"(anchor term unweighted)", flush=True)
            for buf, pp in zip(gprobe, head_params):
                if pp.grad is not None:
                    pp.grad.copy_(buf)
        if world > 1:
            for p in head_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
        gnorm = torch.nn.utils.clip_grad_norm_(head_params, args.grad_clip)
        opt.step()
        if rank == 0 and (step % 5 == 0 or step == 1):
            p0(rank, f"[step {step}] loss={loss_sum / task_denom:.4f} "
                     f"flow={flow_sum / task_denom:.4f} "
                     f"anchor={anchor_val:.4f} "
                     f"batch_minADE={min_sum / task_denom:.4f} "
                     f"gnorm={float(gnorm):.3f} secs={time.time() - t0:.2f}")
        if step % args.probe_every == 0:
            run_probes(step)
        if step % args.save_head_every == 0:
            save_head(step)
        if world > 1 and step % args.probe_every == 0:
            dist.barrier()

    save_head(args.steps)
    if args.save_full_at_end and rank == 0:
        if args.fp32_master:
            for p in head_params:
                p.data = p.data.to(torch.bfloat16)
        model.save_pretrained(out_dir / "full_final", safe_serialization=True)
        p0(rank, f"[draft-ade] full model saved -> {out_dir / 'full_final'}")
    if world > 1:
        dist.barrier()
    p0(rank, "[draft-ade] DONE")


if __name__ == "__main__":
    main()
