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

"""Frozen-target prefill and SpecForge draft training, one sample per GPU microstep."""

import json
import math
import os
import time
from contextlib import ExitStack, nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

from .cache import load_cache
from .contracts import (
    accumulation_steps,
    check_resume_settings,
    check_training_output,
    load_samples,
    mixture_weight,
    sample_key,
    training_samples,
)
from .data import InputData
from .draft import DRAFT_CLASSES, build_engine, make_config
from .runtime import Runtime
from .target import Target


def save_checkpoint(runtime, draft, optimizer, scheduler, step, run_info):
    """Save the draft and optimizer state; never save the frozen target."""
    pending = runtime.output / f"step_{step}.tmp"
    final = runtime.output / f"step_{step}"
    if runtime.rank == 0:
        if final.exists():
            raise FileExistsError(final)
        pending.mkdir(exist_ok=True)
        draft.save_pretrained(pending, safe_serialization=True, max_shard_size="100GB")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            pending / "training_state.pt",
        )
        (pending / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
        os.rename(pending, final)
        (runtime.output / "latest.json").write_text(
            json.dumps({"checkpoint": final.name, "step": step}) + "\n"
        )
        print(f"Saved draft checkpoint: {final}", flush=True)
    runtime.barrier()


class SampleOrder:
    """A reproducible shuffled stream indexed by the absolute sample offset."""

    def __init__(self, size, seed):
        self.size, self.seed = size, seed
        self.epoch, self.order = None, None

    def __getitem__(self, offset):
        epoch, index = divmod(offset, self.size)
        if epoch != self.epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.order = torch.randperm(self.size, generator=generator).tolist()
            self.epoch = epoch
        return self.order[index]


def run(args):
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    samples = load_samples(args.manifest, args.limit, source=args.data_source)
    metadata, cache = load_cache(args.cache)
    requested_count = len(samples)
    samples = training_samples(samples, cache)
    with Runtime(args.output, args.seed) as runtime, ExitStack() as logging:
        start = check_training_output(runtime.output, args.resume)
        if start > args.steps:
            raise ValueError("Resume step exceeds --steps")
        excluded = requested_count - len(samples)
        if runtime.rank == 0:
            print(
                f"Training selection: {len(samples)} complete CoCs; excluded {excluded}", flush=True
            )
        data = InputData(args)
        if data.contract != metadata["data"]:
            raise ValueError("Training data source differs from the cache")
        accumulation = accumulation_steps(args.global_batch_size, runtime.world_size)
        target = Target(args, runtime)
        if target.contract != metadata["target"]:
            raise ValueError("Training target differs from the cache")
        run_info = {
            "target": target.contract,
            "data": data.contract,
            "method": args.method,
            "steps": args.steps,
            "global_batch_size": args.global_batch_size,
            "world_size": runtime.world_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "samples": len(samples),
            "incomplete_exclusions": excluded,
        }
        if args.resume:
            previous = json.loads((args.resume / "run.json").read_text())
            check_resume_settings(previous, run_info)
        if runtime.rank == 0:
            (runtime.output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
        draft = (
            DRAFT_CLASSES[args.method](make_config(args, target)).to(runtime.device).float().train()
        )
        engine = build_engine(draft, target, args.num_anchors)
        model = engine
        if runtime.world_size > 1:
            model = DistributedDataParallel(
                engine, device_ids=[runtime.local_rank], broadcast_buffers=False
            )
        optimizer = torch.optim.AdamW(draft.parameters(), lr=args.learning_rate, weight_decay=0.0)
        warmup = max(1, int(args.steps * 0.04))

        def lr_scale(step):
            if step < warmup:
                return (step + 1) / warmup
            fraction = (step - warmup) / max(1, args.steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, fraction)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        if args.resume:
            from safetensors.torch import load_file

            # Only resume trusted checkpoints: optimizer state is a pickle.
            state = torch.load(
                args.resume / "training_state.pt", map_location="cpu", weights_only=False
            )
            if state["step"] != start:
                raise ValueError("Checkpoint step does not match its directory name")
            draft.load_state_dict(
                load_file(args.resume / "model.safetensors", device=str(runtime.device))
            )
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        wandb = None
        if runtime.rank == 0:
            metrics_file = logging.enter_context((runtime.output / "metrics.jsonl").open("a"))
            if args.wandb_project:
                import wandb

                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or runtime.output.resolve().name,
                    config=run_info,
                    dir=str(runtime.output),
                )
                logging.callback(wandb.finish)
        order = SampleOrder(len(samples), args.seed)
        started = time.monotonic()
        for step in range(start, args.steps):
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            weight = mixture_weight(args.method, step, args.steps)
            schedule = {"lambda_base" if args.method == "domino" else "selector_loss_alpha": weight}
            totals = torch.zeros(3, device=runtime.device, dtype=torch.float64)
            for micro in range(accumulation):
                offset = step * args.global_batch_size + micro * runtime.world_size + runtime.rank
                # SpecForge samples anchors with torch.rand, even without dropout.
                # Make that stream a function of the sample offset, including after resume.
                torch.manual_seed(args.seed + offset)
                row = samples[order[offset]]
                cached = cache[sample_key(row)]
                inputs, fingerprint = target.prepare(row, data)
                if fingerprint != cached["input_hash"]:
                    raise ValueError(f"{cached['key']}: fused prompt/grid changed since caching")
                if target.mask_id in cached["gen_ids"]:
                    raise ValueError("Draft mask collides with generated CoC tokens")
                ids, hidden, mask = target.prefill_features(
                    inputs, cached["gen_ids"], draft.target_layer_ids
                )
                sync = (
                    model.no_sync()
                    if runtime.world_size > 1 and micro + 1 < accumulation
                    else nullcontext()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, accuracy, metrics = model(
                        input_ids=ids, hidden_states=hidden, loss_mask=mask, **schedule
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"Nonfinite loss at step {step}, sample {cached['key']}"
                        )
                    (loss / accumulation).backward()
                totals += torch.stack(
                    (
                        loss.detach().double(),
                        accuracy.detach().double(),
                        torch.as_tensor(
                            metrics.get("accept_len", 0), device=runtime.device
                        ).double(),
                    )
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                draft.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            totals = runtime.sum(totals) / (runtime.world_size * accumulation)
            if runtime.rank == 0:
                acceptance = (
                    {"teacher_forced_accept_len": totals[2].item()}
                    if args.method == "domino"
                    else {}
                )
                record = {
                    "step": step + 1,
                    "steps": args.steps,
                    "loss": totals[0].item(),
                    "teacher_forced_accuracy": totals[1].item(),
                    **acceptance,
                    **schedule,
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item(),
                    "samples_per_second": args.global_batch_size
                    / (time.monotonic() - step_started),
                    "max_memory_allocated_gb": torch.cuda.max_memory_allocated(runtime.device)
                    / 1e9,
                    "elapsed_seconds": time.monotonic() - started,
                }
                line = json.dumps(record)
                print(line, flush=True)
                metrics_file.write(line + "\n")
                metrics_file.flush()
                if wandb is not None:
                    wandb.log(record, step=step + 1)
            if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
                save_checkpoint(runtime, draft, optimizer, scheduler, step + 1, run_info)


if __name__ == "__main__":
    from .cli import main

    main()
