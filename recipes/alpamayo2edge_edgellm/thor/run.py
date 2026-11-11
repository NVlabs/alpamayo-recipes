#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one clip through the recipe runtime (stock Edge-LLM engines, our Python TensorRT driver) on DRIVE AGX Thor.

Run after ``source $WORK/env.sh``. Usage:
  python run.py ENGINES_DIR CLIP_DIR [--candidates 6] [--steps 10] [--rolling] [--seed 0] [--max-tokens 256]
  ENGINES_DIR  from thor/build.py: llm/ visual/ action/ (e.g. $WORK/engines/fp16)
  CLIP_DIR     from host/make_inputs.py: prompt.json + images/ (+ clip.json, reference/)
  --rolling    encode every frame but the newest of each camera before the clock starts (a streaming runtime has
               them from earlier ticks); the timed ViT then runs only the current frames
  --warmup N   untimed full runs first (engine and allocator warm-up; default 1)

Stages: ViT -> prefill -> greedy chain-of-causation decode (stops after ``<|traj_future_start|>``) -> action
expert (``--candidates`` noise samples, ``--steps`` Euler steps). Writes $WORK/results/<engines>/<clip>/:
output.json (CoC text + 64 (acceleration, curvature) pairs per candidate), profile.json (stage ms), run.log.
"""

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime.action import action_expert  # noqa: E402
from runtime.engine import Engine  # noqa: E402
from runtime.llm import LanguageModel  # noqa: E402
from runtime.rope import cos_sin, mrope_positions  # noqa: E402
from runtime.speculative import Speculative  # noqa: E402
from runtime.vision import VisionEncoder  # noqa: E402


class Clock:
    """Stage timer on the current stream (synchronizes at every mark)."""

    def __init__(self):
        self.ms = {}
        torch.cuda.synchronize()
        self.t = time.perf_counter()

    def mark(self, name):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.ms[name] = self.ms.get(name, 0.0) + (now - self.t) * 1000
        self.t = now


class Runtime(Speculative):
    def __init__(self, engines: Path, candidates: int):
        plugin = os.environ.get("EDGELLM_PLUGIN_PATH") or str(
            Path(os.environ["WORK"]) / "edgellm" / "build" / "libNvInfer_edgellm_plugin.so"
        )
        self.plugin = ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        load = lambda sub: (
            Engine(runtime, engines / sub / f"{sub}.engine"),
            json.load(open(engines / sub / "config.json")),
        )
        visual, visual_cfg = load("visual")
        llm, llm_cfg = load("llm")
        action, action_cfg = load("action")
        self.vision = VisionEncoder(visual, visual_cfg)
        self.lm = LanguageModel(llm, engines / "llm", llm_cfg)
        self.action = action_expert(action, action_cfg, candidates)
        self.eos = set(
            llm_cfg["eos_token_id"]
            if isinstance(llm_cfg["eos_token_id"], list)
            else [llm_cfg["eos_token_id"]]
        )
        self.tokenizer = None
        try:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(engines / "llm" / "tokenizer.json"))
        except Exception as e:  # noqa: BLE001 - text decoding is a convenience, ids are always written
            print(f"tokenizer unavailable ({e}); output.json carries token ids only")
        self.stream = torch.cuda.Stream()  # TensorRT synchronizes extra on the default stream
        self.drafter = None
        self.spec_ms = None

    def run(
        self, prompt: dict, images: dict, *, steps: int, max_tokens: int, noise: torch.Tensor
    ) -> dict:
        with torch.cuda.stream(self.stream):
            return self._run(prompt, images, steps=steps, max_tokens=max_tokens, noise=noise)

    def _run(
        self, prompt: dict, images: dict, *, steps: int, max_tokens: int, noise: torch.Tensor
    ) -> dict:
        lm, clock = self.lm, Clock()
        ids = prompt["token_ids"]
        future_start = prompt["special"]["<|traj_future_start|>"]
        pages = list(range(lm.max_pages))
        table = lm.page_table(pages)

        # ViT: every image (cache hits are free), merged tokens straight into the prompt rows
        features = self.vision.encode(images)
        clock.mark("vit_ms")

        embeds = lm.embed(torch.tensor(ids))
        for image in prompt["images"]:
            embeds[image["token_start"] : image["token_start"] + image["tokens"]] = features[
                image["image"]
            ]
        pos, rope_delta = mrope_positions(ids, prompt["images"])
        lm.begin_request(
            rope_delta, lm.max_pages * 128
        )  # text RoPE rows for every later token, once on the GPU
        logits = lm.prefill(embeds, cos_sin(pos, lm.theta, lm.section).cuda(), table)
        clock.mark("prefill_ms")

        # greedy CoC: the token after <|traj_future_start|> is never needed, but <|traj_future_start|> itself is
        # fed so its K/V exists (the expert attends to it, as in the stock runner)
        generated, past, rounds = [], len(ids), []
        if self.drafter is None:
            while True:
                token = int(lm.argmax(logits))
                generated.append(token)
                if token in self.eos or len(generated) >= max_tokens:
                    break
                logits = lm.decode(lm.embed([token]), lm.text_rope(past, 1), table, past)
                past += 1
                if token == future_start:
                    break
        else:
            generated, past, rounds = self.speculate(
                logits, past, rope_delta, table, future_start, max_tokens
            )
        clock.mark("decode_ms")

        self.action.load_prefix(lm.pool, pages, past)
        clock.mark("action_kv_ms")
        actions = self.action.sample(past, rope_delta, steps, noise)
        clock.mark("action_ms")

        text = (
            self.tokenizer.decode(generated, skip_special_tokens=False) if self.tokenizer else None
        )
        return {
            "generated": generated,
            "text": text,
            "kv_length": past,
            "rope_delta": rope_delta,
            "rounds": rounds,
            "spec_ms": self.spec_ms,
            "actions": actions.cpu().tolist(),
            "ms": clock.ms,
            "decode_steps": len(generated) - (1 if generated and generated[-1] in self.eos else 0),
            "vit_calls": self.vision.calls,
            "total_ms": sum(clock.ms.values()),
        }


def load_images(clip: Path, prompt: dict) -> dict:
    out = {}
    for image in prompt["images"]:
        arr = np.asarray(Image.open(clip / image["image"]).convert("RGB"))
        if arr.shape[:2] != (image["height"], image["width"]):
            raise ValueError(
                f"{image['image']} is {arr.shape[:2]}, prompt says {(image['height'], image['width'])}"
            )
        out[image["image"]] = torch.from_numpy(arr.copy()).cuda()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("engines", type=Path)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--candidates", type=int, default=6)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rolling", action="store_true")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--draft-engine",
        type=Path,
        default=None,
        help="DFlash2 draft engine dir (thor/build_draft.py); without it the chain is decoded greedily",
    )
    args = ap.parse_args()
    engines, clip = args.engines.resolve(), args.clip.resolve()
    out = args.out or Path(os.environ["WORK"]) / "results" / engines.name / (
        clip.name + ("-rolling" if args.rolling else "")
    )
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rt = Runtime(engines, args.candidates)
    if args.draft_engine:
        rt.load_draft_engine(args.draft_engine.resolve())
    load_s = time.time() - t0
    prompt = json.load(open(clip / "prompt.json"))
    images = load_images(clip, prompt)
    noise = torch.randn(
        args.candidates, rt.action.tokens, 2, generator=torch.Generator().manual_seed(args.seed)
    )

    for _ in range(args.warmup):
        rt.run(prompt, images, steps=args.steps, max_tokens=args.max_tokens, noise=noise)
        rt.vision.cache.clear()
    if args.rolling:
        current = {
            f["image"] for f in json.load(open(clip / "clip.json"))["frames"] if f["t"] == 0.0
        }
        with torch.cuda.stream(rt.stream):
            rt.vision.encode({k: v for k, v in images.items() if k not in current})
        torch.cuda.synchronize()
    encoded = sum(1 for k in images if k not in rt.vision.cache)
    result = rt.run(prompt, images, steps=args.steps, max_tokens=args.max_tokens, noise=noise)

    responses = [
        {
            "request_idx": b,
            "output_text": result["text"],
            "output_tokens": result["generated"],
            "output_trajectory": result["actions"][b],
        }
        for b in range(args.candidates)
    ]
    json.dump(
        {
            "input_file": "prompt.json",
            "prompt_tokens": prompt["tokens"],
            "kv_length": result["kv_length"],
            "rope_delta": result["rope_delta"],
            "responses": responses,
        },
        open(out / "output.json", "w"),
        indent=1,
    )
    profile = {
        "platform": "DRIVE AGX Thor",
        "engines": str(engines),
        "clip": str(clip),
        "rolling": args.rolling,
        "candidates": args.candidates,
        "steps": args.steps,
        "warmup": args.warmup,
        "load_s": round(load_s, 1),
        "prompt_tokens": prompt["tokens"],
        "images": len(images),
        "vit_images_encoded_in_timed_run": encoded,
        "generated_tokens": len(result["generated"]),
        "decode_steps": result["decode_steps"],
        "stages_ms": {k: round(v, 2) for k, v in result["ms"].items()},
        "total_ms": round(result["total_ms"], 2),
        "decode_ms_per_step": round(result["ms"]["decode_ms"] / max(1, result["decode_steps"]), 2),
        "gpu_mem_allocated_mb": round(torch.cuda.memory_allocated() / 2**20),
        "tap_layers": rt.lm.taps,
        "drafter": str(args.draft_engine) if args.draft_engine else None,
        "rounds": result["rounds"],
        "noise": f"torch.randn seed {args.seed}",
        "spec_ms": result["spec_ms"],
        "accepted_per_round": (sum(r["accepted"] for r in result["rounds"]) / len(result["rounds"]))
        if result["rounds"] and "accepted" in result["rounds"][0]
        else None,
    }
    json.dump(profile, open(out / "profile.json", "w"), indent=1)
    print(
        json.dumps(
            {
                "coc": result["text"],
                "tokens": result["generated"][:8],
                **profile["stages_ms"],
                "total_ms": profile["total_ms"],
                "rounds": result["rounds"],
                "spec_ms": result["spec_ms"],
                "out": str(out),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
