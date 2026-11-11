#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn one driving clip into a TensorRT Edge-LLM ``action_inference`` request.

Usage: python make_inputs.py SAMPLE_DIR MODEL_DIR OUT
       python make_inputs.py --clip CLIP_ID T0_US "NAV TEXT" MODEL_DIR OUT
  SAMPLE_DIR     an Edge inference sample on disk: sample.json + tensors.safetensors
                 (22 camera frames, camera ids, timestamps, 20 ego poses, nav text)
  --clip         the same sample read from the PhysicalAI-AV dataset through the public
                 ``alpamayo2_edge`` package (``load_physical_aiavdataset``; needs its ``dataset``
                 extra and dataset access), e.g. the clip of its examples/nav_samples.json
  MODEL_DIR      nvidia/Alpamayo2-Edge checkpoint (tokenizer + chat template)
  OUT            clip directory: images/*.png, request.json, prompt.json, clip.json

The request reproduces the Alpamayo2-Edge prompt: per camera a label, then per
frame a ``<t=..s>`` tag and the image (pre-resized to the camera's pixel budget,
``do_resize=false``), then the 57-token history block, navigation text and task.
The history block is written as text placeholders; the stock runner fills them
from the ``trajectory`` item (19 poses relative to the oldest one, so the R1
"first point then deltas" encoding equals the Alpamayo2-Edge delta tokens).

``prompt.json`` is the same prompt for ``thor/runtime``: the chat template rendered
and tokenized here, every ``<|image_pad|>`` expanded to the image's token count and
the history placeholders replaced by the history tokens, plus the image list.
"""

import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer

CAMERAS = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}
OFFSETS = {"full": [-2.0, -1.0, -0.5, -0.1, 0.0], "sub": [-1.0, -0.1, 0.0]}
PIXEL_BOUNDS = {c: [163840, 196608] if c in (1, 6) else [81920, 98304] for c in CAMERAS}
FACTOR = 32  # patch 16 x merge 2
SYSTEM = "You are a driving assistant that generates safe and accurate actions."
TASK = "Your task: output the chain-of-thought reasoning of the driving process, then output the future trajectory."
HIST_MIN, HIST_MAX, HIST_BINS = (
    torch.tensor([-4.0, -4.0, -10.0]),
    torch.tensor([4.0, 4.0, 10.0]),
    1000,
)
CANDIDATES = 6  # trajectories per clip = request batch
IMAGE_PAD, HISTORY_PAD = "<|image_pad|>", "<|traj_history|>"


def smart_resize(h, w, min_pixels, max_pixels, factor=FACTOR):
    """HF Qwen-family smart_resize (also what the stock runner implements)."""
    h_bar, w_bar = round(h / factor) * factor, round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar, w_bar = (
            max(factor, math.floor(h / beta / factor) * factor),
            max(factor, math.floor(w / beta / factor) * factor),
        )
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def history_tokens(xyz: torch.Tensor) -> list:
    """Alpamayo2-Edge DeltaTrajectoryTokenizer (pad_origin_at_beginning=False): 19 deltas x 3 bins."""
    delta = xyz[1:] - xyz[:-1]
    bins = (
        ((delta - HIST_MIN) / (HIST_MAX - HIST_MIN) * (HIST_BINS - 1))
        .round()
        .long()
        .clamp(0, HIST_BINS - 1)
    )
    return bins.flatten().tolist()


def prompt_tokens(model_dir: Path, request: dict, frames: list, history: list) -> dict:
    """Render + tokenize the request as the stock runner does, then expand the image pads (one per image ->
    that image's token count) and fill the history pads with the history tokens (``<i{bin}>`` ids)."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text = tokenizer.apply_chat_template(
        request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    image_pad, history_pad = tokenizer.convert_tokens_to_ids([IMAGE_PAD, HISTORY_PAD])
    history_ids = tokenizer.convert_tokens_to_ids([f"<i{t}>" for t in history])
    expanded, images, image, hist = [], [], 0, 0
    for i in ids:
        if i == image_pad:
            images.append({**frames[image], "token_start": len(expanded)})
            expanded.extend([image_pad] * frames[image]["tokens"])
            image += 1
        elif i == history_pad:
            expanded.append(history_ids[hist])
            hist += 1
        else:
            expanded.append(i)
    if image != len(frames) or hist != len(history):
        sys.exit(
            f"prompt has {image} image pads and {hist} history pads, expected {len(frames)} / {len(history)}"
        )
    return {
        "text": text,
        "token_ids": expanded,
        "images": images,
        "image_pad_id": image_pad,
        "tokens": len(expanded),
        "special": {
            n: tokenizer.convert_tokens_to_ids(n)
            for n in ("<|vision_start|>", "<|vision_end|>", "<|traj_future_start|>", "<|im_end|>")
        },
    }


def load_sample(args: list[str]) -> tuple[dict, dict, str]:
    """(metadata, tensors, source) from a sample directory or ``--clip CLIP_ID T0_US NAV_TEXT``."""
    if args[0] == "--clip":
        from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset

        clip_id, t0_us, nav_text = args[1], int(args[2]), args[3]
        sample = load_physical_aiavdataset(clip_id, t0_us, nav_text=nav_text)
        meta = {k: v for k, v in sample.items() if not hasattr(v, "shape")}
        meta.setdefault("__key__", f"{clip_id}_{t0_us}")
        return (
            meta,
            {k: v for k, v in sample.items() if hasattr(v, "shape")},
            f"{clip_id} t0 {t0_us} us",
        )
    sample_dir = Path(args[0])
    return (
        json.load(open(sample_dir / "sample.json")),
        load_file(sample_dir / "tensors.safetensors"),
        str(sample_dir),
    )


def main(meta: dict, t: dict, source: str, model_dir: Path, out: Path) -> None:
    frames, cams, stamps = (
        t["image_frames"],
        t["camera_indices"].flatten().tolist(),
        t["absolute_timestamps"].flatten(),
    )
    xyz = t["ego_history_xyz"][0]
    bounds = {int(k): v for k, v in meta.get("camera_pixel_bounds", {}).items()} or PIXEL_BOUNDS
    (out / "images").mkdir(parents=True, exist_ok=True)

    content, clip_frames = [{"type": "text", "text": ""}], []
    for cam, name in CAMERAS.items():
        idx = sorted((i for i, c in enumerate(cams) if c == cam), key=lambda i: stamps[i].item())
        labels = OFFSETS["full"] if len(idx) == 5 else OFFSETS["sub"]
        if len(idx) != len(labels):
            sys.exit(f"camera {cam}: {len(idx)} frames, expected {len(labels)}")
        content.append({"type": "text", "text": f"{name}: "})
        for i, label in zip(idx, labels):
            img = Image.fromarray(frames[i, 0].permute(1, 2, 0).numpy())
            h, w = smart_resize(img.height, img.width, *bounds[cam])
            path = f"images/cam{cam}_t{label:+.1f}s.png"
            img.resize((w, h), Image.BICUBIC).save(out / path)
            content.append({"type": "text", "text": f"<t={label:.1f}s>"})
            content.append({"type": "image", "image": path, "do_resize": False})
            clip_frames.append(
                {
                    "camera": cam,
                    "t": label,
                    "image": path,
                    "height": h,
                    "width": w,
                    "tokens": (h // FACTOR) * (w // FACTOR),
                }
            )
    content.pop(0)

    tokens = history_tokens(xyz)
    placeholder = (
        "<|traj_history_start|>" + "<|traj_history|>" * len(tokens) + "<|traj_history_end|>"
    )
    nav = meta["nav_text"][0] if isinstance(meta["nav_text"], list) else meta["nav_text"]
    content.append({"type": "text", "text": f"\n{placeholder}\nNavigation input is: {nav}\n{TASK}"})
    content.append({"type": "trajectory", "trajectory": (xyz[1:] - xyz[0]).tolist()})

    request = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content},
        ]
    }
    json.dump(
        {
            "batch_size": CANDIDATES,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_generate_length": 256,
            "enable_thinking": False,
            "requests": [request] * CANDIDATES,
        },
        open(out / "request.json", "w"),
        indent=1,
    )

    prompt = prompt_tokens(model_dir, request, clip_frames, tokens)
    json.dump(prompt, open(out / "prompt.json", "w"))
    clip = {
        "source": source,
        "key": meta.get("__key__"),
        "nav_text": nav,
        "frames": clip_frames,
        "image_tokens": sum(f["tokens"] for f in clip_frames),
        "prompt_tokens": prompt["tokens"],
        "ego_history_xyz": xyz.tolist(),
        "ego_history_rot": t["ego_history_rot"][0].tolist(),
        "history_tokens": tokens,
    }
    json.dump(clip, open(out / "clip.json", "w"), indent=1)
    print(
        json.dumps(
            {k: clip[k] for k in clip if k not in ("frames", "ego_history_xyz", "history_tokens")}
        )
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) not in (3, 6) or (len(argv) == 6) != (argv[0] == "--clip"):
        sys.exit(__doc__)
    meta, tensors, source = load_sample(argv[:-2])
    main(meta, tensors, source, Path(argv[-2]), Path(argv[-1]))
