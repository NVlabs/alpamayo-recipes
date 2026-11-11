# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP8 quantization of the Cosmos3-Edge VLM with TensorRT Edge-LLM's quantizer, calibrated on driving frames.

    python quantize.py SHIM_HF OUT_DIR CLIP_DIR [CLIP_DIR ...] [--kv-fp8]
    python quantize.py --draft SHIM_HF DRAFT_SNAPSHOT OUT_DIR CLIP_DIR [CLIP_DIR ...]

SHIM_HF is ``host/shim.py --hf`` output (HF ``cosmos3_edge`` layout). Every frame under ``CLIP_DIR/images`` becomes
one (image, instruction) calibration sample through the model's own processor, so the SigLIP2 tower and the language
model record activation ranges on camera frames rather than on the quantizer's default MMMU/news-text sets (which
cost 0.7 m ADE on this model, see README). Both towers are quantized (per-tensor FP8 weights and activations,
ModelOpt ``FP8_DEFAULT_CFG``; lm_head and embeddings stay FP16); ``host/shim.py --fp16-modules`` restores
chosen modules to FP16 afterwards. ``--kv-fp8`` also calibrates FP8 scales for the
language model's K/V cache (``k_proj.k_scale`` / ``v_proj.v_scale``): the stock exporter then marks every attention
plugin ``enable_fp8_kv_cache`` and the engines keep the paged pool in FP8, halving the K/V bytes that the prefill,
verify and tree-action attention kernels stream (see README, FP8).

``--draft``: the DFlash2 drafter through Edge-LLM's draft quantizer (``quantize_and_export_dflash_draft``: FP8 layers,
the head, the selector and the tap projection stay FP16), calibrated on the clips' prompt text (special tokens
stripped) and reference chains. The stock quantizer loads the base model with ``AutoModelForCausalLM``, which does
not know ``cosmos3_edge``; the VLM class is used instead. OUT_DIR is a drafter snapshot for ``host/export.py draft``.

Run in a venv with Edge-LLM's ``tools`` extras and a transformers that knows ``cosmos3_edge`` (5.17.0; Edge-LLM
v0.11.1 pins 5.14.1, which does not).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image
from tensorrt_edgellm.quantization.quantize import quantize_and_export

INSTRUCTION = (
    "You are the driving assistant of this ego vehicle. Describe the traffic situation in this camera "
    "frame and state the ego vehicle's next action."
)


def driving_frames(clips: list[Path]):
    """Image calibration dataset: (PIL image, instruction) per frame, clip by clip, all cameras and times."""

    def generate():
        for clip in clips:
            for path in sorted((clip / "images").iterdir()):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    yield Image.open(path).convert("RGB"), INSTRUCTION

    generate.calib_name = "driving_frames:" + ",".join(c.name for c in clips)
    return generate


def main(shim_hf: Path, out: Path, clips: list[Path], kv_fp8: bool = False) -> None:
    frames = sum(len(list((c / "images").iterdir())) for c in clips)
    quantize_and_export(
        model_dir=str(shim_hf),
        output_dir=str(out),
        quantization="fp8",
        visual_quantization="fp8",
        kv_cache_quantization="fp8" if kv_fp8 else None,
        image_dataset=driving_frames(clips),
        num_samples=frames,
    )


def clip_text(clips: list[Path]) -> list[str]:
    """Text calibration strings: each clip's rendered prompt without special tokens, plus its reference chains."""
    texts = []
    for clip in clips:
        texts.append(
            re.sub(r"<\|[^|]*\|>", "", json.load(open(clip / "prompt.json"))["text"]).strip()
        )
        coc = clip / "reference" / "coc.json"
        if coc.is_file():
            texts += [
                c for c in json.load(open(coc)).get("coc", []) if isinstance(c, str) and c.strip()
            ]
    return list(dict.fromkeys(texts))


def quantize_draft(shim_hf: Path, snapshot: Path, out: Path, clips: list[Path]) -> None:
    import torch
    import tensorrt_edgellm.quantization.models.dflash_draft as dflash_draft
    from tensorrt_edgellm.quantization import quantize_and_export_dflash_draft
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    def load_base(model_dir, dtype, device):
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.float16, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model.to(device).eval(), tokenizer

    dflash_draft._load_base_for_calib = load_base
    texts = clip_text(clips)
    quantize_and_export_dflash_draft(
        str(shim_hf),
        str(snapshot),
        str(out),
        quantization="fp8",
        dtype="fp16",
        text_dataset=lambda: iter(texts),
        num_samples=len(texts),
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--kv-fp8", "--draft")]
    if "--draft" in sys.argv:
        if len(args) < 4:
            sys.exit(__doc__)
        quantize_draft(Path(args[0]), Path(args[1]), Path(args[2]), [Path(c) for c in args[3:]])
    elif len(args) < 3:
        sys.exit(__doc__)
    else:
        main(
            Path(args[0]), Path(args[1]), [Path(c) for c in args[2:]], kv_fp8="--kv-fp8" in sys.argv
        )
