#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-key the public nvidia/Alpamayo2-Edge checkpoint for stock TensorRT Edge-LLM.

The release stores the Cosmos3-Edge VLM under ``vlm.*`` and the action expert
under ``expert.*``. Edge-LLM's ``cosmos3_edge`` exporter reads a plain
Cosmos3-Edge checkpoint, so this writes one:

  config.json                 the release's ``vlm_config`` block (model_type cosmos3_edge)
  model.safetensors (+index)  text tower flat (layers.N.*, embed_tokens, norm, lm_head)
                              with mlp.fc1/fc2 -> mlp.up_proj/down_proj (text tower only);
                              vision tower + projector as model.visual.* / model.projector.*
  expert.safetensors          action expert as expert.layers.N.*, expert.norm.*,
                              action_in_proj.*, action_out_proj.* (same mlp rename);
                              not listed in the index, so the VLM exporter never sees it
  alpamayo2_edge.config.json  the release config, untouched (expert + trajectory settings)
  tokenizer / chat template / generation + processor config copied verbatim
  shim.json                   source file hashes and key counts

Usage: python shim.py SRC OUT [--hf] [--expert SHIM] [--fp16-modules SHIM:SUBSTR[,SUBSTR...]]
  SRC       the downloaded release snapshot (``vlm.*`` / ``expert.*`` keys) or a quantized HF-layout
            Cosmos3-Edge checkpoint written by ``tensorrt-edgellm-quantize`` from a ``--hf`` shim
            (``model.language_model.*`` keys plus ``*_scale`` tensors and ``hf_quant_config.json``)
  --hf      write the VLM in the HF ``cosmos3_edge`` layout (``model.language_model.layers.N.mlp.fc1``)
            instead of the exporter layout: the input ``tensorrt-edgellm-quantize`` loads
  --expert  a shim made from the release: its expert.safetensors and alpamayo2_edge.config.json are
            copied when SRC has no expert (a quantized VLM)
  --fp16-modules SHIM:SUBSTR[,SUBSTR...]  take every tensor whose exporter-layout key contains one of the
            substrings from that shim (e.g. ``shim:visual.embeddings.patch_embedding,visual.encoder.layers.0.``)
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

COPY = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "hf_quant_config.json",
)
MLP = ((".mlp.fc1.", ".mlp.up_proj."), (".mlp.fc2.", ".mlp.down_proj."))
# (source prefix, replacement, destination file, rename relu2 MLP fc1/fc2); the release's ``vlm.`` prefix is
# stripped first, so the same rules read a quantized HF-layout VLM. The SigLIP vision tower keeps fc1/fc2:
# that is what the stock visual model is named.
RULES = (
    ("model.language_model.", "", "model", True),
    ("lm_head.", "lm_head.", "model", False),
    ("model.visual.", "model.visual.", "model", False),
    ("model.projector.", "model.projector.", "model", False),
    ("expert.expert.", "expert.", "expert", True),
    ("expert.action_in_proj.", "action_in_proj.", "expert", False),
    ("expert.action_out_proj.", "action_out_proj.", "expert", False),
)


def rekey(key: str, hf: bool = False):
    if key.startswith("vlm."):
        key = key[len("vlm.") :]
    if hf and not key.startswith("expert."):
        return key, "model"
    for src, dst, dest, relu2 in RULES:
        if key.startswith(src):
            key = dst + key[len(src) :]
            if relu2:
                for a, b in MLP:
                    key = key.replace(a, b)
            return key, dest
    raise KeyError(f"no rule for checkpoint key {key!r}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(
    src: Path,
    out: Path,
    hf: bool = False,
    expert: Path | None = None,
    fp16_modules: str | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    release = json.load(open(src / "config.json"))
    if release.get("model_type") == "alpamayo2_edge":
        json.dump(release["vlm_config"], open(out / "config.json", "w"), indent=2)
        shutil.copy2(src / "config.json", out / "alpamayo2_edge.config.json")
    elif release.get("model_type") == "cosmos3_edge":  # quantized VLM from a --hf shim
        shutil.copy2(src / "config.json", out / "config.json")
    else:
        sys.exit(
            f"{src}/config.json model_type is {release.get('model_type')!r}, "
            "expected 'alpamayo2_edge' or 'cosmos3_edge'"
        )
    for name in COPY:
        for base in (src, expert):
            if base is not None and (base / name).is_file():
                shutil.copy2(base / name, out / name)
                break
    if hf:
        # transformers takes processor text-kwarg defaults from the tokenizer's init kwargs; the HF Cosmos3-Edge
        # model needs ``mm_token_type_ids`` for M-RoPE, and the stock calibration loop does not request them.
        tok = json.load(open(out / "tokenizer_config.json"))
        tok["return_mm_token_type_ids"] = True
        json.dump(tok, open(out / "tokenizer_config.json", "w"), indent=2)

    if (src / "model.safetensors.index.json").is_file():
        shards = sorted(
            set(json.load(open(src / "model.safetensors.index.json"))["weight_map"].values())
        )
    else:  # single-shard checkpoint
        shards = ["model.safetensors"]
    tensors = {"model": {}, "expert": {}}
    for shard in shards:
        with safe_open(src / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                new, dest = rekey(key, hf)
                if new in tensors[dest]:
                    sys.exit(f"duplicate key after rekey: {new}")
                tensors[dest][new] = f.get_tensor(key)
    if (
        fp16_modules
    ):  # take the listed modules from an FP16 shim (their *_scale tensors go with them)
        shim_path, subs = fp16_modules.split(":", 1)
        source, match = (
            Path(shim_path),
            lambda k, subs=tuple(subs.split(",")): any(sub in k for sub in subs),
        )
        tensors["model"] = {
            k: v for k, v in tensors["model"].items() if not match(k)
        }  # drops the *_scale too
        with safe_open(source / "model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                if match(key):
                    tensors["model"][key] = f.get_tensor(key)
    if not tensors["expert"]:
        del tensors["expert"]
        if expert is not None:
            for name in ("expert.safetensors", "alpamayo2_edge.config.json"):
                shutil.copy2(expert / name, out / name)
    for dest, t in tensors.items():
        save_file(t, out / f"{dest}.safetensors", metadata={"format": "pt"})
    json.dump(
        {
            "metadata": {
                "total_size": sum(v.numel() * v.element_size() for v in tensors["model"].values())
            },
            "weight_map": {k: "model.safetensors" for k in tensors["model"]},
        },
        open(out / "model.safetensors.index.json", "w"),
        indent=2,
    )

    json.dump(
        {
            "source": str(src),
            "layout": "hf" if hf else "exporter",
            "fp16_modules": fp16_modules,
            "source_sha256": {
                p.name: sha256(p)
                for p in sorted(src.iterdir())
                if p.is_file() and not p.name.startswith(".")
            },
            "keys": {dest: len(t) for dest, t in tensors.items()},
            "dtypes": sorted({str(v.dtype) for t in tensors.values() for v in t.values()}),
        },
        open(out / "shim.json", "w"),
        indent=2,
    )
    print(json.dumps({dest: len(t) for dest, t in tensors.items()}))


if __name__ == "__main__":
    args = sys.argv[1:]
    hf = "--hf" in args
    valued = {"--expert": None, "--fp16-modules": None}
    for flag in valued:
        if flag in args:
            valued[flag] = (
                args[args.index(flag) + 1]
                if flag == "--fp16-modules"
                else Path(args[args.index(flag) + 1])
            )
    rest = [
        a
        for i, a in enumerate(args)
        if a not in ("--hf", *valued) and not (i > 0 and args[i - 1] in valued)
    ]
    if len(rest) != 2:
        sys.exit(__doc__)
    main(
        Path(rest[0]),
        Path(rest[1]),
        hf,
        valued["--expert"],
        valued["--fp16-modules"],
    )
