# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reduced output vocabulary for the language model and the drafter (TensorRT Edge-LLM ``reduce-vocab`` format).

Usage: python vocab.py SHIM IDS SIZE OUT
  SHIM  shim checkpoint (config.json gives vocab_size and eos_token_id)
  IDS   JSON list of the token ids the chain of causation uses (host/coc_vocab.json)
  SIZE  reduced vocabulary size; the list is padded to it with the lowest unused ids
  OUT   writes vocab_map.safetensors (sorted int32 ids) and reduced_vocab.json

The transformer and the input embedding keep the full vocabulary; only the LM head rows and the logits shrink
(535 MB of FP16 head for 135168 tokens). Greedy decoding is unchanged as long as the argmax is in the list.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def main(shim: Path, ids: Path, size: int, out: Path) -> None:
    cfg = json.load(open(shim / "config.json"))
    text = cfg.get("text_config", cfg)
    vocab_size = text["vocab_size"]
    eos = cfg.get("eos_token_id", text.get("eos_token_id"))
    keep = set(json.load(open(ids)))
    keep.update(eos if isinstance(eos, list) else [eos])
    if len(keep) > size:
        sys.exit(f"{len(keep)} ids exceed SIZE {size}")
    for t in range(vocab_size):  # pad with the lowest unused ids (bytes, digits, short word pieces)
        if len(keep) == size:
            break
        keep.add(t)
    vocab_map = torch.tensor(sorted(keep), dtype=torch.int32)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"vocab_map": vocab_map}, str(out / "vocab_map.safetensors"))
    json.dump(
        {"vocab_size": vocab_size, "reduced_vocab_size": size, "source": str(ids)},
        open(out / "reduced_vocab.json", "w"),
        indent=1,
    )
    print({"vocab_size": vocab_size, "reduced_vocab_size": size, "eos": eos})


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
