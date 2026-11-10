#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the DFlash2 draft engine (host/export.py draft ONNX) with the TensorRT Python API on the DRIVE AGX Thor.

Usage: python build_draft.py ONNX_DRAFT_DIR ENGINE_DIR [CAPACITY]
  ONNX_DRAFT_DIR  model.onnx + model.onnx.data + dflash2_selector.safetensors + config.json (+ draft_vocab_map.safetensors)
  ENGINE_DIR      engines/<name>/draft: draft.engine, config.json and the selector copied, build info appended
  CAPACITY        draft K/V rows (default 3584 = the language model's capacity); pages = capacity / 128

Two optimization profiles over the target-delta rows: 0 = bootstrap (the whole prompt, up to 5120 rows), 1 =
continuation (one block or less). Needs the Edge-LLM plugin library (``EDGELLM_PLUGIN_PATH`` or
``$WORK/edgellm/build/libNvInfer_edgellm_plugin.so``); the K/V pools are updated in place by the plugin.
"""

import ctypes
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import tensorrt as trt

MAX_DELTA, OPT_PROMPT = 5120, 3300


def profiles(builder, cfg: dict, pages: int):
    block, layers = cfg["dflash_config"]["block_size"], cfg["num_hidden_layers"]
    kv, d, hidden, taps = (
        cfg["num_key_value_heads"],
        cfg["head_dim"],
        cfg["hidden_size"],
        cfg["base_model_hidden_size"],
    )
    fixed = {
        "inputs_embeds": (block, hidden),
        "rope_rotary_cos_sin": (block, d),
        "positions": (block,),
        "attention_position_ids": (block,),
        "packed_attention_mask": (block, 1),
        "query_start_offsets": (2,),
        "query_lengths": (1,),
        "past_lengths": (1,),
        "attention_sequence_lengths": (1,),
        "state_indices": (1,),
        "execution_phase_marker": (4,),
        "context_sequence_count_carrier": (0,),
        "kv_page_table": (1, 2, pages),
        **{f"past_key_values_{i}": (2, pages, 128, kv, d) for i in range(layers)},
    }
    out = []
    for lo, opt, hi in ((1, OPT_PROMPT, MAX_DELTA), (1, block, block)):
        p = builder.create_optimization_profile()
        for name, shape in fixed.items():
            p.set_shape(name, shape, shape, shape)
        p.set_shape("dflash_target_hidden_concat", (lo, taps), (opt, taps), (hi, taps))
        p.set_shape("dflash_delta_rope_cos_sin", (lo, d), (opt, d), (hi, d))
        for name in ("dflash_delta_positions", "dflash_delta_token_to_sequence"):
            p.set_shape(name, (lo,), (opt,), (hi,))
        out.append(p)
    return out


def main(onnx_dir: Path, engine_dir: Path, capacity: int) -> None:
    cfg = json.load(open(onnx_dir / "config.json"))
    pages = -(-capacity // 128)
    logger = trt.Logger(trt.Logger.WARNING)
    plugin = os.environ.get("EDGELLM_PLUGIN_PATH") or str(
        Path(os.environ["WORK"]) / "edgellm" / "build" / "libNvInfer_edgellm_plugin.so"
    )
    ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_dir / "model.onnx")):
        sys.exit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    bc = builder.create_builder_config()
    bc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    bc.builder_optimization_level = 3
    if hasattr(trt.PreviewFeature, "ALIASED_PLUGIN_IO_10_03"):
        bc.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)
    for p in profiles(builder, cfg, pages):
        bc.add_optimization_profile(p)
    t0 = time.monotonic()
    serialized = builder.build_serialized_network(network, bc)
    if serialized is None:
        sys.exit("engine build failed")
    engine_dir.mkdir(parents=True, exist_ok=True)
    path = engine_dir / "draft.engine"
    path.write_bytes(bytes(serialized))
    shutil.copy2(onnx_dir / cfg["dflash_config"]["selector_file"], engine_dir)
    if (
        onnx_dir / "draft_vocab_map.safetensors"
    ).is_file():  # reduced draft head (host/export.py draft --vocab)
        shutil.copy2(onnx_dir / "draft_vocab_map.safetensors", engine_dir)
    cfg["build"] = {
        "tensorrt": trt.__version__,
        "seconds": round(time.monotonic() - t0),
        "capacity": capacity,
        "pages": pages,
        "onnx_sha256": hashlib.sha256((onnx_dir / "model.onnx").read_bytes()).hexdigest(),
        "engine_bytes": path.stat().st_size,
        "delta_profiles": [
            [1, OPT_PROMPT, MAX_DELTA],
            [1, cfg["dflash_config"]["block_size"], cfg["dflash_config"]["block_size"]],
        ],
    }
    (engine_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg["build"], indent=1))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(
        Path(sys.argv[1]).resolve(),
        Path(sys.argv[2]).resolve(),
        int(sys.argv[3]) if len(sys.argv) == 4 else 3584,
    )
