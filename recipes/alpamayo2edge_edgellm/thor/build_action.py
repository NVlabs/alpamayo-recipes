#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the action engine (host/action_graph.py ONNX) with the TensorRT Python API on the DRIVE AGX Thor.

Usage: python build_action.py ONNX_ACTION_DIR ENGINE_DIR
  ONNX_ACTION_DIR  host/action_graph.py output (model.onnx, weights.bin, config.json)
  ENGINE_DIR       engines/<name>/action: writes action.engine, copies config.json, appends build info

The network is strongly typed (the graph casts explicitly: FP16 GEMMs, FP32 norm statistics and Euler update).
The Edge-LLM plugin library is loaded (``EDGELLM_PLUGIN_PATH`` or ``$WORK/edgellm/build/libNvInfer_edgellm_plugin.so``)
and the dynamic dims get one profile: pool pages, page-table width, the shape-only phase marker (extent 5) and
context carrier (extent 0).
"""

import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import tensorrt as trt

PAGE_PROFILE = {
    "pages": (28, 448, 512),
    "max_pages": (1, 32, 64),
}  # sequences x [prefix pages + private]


def main(onnx_dir: Path, engine_dir: Path) -> None:
    config = json.load(open(onnx_dir / "config.json"))
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
    bc.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 16 << 30
    )  # the graph is one Myelin region over 28 pools
    bc.builder_optimization_level = 3
    profile = builder.create_optimization_profile()
    layers, kv, d = (
        config["num_hidden_layers"],
        config["num_key_value_heads"],
        config["head_dim"],
    )
    for i in range(layers):
        profile.set_shape(f"pool_{i}", *[(2, pages, 128, kv, d) for pages in PAGE_PROFILE["pages"]])
    sequences = config["candidates"] // (config.get("candidates_per_sequence") or 1)
    profile.set_shape("kv_page_table", *[(sequences, 2, w) for w in PAGE_PROFILE["max_pages"]])
    profile.set_shape(
        "execution_phase_marker", (5,), (5,), (5,)
    )  # kSpecTargetVerify: tree decoding
    profile.set_shape(
        "context_sequence_count_carrier", (0,), (0,), (0,)
    )  # not a context phase: no context sequences
    if hasattr(trt.PreviewFeature, "ALIASED_PLUGIN_IO_10_03"):
        bc.set_preview_feature(
            trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True
        )  # pool_i -> present_i in place
    bc.add_optimization_profile(profile)
    t0 = time.monotonic()
    serialized = builder.build_serialized_network(network, bc)
    if serialized is None:
        sys.exit("engine build failed")
    engine_dir.mkdir(parents=True, exist_ok=True)
    path = engine_dir / "action.engine"
    path.write_bytes(bytes(serialized))
    config["build"] = {
        "tensorrt": trt.__version__,
        "seconds": round(time.monotonic() - t0),
        "onnx_sha256": hashlib.sha256((onnx_dir / "model.onnx").read_bytes()).hexdigest(),
        "engine_bytes": path.stat().st_size,
        "page_profile": PAGE_PROFILE,
    }
    (engine_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config["build"], indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
