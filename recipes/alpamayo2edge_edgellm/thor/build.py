#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build TensorRT Edge-LLM and the language and vision engines on the DRIVE AGX Thor.

Run after ``source $WORK/env.sh`` (see thor/setup.sh).

Usage: python build.py ONNX_DIR [LOGITS_ROWS]
  ONNX_DIR     an exported set with llm/, visual/, action/ (e.g. $WORK/onnx/fp8)
  LOGITS_ROWS  logits rows per LLM pass (llm_build --maxBatchSize; default 6): 16 for speculative decoding
               with the block-16 drafter (anchor + 15 drafts scored in one verify pass)

Writes:
  $WORK/edgellm/src        copy of the pinned Edge-LLM source
  $WORK/edgellm/build      Ninja tree: the plugin library, llm_build, visual_build
  $WORK/engines/<name>/    llm/ visual/ engines (<name> = basename of ONNX_DIR) + build.json

The K/V-cache capacity comes from ONNX_DIR/action/config.json (host/action_graph.py), so the language engine
and the action engine (thor/build_action.py) always agree.
"""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
PINNED_SRC = RECIPE / "tensorrt-edge-llm"
TARGETS = ("NvInfer_edgellm_plugin", "llm_build", "visual_build")

MAX_INPUT_LEN = 5120  # 22 images x <=192 tokens + text + trajectory history
MAX_BATCH = 6  # trajectory candidates
IMAGE_TOKENS = {
    "min": 64,
    "max": MAX_BATCH * 22 * 192,
    "per_image": 192,
}  # max is per batch; clips use 91 (side) / 180 (front) tokens per image


def run(cmd, log, **kw):
    """Edge-LLM tools dlopen the plugin library from EDGELLM_PLUGIN_PATH (default: build/ under the cwd)."""
    env = dict(os.environ)
    env["EDGELLM_PLUGIN_PATH"] = str(
        Path(os.environ["WORK"]) / "edgellm" / "build" / "libNvInfer_edgellm_plugin.so"
    )
    kw.setdefault("env", env)
    print("+", " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as f:
        f.write("+ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        subprocess.run([str(c) for c in cmd], check=True, stdout=f, stderr=subprocess.STDOUT, **kw)


def curand_include(work: Path) -> Path:
    """DRIVE OS ships no cuRAND headers; thor/setup.sh installs them via the nvidia-curand wheel."""
    toolkit = Path("/usr/local/cuda/include")
    if (toolkit / "curand_kernel.h").is_file():
        return toolkit
    wheel = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "include"
    headers = sorted(wheel.glob("curand*.h"))
    if not headers:
        sys.exit(f"curand_kernel.h not found in {toolkit} or {wheel}; run thor/setup.sh")
    target = (
        work / "edgellm" / "curand-include"
    )  # only cuRAND headers: nothing else may shadow the toolkit
    target.mkdir(parents=True, exist_ok=True)
    for h in headers:
        shutil.copyfile(h, target / h.name)
    return target


def nvcc_version() -> str:
    out = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True).stdout
    return out.split("release ")[1].split(",")[0]


def main(onnx_dir: Path, logits_rows: int = MAX_BATCH) -> None:
    work, trt = Path(os.environ["WORK"]), Path(os.environ["TENSORRT_ROOT"])
    src, build = work / "edgellm" / "src", work / "edgellm" / "build"
    engines = work / "engines" / onnx_dir.name
    engines.mkdir(parents=True, exist_ok=True)
    log = engines / "build.log"
    report = {
        "onnx_dir": str(onnx_dir),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }

    if not (src / "CMakeLists.txt").exists():
        shutil.copytree(PINNED_SRC, src, symlinks=True, ignore=shutil.ignore_patterns(".git"))

    if not (build / "build.ninja").exists():
        run(
            [
                "cmake",
                "-S",
                src,
                "-B",
                build,
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_TOOLCHAIN_FILE={src / 'cmake/aarch64_linux_toolchain.cmake'}",
                "-DEMBEDDED_TARGET=auto-thor",
                f"-DCUDA_CTK_VERSION={nvcc_version()}",
                f"-DTRT_PACKAGE_DIR={trt}",
                f"-DCURAND_KERNEL_INCLUDE_DIR={curand_include(work)}",
            ],
            log,
        )
    t0 = time.time()
    run(
        ["cmake", "--build", build, "-j", str(max(2, os.cpu_count() - 2)), "--target", *TARGETS],
        log,
    )
    report["edgellm_build_s"] = round(time.time() - t0)

    max_kv = json.load(open(onnx_dir / "action" / "config.json"))[
        "capacity"
    ]  # host/action_graph.py
    report["max_kv_cache_capacity"] = max_kv
    bins = {
        "llm_build": build / "examples/llm/llm_build",
        "visual_build": build / "examples/multimodal/visual_build",
    }
    steps = {
        "llm": [
            bins["llm_build"],
            "--onnxDir",
            onnx_dir / "llm",
            "--engineDir",
            engines / "llm",
            "--maxInputLen",
            MAX_INPUT_LEN,
            "--maxKVCacheCapacity",
            max_kv,
            "--maxBatchSize",
            logits_rows,
        ],
        "visual": [
            bins["visual_build"],
            "--onnxDir",
            onnx_dir / "visual",
            "--engineDir",
            engines,
            "--minImageTokens",
            IMAGE_TOKENS["min"],
            "--maxImageTokens",
            IMAGE_TOKENS["max"],
            "--maxImageTokensPerImage",
            IMAGE_TOKENS["per_image"],
        ],
    }
    for name, cmd in steps.items():
        if list((engines / name).glob("*.engine")):
            report[f"{name}_engine_s"] = "kept"
            continue
        t0 = time.time()
        run(cmd, log)
        report[f"{name}_engine_s"] = round(time.time() - t0)

    report["engines"] = {
        str(p.relative_to(engines)): p.stat().st_size for p in sorted(engines.rglob("*.engine"))
    }
    (engines / "build.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), int(sys.argv[2]) if len(sys.argv) == 3 else MAX_BATCH)
