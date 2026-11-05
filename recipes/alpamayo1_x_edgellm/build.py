# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the plugin library and the six TensorRT engines on DRIVE Thor.

    python -m alpamayo1_x_edgellm.build --onnx-dir <export> --output-dir <engines> [--tensorrt-root <TRT>]

Builds TensorRT Edge-LLM's plugin library from the submodule with CMake: ``AlpamayoFusedGateUpPlugin`` and
Edge-LLM's own ``AttentionPlugin`` (the verify and action engines use its tree mode; its tree-decode
kernel comes from the CuTe DSL prebuilt group ``fmha_decode_ctm``, extracted from the submodule's
``kernelSrcs/cuteDSLPrebuilt`` tarball). Then builds ``visual``, ``prefill``, ``verify_q17``, ``verify_q1``,
``domino`` and ``action`` engines from ``<export>`` with builder optimization level 3. Engines are
specific to the board, CUDA and TensorRT version; build them on the target.

DRIVE OS ships no cuRAND headers, which Edge-LLM's CMake requires: ``--curand-include`` names a directory
holding them; by default the toolkit's ``include/`` is used when it has ``curand_kernel.h``, else the
``curand*.h`` of the environment's ``nvidia-curand`` wheel are copied to ``<engines>/curand-include``.
nvcc's temporary files go to ``<engines>/tmp`` (the board's root volume is small).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sysconfig
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGELLM = HERE / "tensorrt-edge-llm"
TENSORRT_MAJOR = "11"  # V3 plugin API; measured with 11.2 and 11.3
EDGELLM_PLUGINS = "NvInfer_edgellm_plugin"
TENSORRT_PLUGINS = (EDGELLM_PLUGINS,)
CUTE_DSL_GROUP = "fmha_decode_ctm"  # AttentionPlugin tree-decode kernel (SM 11.0, CUDA 13)
CUTE_DSL_TAG = "sm_110"  # explicit: CuteDsl.cmake infers the tag from EMBEDDED_TARGET "auto_thor", the toolchain file takes "auto-thor"
CUTE_DSL_TARBALL = EDGELLM / "kernelSrcs/cuteDSLPrebuilt/cutedsl_aarch64_sm_110_cuda13.tar.gz"
CUTE_DSL_ARTIFACT = EDGELLM / "cpp/kernels/cuteDSLArtifact/aarch64/sm_110/metadata.json"  # CMake extracts the tarball here

# engine -> ONNX directory under the export root
ENGINES = {"visual": "visual", "prefill": "prefill", "verify_q17": "verify/q17", "verify_q1": "verify/q1",
           "domino": "domino", "action": "action"}

# Domino: one engine, three shape profiles (seed / sliding window / incremental speculative rounds).
DOMINO_PROFILES = [
    {"features": [(3006, 10240)] * 3, "past": [(0, 10240)] * 3, "context_rope": [(3006, 128)] * 3},
    {"features": [(802, 10240)] * 3, "past": [(2204, 10240)] * 3, "context_rope": [(3006, 128)] * 3},
    {"features": [(1, 10240), (8, 10240), (17, 10240)], "past": [(3006, 10240), (3022, 10240), (3247, 10240)],
     "context_rope": [(3007, 128), (3030, 128), (3264, 128)]},
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def cuda_version(nvcc: str = "nvcc") -> str:
    text = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    return text.split("release ")[1].split(",")[0]


def curand_include(output: Path, requested: Path | None) -> Path:
    """Directory holding ``curand_kernel.h`` for Edge-LLM's CMake (``CURAND_KERNEL_INCLUDE_DIR``)."""
    if requested is not None:
        if not (requested / "curand_kernel.h").is_file():
            raise FileNotFoundError(f"{requested}: no curand_kernel.h")
        return requested
    toolkit = Path(shutil.which("nvcc") or "nvcc").resolve().parent.parent / "include"
    if (toolkit / "curand_kernel.h").is_file():
        return toolkit
    wheel = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/include"
    headers = sorted(wheel.glob("curand*.h"))
    if not any(h.name == "curand_kernel.h" for h in headers):
        raise FileNotFoundError(f"curand_kernel.h: neither {toolkit} nor {wheel} has the cuRAND headers; "
                                "pass --curand-include <dir holding them>")
    target = output / "curand-include"  # only the cuRAND headers: the wheel's other CUDA headers must not shadow the toolkit's
    target.mkdir(parents=True, exist_ok=True)
    for header in headers:
        shutil.copyfile(header, target / header.name)
    return target


def build_edgellm_plugins(tensorrt_root: Path, output: Path, curand: Path | None) -> Path:
    """TensorRT Edge-LLM's plugin library for DRIVE Thor (110a), built from the submodule, with the CuTe DSL
    ``fmha_decode_ctm`` kernels of the AttentionPlugin's tree mode."""
    if not CUTE_DSL_ARTIFACT.is_file() and not CUTE_DSL_TARBALL.is_file():
        raise FileNotFoundError(f"{CUTE_DSL_TARBALL}: the CuTe DSL prebuilt kernels (group {CUTE_DSL_GROUP}, the "
                                "AttentionPlugin's tree-decode kernel) are not in the submodule checkout; update the submodule")
    build, tmp = output / "edgellm-build", output / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(tmp))
    configure = ["cmake", "-S", str(EDGELLM), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                 f"-DTRT_PACKAGE_DIR={tensorrt_root}", f"-DCUDA_CTK_VERSION={cuda_version()}",
                 f"-DCMAKE_TOOLCHAIN_FILE={EDGELLM / 'cmake/aarch64_linux_toolchain.cmake'}", "-DEMBEDDED_TARGET=auto-thor",
                 f"-DENABLE_CUTE_DSL={CUTE_DSL_GROUP}", f"-DCUTE_DSL_ARTIFACT_TAG={CUTE_DSL_TAG}",
                 f"-DCURAND_KERNEL_INCLUDE_DIR={curand_include(output, curand)}"]
    for command in (configure, ["cmake", "--build", str(build), "--target", EDGELLM_PLUGINS, "-j", str(os.cpu_count())]):
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    library = output / f"lib{EDGELLM_PLUGINS}.so"
    library.write_bytes((build / f"lib{EDGELLM_PLUGINS}.so").resolve().read_bytes())
    return library


def compile_plugins(tensorrt_root: Path, output: Path, curand: Path | None) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    return {EDGELLM_PLUGINS: build_edgellm_plugins(tensorrt_root, output, curand)}


def build_engine(trt, logger, onnx_path: Path, engine_path: Path, *, profiles=(), strongly_typed=False) -> dict:
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.builder_optimization_level = 3
    for shapes in profiles:
        profile = builder.create_optimization_profile()
        for name, bounds in shapes.items():
            profile.set_shape(name, *bounds)
        config.add_optimization_profile(profile)
    begin = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"engine build failed: {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    return {"onnx": str(onnx_path), "onnx_sha256": sha256(onnx_path), "engine_sha256": sha256(engine_path),
            "engine_bytes": engine_path.stat().st_size, "seconds": round(time.monotonic() - begin, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx-dir", type=Path, required=True, help="Output directory of the export step.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plugins and engines (rebuilt in place).")
    parser.add_argument("--tensorrt-root", type=Path, default=Path(os.environ.get("TENSORRT_ROOT", "/usr")),
                        help="TensorRT SDK root (include/, lib/); default $TENSORRT_ROOT.")
    parser.add_argument("--curand-include", type=Path, help="Directory holding curand_kernel.h (default: detected).")
    parser.add_argument("--only", nargs="+", choices=list(ENGINES), help="Build only these engines.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)  # a failed earlier build may be rerun in place

    import tensorrt as trt
    if trt.__version__.split(".")[0] != TENSORRT_MAJOR:
        raise RuntimeError(f"recipe needs TensorRT {TENSORRT_MAJOR}.x, found {trt.__version__}")
    libraries = compile_plugins(args.tensorrt_root, args.output_dir / "plugins", args.curand_include)
    handles = [ctypes.CDLL(str(libraries[name]), mode=ctypes.RTLD_GLOBAL) for name in TENSORRT_PLUGINS]  # static registration
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    receipt = {"tensorrt": trt.__version__, "plugins": {n: {"path": str(p), "sha256": sha256(p)} for n, p in libraries.items()},
               "engines": {}}
    for name, relative in ENGINES.items():
        if args.only and name not in args.only:
            continue
        print(f"building {name}", flush=True)
        receipt["engines"][name] = build_engine(
            trt, logger, args.onnx_dir / relative / "model.onnx", args.output_dir / f"{name}.engine",
            profiles=DOMINO_PROFILES if name == "domino" else (), strongly_typed=name == "visual")
        (args.output_dir / "build.json").write_text(json.dumps(receipt, indent=1) + "\n")
    del handles
    print(json.dumps(receipt["engines"], indent=1))


if __name__ == "__main__":
    main()
