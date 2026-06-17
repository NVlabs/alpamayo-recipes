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

import json
import logging
import os

import huggingface_hub
import pandas as pd
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

try:
    import modelopt.torch.quantization as mtq  # noqa: F401f

    assert torch.ops.tensorrt.quantize_op.default
except Exception:
    logger.warning("Unable to import quantization op. Please install modelopt library")

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from safetensors import safe_open

# FP8 E4M3 format has a maximum representable value of 448.0
MAX_BOUND_FP8 = 448.0
# Additional scaling factor for NVFP4
MAX_BOUND_NVFP4 = 6.0

FP8_CONFIG = {
    "quant_cfg": {
        "*": {"enable": False},
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"enable": False},
        "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*softmax_quantizer": {"num_bits": (4, 3), "axis": None},
        "*bmm2_output_quantizer": {"num_bits": (4, 3), "axis": None},
    },
    "algorithm": "max",
}


# def _disable_modelopt_fp8_cuda_extension() -> None:
#     """
#     Disable ModelOpt FP8 CUDA extension and force eager FP8 fake-quant path.

#     This avoids known illegal-memory-access failures in some environments when
#     modelopt_cuda_ext_fp8 kernels are exercised during export/compile flows.
#     """
#     try:
#         import modelopt.torch.quantization.extensions as mtq_ext
#         import modelopt.torch.quantization.tensor_quant as mtq_tq
#     except Exception:
#         return

#     if getattr(mtq_ext, "_alpamayo_fp8_ext_disabled", False):
#         return

#     def _no_fp8_ext(raise_if_failed: bool = False):
#         return None

#     mtq_ext.get_cuda_ext_fp8 = _no_fp8_ext
#     mtq_tq.get_cuda_ext_fp8 = _no_fp8_ext
#     mtq_ext._alpamayo_fp8_ext_disabled = True
#     logger.info("Disabled modelopt FP8 CUDA extension; using eager FP8 fake quantization")


def quantize_model(model, args, tokenizer=None, calibration_forward_loop=None):
    """
    Quantize a PyTorch model using ModelOpt post-training quantization (PTQ).

    This function applies quantization to reduce model precision for faster inference
    while maintaining acceptable accuracy. It uses calibration data generated from
    the provided tokenizer to determine optimal quantization parameters.

    Supported quantization formats:
        - fp8: 8-bit floating point quantization
        - nvfp4: 4-bit NVIDIA floating point quantization
    Args:
        model: PyTorch model to quantize. Must be in evaluation mode.
        args: Command line arguments containing quant_format and debug.
        tokenizer: Hugging Face tokenizer for creating calibration data.
            Required only when `calibration_forward_loop` is not provided.
        calibration_forward_loop: Optional callable taking `model` and running
            calibration forward passes. Use this for non-text modules whose
            forward signature is not compatible with dataset_utils batches.

    Returns:
        Quantized model
    """
    # Create calibration forward loop. For standard text models we can build
    # it from tokenizer-based data, but vision modules often need custom args.
    if calibration_forward_loop is None:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when calibration_forward_loop is None")
        calib_dataloader = get_dataset_dataloader(
            tokenizer=tokenizer,
            batch_size=32,
            num_samples=512,
            device="cuda:0",
        )
        calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    else:
        calibrate_loop = calibration_forward_loop
    if args.quant_format == "int8":
        if args.quant_algo == "smoothquant":
            if args.weight_only:
                raise RuntimeError(
                    "SmoothQuant is supported for weight-and-activation quantization, weight-only flag should not be set"
                )
            quant_cfg = mtq.INT8_SMOOTHQUANT_CFG
        elif args.weight_only:
            quant_cfg = mtq.INT8_WEIGHT_ONLY_CFG
        else:
            raise RuntimeError(
                f"Unsupported args.quant_algo: {args.quant_algo} and args.weight_only: {args.weight_only} for int8 quantization"
            )
    elif args.quant_format == "fp8":
        # _disable_modelopt_fp8_cuda_extension()
        if args.weight_only:
            quant_cfg = mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG
        else:
            quant_cfg = FP8_CONFIG
    elif args.quant_format == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG
        quant_cfg["quant_cfg"]["*action_in_proj.encoder.trunk.0.input_quantizer"] = {
            "enable": False
        }
        quant_cfg["quant_cfg"]["*action_in_proj.encoder.trunk.0.weight_quantizer"] = {
            "enable": False
        }
    elif args.quant_format == "w4a8_nvfp4_fp8":
        quant_cfg = mtq.W4A8_NVFP4_FP8_CFG
    else:
        raise RuntimeError("Unsupported quantization format")

    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    # For weight-only PTQ, fold quantized weights back into module parameters and
    # disable weight quantizers. This avoids runtime quantize_op on exported graphs,
    # which can otherwise fail TRT conversion due to lifted fake scale tensors.
    if args.debug:
        logger.info("================== quantize_model summary ==================")
        mtq.print_quant_summary(model)

    if args.weight_only:
        mtq.fold_weight(model)

    return model


def auto_quantize_model(
    model,
    args,
    *,
    clip_ids,
    processor,
    t0_us: int,
    top_p: float,
    temperature: float,
    max_generation_length: int,
    calibration_traj_samples: int,
    device: str,
):
    """
    Quantize a PyTorch model using ModelOpt's AutoQuantize API.

    Searches per-layer across [NVFP4_DEFAULT_CFG, FP8_DEFAULT_CFG] under the
    effective-bits budget in args.auto_quantize_bits. Calibration data is built
    from the same joint VLM + diffusion rollout used by
    alpamayo_r1.eval.make_joint_calibration_forward_loop.

    Args:
        model: PyTorch model to quantize. Must be in eval mode.
        args: Namespace with `auto_quantize_bits` (float) and `debug` (bool).
        clip_ids: Iterable of clip_ids for calibration.
        processor: HF processor used for chat-template tokenization.
        t0_us, top_p, temperature, max_generation_length, calibration_traj_samples,
        device: Same semantics as make_joint_calibration_forward_loop.

    Returns:
        Quantized model (the search_state from mtq.auto_quantize is discarded).
    """
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    def _one_epoch():
        for clip_id in clip_ids:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
            messages = helper.create_message(
                data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
                "ego_future_xyz": data["ego_future_xyz"],
                "ego_future_rot": data["ego_future_rot"],
            }
            yield helper.to_device(model_inputs, device)

    class _ReusableLoader:
        """Re-iterable wrapper so modelopt can run calibration + scoring passes."""

        def __iter__(self):
            return _one_epoch()

    data_loader = _ReusableLoader()

    # --- Patch _fp8_eager to handle amax=0 without producing NaN ---
    # The eager FP8 fallback (used when the CUDA extension can't load) computes
    # scale = 448 / amax. For quantizers whose calibrated amax is 0, this becomes
    # inf, and `0 * inf` in the subsequent `x * scale` produces NaN. Replace zero
    # amax with a tiny positive value so the fake-quant output is 0 in that
    # channel (which is what a real FP8 kernel would produce).
    from modelopt.torch.quantization import tensor_quant as _mtq_tq

    _orig_fp8_eager = _mtq_tq._fp8_eager

    def _safe_fp8_eager(x, amax=None):
        if amax is not None:
            amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
        return _orig_fp8_eager(x, amax)

    _mtq_tq._fp8_eager = _safe_fp8_eager
    _mtq_tq.fp8_eager = _safe_fp8_eager

    # --- Diagnostic: also clamp NaN/inf scores as a belt-and-suspenders fallback ---
    from modelopt.torch.quantization import algorithms as _mtq_algos

    _orig_get_score = _mtq_algos._get_auto_quantize_score
    _nan_counter = {"count": 0, "examples": []}

    def _patched_get_score(grad_output, output_diff):
        score = _orig_get_score(grad_output, output_diff)
        if not torch.isfinite(score):
            _nan_counter["count"] += 1
            if len(_nan_counter["examples"]) < 3:
                _nan_counter["examples"].append(
                    {
                        "grad_finite": torch.isfinite(grad_output).all().item(),
                        "diff_finite": torch.isfinite(output_diff).all().item(),
                        "shape": tuple(grad_output.shape),
                    }
                )
            # Replace with 0 — treat this candidate as if it had no measurable
            # impact for this sample. If all samples for this (layer, recipe)
            # produce NaN, the aggregate score is 0 and LP picks the cheapest
            # option, which is fine.
            return torch.zeros_like(score)
        return score

    _mtq_algos._get_auto_quantize_score = _patched_get_score

    def forward_step(runtime_model, data):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = runtime_model.teacher_forced_flow_loss_forward(data=data)
        v_pred, v_target = out["v_pred"], out["v_target"]
        logger.info(
            f"[autoquant-fwd] v_pred: finite={torch.isfinite(v_pred).all().item()} "
            f"min={v_pred.min().item():.4g} max={v_pred.max().item():.4g} "
            f"abs_mean={v_pred.abs().mean().item():.4g} | "
            f"v_target: finite={torch.isfinite(v_target).all().item()} "
            f"min={v_target.min().item():.4g} max={v_target.max().item():.4g}"
        )
        return out

    def loss_func(output, batch):
        loss = torch.nn.functional.mse_loss(output["v_pred"], output["v_target"])
        logger.info(f"[autoquant-loss] loss={loss.item():.6g} finite={torch.isfinite(loss).item()}")
        return loss

    try:
        model, search_state = mtq.auto_quantize(
            model,
            constraints={"effective_bits": args.auto_quantize_bits},
            quantization_formats=["NVFP4_DEFAULT_CFG", FP8_CONFIG],
            data_loader=data_loader,
            forward_step=forward_step,
            loss_func=loss_func,
            disabled_layers="*lm_head*",
            num_calib_steps=512,
            num_score_steps=128,
            verbose=True,
        )
    finally:
        logger.info(f"[autoquant-nan] total non-finite score calls: {_nan_counter['count']}")
        for i, ex in enumerate(_nan_counter["examples"]):
            logger.info(f"[autoquant-nan][{i}] {ex}")
        _mtq_algos._get_auto_quantize_score = _orig_get_score
        _mtq_tq._fp8_eager = _orig_fp8_eager
        _mtq_tq.fp8_eager = _orig_fp8_eager

    logger.info("================== auto_quantize search_state ==================")
    logger.info(search_state)

    if args.debug:
        logger.info("================== auto_quantize_model summary ==================")
        mtq.print_quant_summary(model)

    return model


def read_clip_ids_from_parquet(parquet_path: str) -> list[str]:
    """Reads clip_ids from parquet. Returns unique clip_ids preserving first-occurrence order."""
    df = pd.read_parquet(str(parquet_path))
    cols_lower = {c.lower(): c for c in df.columns}
    clip_ids = df[cols_lower["key"]].astype(str).tolist()
    seen = set()
    uniq = []
    for cid in clip_ids:
        if cid not in seen:
            seen.add(cid)
            uniq.append(cid)
    return uniq
