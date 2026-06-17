#!/usr/bin/env python3
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

import argparse
import os
from pathlib import Path
from typing import Any

import einops
import modelopt.torch.opt as mto
import torch
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.token_utils import to_special_token
from alpamayo_r1.common import logging
from alpamayo_r1.common.logging import setup_logging
from tqdm import tqdm

from alpamayo1_5_quant.utils import read_clip_ids_from_parquet

setup_logging()

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


# Additional method that needs to be added to the Alpamayo1_5 class. We add this via setattr when this file is loaded
def _am1_5_ext_teacher_forced_flow_loss_forward(
    self,
    data: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Differentiable forward that returns the flow-matching training targets.

    Bypasses autoregressive reasoning generation and diffusion sampling.
    The VLM runs in a single non-sampling forward pass (with ``<traj_future_start>``
    appended to the prompt) to build the prompt KV cache; the expert then runs once
    on a linearly-interpolated noisy action and returns the predicted velocity field.

    Args:
        data: dict with ``tokenized_data`` (input_ids + other processor outputs),
            ``ego_history_xyz``, ``ego_history_rot``, ``ego_future_xyz``,
            ``ego_future_rot``.

    Returns:
        dict with keys ``v_pred`` and ``v_target``, both shape
        ``(B, n_diffusion_tokens, action_dim)``. Callers compute MSE between them.
    """
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    ego_future_xyz = data["ego_future_xyz"]
    ego_future_rot = data["ego_future_rot"]
    B, n_traj_group, _, _ = ego_history_xyz.shape
    assert n_traj_group == 1, "Only one trajectory group is supported."

    tokenized_data = dict(data["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
    device = input_ids.device

    # Append <traj_future_start> so the expert attends through the full prompt
    # that inference would have generated up to the action block.
    traj_future_start_id = self.tokenizer.convert_tokens_to_ids(
        to_special_token("traj_future_start")
    )
    start_col = torch.full(
        (input_ids.shape[0], 1),
        traj_future_start_id,
        dtype=input_ids.dtype,
        device=device,
    )
    input_ids = torch.cat([input_ids, start_col], dim=1)
    if "attention_mask" in tokenized_data and tokenized_data["attention_mask"] is not None:
        am = tokenized_data["attention_mask"]
        tokenized_data["attention_mask"] = torch.cat(
            [am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1
        )

    vlm_outputs = self.vlm(
        input_ids=input_ids,
        use_cache=True,
        return_dict=True,
        **tokenized_data,
    )
    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()
    rope_deltas = self.vlm.model.rope_deltas

    n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
    offset = torch.full((B,), prefill_seq_len, device=device, dtype=torch.long)

    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=B).clone()
    delta = rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    # No padding between prompt cache and action block: full attention mask.
    attention_mask = torch.zeros(
        (B, 1, n_diffusion_tokens, prefill_seq_len + n_diffusion_tokens),
        dtype=torch.float32,
        device=device,
    )

    forward_kwargs = {}
    if self.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    # Build flow-matching target: x_1 = GT action, x_0 ~ N(0, I).
    x_1 = self.action_space.traj_to_action(
        traj_history_xyz=ego_history_xyz[:, 0],
        traj_history_rot=ego_history_rot[:, 0],
        traj_future_xyz=ego_future_xyz[:, 0],
        traj_future_rot=ego_future_rot[:, 0],
    )  # (B, n_diffusion_tokens, 2)
    x_1 = x_1.to(device=device, dtype=torch.float32)

    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, 1, 1, device=device, dtype=x_1.dtype)
    x_t = (1.0 - t) * x_0 + t * x_1
    v_target = x_1 - x_0

    # Cast to action-module dtype to match action_in_proj / expert weights.
    proj_dtype = next(self.action_in_proj.parameters()).dtype
    x_t_cast = x_t.to(dtype=proj_dtype)
    t_cast = t.to(dtype=proj_dtype)

    future_token_embeds = self.action_in_proj(x_t_cast, t_cast)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(B, n_diffusion_tokens, -1)

    expert_out = self.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=prompt_cache,
        attention_mask=attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    prompt_cache.crop(prefill_seq_len)
    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    v_pred = self.action_out_proj(last_hidden).view(B, *self.action_space.get_action_space_dims())

    return {"v_pred": v_pred.to(torch.float32), "v_target": v_target}


setattr(
    Alpamayo1_5, "teacher_forced_flow_loss_forward", _am1_5_ext_teacher_forced_flow_loss_forward
)


def make_joint_calibration_forward_loop(
    *,
    clip_ids: list[str],
    processor,
    t0_us: int,
    top_p: float,
    temperature: float,
    max_generation_length: int,
    calibration_traj_samples: int,
    device: str,
):
    """
    Build a calibration loop that exercises both VLM generation and diffusion.

    This avoids text-only calibration and ensures quantizers in the rollout path
    (vlm/expert/diffusion-related modules) observe representative activations.
    """

    def _calibration_loop(runtime_model):
        runtime_model.eval()
        with torch.no_grad():
            for clip_id in tqdm(clip_ids, desc="calibration"):
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
                }
                model_inputs = helper.to_device(model_inputs, device)

                with torch.autocast("cuda", dtype=torch.float16):
                    runtime_model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=top_p,
                        temperature=temperature,
                        num_traj_samples=calibration_traj_samples,
                        max_generation_length=max_generation_length,
                    )

    return _calibration_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="nvidia/Alpamayo-1.5-10B")
    ap.add_argument("--t0_us", type=int, default=5_100_000)
    ap.add_argument("--num_traj_samples", type=int, default=6)
    ap.add_argument("--max_generation_length", type=int, default=256)
    ap.add_argument("--top_p", type=float, default=0.98)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument(
        "--quant_format",
        type=str,
        required=True,
        choices=["fp8", "nvfp4", "w4a8_nvfp4_fp8", "auto"],
    )
    ap.add_argument(
        "--auto_quantize_bits",
        type=float,
        default=4.8,
        help="Effective-bits budget for AutoQuantize (only used when --quant_format auto)",
    )
    ap.add_argument("--quant_algo", type=str, default="max", choices=["max", "smoothquant"])
    ap.add_argument("--quant_weight_only", action="store_true")
    ap.add_argument(
        "--calib_parquet", type=str, default="0417_5k_train_set_for_calibration_25.10.parquet"
    )
    ap.add_argument("--num_of_calib_clips", type=int, default=100)
    ap.add_argument("--save_model_dir", type=str, required=True)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    calib_parquet_path = (script_dir / args.calib_parquet).resolve()
    calib_clip_ids = read_clip_ids_from_parquet(str(calib_parquet_path))
    calib_clip_ids = calib_clip_ids[: args.num_of_calib_clips]
    logger.info(f"Loaded {len(calib_clip_ids)} calibration clip_ids from: {calib_parquet_path}")

    device = "cuda"
    mto.enable_huggingface_checkpointing()
    model = Alpamayo1_5.from_pretrained(args.ckpt, dtype=torch.float16).to(
        device=device, dtype=torch.float16
    )
    model.eval()

    processor = helper.get_processor(model.tokenizer)

    from alpamayo1_5_quant.utils import auto_quantize_model, quantize_model

    logger.info(f"Quantizing model ({args.quant_format}) ...")

    quantization_args = argparse.Namespace(
        quant_format=args.quant_format,
        quant_algo=args.quant_algo,
        weight_only=args.quant_weight_only,
        debug=True,
        auto_quantize_bits=args.auto_quantize_bits,
    )

    if args.quant_format == "auto":
        with torch.enable_grad():
            model = auto_quantize_model(
                model,
                quantization_args,
                clip_ids=calib_clip_ids,
                processor=processor,
                t0_us=args.t0_us,
                top_p=args.top_p,
                temperature=args.temperature,
                max_generation_length=args.max_generation_length,
                calibration_traj_samples=args.num_traj_samples,
                device=device,
            )
    else:
        calibration_forward_loop = make_joint_calibration_forward_loop(
            clip_ids=calib_clip_ids,
            processor=processor,
            t0_us=args.t0_us,
            top_p=args.top_p,
            temperature=args.temperature,
            max_generation_length=args.max_generation_length,
            calibration_traj_samples=args.num_traj_samples,
            device=device,
        )
        model = quantize_model(
            model,
            quantization_args,
            calibration_forward_loop=calibration_forward_loop,
        )
    model.eval()

    save_dir = os.path.join(
        args.save_model_dir,
        f"alpamayo1.5_{args.quant_format}"
        f"{'_' + str(args.auto_quantize_bits) + 'bits' if args.quant_format == 'auto' else ''}"
        f"{'_weight_only' if args.quant_weight_only else ''}"
        f"_calib{args.num_of_calib_clips}",
    )
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving quantized model to: {save_dir}")
    model.save_pretrained(save_dir)
    logger.info(f"Quantized model saved to: {save_dir}")


if __name__ == "__main__":
    main()
