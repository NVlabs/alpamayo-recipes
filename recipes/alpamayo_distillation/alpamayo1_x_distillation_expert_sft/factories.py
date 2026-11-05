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

"""Model factories for the expert-SFT stage (stage 2).

Stage 2 fine-tunes ONLY the action expert of an already-distilled student
(stage-1 output) on ground-truth future trajectories, conditioned on the
model's own greedily generated Chain-of-Causation reasoning. Two deliberate
deviations from the stage-1 student build, both required to reproduce the
validated recipe:

1. The student is LOADED FROM the stage-1 checkpoint (``from_pretrained``),
   not re-assembled from a teacher + fresh VLM.
2. The whole VLM is frozen (``vlm.*``); the remaining trainable parameters
   (action expert, action in/out projections, trajectory tokenizers) stay in
   bfloat16 master precision — the validated runs loaded the full checkpoint
   with ``torch_dtype=bfloat16`` and trained the expert head in bf16.
   (``trainable_params_fp32=True`` upcasts them to fp32 instead; measured
   effect: ~1.8x larger weight displacement per step and strong
   checkpoint-to-checkpoint eval oscillation, so it is NOT the validated
   recipe — keep it off to reproduce the published trajectory statistics.)

The "teacher" of this stage is the stage-1 student itself (self-conditioning:
it generates the reasoning prefix that the expert attends to). Point
``teacher.kwargs.name_or_path`` at the SAME stage-1 checkpoint; the stock
``alpamayo1_5_distill.training.factory.build_teacher_model`` loads it as-is.
"""

from __future__ import annotations

import torch

from alpamayo1_5_distill.models.modeling_distill import DistillAlpamayo1_5


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def build_expert_sft_student(
    name_or_path: str,
    vlm_name_or_path: str | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    freeze_vlm: bool = True,
    trainable_params_fp32: bool = False,
) -> DistillAlpamayo1_5:
    """Load the stage-1 student checkpoint and prepare it for expert-only SFT.

    Args:
        name_or_path: stage-1 distilled checkpoint (local dir or hub id).
        vlm_name_or_path: optional override when the checkpoint's embedded
            VLM path is machine-specific.
        freeze_vlm: freeze every ``vlm.*`` parameter (the validated recipe's
            ``--diffusion_head_only``). Trainable remainder: ``expert.*``,
            ``action_in_proj.*``, ``action_out_proj.*`` and the trajectory
            tokenizers.
        trainable_params_fp32: upcast trainable parameters to float32 master
            weights (forward still runs under bf16 autocast). The validated
            runs kept bf16 masters — leave this False to reproduce them.
    """
    config = DistillAlpamayo1_5.config_with_vlm_override(
        name_or_path,
        vlm_name_or_path,
    )
    model = DistillAlpamayo1_5.from_pretrained(
        name_or_path,
        config=config,
        dtype=_dtype(dtype),
        attn_implementation=attn_implementation,
    )
    if freeze_vlm:
        n_frozen = 0
        for name, parameter in model.named_parameters():
            if name.startswith("vlm."):
                parameter.requires_grad_(False)
                n_frozen += 1
        if n_frozen == 0:
            raise AttributeError("freeze_vlm=True but no 'vlm.*' parameters found")
    if trainable_params_fp32:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters left after freezing")
    prefixes = sorted({n.split(".")[0] for n in trainable})
    print(f"[expert-sft] trainable prefixes: {prefixes} "
          f"({len(trainable)} tensors, fp32={trainable_params_fp32}, "
          f"vlm frozen={freeze_vlm})", flush=True)
    return model
