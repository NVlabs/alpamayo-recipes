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

"""Normal SpecForge imports; shared target embedding/head, no target duplication."""

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel, OnlineDominoModel
from specforge.modeling.draft.dflash import build_target_layer_ids
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from transformers import Qwen3Config

from .contracts import SPECFORGE_REVISION

DRAFT_CLASSES = {"domino": DominoDraftModel, "dflash2": DFlash2DraftModel}


def make_config(args, target):
    tc = target.text_config
    layers = args.target_layers
    if layers is None:
        layers = build_target_layer_ids(tc.num_hidden_layers, args.draft_layers)
    if (
        not layers
        or layers != sorted(set(layers))
        or min(layers) < 0
        or max(layers) >= tc.num_hidden_layers - 1
    ):
        raise ValueError("Target layers must be increasing, unique and exclude the final layer")
    if args.block_size < 2:
        raise ValueError("Block size must be >= 2")
    config = Qwen3Config(
        hidden_size=tc.hidden_size,
        intermediate_size=tc.intermediate_size,
        num_hidden_layers=args.draft_layers,
        num_attention_heads=tc.num_attention_heads,
        num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim,
        rms_norm_eps=tc.rms_norm_eps,
        rope_theta=tc.rope_theta,
        max_position_embeddings=tc.max_position_embeddings,
        vocab_size=tc.vocab_size,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        layer_types=["full_attention"] * args.draft_layers,
        use_sliding_window=False,
    )
    config.block_size = args.block_size
    config.num_target_layers = tc.num_hidden_layers
    config.specdec_method = args.method
    config.specforge_revision = SPECFORGE_REVISION
    config.dflash_config = {
        "block_size": args.block_size,
        "mask_token_id": target.mask_id,
        "target_layer_ids": layers,
    }
    if args.method == "domino":
        config.dflash_config.update(
            emb_dim=256, gru_hidden_dim=1024, pure_draft_prefix_len=1, shift_label=True
        )
    else:
        config.dflash_config.update(
            conv_kernel_size=2, conv_group_size=16, selector_rank=256, selector_top_k=16
        )
    config._attn_implementation = "sdpa"
    return config


def build_engine(draft, target, num_anchors):
    common = dict(
        draft_model=draft,
        target_lm_head=target.vlm.get_output_embeddings(),
        target_embed_tokens=target.vlm.get_input_embeddings(),
        mask_token_id=target.mask_id,
        block_size=draft.config.block_size,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=7.0,
        objective_chunk_blocks=8,
    )
    if draft.config.specdec_method == "domino":
        return OnlineDominoModel(**common, shift_label=True)
    return OnlineDFlashModel(
        **common,
        selector_loss_alpha=1.0,
        selector_warmup_ratio=0.04,
        selector_ramp_ratio=0.04,
        selector_stop_gradient=False,
    )
