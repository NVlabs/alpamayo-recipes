#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the Alpamayo2-Edge pieces the stock exporter does not cover, with stock Edge-LLM machinery.

``tensorrt-edgellm-export --task reasoning`` covers the VLM (visual + text engine); host/action_graph.py the expert.

``llm``: the text engine again, with one extra output ``hidden_states`` = the decoder outputs of the drafter's tap
layers (default 1, 7, 13, 19, 25) concatenated per token, for speculative decoding with a DFlash2 drafter. Same
graph otherwise: the stock exporter with a two-line model subclass.

``draft``: the DFlash2 drafter through the stock ``--dflash-draft`` exporter (ONNX + selector codebooks). The
stock contract insists on five draft layers (the public z-lab drafts); the Alpamayo2-Edge drafter has four, so the
config builder is wrapped without that check. The snapshot's ``draft.safetensors`` is linked as ``model.safetensors``.

Usage: python export.py llm SHIM OUT [TAPS] [--vocab DIR]
       python export.py draft SHIM SNAPSHOT OUT [--vocab DIR]
  SHIM      shim.py output directory
  OUT       destination, e.g. onnx/fp8/llm or onnx/draft
  TAPS      comma-separated tap layers (default 1,7,13,19,25)
  SNAPSHOT  drafter directory: config.json (dflash_config block) + draft.safetensors
"""

import json
import logging
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def export_llm(shim: Path, out: Path, taps: list, vocab: Path | None = None) -> None:
    from tensorrt_edgellm import _export_api as api
    from tensorrt_edgellm.model import register_model
    from tensorrt_edgellm.models.cosmos3_reasoner import Cosmos3ReasonerCausalLM
    from tensorrt_edgellm.scripts.export import _export_llm

    class TappedCausalLM(Cosmos3ReasonerCausalLM):
        """The stock model plus a ``hidden_states`` output: the tap layers' outputs, [tokens, len(taps) * hidden].
        ``logits_indices`` gets its own dynamic dimension: the stock graph ties it to the sequence count (one logits
        row per sequence), but speculative verification scores every row of one sequence."""

        emit_hidden_states = True

        def _target_hidden_layer_ids(self):
            return taps

        def onnx_export_spec(self):
            spec = super().onnx_export_spec()
            index = spec.input_names.index("logits_indices")
            spec.dynamic_shapes[index] = {0: torch.export.Dim("logits_rows", min=1, max=8_388_608)}
            return spec

    for model_type in ("cosmos3_edge", "cosmos3_edge_text"):
        register_model(model_type, TappedCausalLM, api.standard_attention_scale)
    _export_llm(str(shim), str(out), model_type="cosmos3_edge", reduced_vocab_dir=str(vocab or ""))
    cfg_path = out / "config.json"
    cfg = json.load(open(cfg_path))
    cfg["hidden_states_tap_layers"] = taps  # read by thor/runtime (the stock runtime ignores it)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print({"hidden_states_tap_layers": taps, "reduced_vocab_size": cfg.get("reduced_vocab_size")})


def export_draft(shim: Path, snapshot: Path, out: Path, vocab: Path | None = None) -> None:
    from dataclasses import replace
    from tensorrt_edgellm import config as cfgmod, model as modelmod
    from tensorrt_edgellm.scripts.export import _export_dflash_draft

    staged = out.parent / (out.name + ".checkpoint")
    staged.mkdir(parents=True, exist_ok=True)
    weights = (
        snapshot / "draft.safetensors"
    )  # a training snapshot, or the quantizer's model.safetensors
    if not weights.is_file():
        weights = snapshot / "model.safetensors"
    for name, source in (
        ("model.safetensors", weights),
        ("hf_quant_config.json", snapshot / "hf_quant_config.json"),
    ):
        link = staged / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if source.is_file():
            link.symlink_to(source.resolve())
    config = json.load(open(snapshot / "config.json"))
    config["dflash_config"].setdefault(
        "is_causal", False
    )  # the stock contract wants it spelled out; the block is non-causal
    json.dump(config, open(staged / "config.json", "w"), indent=2)

    def four_layer_ok(draft_dir, default_attention_scale):
        root, llm = cfgmod.load_checkpoint_config_dicts(draft_dir)
        resolved = cfgmod.resolve_dflash_contract(root, llm)
        if resolved.version != cfgmod.DFlashVersion.V2:
            raise ValueError("DFlash2 draft checkpoint requires architecture DFlash2DraftModel")
        config = cfgmod.make_dflash_draft_config(draft_dir, default_attention_scale)
        return replace(
            config,
            dflash_version=resolved.version,
            dflash_target_layer_ids=list(resolved.target_layer_ids),
            dflash_block_size=resolved.block_size,
            dflash_mask_token_id=resolved.mask_token_id,
            is_dflash_draft_flag=True,
            dflash2_target_layer_ids=list(resolved.target_layer_ids),
            dflash2_block_size=resolved.block_size,
            dflash2_mask_token_id=resolved.mask_token_id,
            dflash2_is_causal=resolved.is_causal,
            dflash2_conv_kernel_size=resolved.conv_kernel_size,
            dflash2_conv_group_size=resolved.conv_group_size,
            dflash2_selector_rank=resolved.selector_rank,
            dflash2_selector_top_k=resolved.selector_top_k,
        )

    modelmod.make_dflash2_draft_config = (
        four_layer_ok  # the stock function minus "exactly five draft layers"
    )
    _export_dflash_draft(str(shim), str(out), str(staged))
    if vocab:
        reduce_draft_head(out, vocab)
    print(sorted(p.name for p in out.iterdir()))


def reduce_draft_head(out: Path, vocab: Path) -> None:
    """Keep only the reduced vocabulary's columns of the drafter's LM head; the proposals come out as reduced ids
    and ``draft_vocab_map.safetensors`` (the stock sidecar) maps them back. The stock exporter declines this for
    DFlash2 because its C++ decoder would not remap; thor/runtime/drafter_trt.py does."""
    import onnx
    import shutil

    vocab_map = load_file(str(vocab / "vocab_map.safetensors"))["vocab_map"].long()
    model = onnx.load(str(out / "model.onnx"))
    g = model.graph
    producer = {o: n for n in g.node for o in n.output}
    topk = next(
        n for n in g.node if n.op_type == "TopK" and "spec_proposal_unary_values" in n.output
    )
    matmul = producer[producer[topk.input[0]].input[0]]  # TopK <- Cast <- MatMul(hidden, head)
    head = next(t for t in g.initializer if t.name == matmul.input[1])
    weight = torch.from_numpy(onnx.numpy_helper.to_array(head).copy())  # [hidden, vocab]
    reduced = onnx.numpy_helper.from_array(weight[:, vocab_map].contiguous().numpy(), head.name)
    head.CopyFrom(reduced)
    data = out / "model.onnx.data"
    data.unlink()
    onnx.save(
        model,
        str(out / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data.name,
        size_threshold=0,
    )
    shutil.copy(vocab / "vocab_map.safetensors", out / "draft_vocab_map.safetensors")
    json.dump(
        {
            "vocab_size": weight.shape[1],
            "reduced_vocab_size": int(vocab_map.numel()),
            "source": str(vocab),
        },
        open(out / "draft_reduced_vocab.json", "w"),
        indent=1,
    )
    cfg = json.load(open(out / "config.json"))
    cfg["reduced_vocab_size"] = int(vocab_map.numel())
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    print({"draft_head": list(weight.shape), "reduced_vocab_size": int(vocab_map.numel())})


if __name__ == "__main__":
    args = sys.argv[1:]
    vocab = Path(args.pop(args.index("--vocab") + 1)) if "--vocab" in args else None
    if vocab:
        args.remove("--vocab")
    if len(args) == 4 and args[0] == "draft":
        export_draft(Path(args[1]), Path(args[2]), Path(args[3]), vocab)
    elif len(args) in (3, 4) and args[0] == "llm":
        export_llm(
            Path(args[1]),
            Path(args[2]),
            [int(v) for v in (args[3] if len(args) == 4 else "1,7,13,19,25").split(",")],
            vocab,
        )
    else:
        sys.exit(__doc__)
