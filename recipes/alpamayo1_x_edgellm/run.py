# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the six DRIVE Thor engines over a job list; one result row per request.

    python -m alpamayo1_x_edgellm.run --engines <build> --model <bundle> \\
        --vlm-metadata <Cosmos-Reason2 dir> --jobs <fixture>/jobs.json --output results.json \\
        [--reference gold20-reference.json]

One request: newest-tick vision (older ticks come from a per-clip cache) -> embeddings ->
full-attention prefill -> Domino draft / Q17 verify rounds until the trajectory start token
(Q1 appends it when it was not an accepted draft) -> five-step action over 6 candidates -> unicycle
trajectory. Decoding is greedy and the action noise is the bundle's fixed ``action-noise.bin``.

The language K/V cache is one paged FP16 pool per layer (TensorRT Edge-LLM ``AttentionPlugin``
layout, ``export/graph.py``): the prefill fills pages 0..23, the verifier extends rows in place up to
page 24, and the action expert gives each of its six candidates two private pages after them. ``request_ms`` is wall
time from the token handoff to the trajectory; prompt building, older vision ticks and the metric
are outside it. Latency is summarized over non-seed requests (each clip's first request is
excluded; p90 nearest-rank).

With ``--reference``, rows pair by (eval_index, clip_id, horizon) with the reference run and, for
delta = minADE6 - reference minADE6, the summary reports mean(delta), the linearly interpolated
90th percentile and the endpoint (horizon 16) mean difference against guideline limits of 0.02 /
0.10 / 0.05 m. The limits are guidance, not a hard gate: the 90th percentile varies by a few
centimetres between boards and engine builds.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from alpamayo1_x_edgellm.core import Bundle, inference_config, write_json
from alpamayo1_x_edgellm.export.action import CANDIDATES, MAX_PAGES as ACTION_PAGES, STEPS
from alpamayo1_x_edgellm.export.graph import NUM_PAGES, PAGE, PHASE_SPEC_TARGET_VERIFY
from alpamayo1_x_edgellm.export.prefill import PAGES as PREFILL_PAGES
from alpamayo1_x_edgellm.export.verify import LANGUAGE_PAGES

ROWS, HIDDEN, LAYERS, FEATURES = 3006, 2048, 28, 10240
IMAGES, PATCHES, MERGED = 16, 720, 180
PROPOSALS, MAX_NEW_TOKENS, ACTION_PREFIX = 16, 128, 3072  # ACTION_PREFIX: the action page table holds 24 prefix pages
ROPE_ROWS = LANGUAGE_PAGES * PAGE  # M-RoPE rows prepared per prompt layout
ACTION_PRIVATE = LANGUAGE_PAGES  # first of the 6 x 2 private action pages
MASK_TOKEN = 155696  # Domino proposal rows 1..15
TENSORRT_PLUGINS = ("NvInfer_edgellm_plugin",)
GATES = {"mean_m": 0.02, "p90_m": 0.10, "endpoint_m": 0.05}


# --------------------------------------------------------------------------- #
# TensorRT
# --------------------------------------------------------------------------- #
class Engine:
    """One deserialized engine; one execution context per optimization profile."""

    def __init__(self, trt, runtime, path: Path, profiles: int = 1):
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {path}")
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = {n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT}
        self.placeholder = torch.zeros(256, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        self.contexts = []
        for profile in range(profiles):
            context = self.engine.create_execution_context()
            context.set_optimization_profile_async(profile, stream)
            self.contexts.append(context)
        torch.cuda.synchronize()

    def __call__(self, tensors: dict[str, torch.Tensor], profile: int = 0) -> None:
        context = self.contexts[profile]
        for name in self.names:
            tensor = tensors[name]
            if name in self.inputs:
                context.set_input_shape(name, tuple(tensor.shape))
            # An empty tensor (Domino's first-round ``past``) still needs a valid address.
            context.set_tensor_address(name, tensor.data_ptr() if tensor.numel() else self.placeholder.data_ptr())
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")


# --------------------------------------------------------------------------- #
# Model pieces used outside the engines: processor, embeddings, M-RoPE, trajectory
# --------------------------------------------------------------------------- #
def load_model(checkpoint: Path, vlm_metadata: Path):
    """BF16 AlpamayoR1 without its language/vision layers (the engines replace them).

    Kept: tokenizer + slow Qwen3-VL image processor (576x320 -> 720 patches), embedding table,
    M-RoPE module, history-trajectory tokenizer and the action space (BF16 statistics)."""
    import alpamayo_r1.models.base_model as base_model
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    class ImageOnlyProcessor:
        @staticmethod
        def from_pretrained(name, **kwargs):
            pixels = {k: v for k, v in kwargs.items() if k in ("min_pixels", "max_pixels")}
            images = AutoImageProcessor.from_pretrained(name, use_fast=False, **pixels)
            tokenizer = AutoTokenizer.from_pretrained(name)
            attributes, Qwen3VLProcessor.attributes = Qwen3VLProcessor.attributes, ["image_processor", "tokenizer"]
            try:
                processor = Qwen3VLProcessor(image_processor=images, tokenizer=tokenizer, video_processor=None,
                                             chat_template=getattr(tokenizer, "chat_template", None))
            finally:
                Qwen3VLProcessor.attributes = attributes
            processor.video_processor = None
            return processor

    config = inference_config(json.loads((checkpoint / "config.json").read_text()))
    config.update(vlm_name_or_path=str(vlm_metadata), attn_implementation="sdpa")
    original, base_model.AutoProcessor = base_model.AutoProcessor, ImageOnlyProcessor
    AlpamayoR1._supports_sdpa = True
    try:
        model = AlpamayoR1.from_pretrained(checkpoint, config=AlpamayoR1Config(**config), dtype=torch.bfloat16,
                                           attn_implementation="sdpa", local_files_only=True).cuda().eval()
    finally:
        base_model.AutoProcessor = original
    model.requires_grad_(False)
    processor = model.processor
    processor.tokenizer = model.tokenizer
    multimodal = model.vlm.model
    multimodal.language_model.layers = torch.nn.ModuleList()
    visual = multimodal.visual
    visual.blocks, visual.deepstack_merger_list = torch.nn.ModuleList(), torch.nn.ModuleList()
    visual.patch_embed = visual.pos_embed = visual.merger = None
    torch.cuda.empty_cache()
    return model, processor


def visual_geometry(grid, *, side=48, merge=2, theta=10_000.0) -> dict[str, torch.Tensor]:
    """Vision engine position inputs for images of one grid, in 2x2-merge token order: rotary
    angles (vision theta 10000) and the four bilinear taps into the 48x48 position table."""
    index_rows, weight_rows, rotary = [[] for _ in range(4)], [[] for _ in range(4)], []
    inverse = 1.0 / (theta ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32.0))
    table = torch.outer(torch.arange(max(max(h, w) for _, h, w in grid), dtype=torch.float32), inverse)
    for frames, height, width in grid:
        hv, wv = torch.linspace(0, side - 1, height), torch.linspace(0, side - 1, width)
        hf, wf = hv.int(), wv.int()
        hc, wc = (hf + 1).clip(max=side - 1), (wf + 1).clip(max=side - 1)
        dh, dw = hv - hf, wv - wf
        mh, mw = height // merge, width // merge
        indices = [(b.reshape(mh, 1, merge, 1) + w.reshape(1, mw, 1, merge)).flatten()
                   for b in (hf * side, hc * side) for w in (wf, wc)]
        weights = [(v.reshape(mh, 1, merge, 1) * w.reshape(1, mw, 1, merge)).flatten()
                   for v in (1 - dh, dh) for w in (1 - dw, dw)]
        for tap in range(4):
            index_rows[tap].extend(indices[tap].repeat(frames).tolist())
            weight_rows[tap].extend(weights[tap].repeat(frames).tolist())
        rows = (torch.arange(mh)[:, None, None, None] * merge + torch.arange(merge)[None, None, :, None]).expand(mh, mw, merge, merge)
        cols = (torch.arange(mw)[None, :, None, None] * merge + torch.arange(merge)[None, None, None, :]).expand(mh, mw, merge, merge)
        rotary.append(table[torch.stack((rows.reshape(-1), cols.reshape(-1)), -1).repeat(frames, 1)].flatten(1))
    return {"rotary_pos_emb": torch.cat(rotary).cuda(),
            "fast_pos_embed_idx": torch.tensor(index_rows, dtype=torch.int64).cuda(),
            "fast_pos_embed_weight": torch.tensor(weight_rows, dtype=torch.float16).cuda()}


def plain_rope(positions: torch.Tensor) -> torch.Tensor:
    """1-D RoPE [rows, 128] (cos | sin) at language theta 5e6, for Domino and the action expert."""
    inverse = 5e6 ** (-2 * torch.arange(64, device="cuda", dtype=torch.float32) / 128)
    phase = positions.float()[:, None] * inverse
    return torch.cat((phase.cos(), phase.sin()), -1).contiguous()


# --------------------------------------------------------------------------- #
# The request pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, engines: Path, model: Bundle, vlm_metadata: Path):
        import tensorrt as trt

        plugins = engines / "plugins"
        self.handles = [ctypes.CDLL(str(plugins / f"lib{n}.so"), mode=ctypes.RTLD_GLOBAL) for n in TENSORRT_PLUGINS]
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        load = lambda name, profiles=1: Engine(trt, runtime, engines / f"{name}.engine", profiles)
        self.visual, self.prefill, self.q17, self.q1, self.action = (
            load(n) for n in ("visual", "prefill", "verify_q17", "verify_q1", "action"))
        self.domino = load("domino", 3)
        self.build = json.loads((engines / "build.json").read_text())

        self.model, self.processor = load_model(model.directory, vlm_metadata)
        self.multimodal = self.model.vlm.model
        self.embedding = self.multimodal.get_input_embeddings()
        self.rotary = next(m for n, m in self.model.vlm.named_modules() if n.endswith("rotary_emb") and hasattr(m, "mrope_section"))
        self.stop = int(self.model.tokenizer.traj_token_ids["future_start"])
        self.image_token = int(self.model.vlm.config.image_token_id)

        empty = lambda *shape, dtype=torch.float16: torch.zeros(shape, dtype=dtype, device="cuda")
        self.pool = empty(LAYERS, 2, NUM_PAGES, PAGE, 8, 128)  # paged K/V pool of every layer, FP16
        self.prefill_kv = empty(LAYERS, 2, PREFILL_PAGES, PAGE, 8, 128)  # the prefill's pages, copied into the pool
        self.phase = empty(PHASE_SPEC_TARGET_VERIFY, dtype=torch.int32)  # shape-only: tree-decode phase
        self.carrier = empty(0, dtype=torch.int32)  # shape-only: no context sequences
        self.rope = empty(1, ROPE_ROWS, 128, dtype=torch.float32)  # M-RoPE rows for prefill and decode
        self.layout = None
        self.drope = plain_rope(torch.arange(3264 + 3 * 728, device="cuda"))
        self.history = empty(3264, FEATURES)
        self.fresh_context = empty(ROWS, FEATURES)
        self.noise = empty(PROPOSALS, HIDDEN)
        self.noise[1:] = self.embed([MASK_TOKEN] * (PROPOSALS - 1))
        self.patches = empty(4 * PATCHES, 1536)
        self.geometry = visual_geometry([(1, 20, 36)] * 4)
        self.vision_cache: dict[str, dict[str, torch.Tensor]] = {}
        self.clip = None

        self.action_noise = torch.from_numpy(np.frombuffer(model.noise(), "<f4").copy()).reshape(384, 2).cuda()

    def embed(self, ids) -> torch.Tensor:
        """BF16 embedding rows rounded to FP16, as the engines consume them."""
        ids = torch.as_tensor(ids, device="cuda").reshape(-1)
        return self.embedding(ids).to(torch.float16)

    # --- prompt (outside the timer) ------------------------------------------------------------
    def prompt(self, request_path: Path):
        from alpamayo_r1 import helper
        from PIL import Image

        request = json.loads(request_path.read_text())
        content = request["requests"][0]["messages"][1]["content"]
        paths = [self._image_path(request_path, c["image"]) for c in content if c["type"] == "image"]
        trajectory = next(c["trajectory"] for c in content if c["type"] == "trajectory")
        if len(paths) != IMAGES:
            raise ValueError(f"{request_path}: expected 16 camera-major images")
        frames = torch.stack([torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1) for p in paths])
        inputs = self.processor.apply_chat_template(helper.create_message(frames), tokenize=True,
                                                    add_generation_prompt=False, continue_final_message=True,
                                                    return_dict=True, return_tensors="pt")
        if inputs["input_ids"].shape != (1, ROWS) or inputs["pixel_values"].shape != (IMAGES * PATCHES, 1536):
            raise ValueError(f"{request_path}: prompt is not the 3006-row, 16 x 720-patch layout")
        ego = {"ego_history_xyz": torch.tensor(trajectory, dtype=torch.float32).reshape(1, 1, 16, 3),
               "ego_history_rot": torch.eye(3).expand(1, 1, 16, 3, 3).contiguous()}  # the fixture has no rotations
        pixels = inputs["pixel_values"].to(torch.bfloat16)  # the vision input is the BF16-rounded patch
        return [str(p) for p in paths], inputs["input_ids"], inputs["image_grid_thw"], pixels, ego

    @staticmethod
    def _image_path(request_path: Path, image: str) -> Path:
        path = Path(image)
        if path.is_absolute():
            return path
        for root in (request_path.parent.parent, request_path.parent):
            if (root / path).is_file():
                return root / path
        raise FileNotFoundError(image)

    # --- vision -----------------------------------------------------------------------------------
    def vision_tick(self, paths: list[str], pixels: torch.Tensor, tick: int) -> None:
        """Encode tick ``tick`` of the four cameras (images c*4+tick) and cache each camera's rows."""
        images = [camera * 4 + tick for camera in range(4)]
        for camera, image in enumerate(images):
            self.patches[camera * PATCHES:(camera + 1) * PATCHES].copy_(pixels[image * PATCHES:(image + 1) * PATCHES])
        outputs = {n: torch.empty(4 * MERGED, HIDDEN, dtype=torch.float16, device="cuda")
                   for n in ("output", "deepstack_features_0", "deepstack_features_1", "deepstack_features_2")}
        self.visual({"input": self.patches, **self.geometry, **outputs})
        for camera, image in enumerate(images):
            self.vision_cache[paths[image]] = {n: v[camera * MERGED:(camera + 1) * MERGED] for n, v in outputs.items()}

    # --- M-RoPE ---------------------------------------------------------------------------------
    def pack_rope(self, position_ids: torch.Tensor, start: int) -> None:
        """Interleaved M-RoPE (sections T/H/W) for position_ids [3, 1, n] into rope rows [start, start+n)."""
        n = position_ids.shape[-1]
        inverse = self.rotary.inv_freq.float()[None, None, :, None].expand(3, 1, -1, 1)
        freqs = (inverse @ position_ids[:, :, None, :].float()).transpose(2, 3)
        interleaved = freqs[0].clone()
        sections = list(self.rotary.mrope_section)
        for axis, offset in ((1, 1), (2, 2)):
            interleaved[..., offset:sections[axis] * 3:3] = freqs[axis, ..., offset:sections[axis] * 3:3]
        scale = self.rotary.attention_scaling
        self.rope[:, start:start + n] = torch.cat((interleaved.cos() * scale, interleaved.sin() * scale), -1).float()

    def prepare_layout(self, fused_ids: torch.Tensor, grid: torch.Tensor) -> None:
        """Image row indices and the rope table depend only on the prompt layout: compute once."""
        key = (fused_ids == self.image_token).nonzero()[:, 1]
        if self.layout is not None and torch.equal(self.layout["image_rows"], key):
            return
        positions, deltas = self.multimodal.get_rope_index(fused_ids, grid.cuda(), None, attention_mask=None)
        self.pack_rope(positions, 0)
        decode = torch.arange(ROPE_ROWS - ROWS, device="cuda").view(1, -1).add((ROWS + deltas).cuda()).unsqueeze(0).expand(3, -1, -1)
        self.pack_rope(decode, ROWS)
        self.layout = {"image_rows": key, "rope_deltas": int(deltas.reshape(-1)[0])}

    # --- language -------------------------------------------------------------------------------
    def run_prefill(self, paths, fused_ids) -> tuple[int, torch.Tensor]:
        rows = self.layout["image_rows"]
        hidden = self.embedding(fused_ids[0]).to(torch.float16)
        cached = [self.vision_cache[p] for p in paths]  # image order c*4+t == token order
        hidden.index_copy_(0, rows, torch.cat([c["output"] for c in cached]).to(torch.bfloat16).to(torch.float16))
        deepstack = {}
        for i in range(3):
            d = torch.zeros(ROWS, HIDDEN, dtype=torch.float16, device="cuda")
            d.index_copy_(0, rows, torch.cat([c[f"deepstack_features_{i}"] for c in cached]).to(torch.bfloat16).to(torch.float16))
            deepstack[f"deepstack_{i}"] = d
        kv = {f"kv_{i}": self.prefill_kv[i] for i in range(LAYERS)}
        features = torch.empty(ROWS, FEATURES, dtype=torch.float16, device="cuda")
        logits = torch.empty(1, self.embedding.weight.shape[0], dtype=torch.float16, device="cuda")
        self.prefill({"hidden": hidden, **deepstack, "rope": self.rope[0, :ROWS], **kv, "features": features, "logits": logits})
        self.pool[:, :, :PREFILL_PAGES].copy_(self.prefill_kv)
        return int(logits.argmax(-1).item()), features

    def draft(self, features: torch.Tensor, context_start: int, anchor: int) -> list[int]:
        """Domino: project the new target features into the drafter context and propose 16 tokens."""
        rows = features.shape[0]
        proposal = context_start + rows
        self.noise[0] = self.embed([anchor])[0]
        tokens = torch.empty(PROPOSALS, dtype=torch.int64, device="cuda")
        self.domino({"features": features, "past": self.history[:context_start], "noise": self.noise,
                     "context_rope": self.drope[:proposal], "proposal_rope": self.drope[proposal:proposal + PROPOSALS],
                     "fresh_context": self.fresh_context[:rows], "tokens": tokens}, profile=0 if context_start == 0 else 2)
        self.history[context_start:proposal].copy_(self.fresh_context[:rows])
        return tokens.tolist()

    def tree_inputs(self, past: list[int], rows: int) -> dict[str, torch.Tensor]:
        """AttentionPlugin tree-mode inputs for sequences with prefix lengths ``past`` and ``rows`` new rows each:
        the new rows take positions [past, past + rows) and the cache length becomes past + rows."""
        past = torch.tensor(past, dtype=torch.int32, device="cuda")
        positions = (torch.arange(rows, dtype=torch.int32, device="cuda")[None] + past[:, None]).reshape(-1)
        return {"past_lengths": past, "sequence_lengths": past + rows, "positions": positions,
                "phase_marker": self.phase, "sequence_count_carrier": self.carrier}

    def verify(self, engine, tokens: list[int], start: int, attention_layers: int, **outputs) -> None:
        """Q17/Q1: score ``tokens`` at rows [start, start + len(tokens)); the plugin writes their K/V into the
        pool in place (pool_i == present_i)."""
        rows = len(tokens)
        pools = {f"pool_{i}": self.pool[i] for i in range(attention_layers)}
        presents = {f"present_{i}": self.pool[i] for i in range(attention_layers)}
        engine({"activation": self.embed(tokens), "rope": self.rope[0, start:start + rows],
                **self.tree_inputs([start], rows), **pools, **presents, **outputs})

    def decode(self, anchor: int, features: torch.Tensor) -> dict:
        """Greedy speculative decoding up to and including the trajectory start token."""
        generated, length, context_start, rounds, committed_total = [anchor], ROWS + 1, 0, 0, 0
        stopped, stop_cached = anchor == self.stop, False
        while not stopped and len(generated) < MAX_NEW_TOKENS:
            start = length - 1  # the anchor's row
            draft = self.draft(features, context_start, generated[-1])
            q17_features = torch.empty(1, PROPOSALS + 1, FEATURES, dtype=torch.float16, device="cuda")
            posterior = torch.empty(PROPOSALS + 1, dtype=torch.int64, device="cuda")
            self.verify(self.q17, [generated[-1]] + draft, start, LAYERS, features=q17_features, posterior=posterior)
            posterior = posterior.tolist()
            accepted = 0
            while accepted < PROPOSALS and draft[accepted] == posterior[accepted]:
                accepted += 1
            committed = (draft[:accepted] + [posterior[accepted]])[:MAX_NEW_TOKENS - len(generated)]
            if self.stop in committed:
                committed = committed[:committed.index(self.stop) + 1]
                stopped, stop_cached = True, len(committed) <= accepted
            generated += committed
            length += len(committed)
            rounds, committed_total = rounds + 1, committed_total + len(committed)
            features, context_start = q17_features[0, :accepted + 1], start
        if not stopped:
            raise RuntimeError(f"no trajectory start token within {MAX_NEW_TOKENS} tokens")
        if not stop_cached:  # the stop token was the bonus token: write its K/V with Q1
            row = length - 1
            fresh = torch.empty(2, 8, 1, 128, dtype=torch.float16, device="cuda")  # layer 27 has no attention in Q1
            self.verify(self.q1, [self.stop], row, LAYERS - 1, **{f"fresh_{LAYERS - 1}": fresh})
            self.pool[LAYERS - 1, :, row // PAGE, row % PAGE].copy_(fresh[:, :, 0])
        return {"generated_token_ids": generated, "prefix_length": length, "rounds": rounds, "committed_tokens": committed_total}

    # --- action ---------------------------------------------------------------------------------
    def action_page_table(self, prefix_length: int) -> torch.Tensor:
        """[6, 2, 25] page table: every candidate reads the full prefix pages and owns two private pages after
        the language pages, where its 64 rows go. A partly filled last prefix page is copied into each
        candidate's first private page (once per request; the plugin must not write into a shared page)."""
        shared, partial = divmod(prefix_length, PAGE)
        pages = np.zeros((CANDIDATES, ACTION_PAGES), np.int32)
        for b in range(CANDIDATES):
            own = list(range(shared)) + [ACTION_PRIVATE + 2 * b, ACTION_PRIVATE + 2 * b + 1]
            pages[b] = (own + [own[-1]] * ACTION_PAGES)[:ACTION_PAGES]  # unused entries repeat a valid page
        if partial:
            self.pool[:, :, ACTION_PRIVATE:ACTION_PRIVATE + 2 * CANDIDATES:2] = self.pool[:, :, shared:shared + 1]
        return torch.from_numpy(np.stack((pages, pages + NUM_PAGES), 1)).cuda()

    def run_action(self, prefix_length: int) -> torch.Tensor:
        if not ROWS <= prefix_length <= ACTION_PREFIX:
            raise ValueError(f"action prefix must be at most {ACTION_PREFIX} rows, got {prefix_length}")
        pools = {f"pool_{i}": self.pool[i] for i in range(LAYERS)}
        presents = {f"present_{s}_{i}": self.pool[i] for s in range(STEPS) for i in range(LAYERS)}  # in place
        positions = torch.arange(64, device="cuda") + self.layout["rope_deltas"] + prefix_length
        actions = torch.empty(384, 2, dtype=torch.float32, device="cuda")
        self.action({"noise": self.action_noise, "rope": plain_rope(positions).repeat(CANDIDATES, 1),
                     "page_table": self.action_page_table(prefix_length), **self.tree_inputs([prefix_length] * CANDIDATES, 64),
                     **pools, **presents, "actions": actions})
        return actions.reshape(CANDIDATES, 64, 2)

    # --- one request ------------------------------------------------------------------------------
    def __call__(self, job: dict, root: Path) -> dict:
        request = root / job["input"]
        self.requests_done = getattr(self, "requests_done", 0) + 1
        if job["clip_id"] != self.clip:  # the rolling vision cache is per clip
            self.vision_cache.clear()
            self.clip = job["clip_id"]
        paths, input_ids, grid, pixels, ego = self.prompt(request)
        pixels, ego = pixels.cuda(), {k: v.cuda() for k, v in ego.items()}
        for tick in range(3):  # older ticks: normally cached from the previous request
            if any(paths[c * 4 + tick] not in self.vision_cache for c in range(4)):
                self.vision_tick(paths, pixels, tick)
        input_ids = input_ids.cuda()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        self.vision_tick(paths, pixels, 3)
        fused_ids = self.model.fuse_traj_tokens(input_ids, ego)
        self.prepare_layout(fused_ids, grid)
        anchor, features = self.run_prefill(paths, fused_ids)
        decoded = self.decode(anchor, features)
        actions = self.run_action(decoded["prefix_length"])
        xyz = ego["ego_history_xyz"][:, -1].repeat_interleave(6, 0)
        rot = ego["ego_history_rot"][:, -1].repeat_interleave(6, 0)
        trajectory, _ = self.model.action_space.action_to_traj(actions, xyz, rot)
        torch.cuda.synchronize()
        request_ms = (time.perf_counter() - begin) * 1e3

        truth = torch.tensor(json.loads((root / job["metric"]).read_text())["ground_truth_xy"], dtype=torch.float32)
        ades = (trajectory[..., :2].float().cpu() - truth).norm(dim=-1).mean(-1)
        return {"eval_index": job["eval_index"], "clip_id": job["clip_id"], "horizon": job.get("horizon"),
                # Seed rows (excluded from latency): each clip's first window, or the run's first request
                # (one-time setup) when the jobs carry no horizon.
                "is_seed": job.get("horizon") == 4 if "horizon" in job else self.requests_done == 1, "minade6_m": float(ades.min()), "best_sample": int(ades.argmin()),
                "ades_m": ades.tolist(), **decoded, "request_ms": request_ms}


def summarize(rows: list[dict], reference: dict | None) -> dict:
    """Quality, acceptance and steady-state latency; paired comparison with a reference run if given."""
    steady = sorted(r["request_ms"] for r in rows if not r["is_seed"])
    def endpoint(rs):  # windows ending at t0 (horizon 16); None when the jobs carry no horizon
        values = [r["minade6_m"] for r in rs if r["horizon"] == 16]
        return float(np.mean(values)) if values else None
    summary = {"requests": len(rows), "minade6_mean_m": float(np.mean([r["minade6_m"] for r in rows])),
               "endpoint_minade6_mean_m": endpoint(rows),
               "acceptance": sum(r["committed_tokens"] for r in rows) / max(1, sum(r["rounds"] for r in rows)),
               "latency_ms": {"median": float(np.median(steady)), "p90": steady[math.ceil(0.9 * len(steady)) - 1]} if steady else None}
    if reference is not None:
        arm = reference["gate"]["reference_arm"]
        by_key = {(r["eval_index"], r["clip_id"], r["horizon"]): r for r in reference["arms"][arm]["rows"]}
        paired = [by_key[(r["eval_index"], r["clip_id"], r["horizon"])] for r in rows]
        delta = np.array([r["minade6_m"] - b["minade6_m"] for r, b in zip(rows, paired)])
        measured = {"mean_m": float(delta.mean()), "p90_m": float(np.quantile(delta, 0.9)),
                    "endpoint_m": endpoint(rows) - endpoint(paired) if endpoint(rows) is not None else 0.0}
        summary["gate"] = {"reference": arm, "limits": GATES, "measured": measured,
                           "tokens_exact_rows": sum(r["generated_token_ids"] == b["generated_token_ids"] for r, b in zip(rows, paired)),
                           "within_limits": bool(np.isfinite(delta).all()) and all(measured[k] <= GATES[k] for k in GATES)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", type=Path, required=True, help="Output directory of the build step.")
    parser.add_argument("--model", type=Path, required=True, help="Model bundle directory (README).")
    parser.add_argument("--vlm-metadata", type=Path, required=True, help="Cosmos-Reason2-2B tokenizer/processor files.")
    parser.add_argument("--jobs", type=Path, required=True, help="jobs.json of the evaluation fixture.")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON (must not exist).")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N jobs.")
    parser.add_argument("--reference", type=Path, help="Reference rows {gate, arms} to compare minADE6 against.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    jobs = json.loads(args.jobs.read_text())["jobs"]
    jobs = jobs[:args.limit] if args.limit else jobs
    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("the engines are built for DRIVE Thor (SM 11.0)")
    bundle = Bundle(args.model)

    torch.manual_seed(0)
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):  # TensorRT syncs extra on the default stream
        pipeline = Pipeline(args.engines, bundle, args.vlm_metadata)
        rows = []
        for job in jobs:
            rows.append(pipeline(job, args.jobs.parent))
            print(json.dumps({k: rows[-1][k] for k in ("eval_index", "horizon", "minade6_m", "rounds", "request_ms")}), flush=True)
    summary = summarize(rows, json.loads(args.reference.read_text()) if args.reference else None)
    write_json(args.output, {"jobs": str(args.jobs), "jobs_sha256": hashlib.sha256(args.jobs.read_bytes()).hexdigest(),
                             "model": {"bundle": str(bundle.directory), "manifest_sha256": bundle.manifest_sha256},
                             "build": pipeline.build, "summary": summary, "rows": rows})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
