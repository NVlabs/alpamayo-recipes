# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen-VLM conditioning utilities for stage-3 (DRaFT).

Extracted VERBATIM from the project's evaluation harness so that training-time
conditioning (prompt layout, CoC generation, KV-cache construction) is
byte-identical to evaluation and to the product inference path
(`alpamayo_r1.models.alpamayo_r1.AlpamayoR1.sample_trajectories_from_data_with_
vlm_rollout` — `vlm_stage` below is its first half, split at the diffusion
stage). The training entry carries a `--parity-check` gate that bit-compares
the gradient-enabled Euler loop against `model.diffusion.sample` on a real
conditioning context before any arm is trusted.

Contents:
- `create_message_r15` / `CAMERA_DISPLAY_NAMES`: AM1.5 camera-annotated prompt
  (ported verbatim from `alpamayo1_5.helper.create_message`).
- `install_offline_avdi`: local-dir-only PhysicalAI-AV dataset interface.
- `ClipDataset`: video decode inside DataLoader workers, per-clip t0 support.
- `vlm_stage`: frozen-VLM CoC sampling + KV cache + expert position ids/masks.
- `HEAD_PREFIXES`: the trainable action-head tensor namespace.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import einops
import torch
from torch.utils.data import Dataset

import alpamayo_r1.models.alpamayo_r1 as M
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

HEAD_PREFIXES = ("expert.", "action_in_proj.", "action_out_proj.")

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


def create_message_r15(frames, camera_indices, num_frames_per_camera=4,
                       nav_text=None):
    """alpamayo1_5.helper.create_message equivalent. nav_text inserts the
    <|route_start|>...<|route_end|> section exactly where the official helper
    puts it (between the traj-history placeholder and the CoC instruction);
    None keeps the basic format byte-identical to before."""
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"
    num_traj_token = 48
    hist = f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    expanded = camera_indices.repeat_interleave(num_frames_per_camera)
    content, prev_cam, frame_idx = [], None, 0
    for i, frame in enumerate(frames):
        cam_id = int(expanded[i])
        if prev_cam is not None and cam_id != prev_cam:
            frame_idx = 0
        if frame_idx == 0:
            cam_name = CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
            content.append({"type": "text", "text": f"{cam_name}: "})
        content.append({"type": "text", "text": f"frame {frame_idx} "})
        content.append({"type": "image", "image": frame})
        prev_cam, frame_idx = cam_id, frame_idx + 1
    route = f"<|route_start|>{nav_text}<|route_end|>" if nav_text else ""
    user_text = (f"{hist}{route}output the chain-of-thought reasoning of the driving "
                 f"process, then output the future trajectory.")
    return [
        {"role": "system", "content": [{"type": "text", "text":
            "You are a driving assistant that generates safe and accurate actions."}]},
        {"role": "user", "content": content + [{"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]


def install_offline_avdi(pai_root: str):
    import contextlib
    import physical_ai_av
    from physical_ai_av.utils import hf_interface as _hfi

    def _local_path(self, filename):
        q = Path(self.local_dir) / filename
        return str(q) if q.exists() else None

    def _is_file_cached(self, filename):
        return _local_path(self, filename) is not None

    def _download_file(self, filename, **kw):
        q = _local_path(self, filename)
        if q is None:
            raise FileNotFoundError(f"{filename} not in local_dir (offline local mode)")
        return q

    def _download_files(self, files, **kw):
        return [_download_file(self, f if isinstance(f, str) else f.path) for f in files]

    @contextlib.contextmanager
    def _open_file(self, filename, mode="rb", maybe_stream=False):
        q = _local_path(self, filename)
        if q is None:
            raise FileNotFoundError(f"{filename} not in local_dir (offline local mode)")
        with open(q, mode) as f:
            yield f

    _hfi.HfRepoInterface.is_file_cached = _is_file_cached
    _hfi.HfRepoInterface.download_file = _download_file
    _hfi.HfRepoInterface.download_files = _download_files
    _hfi.HfRepoInterface.open_file = _open_file

    _meta = Path(pai_root) / ".cache/huggingface/download/features.csv.metadata"
    _rev = _meta.read_text().splitlines()[0].strip() if _meta.exists() else "local"
    return physical_ai_av.PhysicalAIAVDatasetInterface(
        revision=_rev, local_dir=pai_root, confirm_download_threshold_gb=100000)


class ClipDataset(Dataset):
    """Runs load_physical_aiavdataset (video decode) inside DataLoader workers.

    t0 variant: t0_map gives a PER-CLIP relative t0 (event_t0s[0] -
    start_timestamp from the golden metadata); clips without an entry fall
    back to the global --t0_us."""

    def __init__(self, clip_ids, t0_us, avdi, t0_map=None):
        self.clip_ids = clip_ids
        self.t0_us = t0_us
        self.avdi = avdi
        self.t0_map = t0_map or {}

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        cid = self.clip_ids[i]
        last = None
        for attempt in range(3):
            try:
                data = load_physical_aiavdataset(
                    cid, t0_us=int(self.t0_map.get(cid, self.t0_us)),
                    avdi=self.avdi)
                return {"clip_id": cid, "data": data}
            except Exception as e:
                last = e
                retriable = any(s in repr(e) for s in
                                ("429", "Too Many Requests", "BadZipFile",
                                 "IncompleteRead", "ConnectionError", "Timeout"))
                if retriable and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                else:
                    break
        return {"clip_id": cid, "_err": repr(last)}


def vlm_stage(model, data, top_p, temperature, num_traj_samples,
              max_generation_length):
    """First half of helper.sample_trajectories_from_data_with_vlm_rollout,
    up to and including position_ids / attention_mask construction."""
    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    B, n_traj_group, _, _ = ego_history_xyz.shape
    assert n_traj_group == 1, "Only one trajectory group is supported for inference."
    tokenized_data = data["tokenized_data"]
    input_ids = tokenized_data.pop("input_ids")
    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)
    device = input_ids.device

    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = num_traj_samples
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    if os.environ.get("EVAL_OFFICIAL_GEN", "0") == "1":
        # modeling_distill.py student-rollout parity: natural EOS disabled so
        # generation runs through <|im_end|> until <traj_future_start>, and
        # +1 token so the KV cache holds the marker's successor
        generation_config = copy.deepcopy(generation_config)
        generation_config.eos_token_id = None
        generation_config.forced_eos_token_id = None
        generation_config.max_new_tokens = max_generation_length + 1

    eos_token_id = model.tokenizer.convert_tokens_to_ids(
        M.to_special_token("traj_future_start"))
    stopping_criteria = M.StoppingCriteriaList(
        [M.StopAfterEOS(eos_token_id=eos_token_id)])
    logits_processor = M.LogitsProcessorList([
        M.ExpertLogitsProcessor(
            traj_token_offset=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size,
        )
    ])
    vlm_outputs = model.vlm.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
        logits_processor=logits_processor,
        **tokenized_data,
    )
    vlm_outputs.rope_deltas = model.vlm.model.rope_deltas
    vlm_outputs.sequences = M.replace_padding_after_eos(
        token_ids=vlm_outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    prompt_cache = vlm_outputs.past_key_values
    prefill_seq_len = prompt_cache.get_seq_length()

    b_star = vlm_outputs.sequences.shape[0]
    traj_future_start_mask = vlm_outputs.sequences == eos_token_id
    has_traj_future_start = traj_future_start_mask.any(dim=1)
    traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
    last_token_positions = torch.full(
        (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device)
    valid_token_pos_id = torch.where(
        has_traj_future_start, traj_future_start_positions, last_token_positions)
    offset = valid_token_pos_id + 1

    n_diffusion_tokens = model.action_space.get_action_space_dims()[0]
    position_ids = torch.arange(n_diffusion_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
    delta = vlm_outputs.rope_deltas + offset[:, None]
    position_ids += delta.to(position_ids.device)

    attention_mask = torch.zeros(
        (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
        dtype=torch.float32, device=device)
    for i in range(b_star):
        attention_mask[i, :, :, offset[i]: -n_diffusion_tokens] = torch.finfo(
            attention_mask.dtype).min

    forward_kwargs = {}
    if model.config.expert_non_causal_attention:
        forward_kwargs["is_causal"] = False

    return {
        "prompt_cache": prompt_cache,
        "prefill_seq_len": prefill_seq_len,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "forward_kwargs": forward_kwargs,
        "n_diffusion_tokens": n_diffusion_tokens,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "B": B,
        "k": num_traj_samples,
        "device": device,
        "n_miss_fs": int((~has_traj_future_start).sum().item()),
    }
