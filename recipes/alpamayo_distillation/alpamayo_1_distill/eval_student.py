#!/usr/bin/env python3
"""Standalone, distributed student-checkpoint evaluation.

The default protocol evaluates the packaged 644-clip gold set at t0=5.1 s.
Use ``--eval_set full`` to evaluate every clip in the PAI validation split.

Single node::

    torchrun --standalone --nproc_per_node=8 \
        --module alpamayo1_distill.eval_student \
        --ckpt_dir /path/to/ckpts/step_50000 \
        --local_dir /path/to/PhysicalAI-Autonomous-Vehicles

Multiple nodes are launched with the same Python entry point by supplying the
standard torchrun rendezvous arguments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset

from alpamayo1_distill import helper
from alpamayo1_distill.config import ExpertModelConfig
from alpamayo1_distill.dataloader.physical_ai_av.src.physical_ai_av.dataset_local import (
    PhysicalAIAVDatasetInterfaceLocal,
)
from alpamayo1_distill.dataloader.physical_aiavdataset import physicalAiAvDataset
from alpamayo1_distill.defaults import (
    DEFAULT_GOLD_CLIP_IDS_PATH,
    PAI_FULL_VAL_SIZE,
    PAI_GOLD_EVAL_SIZE,
)
from alpamayo1_distill.models.alpamayo_r1_distill_2b import ExpertModelDistill
from alpamayo1_distill.vla_constant import TRAJ_TOKEN


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GOLD_CLIP_IDS = DEFAULT_GOLD_CLIP_IDS_PATH
DEFAULT_GOLD_PARQUET = SCRIPT_DIR / "config" / "1005_7cam_gold_eval_metadb_public.parquet"


def setup_distributed() -> tuple[int, int, int]:
    """Initialize torch.distributed when launched by torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        torch.cuda.set_device(0)
        return 0, 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=2),
    )
    return local_rank, dist.get_rank(), dist.get_world_size()


def _sample_identity(dataset: Dataset, index: int) -> tuple[str | None, int | None]:
    """Best-effort clip/timestamp lookup without decoding the sample."""
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return None, None
    clip_id, t0_us = samples[index]
    return str(clip_id), int(t0_us)


class SafeEvalDataset(Dataset):
    """Catch ordinary sample decode errors inside DataLoader workers."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, Any]:
        global_index = self.indices[local_index]
        clip_id, t0_us = _sample_identity(self.dataset, global_index)
        try:
            sample = dict(self.dataset[global_index])
        except Exception as exc:
            return {
                "_eval_error": repr(exc),
                "_eval_index": global_index,
                "clip_id": clip_id,
                "t0_us": t0_us,
            }
        sample["_eval_index"] = global_index
        return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) == 1 and "_eval_error" in batch[0]:
        return batch[0]
    out: dict[str, Any] = {}
    out["image_frames"] = torch.stack([b["image_frames"] for b in batch], dim=0)
    out["camera_indices"] = torch.stack([b["camera_indices"] for b in batch], dim=0)
    out["relative_timestamps"] = torch.stack(
        [b["relative_timestamps"] for b in batch], dim=0
    )

    def squeeze_11(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(0).squeeze(0)

    out["ego_history_xyz"] = torch.stack(
        [squeeze_11(b["ego_history_xyz"]) for b in batch], dim=0
    ).unsqueeze(1)
    out["ego_history_rot"] = torch.stack(
        [squeeze_11(b["ego_history_rot"]) for b in batch], dim=0
    ).unsqueeze(1)
    out["ego_future_xyz"] = torch.stack(
        [squeeze_11(b["ego_future_xyz"]) for b in batch], dim=0
    ).unsqueeze(1)
    out["ego_future_rot"] = torch.stack(
        [squeeze_11(b["ego_future_rot"]) for b in batch], dim=0
    ).unsqueeze(1)
    out["t0_us"] = [int(b["t0_us"]) for b in batch]
    out["clip_id"] = [str(b["clip_id"]) for b in batch]
    out["_eval_index"] = [int(b["_eval_index"]) for b in batch]
    return out


def _minade_values(pred_xyz: torch.Tensor, gt_xy: torch.Tensor) -> list[float]:
    """Return one minADE value per sample, preserving full aggregation precision."""
    pred_xy = pred_xyz[..., :2]
    batch_size, num_steps = pred_xy.shape[0], pred_xy.shape[-2]
    pred_xy = pred_xy.reshape(batch_size, -1, num_steps, 2)
    distance = torch.linalg.vector_norm(pred_xy - gt_xy[:, None, :, :], dim=-1)
    ade = distance.mean(dim=-1)
    return ade.min(dim=1).values.detach().to(torch.float64).cpu().tolist()


def _sample(
    model: ExpertModelDistill,
    model_inputs: dict[str, Any],
    *,
    greedy: bool,
    max_generation_length: int,
    num_traj_samples: int,
    top_p: float,
    temperature: float,
):
    kwargs: dict[str, Any] = {
        "data": model_inputs,
        "max_generation_length": max_generation_length,
        "return_extra": True,
    }
    if greedy:
        kwargs.update(top_p=1.0, temperature=1.0, num_traj_samples=1, do_sample=False)
    else:
        kwargs.update(
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_traj_samples,
        )
    fn = getattr(model, "_sample_trajectories_from_data_with_vlm_rollout", None)
    if fn is None:
        fn = getattr(model, "sample_trajectories_from_data_with_vlm_rollout")
    return fn(**kwargs)


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            value = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON metadata from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def _nonempty(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return None


def _validate_resolved_path(value: str, *, label: str, override_flag: str) -> str:
    """Fail early for stale absolute paths while allowing cached Hugging Face IDs."""
    path = Path(value).expanduser()
    if path.is_absolute():
        if not path.is_dir():
            raise FileNotFoundError(
                f"Resolved {label} path does not exist: {path}. "
                f"Supply a valid path with {override_flag}."
            )
        return str(path.resolve())
    return value


def _validate_checkpoint_dir(ckpt_dir: Path) -> None:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
    if ckpt_dir.name.endswith(".tmp"):
        raise ValueError(f"Refusing to evaluate an incomplete staging checkpoint: {ckpt_dir}")
    if not (ckpt_dir / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint is missing config.json: {ckpt_dir}")
    has_weights = any(ckpt_dir.glob("*.safetensors")) or (
        ckpt_dir / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        raise FileNotFoundError(f"Checkpoint has no safetensors model weights: {ckpt_dir}")


def _read_clip_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Clip-ID file does not exist: {path}")
    with path.open() as f:
        clip_ids = [line.strip() for line in f if line.strip()]
    if not clip_ids:
        raise ValueError(f"Clip-ID file is empty: {path}")
    if len(set(clip_ids)) != len(clip_ids):
        seen: set[str] = set()
        duplicates: list[str] = []
        for clip_id in clip_ids:
            if clip_id in seen and clip_id not in duplicates:
                duplicates.append(clip_id)
            seen.add(clip_id)
        raise ValueError(
            f"Clip-ID file contains duplicates ({len(duplicates)} unique duplicates): "
            f"{duplicates[:10]}"
        )
    return clip_ids


def _validate_gold_assets(clip_ids: list[str]) -> None:
    if not DEFAULT_GOLD_PARQUET.is_file():
        raise FileNotFoundError(
            f"Gold evaluation parquet is missing: {DEFAULT_GOLD_PARQUET}. "
            "Download it from "
            "https://github.com/NVlabs/alpamayo-recipes/raw/refs/heads/main/"
            "data/1005_7cam_gold_eval_metadb_public.parquet "
            f"and save it to {DEFAULT_GOLD_PARQUET}."
        )
    import pandas as pd

    frame = pd.read_parquet(DEFAULT_GOLD_PARQUET, columns=["key"])
    parquet_ids = list(dict.fromkeys(frame["key"].dropna().astype(str).tolist()))
    if len(parquet_ids) != PAI_GOLD_EVAL_SIZE:
        raise ValueError(
            f"Gold evaluation parquet must contain {PAI_GOLD_EVAL_SIZE} unique keys; "
            f"found {len(parquet_ids)}"
        )
    if set(parquet_ids) != set(clip_ids):
        only_parquet = sorted(set(parquet_ids) - set(clip_ids))
        only_text = sorted(set(clip_ids) - set(parquet_ids))
        raise ValueError(
            "Gold clip list does not match 1005_7cam_gold_eval_metadb_public.parquet. "
            f"Only in parquet: {only_parquet[:10]}; only in txt: {only_text[:10]}"
        )


def _validate_clip_ids(
    avdi: PhysicalAIAVDatasetInterfaceLocal,
    clip_ids: list[str],
) -> dict[str, int]:
    available = set(str(value) for value in avdi.clip_index.index.tolist())
    missing = [clip_id for clip_id in clip_ids if clip_id not in available]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(clip_ids)} requested clips are absent from the local clip "
            f"index. First missing IDs: {missing[:10]}"
        )

    split_counts: dict[str, int] = {}
    if "split" in avdi.clip_index.columns:
        for clip_id in clip_ids:
            split = str(avdi.clip_index.loc[clip_id, "split"])
            split_counts[split] = split_counts.get(split, 0) + 1
    return split_counts


def _seed_for_sample(
    base_seed: int,
    *,
    clip_id: str,
    t0_us: int,
    greedy: bool,
    strategy: str,
) -> int | None:
    if base_seed < 0:
        return None
    mode_offset = 1 if greedy else 0
    if strategy == "reference":
        # The reference SFT evaluator resets to the same seed before each clip.
        # Keep the one-trajectory diffusion stream independent from minADE@K.
        return base_seed + mode_offset
    payload = f"{base_seed}\0{clip_id}\0{t0_us}\0{int(greedy)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _run_with_seed(seed: int | None, device: torch.device, fn):
    if seed is None:
        return fn()
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.random.default_generator.manual_seed(seed)
        with torch.cuda.device(device_index):
            torch.cuda.manual_seed(seed)
        return fn()


def _validate_model_tokenizer(
    model: ExpertModelDistill,
    config: ExpertModelConfig,
    checkpoint_config: dict[str, Any],
) -> None:
    tokenizer = model.tokenizer
    embedding_vocab = model.vlm.get_input_embeddings().num_embeddings
    if len(tokenizer) != embedding_vocab:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer={len(tokenizer)}, "
            f"embeddings={embedding_vocab}"
        )

    checkpoint_vocab = checkpoint_config.get("vocab_size")
    if checkpoint_vocab is not None and embedding_vocab != int(checkpoint_vocab):
        raise ValueError(
            f"Checkpoint/model vocabulary mismatch: checkpoint={checkpoint_vocab}, "
            f"embeddings={embedding_vocab}"
        )

    configured_ids = checkpoint_config.get("traj_token_ids") or {}
    mismatches: list[str] = []
    for name, token in TRAJ_TOKEN.items():
        if name not in configured_ids:
            continue
        actual = tokenizer.convert_tokens_to_ids(token)
        expected = int(configured_ids[name])
        if actual != expected:
            mismatches.append(f"{name}: config={expected}, tokenizer={actual}")
    if mismatches:
        raise ValueError("Trajectory-token ID mismatch: " + "; ".join(mismatches))

    checkpoint_traj_start = checkpoint_config.get("traj_token_start_idx")
    actual_traj_start = tokenizer.convert_tokens_to_ids("<i0>")
    if checkpoint_traj_start is not None and actual_traj_start != int(checkpoint_traj_start):
        raise ValueError(
            "Trajectory vocabulary start mismatch: "
            f"checkpoint={checkpoint_traj_start}, tokenizer={actual_traj_start}"
        )

    # helper.create_message currently emits exactly 48 history placeholders.
    tokens_per_history = int(
        checkpoint_config.get("tokens_per_history_traj", config.tokens_per_history_traj)
    )
    if tokens_per_history != 48:
        raise ValueError(
            "Online evaluation prompt construction currently requires "
            f"tokens_per_history_traj=48, got {tokens_per_history}"
        )


def _validate_loading_info(loading_info: dict[str, Any]) -> int:
    missing = list(loading_info.get("missing_keys") or [])
    unexpected = list(loading_info.get("unexpected_keys") or [])
    mismatched = list(loading_info.get("mismatched_keys") or [])
    errors = list(loading_info.get("error_msgs") or [])

    # Online-training checkpoints can contain the frozen teacher and the
    # hidden-state projection. Student-only inference intentionally does not
    # instantiate either module.
    allowed_unexpected_prefixes = ("teacher_model.", "hidden_proj.")
    disallowed_unexpected = [
        key for key in unexpected if not key.startswith(allowed_unexpected_prefixes)
    ]
    if missing or disallowed_unexpected or mismatched or errors:
        raise RuntimeError(
            "Checkpoint did not load exactly into the student evaluator. "
            f"missing={missing[:20]}, unexpected={disallowed_unexpected[:20]}, "
            f"mismatched={mismatched[:20]}, errors={errors[:20]}"
        )
    return len(unexpected)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp_path.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("eval_student")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Student checkpoint directory, for example .../ckpts/step_50000.",
    )
    parser.add_argument(
        "--student_vlm_path",
        type=str,
        default=None,
        help=(
            "Cosmos-Reason2-2B relocation/local-mirror path. Resolution: CLI, "
            "train_args.json, checkpoint config. Alternate architectures are rejected."
        ),
    )
    parser.add_argument(
        "--student_processor_path",
        type=str,
        default=None,
        help="Student processor path. Resolution: CLI, train_args.json, student VLM path.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="PhysicalAI-AV root. Defaults to the value saved in train_args.json.",
    )
    parser.add_argument("--local_files_only", action="store_true", default=False)

    parser.add_argument(
        "--eval_set",
        choices=("gold", "full"),
        default="gold",
        help=(
            "gold: packaged 644-clip set (default); full: full PAI evaluation set."
        ),
    )
    parser.add_argument("--t0_us", type=int, default=5_100_000)

    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed reset before every clip. Set -1 to disable reseeding.",
    )
    parser.add_argument(
        "--seed_strategy",
        choices=("reference", "per_clip"),
        default="reference",
        help="reference reuses the SFT recipe's seed per clip; per_clip hashes clip/t0/mode.",
    )
    parser.add_argument("--num_traj_samples", type=int, default=6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument(
        "--strict_determinism",
        action="store_true",
        help="Disable TF32 and require deterministic PyTorch kernels (may reject flash-attn ops).",
    )
    parser.add_argument(
        "--max_generation_length",
        type=int,
        default=None,
        help="Defaults to reasoning_max_generation_length from the checkpoint config.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional rank-0 JSON output containing summary and per-clip metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.strict_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for student checkpoint evaluation")
    if args.batch_size_per_gpu != 1:
        raise ValueError(
            "--batch_size_per_gpu must be 1: the current online prompt builder is not "
            "batch-safe. Scale evaluation with more GPUs/nodes instead."
        )
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch_factor must be positive")
    if args.log_every < 0:
        raise ValueError("--log_every must be non-negative")
    if args.num_traj_samples <= 0:
        raise ValueError("--num_traj_samples must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top_p must satisfy 0 < top_p <= 1")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.seed < -1:
        raise ValueError("--seed must be -1 (disabled) or non-negative")
    if args.t0_us <= 1_600_000:
        raise ValueError("--t0_us must be greater than 1,600,000 for the history window")
    # Evaluation favors reproducibility over training throughput autotuning.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = not args.strict_determinism
    if args.strict_determinism:
        torch.use_deterministic_algorithms(True)

    local_rank, rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    _validate_checkpoint_dir(ckpt_dir)
    train_args = _read_json_if_present(ckpt_dir / "train_args.json")
    checkpoint_config = _read_json_if_present(ckpt_dir / "config.json")

    config_source = str(ckpt_dir)
    student_vlm_path = _nonempty(
        args.student_vlm_path,
        train_args.get("student_vlm_path"),
        checkpoint_config.get("vlm_name_or_path"),
    )
    if student_vlm_path is None:
        raise ValueError(
            "Could not resolve the student VLM path. Pass --student_vlm_path explicitly."
        )
    student_vlm_path = _validate_resolved_path(
        student_vlm_path,
        label="student VLM",
        override_flag="--student_vlm_path",
    )
    student_processor_path = _nonempty(
        args.student_processor_path,
        train_args.get("student_processor_path"),
        student_vlm_path,
    )
    assert student_processor_path is not None
    student_processor_path = _validate_resolved_path(
        student_processor_path,
        label="student processor",
        override_flag="--student_processor_path",
    )
    # Config initialization derives vocabulary/token IDs from its VLM path.
    # Match training by using the processor path for that derivation; the model
    # constructor below then switches the backbone to student_vlm_path.
    config = ExpertModelConfig.from_pretrained(
        config_source,
        local_files_only=True,
        vlm_name_or_path=student_processor_path,
    )
    local_dir = _nonempty(args.local_dir, train_args.get("local_dir"))
    if local_dir is None:
        raise ValueError("Could not resolve the dataset root. Pass --local_dir explicitly.")
    local_dir = _validate_resolved_path(
        local_dir,
        label="dataset root",
        override_flag="--local_dir",
    )

    # Keep config/model construction on the same resolved backbone. This is
    # required because ExpertModelDistill initializes a base VLM before loading
    # the complete student checkpoint state.
    config.vlm_name_or_path = student_vlm_path
    max_generation_length = (
        args.max_generation_length
        if args.max_generation_length is not None
        else int(config.reasoning_max_generation_length)
    )
    if max_generation_length <= 0:
        raise ValueError("--max_generation_length must be positive")

    if rank == 0:
        print(
            "[eval_student] resolved setup\n"
            f"  checkpoint={ckpt_dir}\n"
            f"  config={config_source}\n"
            f"  student_vlm={student_vlm_path}\n"
            f"  student_processor={student_processor_path}\n"
            f"  dataset_root={local_dir}\n"
            f"  topology=world_size:{world_size}, local_rank:{local_rank}\n"
            f"  seed={args.seed}, seed_strategy={args.seed_strategy}, t0_us={args.t0_us}"
        )

    model, loading_info = ExpertModelDistill.from_pretrained(
        str(ckpt_dir),
        local_files_only=True,
        config=config,
        dtype=torch.bfloat16,
        student_model_path=student_vlm_path,
        processor_model_path=student_processor_path,
        output_loading_info=True,
    )
    allowed_unexpected_count = _validate_loading_info(loading_info)
    model = model.to(device)
    if rank == 0 and allowed_unexpected_count:
        print(
            "[eval_student] ignored "
            f"{allowed_unexpected_count} teacher/hidden-distillation-only checkpoint keys"
        )
    model.eval()
    model.requires_grad_(False)
    _validate_model_tokenizer(model, config, checkpoint_config)

    # The model's processor already comes from the resolved student processor
    # path and contains the checkpoint-compatible trajectory/special tokens.
    # No 10B teacher weights are needed for student-only evaluation.
    tokenizer = model.tokenizer
    processor = model.processor.processor

    avdi = PhysicalAIAVDatasetInterfaceLocal(
        local_dir=local_dir,
        local_files_only=args.local_files_only,
    )
    ds_kwargs: dict[str, Any] = {
        "t0_us": args.t0_us,
        "tokenizer": tokenizer,
        "local_dir": local_dir,
    }
    if args.local_files_only:
        ds_kwargs["local_files_only"] = True

    clip_ids_file: Path | None = None
    eval_set_label = args.eval_set
    if args.eval_set == "gold":
        clip_ids_file = DEFAULT_GOLD_CLIP_IDS

    split_counts: dict[str, int] = {}
    if clip_ids_file is not None:
        clip_ids = _read_clip_ids(clip_ids_file)
        if len(clip_ids) != PAI_GOLD_EVAL_SIZE:
            raise ValueError(
                f"Packaged gold list must contain {PAI_GOLD_EVAL_SIZE} unique clips; "
                f"found {len(clip_ids)} in {clip_ids_file}"
            )
        _validate_gold_assets(clip_ids)
        split_counts = _validate_clip_ids(avdi, clip_ids)
        ds_kwargs["gold_eval"] = True

    ds_val = physicalAiAvDataset(split="val", avdi=avdi, **ds_kwargs)
    selected_samples = len(ds_val)
    required_samples = (
        PAI_GOLD_EVAL_SIZE if args.eval_set == "gold" else PAI_FULL_VAL_SIZE
    )
    if selected_samples != required_samples:
        raise ValueError(
            "The selected PAI evaluation set has the wrong cardinality: "
            f"expected={required_samples}, constructed={selected_samples}. "
            "Verify that the local dataset is the supported snapshot and is complete."
        )
    expected_samples = len(ds_val)

    # Exact, unpadded sharding. It works across global ranks regardless of
    # whether torchrun spans one node or many nodes.
    local_indices = list(range(rank, expected_samples, world_size))
    ds_local = SafeEvalDataset(ds_val, local_indices)
    loader = DataLoader(
        ds_local,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        # Pinning multiplied across ranks/workers previously caused /dev/shm
        # SIGBUS failures in the online train/validation loaders.
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    if rank == 0:
        print(
            f"[eval_student] eval_set={eval_set_label} split=val "
            f"selected_samples={selected_samples} evaluated_samples={expected_samples} "
            f"clip_ids_file={clip_ids_file} split_counts={split_counts}"
        )
        print(
            "[eval_student] camera protocol: cross-left, front-wide, cross-right, "
            "front-tele; fixed t0_us (parquet event_t0s are not used)"
        )

    local_records: list[dict[str, Any]] = []
    local_failures: list[dict[str, Any]] = []
    with torch.inference_mode():
        for local_step, batch in enumerate(loader, start=1):
            if "_eval_error" in batch:
                local_failures.append(
                    {
                        "index": int(batch["_eval_index"]),
                        "clip_id": batch["clip_id"],
                        "t0_us": batch["t0_us"],
                        "error": batch["_eval_error"],
                    }
                )
                print(
                    f"[eval_student][rank={rank}] DATA FAILED "
                    f"index={batch['_eval_index']} clip_id={batch['clip_id']}: "
                    f"{batch['_eval_error']}",
                    flush=True,
                )
                continue

            global_index = int(batch["_eval_index"][0])
            clip_id = batch["clip_id"][0]
            t0_us = int(batch["t0_us"][0])
            try:
                ego_future_xyz = batch["ego_future_xyz"].to(device, non_blocking=True)
                gt_xy = ego_future_xyz[:, 0, :, :2]
                tokenized = model._get_generation_mode_tokenized_data_online_distill(
                    batch, helper, processor, device
                )
                model_inputs = helper.to_device(
                    {
                        "tokenized_data": tokenized,
                        "ego_history_xyz": batch["ego_history_xyz"],
                        "ego_history_rot": batch["ego_history_rot"],
                    },
                    device,
                )
                model_inputs_greedy = copy.deepcopy(model_inputs)

                stochastic_seed = _seed_for_sample(
                    args.seed,
                    clip_id=clip_id,
                    t0_us=t0_us,
                    greedy=False,
                    strategy=args.seed_strategy,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz_stochastic, _, _, _ = _run_with_seed(
                        stochastic_seed,
                        device,
                        lambda: _sample(
                            model,
                            model_inputs,
                            greedy=False,
                            max_generation_length=max_generation_length,
                            num_traj_samples=args.num_traj_samples,
                            top_p=args.top_p,
                            temperature=args.temperature,
                        ),
                    )
                stochastic_minade = _minade_values(pred_xyz_stochastic, gt_xy)[0]

                greedy_seed = _seed_for_sample(
                    args.seed,
                    clip_id=clip_id,
                    t0_us=t0_us,
                    greedy=True,
                    strategy=args.seed_strategy,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz_greedy, _, _, _ = _run_with_seed(
                        greedy_seed,
                        device,
                        lambda: _sample(
                            model,
                            model_inputs_greedy,
                            greedy=True,
                            max_generation_length=max_generation_length,
                            num_traj_samples=args.num_traj_samples,
                            top_p=args.top_p,
                            temperature=args.temperature,
                        ),
                    )
                greedy_minade = _minade_values(pred_xyz_greedy, gt_xy)[0]
                if not math.isfinite(stochastic_minade) or not math.isfinite(greedy_minade):
                    raise FloatingPointError(
                        "Non-finite metric: "
                        f"stochastic_minade={stochastic_minade}, greedy_ade={greedy_minade}"
                    )

                local_records.append(
                    {
                        "index": global_index,
                        "clip_id": clip_id,
                        "t0_us": t0_us,
                        "stochastic_minade_m": stochastic_minade,
                        "greedy_vlm_single_ade_m": greedy_minade,
                    }
                )
            except Exception as exc:
                local_failures.append(
                    {
                        "index": global_index,
                        "clip_id": clip_id,
                        "t0_us": t0_us,
                        "error": repr(exc),
                    }
                )
                print(
                    f"[eval_student][rank={rank}] FAILED index={global_index} "
                    f"clip_id={clip_id}: {exc}",
                    flush=True,
                )
                torch.cuda.empty_cache()

            # Do not place collectives here: exact strided shards can have
            # different lengths, and in-loop reductions can deadlock.
            if rank == 0 and args.log_every and local_step % args.log_every == 0:
                local_avg = math.fsum(
                    record["stochastic_minade_m"] for record in local_records
                ) / max(1, len(local_records))
                print(
                    f"[eval_student] rank0_progress={local_step}/{len(loader)} "
                    f"rank0_avg_minADE={local_avg:.4f}m "
                    "(global aggregate is reported at completion)",
                    flush=True,
                )

    local_payload = {"records": local_records, "failures": local_failures}
    gathered_payloads: list[dict[str, Any] | None] | None = (
        [None for _ in range(world_size)] if rank == 0 else None
    )
    if dist.is_initialized():
        dist.gather_object(local_payload, gathered_payloads, dst=0)
    else:
        gathered_payloads = [local_payload]

    fatal_error: str | None = None
    if rank == 0:
        try:
            assert gathered_payloads is not None
            records = [
                record
                for payload in gathered_payloads
                if payload is not None
                for record in payload["records"]
            ]
            failures = [
                failure
                for payload in gathered_payloads
                if payload is not None
                for failure in payload["failures"]
            ]
            records.sort(key=lambda value: value["index"])
            failures.sort(key=lambda value: value["index"])

            attempted_indices = [record["index"] for record in records] + [
                failure["index"] for failure in failures
            ]
            if len(attempted_indices) != len(set(attempted_indices)):
                fatal_error = "Distributed evaluation produced duplicate sample indices"
            elif set(attempted_indices) != set(range(expected_samples)):
                missing = sorted(set(range(expected_samples)) - set(attempted_indices))
                fatal_error = (
                    "Distributed evaluation did not cover the selected dataset exactly; "
                    f"first missing indices: {missing[:10]}"
                )
            elif not records:
                fatal_error = "No evaluation sample completed successfully"

            if records:
                stochastic_minade = math.fsum(
                    record["stochastic_minade_m"] for record in records
                ) / len(records)
                greedy_ade = math.fsum(
                    record["greedy_vlm_single_ade_m"] for record in records
                ) / len(records)
            else:
                stochastic_minade = None
                greedy_ade = None

            summary: dict[str, Any] = {
                "checkpoint": str(ckpt_dir),
                "config": str(config_source),
                "student_vlm_path": student_vlm_path,
                "student_processor_path": student_processor_path,
                "dataset_root": local_dir,
                "eval_set": eval_set_label,
                "split": "val",
                "clip_ids_file": str(clip_ids_file) if clip_ids_file is not None else None,
                "selected_samples": selected_samples,
                "expected_samples": expected_samples,
                "successful_samples": len(records),
                "failed_samples": len(failures),
                "world_size": world_size,
                "t0_us": args.t0_us,
                "seed": args.seed,
                "seed_strategy": args.seed_strategy,
                "strict_determinism": args.strict_determinism,
                "num_traj_samples": args.num_traj_samples,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "max_generation_length": max_generation_length,
                "stochastic_minade_m": stochastic_minade,
                "greedy_vlm_single_ade_m": greedy_ade,
                "records": records,
                "failures": failures,
            }
            if args.output_json is not None:
                output_path = Path(args.output_json).expanduser().resolve()
                _write_json_atomic(output_path, summary)
                print(f"[eval_student] wrote results: {output_path}")

            stochastic_text = (
                f"{stochastic_minade:.4f}m" if stochastic_minade is not None else "unavailable"
            )
            greedy_text = f"{greedy_ade:.4f}m" if greedy_ade is not None else "unavailable"
            print(
                "[eval_student] DONE "
                f"final_minADE(stochastic@{args.num_traj_samples})={stochastic_text} "
                f"final_ADE(greedy_vlm_single)={greedy_text} "
                f"successful_samples={len(records)}/{expected_samples} "
                f"failed_samples={len(failures)}"
            )
            if failures and fatal_error is None:
                fatal_error = (
                    f"{len(failures)} evaluation samples failed; see the result/log output"
                )
        except Exception as exc:
            fatal_error = f"Rank-0 result finalization failed: {exc!r}"
            print(f"[eval_student] {fatal_error}", flush=True)

    status = [fatal_error]
    if dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
        dist.destroy_process_group()
    if status[0] is not None:
        raise RuntimeError(status[0])


if __name__ == "__main__":
    main()
