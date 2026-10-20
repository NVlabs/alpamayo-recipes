# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distributed, student-only trajectory evaluation with Alpamayo 1.5 parity.

Both supported evaluation sets compute their full-set base metrics at a fixed
clip-relative timestamp of 5.1 s:

* ``644`` evaluates the packaged public gold clip list.
* ``90k`` evaluates the full PAI evaluation set.

Inputs use the camera-aware Alpamayo 1.5 prompt built by the training pipeline.
Every clip resets the PyTorch CPU and CUDA RNGs to seed 42 immediately before
each inference call, matching the public Alpamayo 1.5 inference example.

Navigation evaluation is optional. The 644 benchmark requires a valid sidecar
at exactly 5.1 s. Evaluation on the full PAI evaluation set selects the valid
navigation timestamp closest to 5.1 s. Paired unconditional and
navigation-conditioned rollouts use the same selected timestamp, while the
complete-set base metric remains fixed at 5.1 s.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from alpamayo1_5_distill.data.dataset_local import PhysicalAIAVDatasetInterfaceLocal
from alpamayo1_5_distill.data.nav_labels import NavigationLabelStore
from alpamayo1_5_distill.data.physical_aiav_loader import load_physical_aiavdataset
from alpamayo1_5_distill.models.modeling_distill import DistillAlpamayo1_5
from alpamayo1_5_distill.training import processor as helper
from alpamayo1_5_distill.training.factory import build_student_inference_batch
from alpamayo1_5_distill.training.validation import minade_values


PHYSICAL_AI_AV_REVISION = "2ae73f49ffd2b5db43b404201beb7b92889f7afc"
INFERENCE_SEED = 42
FIXED_T0_US = 5_100_000
NUM_TRAJ_SAMPLES = 6
TOP_P = 0.98
TEMPERATURE = 0.6
DEFAULT_MAX_GENERATION_LENGTH = 256
GOLD_SET_SIZE = 644
FULL_VAL_SET_SIZE = 90_928
GOLD_METADB = (
    Path(__file__).resolve().parent
    / "assets"
    / "1005_7cam_gold_eval_metadb_public.parquet"
)
GOLD_METADB_URL = (
    "https://github.com/NVlabs/alpamayo-recipes/raw/refs/heads/main/"
    "data/1005_7cam_gold_eval_metadb_public.parquet"
)
FULL_NAV_DIRECTORY = "eval"
FULL_NAV_ROW_COHORTS = "checkpoint_val_5k,opd_unseen_val_70k"


@dataclass(frozen=True)
class EvalSample:
    """Identity and fixed timestamp for one evaluation sample."""

    index: int
    clip_id: str
    t0_us: int = FIXED_T0_US


class ExactShardSampler(Sampler[int]):
    """Assign every sample to exactly one rank without padding or dropping."""

    def __init__(self, dataset_size: int, rank: int, world_size: int) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.dataset_size - self.rank + self.world_size - 1) // self.world_size)


class PhysicalAIEvalDataset(Dataset[dict[str, Any]]):
    """Decode fixed-timestamp PhysicalAI-AV samples and preserve failures."""

    def __init__(
        self,
        samples: list[EvalSample],
        dataset_root: str,
        *,
        nav_config: dict[str, Any] | None = None,
        nav_samples: dict[int, EvalSample] | None = None,
    ) -> None:
        self.samples = samples
        self.dataset_root = str(dataset_root)
        self.nav_store = NavigationLabelStore(nav_config or {})
        self.nav_samples = nav_samples or {}
        self._interface: PhysicalAIAVDatasetInterfaceLocal | None = None
        self._interface_owner: tuple[int, int | None] | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def _get_interface(self) -> PhysicalAIAVDatasetInterfaceLocal:
        worker = torch.utils.data.get_worker_info()
        owner = (os.getpid(), None if worker is None else worker.id)
        if self._interface is None or self._interface_owner != owner:
            self._interface = PhysicalAIAVDatasetInterfaceLocal(
                revision=PHYSICAL_AI_AV_REVISION,
                local_dir=self.dataset_root,
                local_files_only=True,
            )
            self._interface_owner = owner
        return self._interface

    def __getitem__(self, item: int) -> dict[str, Any]:
        spec = self.samples[item]
        try:
            sample = load_physical_aiavdataset(
                clip_id=spec.clip_id,
                t0_us=spec.t0_us,
                avdi=self._get_interface(),
                maybe_stream=False,
            )
            nav_sample = None
            nav_spec = self.nav_samples.get(spec.index)
            if nav_spec is not None:
                if nav_spec.clip_id != spec.clip_id:
                    raise ValueError(
                        f"navigation sample clip mismatch at index {spec.index}: "
                        f"{nav_spec.clip_id!r} != {spec.clip_id!r}"
                    )
                if nav_spec.t0_us == spec.t0_us:
                    nav_sample = self.nav_store.attach(sample)
                else:
                    nav_sample = load_physical_aiavdataset(
                        clip_id=nav_spec.clip_id,
                        t0_us=nav_spec.t0_us,
                        avdi=self._get_interface(),
                        maybe_stream=False,
                    )
                    nav_sample = self.nav_store.attach(nav_sample)
                if not bool(nav_sample.get("nav_enabled", False)):
                    raise ValueError(
                        f"selected navigation sample is not enabled: "
                        f"{nav_spec.clip_id}@{nav_spec.t0_us}"
                    )
        except Exception as error:
            return {
                "eval_index": spec.index,
                "clip_id": spec.clip_id,
                "t0_us": spec.t0_us,
                "error": repr(error),
            }
        return {"eval_index": spec.index, "sample": sample, "nav_sample": nav_sample}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_interface"] = None
        state["_interface_owner"] = None
        return state


def _single_item(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError(f"trajectory evaluation requires batch size 1, got {len(items)}")
    return items[0]


def _setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        torch.cuda.set_device(0)
        return 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(hours=6),
    )
    return local_rank, dist.get_rank(), dist.get_world_size()


def _gold_samples() -> list[EvalSample]:
    if not GOLD_METADB.is_file():
        raise FileNotFoundError(
            f"Gold evaluation metadata is missing: {GOLD_METADB}. "
            f"Download {GOLD_METADB_URL} to {GOLD_METADB} before running gold evaluation."
        )
    frame = pd.read_parquet(GOLD_METADB, columns=["key"])
    clip_ids = frame["key"].dropna().astype(str).tolist()
    if len(clip_ids) != GOLD_SET_SIZE or len(set(clip_ids)) != GOLD_SET_SIZE:
        raise ValueError(
            f"gold metadb must contain {GOLD_SET_SIZE} unique clips; "
            f"found rows={len(clip_ids)}, unique={len(set(clip_ids))}"
        )
    return [EvalSample(index=index, clip_id=clip_id) for index, clip_id in enumerate(clip_ids)]


def _full_validation_samples(
    dataset_root: str,
    *,
    allow_count_mismatch: bool,
) -> list[EvalSample]:
    interface = PhysicalAIAVDatasetInterfaceLocal(
        revision=PHYSICAL_AI_AV_REVISION,
        local_dir=dataset_root,
        local_files_only=True,
    )
    clip_index = interface.clip_index
    if "split" not in clip_index.columns:
        raise ValueError(f"clip_index.parquet has no split column: {list(clip_index.columns)}")
    clip_ids = (
        clip_index.index[clip_index["split"] == "val"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if len(clip_ids) != FULL_VAL_SET_SIZE and not allow_count_mismatch:
        raise ValueError(
            f"expected {FULL_VAL_SET_SIZE} clips in the PhysicalAI-AV val split at "
            f"revision {PHYSICAL_AI_AV_REVISION}, found {len(clip_ids)}; pass "
            "--allow-count-mismatch only if this dataset revision intentionally differs"
        )
    return [EvalSample(index=index, clip_id=clip_id) for index, clip_id in enumerate(clip_ids)]


def _student_checkpoint_dir(checkpoint: str) -> Path:
    root = Path(checkpoint).expanduser().resolve()
    student = root / "student"
    if student.is_dir():
        root = student
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"student checkpoint has no config.json: {root}")
    if not list(root.glob("*.safetensors")):
        raise FileNotFoundError(f"student checkpoint has no safetensors weights: {root}")
    return root


def _load_student(
    checkpoint: str,
    student_vlm_path: str | None,
    device: torch.device,
) -> DistillAlpamayo1_5:
    student_dir = _student_checkpoint_dir(checkpoint)
    load_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if student_vlm_path:
        load_kwargs["config"] = DistillAlpamayo1_5.config_with_vlm_override(
            str(student_dir),
            str(Path(student_vlm_path).expanduser().resolve()),
        )
    model = DistillAlpamayo1_5.from_pretrained(str(student_dir), **load_kwargs).to(device)
    model.eval().requires_grad_(False)
    return model


def _model_inputs(
    model: DistillAlpamayo1_5,
    image_processor: Any,
    sample: dict[str, Any],
    *,
    use_navigation: bool = False,
) -> dict[str, Any]:
    """Build the same basic or navigation-conditioned prompt as validation."""

    if use_navigation and not bool(sample.get("nav_enabled", False)):
        raise ValueError("navigation inference requires an enabled navigation label")
    prepared_sample = dict(sample)
    prepared_sample["nav_enabled"] = bool(use_navigation)
    return build_student_inference_batch(
        model,
        [prepared_sample],
        processor=image_processor,
    )


def _nav_eligible_samples(
    samples: list[EvalSample],
    nav_store: NavigationLabelStore,
    *,
    t0_mode: str = "exact",
) -> list[EvalSample]:
    """Return samples at an exact or closest valid navigation timestamp."""

    if t0_mode not in {"exact", "closest"}:
        raise ValueError(f"unsupported navigation t0 mode: {t0_mode!r}")

    available = nav_store.available_clip_ids()
    selected: list[EvalSample] = []
    for sample in samples:
        if sample.clip_id.removeprefix("clipgt-") not in available:
            continue
        valid_timestamps = nav_store.valid_t0_us(sample.clip_id)
        if not valid_timestamps:
            continue
        if t0_mode == "exact":
            if sample.t0_us not in valid_timestamps:
                continue
            selected_t0_us = sample.t0_us
        else:
            # Prefer the earlier timestamp when two candidates are equally close.
            selected_t0_us = min(
                valid_timestamps,
                key=lambda value: (abs(value - sample.t0_us), value),
            )
        selected.append(
            EvalSample(
                index=sample.index,
                clip_id=sample.clip_id,
                t0_us=selected_t0_us,
            )
        )
    return selected


def _resolve_navigation_t0_mode(eval_set: str, t0_mode: str | None) -> str:
    """Keep gold644 exact; use the closest valid timestamp for full PAI evaluation."""
    return t0_mode or ("closest" if eval_set == "90k" else "exact")


def _resolve_navigation_cohorts(
    eval_set: str,
    cohort: str | None,
    allowed_row_cohorts: str | None,
) -> tuple[str, str | None]:
    """Select strict gold644 defaults or the full PAI evaluation directory."""
    resolved_cohort = cohort or (FULL_NAV_DIRECTORY if eval_set == "90k" else "gold644")
    resolved_allowed = allowed_row_cohorts
    if (
        resolved_allowed is None
        and eval_set == "90k"
        and resolved_cohort == FULL_NAV_DIRECTORY
    ):
        resolved_allowed = FULL_NAV_ROW_COHORTS
    return resolved_cohort, resolved_allowed


def _reset_inference_seed() -> None:
    torch.manual_seed(INFERENCE_SEED)
    torch.cuda.manual_seed_all(INFERENCE_SEED)


def _sample_predictions(
    model: DistillAlpamayo1_5,
    model_inputs: dict[str, Any],
    *,
    greedy: bool,
    max_generation_length: int,
) -> torch.Tensor:
    _reset_inference_seed()
    kwargs: dict[str, Any] = {
        "data": model_inputs,
        "num_traj_samples": 1 if greedy else NUM_TRAJ_SAMPLES,
        "max_generation_length": max_generation_length,
        "return_extra": True,
        # Match training validation when a rollout exhausts its token budget
        # before emitting <traj_future_start>.
        "allow_missing_future_start": True,
    }
    if greedy:
        kwargs.update(top_p=1.0, temperature=1.0, do_sample=False)
    else:
        kwargs.update(top_p=TOP_P, temperature=TEMPERATURE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.sample_trajectories_from_data_with_vlm_rollout(**kwargs)
    return output[0]


def trajectory_minade(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """Compute minADE with the same implementation as training validation."""

    values = minade_values(pred_xyz, gt_xyz)
    if values.numel() != 1:
        raise ValueError(f"expected one minADE value, got shape {tuple(values.shape)}")
    return float(values.item())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--eval-set", choices=("644", "90k"), required=True)
    parser.add_argument(
        "--student-vlm-path",
        default=None,
        help="Optional local Cosmos-Reason2-2B path overriding the path saved in config.json.",
    )
    parser.add_argument(
        "--processor-path",
        default=None,
        help="Optional processor path; defaults to --student-vlm-path or checkpoint config.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--max-generation-length",
        "--max_generation_length",
        dest="max_generation_length",
        type=int,
        default=DEFAULT_MAX_GENERATION_LENGTH,
        help=(
            "Maximum VLM rollout tokens per trajectory sample; matches the "
            "Alpamayo 1.5 inference default of 256."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke-test only: evaluate the first N selected clips; 0 means all.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help=(
            "Allow full PAI evaluation when the local split size differs "
            "from the reference snapshot."
        ),
    )
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help=(
            "Report metrics over successful samples when any sample fails. "
            "By default failures invalidate the evaluation, as in training validation."
        ),
    )
    parser.add_argument(
        "--include-nav-eval",
        action="store_true",
        help=(
            "Also evaluate navigation-conditioned trajectories on samples with "
            "a valid navigation sidecar at the fixed 5.1-second timestamp."
        ),
    )
    parser.add_argument(
        "--nav-root",
        default=None,
        help="Navigation-sidecar root; required with --include-nav-eval.",
    )
    parser.add_argument(
        "--nav-cohort",
        default=None,
        help="Navigation-sidecar directory under --nav-root; defaults by evaluation set.",
    )
    parser.add_argument(
        "--nav-allowed-row-cohorts",
        default=None,
        help=(
            "Optional comma-separated row-level cohort names accepted inside the "
            "selected sidecar directory. By default only --nav-cohort is accepted."
        ),
    )
    parser.add_argument(
        "--nav-clip-index-path",
        default=None,
        help="Optional prebuilt navigation clip index used by the training loader.",
    )
    parser.add_argument(
        "--nav-max-distance-m",
        type=float,
        default=-1.0,
        help="Maximum navigation-instruction distance; -1 disables the limit.",
    )
    parser.add_argument(
        "--nav-t0-mode",
        choices=("exact", "closest"),
        default=None,
        help=(
            "Navigation timestamp selection; defaults to exact for 644 and "
            "closest-to-5.1-seconds for the full PAI evaluation set."
        ),
    )
    parser.add_argument(
        "--nav-include-unknown",
        action="store_true",
        help="Include UNKNOWN navigation commands (excluded by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for trajectory evaluation")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.max_generation_length <= 0:
        raise ValueError("--max-generation-length must be positive")
    if args.log_every < 0 or args.limit < 0:
        raise ValueError("--log-every and --limit must be non-negative")
    if args.nav_max_distance_m < 0.0 and args.nav_max_distance_m != -1.0:
        raise ValueError("--nav-max-distance-m must be -1 or non-negative")
    if args.include_nav_eval and not args.nav_root:
        raise ValueError("--nav-root is required with --include-nav-eval")

    local_rank, rank, world_size = _setup_distributed()
    device = torch.device("cuda", local_rank)
    try:
        if args.eval_set == "644":
            samples = _gold_samples()
        else:
            samples = _full_validation_samples(
                args.dataset_root,
                allow_count_mismatch=args.allow_count_mismatch,
            )
        selected_samples = len(samples)
        if args.limit:
            samples = samples[: min(args.limit, selected_samples)]
        expected_samples = len(samples)
        if not expected_samples:
            raise ValueError("evaluation set is empty")

        nav_cohort, nav_allowed_row_cohorts = _resolve_navigation_cohorts(
            args.eval_set,
            args.nav_cohort,
            args.nav_allowed_row_cohorts,
        )
        nav_t0_mode = _resolve_navigation_t0_mode(args.eval_set, args.nav_t0_mode)
        nav_config = {
            "enabled": bool(args.include_nav_eval),
            "root": args.nav_root or "",
            "cohort": nav_cohort,
            "allowed_row_cohorts": nav_allowed_row_cohorts,
            "clip_index_path": args.nav_clip_index_path,
            "max_distance_m": args.nav_max_distance_m,
            "exclude_unknown": not args.nav_include_unknown,
        }
        nav_eligible_samples: list[EvalSample] = []
        if args.include_nav_eval:
            nav_store = NavigationLabelStore(nav_config)
            nav_eligible_samples = _nav_eligible_samples(
                samples,
                nav_store,
                t0_mode=nav_t0_mode,
            )
            if not nav_eligible_samples:
                raise ValueError(
                    "navigation evaluation found no valid labels using "
                    f"t0_mode={nav_t0_mode!r} around t0_us={FIXED_T0_US} "
                    f"in directory {nav_cohort!r}"
                )
        nav_samples_by_index = {sample.index: sample for sample in nav_eligible_samples}

        model = _load_student(args.checkpoint, args.student_vlm_path, device)
        processor_path = (
            args.processor_path
            or args.student_vlm_path
            or getattr(model.config, "vlm_name_or_path", None)
        )
        image_processor = helper.get_processor(model.tokenizer, processor_path)

        dataset = PhysicalAIEvalDataset(
            samples,
            args.dataset_root,
            nav_config=nav_config,
            nav_samples=nav_samples_by_index,
        )
        workers = int(args.num_workers)
        loader_kwargs: dict[str, Any] = {}
        if workers:
            loader_kwargs.update(
                prefetch_factor=int(args.prefetch_factor),
                persistent_workers=True,
                multiprocessing_context="spawn",
            )
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=ExactShardSampler(expected_samples, rank, world_size),
            num_workers=workers,
            pin_memory=False,
            collate_fn=_single_item,
            **loader_kwargs,
        )

        if rank == 0:
            print(
                "[trajectory-eval] "
                f"eval_set={args.eval_set} selected={selected_samples} "
                f"evaluated={expected_samples} world_size={world_size} "
                f"t0_us={FIXED_T0_US} seed={INFERENCE_SEED} "
                f"include_nav_eval={args.include_nav_eval} "
                f"nav_t0_mode={nav_t0_mode} "
                f"nav_eligible={len(nav_eligible_samples)}",
                flush=True,
            )

        local_records: list[dict[str, Any]] = []
        local_failures: list[dict[str, Any]] = []
        with torch.inference_mode():
            for local_step, item in enumerate(loader, start=1):
                if "error" in item:
                    local_failures.append(item)
                    print(
                        f"[trajectory-eval][rank={rank}] DATA FAILED "
                        f"index={item['eval_index']} clip_id={item['clip_id']}: {item['error']}",
                        flush=True,
                    )
                    continue

                index = int(item["eval_index"])
                sample = item["sample"]
                nav_sample = item["nav_sample"]
                clip_id = str(sample["clip_id"])
                try:
                    model_inputs = _model_inputs(
                        model,
                        image_processor,
                        sample,
                    )
                    sampled = _sample_predictions(
                        model,
                        model_inputs,
                        greedy=False,
                        max_generation_length=args.max_generation_length,
                    )
                    minade6 = trajectory_minade(sampled, model_inputs["ego_future_xyz"])

                    # Rebuild the prompt because trajectory sampling consumes/mutates
                    # tokenized_data, then reset seed 42 for the one-trajectory run.
                    greedy_inputs = _model_inputs(
                        model,
                        image_processor,
                        sample,
                    )
                    greedy = _sample_predictions(
                        model,
                        greedy_inputs,
                        greedy=True,
                        max_generation_length=args.max_generation_length,
                    )
                    greedy_ade = trajectory_minade(
                        greedy,
                        greedy_inputs["ego_future_xyz"],
                    )
                    if not math.isfinite(minade6) or not math.isfinite(greedy_ade):
                        raise FloatingPointError(
                            f"non-finite metric: minADE@6={minade6}, greedy_ADE={greedy_ade}"
                        )
                    record = {
                        "index": index,
                        "clip_id": clip_id,
                        "t0_us": FIXED_T0_US,
                        "minade_at_6_m": minade6,
                        "greedy_ade_m": greedy_ade,
                    }
                    if args.include_nav_eval and nav_sample is not None:
                        nav_t0_us = int(nav_sample["t0_us"])
                        if nav_t0_us == int(sample["t0_us"]):
                            nav_subset_basic_minade6 = minade6
                            nav_subset_basic_greedy_ade = greedy_ade
                        else:
                            nav_basic_inputs = _model_inputs(
                                model,
                                image_processor,
                                nav_sample,
                            )
                            nav_basic_sampled = _sample_predictions(
                                model,
                                nav_basic_inputs,
                                greedy=False,
                                max_generation_length=args.max_generation_length,
                            )
                            nav_subset_basic_minade6 = trajectory_minade(
                                nav_basic_sampled,
                                nav_basic_inputs["ego_future_xyz"],
                            )
                            nav_basic_greedy_inputs = _model_inputs(
                                model,
                                image_processor,
                                nav_sample,
                            )
                            nav_basic_greedy = _sample_predictions(
                                model,
                                nav_basic_greedy_inputs,
                                greedy=True,
                                max_generation_length=args.max_generation_length,
                            )
                            nav_subset_basic_greedy_ade = trajectory_minade(
                                nav_basic_greedy,
                                nav_basic_greedy_inputs["ego_future_xyz"],
                            )
                        nav_inputs = _model_inputs(
                            model,
                            image_processor,
                            nav_sample,
                            use_navigation=True,
                        )
                        nav_sampled = _sample_predictions(
                            model,
                            nav_inputs,
                            greedy=False,
                            max_generation_length=args.max_generation_length,
                        )
                        nav_minade6 = trajectory_minade(
                            nav_sampled,
                            nav_inputs["ego_future_xyz"],
                        )
                        nav_greedy_inputs = _model_inputs(
                            model,
                            image_processor,
                            nav_sample,
                            use_navigation=True,
                        )
                        nav_greedy = _sample_predictions(
                            model,
                            nav_greedy_inputs,
                            greedy=True,
                            max_generation_length=args.max_generation_length,
                        )
                        nav_greedy_ade = trajectory_minade(
                            nav_greedy,
                            nav_greedy_inputs["ego_future_xyz"],
                        )
                        nav_metrics = (
                            nav_subset_basic_minade6,
                            nav_subset_basic_greedy_ade,
                            nav_minade6,
                            nav_greedy_ade,
                        )
                        if not all(math.isfinite(value) for value in nav_metrics):
                            raise FloatingPointError(
                                "non-finite navigation metric: "
                                f"values={nav_metrics}"
                            )
                        record.update(
                            nav_t0_us=nav_t0_us,
                            nav_t0_offset_us=nav_t0_us - FIXED_T0_US,
                            nav_subset_basic_minade_at_6_m=nav_subset_basic_minade6,
                            nav_subset_basic_greedy_ade_m=nav_subset_basic_greedy_ade,
                            nav_minade_at_6_m=nav_minade6,
                            nav_greedy_ade_m=nav_greedy_ade,
                            nav_text=str(nav_sample["nav_text"]),
                            nav_command=str(nav_sample["nav_command"]),
                            nav_distance_m=float(nav_sample["nav_distance_m"]),
                        )
                    local_records.append(record)
                except Exception as error:
                    local_failures.append(
                        {
                            "eval_index": index,
                            "clip_id": clip_id,
                            "t0_us": FIXED_T0_US,
                            "error": repr(error),
                        }
                    )
                    print(
                        f"[trajectory-eval][rank={rank}] INFERENCE FAILED "
                        f"index={index} clip_id={clip_id}: {error}",
                        flush=True,
                    )
                    torch.cuda.empty_cache()

                if rank == 0 and args.log_every and local_step % args.log_every == 0:
                    print(
                        f"[trajectory-eval] rank0_progress={local_step}/{len(loader)} ",
                        flush=True,
                    )

        payload = {"records": local_records, "failures": local_failures}
        gathered: list[dict[str, Any] | None] | None = (
            [None] * world_size if rank == 0 else None
        )
        if dist.is_initialized():
            dist.gather_object(payload, gathered, dst=0)
        else:
            gathered = [payload]

        if rank == 0:
            assert gathered is not None
            records = sorted(
                (
                    record
                    for rank_payload in gathered
                    if rank_payload is not None
                    for record in rank_payload["records"]
                ),
                key=lambda record: record["index"],
            )
            failures = sorted(
                (
                    failure
                    for rank_payload in gathered
                    if rank_payload is not None
                    for failure in rank_payload["failures"]
                ),
                key=lambda failure: failure["eval_index"],
            )
            attempted = [record["index"] for record in records] + [
                failure["eval_index"] for failure in failures
            ]
            if len(attempted) != len(set(attempted)):
                raise RuntimeError("distributed evaluation produced duplicate sample indices")
            if set(attempted) != set(range(expected_samples)):
                missing = sorted(set(range(expected_samples)) - set(attempted))
                raise RuntimeError(
                    f"distributed evaluation missed sample indices; first missing={missing[:10]}"
                )
            if not records:
                raise RuntimeError("no evaluation samples completed successfully")
            if failures and not args.allow_partial_results:
                raise RuntimeError(
                    f"{len(failures)} evaluation sample(s) failed; refusing to report "
                    "a partial minADE. Inspect the failure logs or pass "
                    "--allow-partial-results explicitly."
                )

            mean_minade6 = sum(record["minade_at_6_m"] for record in records) / len(records)
            mean_greedy_ade = sum(record["greedy_ade_m"] for record in records) / len(records)
            nav_records = [record for record in records if "nav_minade_at_6_m" in record]
            if args.include_nav_eval and not nav_records:
                raise RuntimeError("no navigation evaluation samples completed successfully")
            summary = {
                "checkpoint": str(_student_checkpoint_dir(args.checkpoint)),
                "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
                "dataset_revision": PHYSICAL_AI_AV_REVISION,
                "eval_set": args.eval_set,
                "selected_samples": selected_samples,
                "expected_samples": expected_samples,
                "successful_samples": len(records),
                "failed_samples": len(failures),
                "world_size": world_size,
                "t0_us": FIXED_T0_US,
                "seed": INFERENCE_SEED,
                "num_traj_samples": NUM_TRAJ_SAMPLES,
                "top_p": TOP_P,
                "temperature": TEMPERATURE,
                "max_generation_length": args.max_generation_length,
                "camera_aware_prompt": True,
                "allow_missing_future_start": True,
                "partial_results_allowed": bool(args.allow_partial_results),
                "mean_minade_at_6_m": mean_minade6,
                "mean_greedy_ade_m": mean_greedy_ade,
                "records": records,
                "failures": failures,
            }
            if args.include_nav_eval:
                nav_count = len(nav_records)
                nav_timestamps = [int(record["nav_t0_us"]) for record in nav_records]
                nav_abs_offsets = [
                    abs(int(record["nav_t0_offset_us"])) for record in nav_records
                ]
                summary.update(
                    nav_eval_enabled=True,
                    nav_root=str(Path(args.nav_root).expanduser().resolve()),
                    nav_cohort=nav_cohort,
                    nav_allowed_row_cohorts=sorted(nav_store.allowed_row_cohorts),
                    nav_t0_mode=nav_t0_mode,
                    nav_max_distance_m=args.nav_max_distance_m,
                    nav_exclude_unknown=not args.nav_include_unknown,
                    nav_eligible_samples=len(nav_eligible_samples),
                    successful_nav_samples=nav_count,
                    nav_t0_min_us=min(nav_timestamps),
                    nav_t0_max_us=max(nav_timestamps),
                    mean_nav_t0_abs_offset_us=sum(nav_abs_offsets) / nav_count,
                    mean_nav_subset_basic_minade_at_6_m=sum(
                        record["nav_subset_basic_minade_at_6_m"]
                        for record in nav_records
                    )
                    / nav_count,
                    mean_nav_subset_basic_greedy_ade_m=sum(
                        record["nav_subset_basic_greedy_ade_m"]
                        for record in nav_records
                    )
                    / nav_count,
                    mean_nav_minade_at_6_m=sum(
                        record["nav_minade_at_6_m"] for record in nav_records
                    )
                    / nav_count,
                    mean_nav_greedy_ade_m=sum(
                        record["nav_greedy_ade_m"] for record in nav_records
                    )
                    / nav_count,
                )
            else:
                summary["nav_eval_enabled"] = False
            output_path = Path(args.output_json).expanduser().resolve()
            _write_json_atomic(output_path, summary)
            print(
                "[trajectory-eval] DONE "
                f"minADE@6={mean_minade6:.6f}m "
                f"greedy_ADE={mean_greedy_ade:.6f}m "
                f"successful={len(records)}/{expected_samples} "
                f"failed={len(failures)}"
                + (
                    " nav_minADE@6="
                    f"{summary['mean_nav_minade_at_6_m']:.6f}m "
                    "nav_greedy_ADE="
                    f"{summary['mean_nav_greedy_ade_m']:.6f}m "
                    f"nav_successful={len(nav_records)}/{len(nav_eligible_samples)}"
                    if args.include_nav_eval
                    else ""
                )
                + f" output={output_path}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
