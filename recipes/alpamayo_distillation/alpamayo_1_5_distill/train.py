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

"""Accelerate-based single-phase online Alpamayo distillation trainer."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from alpamayo1_5_distill.data.physical_aiav_dataset import (
    OnlinePhysicalAIAVDataset,
    identity_collate,
)
from alpamayo1_5_distill.training import processor as helper
from alpamayo1_5_distill.training.common import (
    atomic_write_json,
    import_callable,
    load_config,
    tokenizer_sha256,
)
from alpamayo1_5_distill.training.factory import (
    InvalidStudentGenerationError,
    InvalidTeacherGenerationError,
    build_on_policy_batch,
    build_online_teacher_batch,
)
from alpamayo1_5_distill.training.losses import DistillationLoss, DistillationLossConfig
from alpamayo1_5_distill.training.lingoqa import (
    InvalidVQAGenerationError,
    LingoQADataset,
    PAIVQADataset,
    TaskConcatDataset,
    WeightedTaskDistributedBatchSampler,
    build_lingoqa_gt_batch,
    build_lingoqa_on_policy_batch,
    build_lingoqa_teacher_guided_gt_batch,
    build_pai_vqa_teacher_batch,
)
from alpamayo1_5_distill.training.on_policy import OnPolicyDistillConfig
from alpamayo1_5_distill.training.pai_vqa_questions import PAI_VQA_QUESTIONS
from alpamayo1_5_distill.training.resume import (
    ResumeDistributedSampler,
    load_trainer_state,
    make_trainer_state,
)
from alpamayo1_5_distill.training.validation import RankShardSampler, run_validation


class DistillTrainModule(nn.Module):
    """Keep the model and trainable loss projections in one checkpointed module."""

    def __init__(self, model: nn.Module, objective: DistillationLoss):
        super().__init__()
        self.model = model
        self.objective = objective

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if batch.get("task_type") == "vqa":
            tokenized_data = {
                key: value
                for key, value in batch["tokenized_data"].items()
                if not (
                    key.endswith("_mask")
                    and key.removesuffix("_mask") in batch["tokenized_data"]
                )
            }
            outputs = self.model.vlm(
                **tokenized_data,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            language_losses = self.objective.language_terms(
                {"student_logits": outputs.logits},
                batch,
            )
            outputs.logits = None
            return self.objective(language_losses, batch)
        method = getattr(self.model, "forward_distill", None)
        if method is None:
            raise AttributeError("model must implement forward_distill(batch)")
        if self.objective.config.future_traj_vlm_weight != 0.0:
            batch = dict(batch)
            batch["compute_future_traj_vlm_loss"] = True
        outputs = method(batch, language_loss_fn=self.objective.language_terms)
        if not isinstance(outputs, dict):
            raise TypeError("forward_distill(batch) must return a dict")
        return self.objective(outputs, batch)


def _unify_trainable_param_dtype(
    module: nn.Module,
    target_dtype: torch.dtype,
) -> int:
    """Cast trainable parameters to one dtype as required by ZeRO-1."""
    cast_count = 0
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.is_floating_point():
            if parameter.dtype != target_dtype:
                parameter.data = parameter.data.to(target_dtype)
                cast_count += 1
    return cast_count


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move restored optimizer tensors onto the parameter device."""

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    def move_states(states: Any) -> None:
        for state in states:
            if isinstance(state, dict):
                for key, value in state.items():
                    state[key] = move(value)

    move_states(optimizer.state.values())
    inner_optimizer = getattr(optimizer, "optim", None)
    if inner_optimizer is not None:
        move_states(inner_optimizer.state.values())


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None


def train(config: dict[str, Any]) -> None:
    training = config["training"]
    data_section = config.get("data", {})
    lingoqa_requested = bool(data_section.get("lingoqa", {}).get("enabled", False))
    pai_vqa_requested = bool(data_section.get("pai_vqa", {}).get("enabled", False))
    vqa_requested = lingoqa_requested or pai_vqa_requested
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=vqa_requested,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(training.get("mixed_precision", "no")),
        kwargs_handlers=[ddp_kwargs],
    )

    def global_print(message: str) -> None:
        if accelerator.is_main_process:
            print(message, flush=True)

    set_seed(int(training.get("seed", 17)))
    output_dir = Path(training["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_dir / "resolved_config.json", config)
    accelerator.wait_for_everyone()

    data_config = dict(config["data"])
    data_config["local_dir"] = data_config.get("local_dir") or os.getenv("ALPAMAYO_DATASET_DIR")
    t0_us_list = data_config.get("t0_us_list")
    if t0_us_list is None:
        t0_us_list = [int(data_config.get("t0_us", 5_100_000))]
    nav_config = dict(data_config.get("nav", {}))
    nav_config["root"] = nav_config.get("root") or os.getenv("ALPAMAYO_NAV_DATASET_DIR")
    sample_loader_kwargs: dict[str, Any] = {"maybe_stream": False}
    # Explicit mixtures choose navigation t0s from aligned sidecar rows. Only
    # attachment mode needs a shared grid; basic mixture samples stay unrestricted.
    if bool(nav_config.get("enabled", False)) and not bool(
        data_config.get("mixture", {}).get("enabled", False)
    ):
        sample_loader_kwargs["t0_quantum_us"] = 100_000
    pai_dataset = OnlinePhysicalAIAVDataset(
        local_dir=data_config.get("local_dir"),
        token=os.environ.get("HF_TOKEN"),
        split=str(data_config.get("split", "train")),
        t0_us_list=t0_us_list,
        random_t0=bool(data_config.get("random_t0", True)),
        samples_per_clip=int(data_config.get("samples_per_clip", 5)),
        margin_us=int(data_config.get("t0_margin_us", 0)),
        seed=int(training.get("seed", 17)),
        max_clip_retries=int(data_config.get("max_clip_retries", 8)),
        loader_kwargs=sample_loader_kwargs,
        clip_allowlist_path=data_config.get("clip_allowlist_path"),
        exclude_clip_ids_path=data_config.get("exclude_clip_ids_path"),
        nav_config=nav_config,
        mixture_config=data_config.get("mixture"),
    )
    exclude_clip_ids_path = data_config.get("exclude_clip_ids_path")
    if exclude_clip_ids_path is not None:
        global_print(
            f"[data] Loaded {pai_dataset.exclusion_file_clip_count} eval clip IDs from "
            f"{Path(exclude_clip_ids_path).resolve()} — will exclude from training set."
        )
        global_print(
            f"[physicalAiAvDataset] Excluded "
            f"{pai_dataset.excluded_selected_clip_count} eval clips from "
            f"{pai_dataset.split} split ({len(pai_dataset.clip_ids)} remaining)"
        )
    if pai_dataset.nav_store.enabled:
        if pai_dataset.mixture_enabled:
            global_print(
                f"[navigation] explicit mixture requested_basic="
                f"{pai_dataset.basic_ratio / (pai_dataset.basic_ratio + pai_dataset.nav_ratio):.3f}, "
                f"requested_nav="
                f"{pai_dataset.nav_ratio / (pai_dataset.basic_ratio + pai_dataset.nav_ratio):.3f}; "
                f"basic_pool={len(pai_dataset.clip_ids)}, "
                f"nav_pool={len(pai_dataset.nav_clip_ids)}, "
                f"nav_t0_retries={pai_dataset.nav_t0_retries}, "
                f"nav_sample_retries={pai_dataset.nav_sample_retries}, "
                f"fallback_nav_to_basic={pai_dataset.fallback_nav_to_basic}"
            )
        else:
            global_print(
                f"[navigation] cohort={pai_dataset.nav_store.cohort}, "
                f"full_pool={len(pai_dataset.clip_ids)}, "
                f"nav_pool={len(pai_dataset.nav_clip_ids)}, "
                f"max_distance_m={pai_dataset.nav_store.max_distance_m:g}"
            )
    if not pai_dataset:
        raise ValueError("training dataset contains no records")
    num_workers = int(data_config.get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_config.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(data_config.get("persistent_workers", True))
        loader_kwargs["multiprocessing_context"] = "spawn"
    train_seed = int(training.get("seed", 17))
    train_batch_size = int(data_config.get("batch_size", 1))
    lingoqa_config = dict(data_config.get("lingoqa", {}))
    lingoqa_enabled = bool(lingoqa_config.get("enabled", False))
    pai_vqa_config = dict(data_config.get("pai_vqa", {}))
    pai_vqa_enabled = bool(pai_vqa_config.get("enabled", False))
    if pai_vqa_enabled and not lingoqa_enabled:
        raise ValueError("data.pai_vqa.enabled requires data.lingoqa.enabled")
    if lingoqa_enabled:
        lingoqa_dataset = LingoQADataset(
            data_root=lingoqa_config["data_root"],
            parquet_name=str(lingoqa_config.get("parquet_name", "train.parquet")),
            n_frames=lingoqa_config.get("n_frames"),
        )
        pai_vqa_dataset = None
        if pai_vqa_enabled:
            questions = pai_vqa_config.get("questions")
            pai_vqa_dataset = PAIVQADataset(
                pai_dataset,
                questions=(
                    PAI_VQA_QUESTIONS
                    if questions is None
                    else [str(question) for question in questions]
                ),
                seed=train_seed,
            )
        dataset = TaskConcatDataset(pai_dataset, lingoqa_dataset, pai_vqa_dataset)
        sampler = WeightedTaskDistributedBatchSampler(
            dataset,
            batch_size=train_batch_size,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=train_seed,
            pai_weight=lingoqa_config.get("pai_weight", 1),
            lingoqa_weight=lingoqa_config.get("weight", 1),
            pai_vqa_probability=float(pai_vqa_config.get("probability", 0.5)),
            shuffle=bool(data_config.get("shuffle", True)),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=bool(data_config.get("pin_memory", True)),
            collate_fn=identity_collate,
            **loader_kwargs,
        )
        source_exposure = sampler.global_source_samples
        pai_exposure, lingoqa_exposure = source_exposure[:2]
        pai_vqa_exposure = source_exposure[2] if len(source_exposure) == 3 else 0
        frame_limit = lingoqa_config.get("n_frames")
        global_print(
            f"[mixture] LingoQA enabled: PAI weight={lingoqa_config.get('pai_weight', 1)}, "
            f"LingoQA weight={lingoqa_config.get('weight', 1)}, "
            f"epoch exposure trajectory:lingoqa_vqa:pai_vqa="
            f"{pai_exposure}:{lingoqa_exposure}:{pai_vqa_exposure}, "
            f"n_frames={'native' if frame_limit is None else frame_limit}"
        )
        if pai_vqa_enabled:
            global_print(
                "[mixture] PAI VQA enabled: "
                f"selection_probability={float(pai_vqa_config.get('probability', 0.5)):.3f}, "
                f"questions={len(pai_vqa_dataset.questions)}, "
                "supervision=Alpamayo-1.5 teacher-generated answer"
            )
        state_dataset_length = sampler.total_size
    else:
        dataset = pai_dataset
        sampler = ResumeDistributedSampler(
            dataset,
            world_size=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=train_seed,
            shuffle=bool(data_config.get("shuffle", True)),
            drop_last=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=bool(data_config.get("pin_memory", True)),
            drop_last=True,
            collate_fn=identity_collate,
            **loader_kwargs,
        )
        state_dataset_length = len(dataset)
    validation_config = config.get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", False))
    val_loader = None
    if validation_enabled:
        val_nav_config = dict(validation_config.get("nav", {}))
        val_nav_config["root"] = val_nav_config.get("root") or os.getenv("ALPAMAYO_NAV_DATASET_DIR")
        val_dataset = OnlinePhysicalAIAVDataset(
            local_dir=data_config.get("local_dir"),
            token=os.environ.get("HF_TOKEN"),
            split=str(validation_config.get("split", "val")),
            t0_us_list=[int(validation_config.get("t0_us", 5_100_000))],
            random_t0=False,
            samples_per_clip=1,
            margin_us=int(validation_config.get("t0_margin_us", 0)),
            seed=int(training.get("seed", 17)),
            max_clip_retries=int(
                validation_config.get("max_clip_retries", data_config.get("max_clip_retries", 8))
            ),
            loader_kwargs={"maybe_stream": False},
            clip_allowlist_path=validation_config.get("clip_allowlist_path"),
            exclude_clip_ids_path=validation_config.get("exclude_clip_ids_path"),
            nav_config=val_nav_config,
        )
        if val_dataset.nav_store.enabled:
            global_print(
                f"[navigation:validation] cohort={val_dataset.nav_store.cohort} "
                f"full_pool={len(val_dataset.clip_ids)}, "
                f"nav_pool={len(val_dataset.nav_clip_ids)}"
            )
        val_workers = int(validation_config.get("num_workers", 0))
        val_loader_kwargs: dict[str, Any] = {}
        if val_workers > 0:
            val_loader_kwargs["prefetch_factor"] = int(validation_config.get("prefetch_factor", 2))
            val_loader_kwargs["persistent_workers"] = bool(
                validation_config.get("persistent_workers", True)
            )
            val_loader_kwargs["multiprocessing_context"] = "spawn"
        val_sampler = RankShardSampler(
            len(val_dataset),
            accelerator.process_index,
            accelerator.num_processes,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(validation_config.get("batch_size", 1)),
            sampler=val_sampler,
            num_workers=val_workers,
            pin_memory=bool(validation_config.get("pin_memory", True)),
            drop_last=False,
            collate_fn=identity_collate,
            **val_loader_kwargs,
        )
    model_config = config["model"]
    model = import_callable(model_config["factory"])(**model_config.get("kwargs", {}))
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return torch.nn.Module")
    teacher_config = config["teacher"]
    teacher = import_callable(teacher_config["factory"])(**teacher_config.get("kwargs", {}))
    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher factory must return torch.nn.Module")
    teacher.to(accelerator.device)
    teacher.eval()
    teacher.requires_grad_(False)
    if tokenizer_sha256(teacher.tokenizer) != tokenizer_sha256(model.tokenizer):
        raise ValueError("online teacher and student token-to-ID mappings differ")
    teacher_processor = helper.get_processor(
        teacher.tokenizer,
        getattr(teacher.config, "vlm_name_or_path", None),
    )
    student_processor = helper.get_processor(
        model.tokenizer,
        getattr(model.config, "vlm_name_or_path", None),
    )
    on_policy_config = OnPolicyDistillConfig(**config.get("on_policy", {}))
    global_print(on_policy_config.describe())
    objective = DistillationLoss(
        DistillationLossConfig(**config.get("loss", {})),
        on_policy_config,
    )
    module = DistillTrainModule(model, objective)

    epochs = int(training.get("epochs", 1))
    accumulation = accelerator.gradient_accumulation_steps
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    max_steps = int(training.get("max_steps") or (epochs * updates_per_epoch))
    if training.get("max_steps") is not None:
        epochs = max(epochs, math.ceil(max_steps / updates_per_epoch))

    global_step = 0
    start_epoch = 0
    resume_sample_offset = 0
    resume = training.get("resume_from")
    checkpoint = (
        _latest_checkpoint(output_dir) if resume == "latest" else (Path(resume) if resume else None)
    )
    if checkpoint is not None:
        global_print(f"Resuming from checkpoint: {checkpoint}")
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"checkpoint is missing trainer state: {state_path}")
        expected_state = make_trainer_state(
            global_step=0,
            epoch=0,
            sample_offset=0,
            seed=train_seed,
            world_size=accelerator.num_processes,
            dataset_length=state_dataset_length,
            per_rank_batch_size=train_batch_size,
            gradient_accumulation_steps=accumulation,
            sampler_epoch=0,
        )
        state = load_trainer_state(state_path, expected=expected_state)
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        resume_sample_offset = state["sample_offset"]
        if resume_sample_offset % train_batch_size:
            raise ValueError("trainer state sample_offset is not aligned to per-rank batch size")
        if resume_sample_offset > sampler.num_samples:
            raise ValueError("trainer state sample_offset exceeds this rank's epoch shard")
    else:
        global_print(f"Starting fresh run in: {output_dir}")

    if global_step >= max_steps:
        global_print(
            f"Checkpoint is already at global_step={global_step} "
            f"(max_steps={max_steps}); nothing to train"
        )
        accelerator.wait_for_everyone()
        return

    optimizer_config = config.get("optimizer", {})
    base_lr = float(optimizer_config.get("lr", 1e-5))
    vlm_lr = base_lr * float(optimizer_config.get("vlm_lr_scale", 1.0))
    if base_lr <= 0 or vlm_lr <= 0:
        raise ValueError("optimizer learning rates must be positive")
    use_zero1 = bool(optimizer_config.get("use_zero1", False)) and accelerator.num_processes > 1
    if use_zero1:
        dtype_name = str(model_config.get("kwargs", {}).get("dtype", "bfloat16"))
        target_dtype = getattr(torch, dtype_name, None)
        if not isinstance(target_dtype, torch.dtype):
            raise ValueError(f"{dtype_name} is not a torch dtype")
        cast_count = _unify_trainable_param_dtype(module, target_dtype)
        global_print(f"[zero1] cast {cast_count} trainable parameter tensors to {target_dtype}")
        # Match the reference trainer: construct ZeRO only after its parameters are
        # on CUDA. ZeroRedundancyOptimizer caches this as its collective device;
        # constructing it on CPU makes state consolidation try a CPU broadcast on
        # the NCCL-only process group.
        module.to(accelerator.device)
        if checkpoint is not None:
            # accelerator.save_state cannot faithfully serialize parameters after
            # ZeroRedundancyOptimizer packs them into bucket views. The separately
            # saved Hugging Face student checkpoint is complete, so load it before
            # constructing ZeRO or DDP.
            from safetensors.torch import load_model

            student_weights = checkpoint / "student" / "model.safetensors"
            if not student_weights.is_file():
                raise FileNotFoundError(
                    f"checkpoint is missing complete student weights: {student_weights}"
                )
            load_model(
                model,
                student_weights,
                device=str(accelerator.device),
                strict=True,
            )
            objective_path = checkpoint / "objective.pt"
            if objective_path.is_file():
                objective.load_state_dict(
                    torch.load(objective_path, map_location="cpu", weights_only=True)
                )
            else:
                global_print(
                    "[resume] checkpoint has no objective.pt; reinitializing the "
                    "hidden-state projection"
                )
        module = accelerator.prepare(module)
        if checkpoint is not None:
            # Restore per-rank RNG state without asking Accelerate to load its
            # incomplete top-level model.safetensors.
            def skip_model_load(models: list[nn.Module], input_dir: str) -> None:
                del input_dir
                models.clear()

            hook = accelerator.register_load_state_pre_hook(skip_model_load)
            try:
                accelerator.load_state(checkpoint)
            finally:
                hook.remove()
    vlm_parameter_ids = {id(parameter) for parameter in model.vlm.parameters()}
    grouped: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        is_vlm = id(parameter) in vlm_parameter_ids
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or "norm" in name.lower()
        grouped[(is_vlm, no_decay)].append(parameter)
    parameter_groups = []
    for (is_vlm, no_decay), parameters in grouped.items():
        parameter_groups.append(
            {
                "params": parameters,
                "lr": vlm_lr if is_vlm else base_lr,
                "weight_decay": (
                    0.0
                    if no_decay
                    else float(
                        optimizer_config.get(
                            "vlm_weight_decay" if is_vlm else "weight_decay",
                            optimizer_config.get("weight_decay", 0.01),
                        )
                    )
                ),
                "is_vlm": is_vlm,
            }
        )
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer_kwargs = {
        "betas": tuple(optimizer_config.get("betas", [0.9, 0.95])),
        "eps": float(optimizer_config.get("eps", 1e-8)),
        "fused": True,
    }
    if use_zero1:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optimizer = ZeroRedundancyOptimizer(
            parameter_groups,
            optimizer_class=torch.optim.AdamW,
            parameters_as_bucket_view=True,
            **optimizer_kwargs,
        )
    else:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
    scheduler_config = config.get("scheduler", {})
    scheduler_name = str(scheduler_config.get("name", "cosine"))
    if scheduler_name != "cosine":
        raise ValueError("the distillation recipe currently supports scheduler.name=cosine")
    warmup_steps = int(
        scheduler_config.get("warmup_steps")
        or float(scheduler_config.get("warmup_ratio", 0.0)) * max_steps
    )
    minimum_lrs = [
        float(
            scheduler_config.get("vlm_min_lr", 0.0)
            if group["is_vlm"]
            else scheduler_config.get("min_lr", 0.0)
        )
        for group in parameter_groups
    ]

    def lr_lambda(initial_lr: float, minimum_lr: float):
        minimum_ratio = min(max(minimum_lr / initial_lr, 0.0), 1.0)

        def schedule(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(step, 1) / warmup_steps
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        return schedule

    scheduler = LambdaLR(
        optimizer,
        [
            lr_lambda(float(group["lr"]), minimum_lrs[index])
            for index, group in enumerate(parameter_groups)
        ],
    )
    if not use_zero1:
        module, optimizer, scheduler = accelerator.prepare(module, optimizer, scheduler)
        if checkpoint is not None:
            accelerator.load_state(checkpoint)
    if checkpoint is not None:
        if use_zero1:
            optimizer_path = checkpoint / "zero_optimizer.pt"
            scheduler_path = checkpoint / "zero_scheduler.pt"
            if not optimizer_path.is_file() or not scheduler_path.is_file():
                raise FileNotFoundError("ZeRO-1 checkpoint is missing optimizer state")
            optimizer.load_state_dict(
                torch.load(optimizer_path, map_location="cpu", weights_only=False)
            )
            _move_optimizer_state_to_device(optimizer, accelerator.device)
            scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=False)
            )

    save_every = int(training.get("save_every_steps", 5000))
    log_every = int(training.get("log_every_steps", 10))
    validate_every = int(validation_config.get("every_steps", 5000))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    last_saved_step = -1
    last_log_step = global_step
    log_window_start = time.perf_counter()
    loss_window_sums: dict[str, torch.Tensor] = {}
    loss_window_microbatches = 0
    lr_window_sum = 0.0
    vlm_lr_window_sum = 0.0
    grad_norm_window_sum = 0.0
    clipped_grad_norm_window_sum = 0.0
    optimizer_steps_in_window = 0
    skipped_on_policy_batches = 0
    skipped_teacher_batches = 0
    skipped_global_batches = 0
    skipped_batches_by_source = {
        "trajectory": 0,
        "lingoqa_vqa": 0,
        "pai_vqa": 0,
    }
    on_policy_fallback_batches = 0

    def save(epoch: int, sample_offset: int) -> None:
        nonlocal last_saved_step
        if global_step == last_saved_step:
            return
        checkpoint_dir = output_dir / f"checkpoint-{global_step:08d}"
        global_print(f"Saving checkpoint: step={global_step}, path={checkpoint_dir}")
        save_start = time.perf_counter()
        if use_zero1:
            optimizer.consolidate_state_dict(to=0)
            accelerator.wait_for_everyone()

            # ZeRO bucket views make Accelerate's safetensors model snapshot
            # incomplete. Save only its per-rank RNG state; the full student and
            # objective weights are written explicitly below.
            def skip_model_save(
                models: list[nn.Module],
                weights: list[dict[str, torch.Tensor]],
                output_dir: str,
            ) -> None:
                del models, output_dir
                weights.clear()

            hook = accelerator.register_save_state_pre_hook(skip_model_save)
            try:
                accelerator.save_state(checkpoint_dir)
            finally:
                hook.remove()
        else:
            accelerator.save_state(checkpoint_dir)
        if accelerator.is_main_process:
            if use_zero1:
                torch.save(optimizer.state_dict(), checkpoint_dir / "zero_optimizer.pt")
                torch.save(scheduler.state_dict(), checkpoint_dir / "zero_scheduler.pt")
            unwrapped = accelerator.unwrap_model(module)
            torch.save(
                {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in unwrapped.objective.state_dict().items()
                },
                checkpoint_dir / "objective.pt",
            )
            unwrapped.model.save_pretrained(
                checkpoint_dir / "student",
                safe_serialization=True,
            )
            atomic_write_json(
                checkpoint_dir / "trainer_state.json",
                make_trainer_state(
                    global_step=global_step,
                    epoch=epoch,
                    sample_offset=sample_offset,
                    seed=train_seed,
                    world_size=accelerator.num_processes,
                    dataset_length=state_dataset_length,
                    per_rank_batch_size=train_batch_size,
                    gradient_accumulation_steps=accumulation,
                    sampler_epoch=sampler.epoch,
                ),
            )
        accelerator.wait_for_everyone()
        last_saved_step = global_step
        global_print(
            f"Checkpoint saved: step={global_step}, path={checkpoint_dir}, "
            f"elapsed={time.perf_counter() - save_start:.1f}s"
        )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        epoch_sample_offset = resume_sample_offset if epoch == start_epoch else 0
        sampler.set_sample_offset(epoch_sample_offset)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        module.train()
        completed_sample_offset = epoch_sample_offset
        for batch_index, batch in enumerate(loader):
            task_type = str(batch[0].get("task_type", "trajectory"))
            if any(str(sample.get("task_type", "trajectory")) != task_type for sample in batch):
                raise RuntimeError("mixture sampler produced a heterogeneous task batch")
            is_vqa = task_type == "vqa"
            vqa_source = ""
            if is_vqa:
                vqa_sources = {str(sample.get("vqa_source", "lingoqa")) for sample in batch}
                if len(vqa_sources) != 1:
                    raise RuntimeError("mixture sampler produced mixed VQA-source batch")
                vqa_source = vqa_sources.pop()
                if vqa_source not in {"lingoqa", "pai"}:
                    raise ValueError(f"unknown VQA source: {vqa_source!r}")
            is_pai_vqa = is_vqa and vqa_source == "pai"
            is_lingoqa = is_vqa and vqa_source == "lingoqa"
            nav_available_fraction = (
                0.0
                if is_vqa
                else sum(bool(sample.get("nav_available", False)) for sample in batch)
                / len(batch)
            )
            nav_enabled_fraction = (
                0.0
                if is_vqa
                else sum(bool(sample.get("nav_enabled", False)) for sample in batch)
                / len(batch)
            )
            mixture_values: dict[str, float] = {}
            if not is_vqa and pai_dataset.mixture_enabled:
                requested_nav_fraction = sum(
                    bool(sample["mixture_requested_nav"]) for sample in batch
                ) / len(batch)
                actual_nav_fraction = sum(
                    bool(sample["mixture_actual_nav"]) for sample in batch
                ) / len(batch)
                mixture_values = {
                    "mixture_requested_basic": 1.0 - requested_nav_fraction,
                    "mixture_requested_nav": requested_nav_fraction,
                    "mixture_actual_basic": 1.0 - actual_nav_fraction,
                    "mixture_actual_nav": actual_nav_fraction,
                    "mixture_nav_t0_retries": sum(
                        int(sample["mixture_nav_t0_retries"]) for sample in batch
                    )
                    / len(batch),
                    "mixture_nav_clip_retries": sum(
                        int(sample["mixture_nav_clip_retries"]) for sample in batch
                    )
                    / len(batch),
                    "mixture_nav_fallback": sum(
                        bool(sample["mixture_nav_fallback"]) for sample in batch
                    )
                    / len(batch),
                }
            online_teacher = config.get("online_teacher", {})
            teacher_top_k = int(online_teacher.get("top_k", 64))
            step_on_policy = on_policy_config.is_on_policy_step(global_step)
            actual_on_policy = False
            try:
                compute_teacher_vector = (
                    float(config["loss"].get("teacher_vector_weight", 0.0)) != 0.0
                )
                if is_pai_vqa and step_on_policy:
                    try:
                        teacher_batch = build_lingoqa_on_policy_batch(
                            teacher,
                            batch,
                            student=model,
                            teacher_processor=teacher_processor,
                            student_processor=student_processor,
                            sample_temperature=on_policy_config.sample_temperature,
                            sample_top_p=on_policy_config.sample_top_p,
                            max_new_tokens=int(
                                pai_vqa_config.get("max_new_tokens", 256)
                            ),
                        )
                        actual_on_policy = True
                    except InvalidVQAGenerationError:
                        on_policy_fallback_batches += 1
                        teacher_batch = build_pai_vqa_teacher_batch(
                            teacher,
                            batch,
                            student=model,
                            teacher_processor=teacher_processor,
                            student_processor=student_processor,
                            top_p=float(pai_vqa_config.get("teacher_top_p", 0.98)),
                            temperature=float(
                                pai_vqa_config.get("teacher_temperature", 0.6)
                            ),
                            max_new_tokens=int(
                                pai_vqa_config.get("max_new_tokens", 256)
                            ),
                        )
                elif is_pai_vqa:
                    # PhysicalAI-AV has no human VQA label. Match the public
                    # Alpamayo VQA inference path. The teacher's generated
                    # answer and logits provide off-policy CE plus KD targets.
                    teacher_batch = build_pai_vqa_teacher_batch(
                        teacher,
                        batch,
                        student=model,
                        teacher_processor=teacher_processor,
                        student_processor=student_processor,
                        top_p=float(pai_vqa_config.get("teacher_top_p", 0.98)),
                        temperature=float(
                            pai_vqa_config.get("teacher_temperature", 0.6)
                        ),
                        max_new_tokens=int(
                            pai_vqa_config.get("max_new_tokens", 256)
                        ),
                    )
                elif is_lingoqa and step_on_policy:
                    try:
                        teacher_batch = build_lingoqa_on_policy_batch(
                            teacher,
                            batch,
                            student=model,
                            teacher_processor=teacher_processor,
                            student_processor=student_processor,
                            sample_temperature=on_policy_config.sample_temperature,
                            sample_top_p=on_policy_config.sample_top_p,
                            max_new_tokens=int(
                                lingoqa_config.get(
                                    "max_new_tokens",
                                    online_teacher.get("max_new_tokens", 256),
                                )
                            ),
                        )
                        actual_on_policy = True
                    except InvalidVQAGenerationError:
                        on_policy_fallback_batches += 1
                        teacher_batch = build_lingoqa_teacher_guided_gt_batch(
                            teacher,
                            batch,
                            student=model,
                            teacher_processor=teacher_processor,
                            student_processor=student_processor,
                        )
                elif is_lingoqa:
                    # Off-policy VQA is ordinary supervised fine-tuning: the
                    # frozen teacher is deliberately not invoked.
                    teacher_batch = build_lingoqa_gt_batch(
                        model,
                        batch,
                        processor=student_processor,
                    )
                elif step_on_policy:
                    try:
                        teacher_batch = build_on_policy_batch(
                            teacher,
                            batch,
                            student_model=model,
                            teacher_processor=teacher_processor,
                            student_processor=student_processor,
                            sample_temperature=on_policy_config.sample_temperature,
                            sample_top_p=on_policy_config.sample_top_p,
                            divergence_temperature=on_policy_config.temperature,
                            max_new_tokens=int(online_teacher.get("max_new_tokens", 256)),
                            compute_teacher_vector=compute_teacher_vector,
                        )
                        actual_on_policy = True
                    except InvalidStudentGenerationError:
                        on_policy_fallback_batches += 1
                        if (
                            on_policy_fallback_batches in (1, 10, 100)
                            or on_policy_fallback_batches % 1000 == 0
                        ):
                            global_print(
                                "[on_policy] student rollout was invalid; "
                                "falling back to the online teacher rollout "
                                f"(local fallbacks={on_policy_fallback_batches})"
                            )
                        teacher_batch = build_online_teacher_batch(
                            teacher,
                            batch,
                            student_model=model,
                            teacher_processor=teacher_processor,
                            student_processor=student_processor,
                            top_k=teacher_top_k,
                            temperature=float(config["loss"].get("temperature", 0.5)),
                            max_new_tokens=int(online_teacher.get("max_new_tokens", 256)),
                            compute_teacher_vector=compute_teacher_vector,
                        )
                else:
                    teacher_batch = build_online_teacher_batch(
                        teacher,
                        batch,
                        student_model=model,
                        teacher_processor=teacher_processor,
                        student_processor=student_processor,
                        top_k=teacher_top_k,
                        temperature=float(config["loss"].get("temperature", 0.5)),
                        max_new_tokens=int(online_teacher.get("max_new_tokens", 256)),
                        compute_teacher_vector=compute_teacher_vector,
                    )
                local_teacher_valid = True
            except (InvalidTeacherGenerationError, InvalidVQAGenerationError):
                teacher_batch = None
                local_teacher_valid = False

            valid_tensor = torch.tensor(
                int(local_teacher_valid),
                dtype=torch.int32,
                device=accelerator.device,
            )
            if accelerator.num_processes > 1:
                dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
            completed_sample_offset = epoch_sample_offset + (batch_index + 1) * train_batch_size
            if not bool(valid_tensor.item()):
                skipped_global_batches += 1
                if is_pai_vqa:
                    skipped_source = "pai_vqa"
                elif is_lingoqa:
                    skipped_source = "lingoqa_vqa"
                else:
                    skipped_source = "trajectory"
                skipped_batches_by_source[skipped_source] += 1
                if step_on_policy:
                    skipped_on_policy_batches += 1
                else:
                    skipped_teacher_batches += 1
                skipped_vqa_batches = (
                    skipped_batches_by_source["lingoqa_vqa"]
                    + skipped_batches_by_source["pai_vqa"]
                )
                global_print(
                    f"[{'on_policy' if step_on_policy else 'online_teacher'}] "
                    "skipped one global batch because at least one rank produced "
                    f"an invalid rollout (source={skipped_source}, "
                    f"total_skipped={skipped_global_batches}, "
                    f"trajectory_skipped={skipped_batches_by_source['trajectory']}, "
                    f"vqa_skipped={skipped_vqa_batches}, "
                    f"lingoqa_vqa_skipped={skipped_batches_by_source['lingoqa_vqa']}, "
                    f"pai_vqa_skipped={skipped_batches_by_source['pai_vqa']}, "
                    f"online_teacher_skipped={skipped_teacher_batches}, "
                    f"on_policy_skipped={skipped_on_policy_batches})"
                )
                del teacher_batch
                continue
            if teacher_batch is None:
                raise RuntimeError("teacher batch validity synchronization failed")
            batch = teacher_batch
            with accelerator.accumulate(module):
                losses = module(batch)
                accelerator.backward(losses["loss"])
                grad_norm = None
                if accelerator.sync_gradients and max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(
                        module.parameters(),
                        max_grad_norm,
                    )
                if not use_zero1 or accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            logged_values = dict(losses)
            zero_metric = losses["loss"].detach() * 0.0
            for required_name in ("ce_loss", "kd_loss", "hidden_loss", "flow_loss"):
                logged_values.setdefault(required_name, zero_metric)
            logged_values["on_policy_active"] = torch.tensor(
                float(actual_on_policy),
                device=accelerator.device,
            )
            logged_values["nav_available"] = torch.tensor(
                nav_available_fraction,
                device=accelerator.device,
            )
            logged_values["nav_enabled"] = torch.tensor(
                nav_enabled_fraction,
                device=accelerator.device,
            )
            for name, value in mixture_values.items():
                logged_values[name] = torch.tensor(
                    value,
                    device=accelerator.device,
                )
            logged_values["lingoqa_active"] = torch.tensor(
                float(is_lingoqa),
                device=accelerator.device,
            )
            logged_values["pai_vqa_active"] = torch.tensor(
                float(is_pai_vqa),
                device=accelerator.device,
            )
            residual_mass = batch.get("teacher_residual_mass") if teacher_top_k != -1 else None
            if isinstance(residual_mass, torch.Tensor):
                topk_mass = (1.0 - residual_mass.float()).clamp(0.0, 1.0)
                loss_mask = batch.get("loss_mask")
                if isinstance(loss_mask, torch.Tensor):
                    mask = loss_mask.to(device=topk_mass.device, dtype=torch.bool)
                    if bool(config["loss"].get("kd_include_prompt", False)):
                        prompt_mask = torch.as_tensor(
                            batch["prompt_mask"],
                            device=topk_mass.device,
                            dtype=torch.bool,
                        )
                        prompt_prediction_mask = torch.zeros_like(prompt_mask)
                        prompt_prediction_mask[..., :-1] = prompt_mask[..., 1:]
                        mask = mask | prompt_prediction_mask
                    mask = mask.to(dtype=topk_mass.dtype)
                    topk_mass = (topk_mass * mask).sum() / mask.sum().clamp_min(1.0)
                else:
                    topk_mass = topk_mass.mean()
                logged_values["teacher_topk_mass"] = topk_mass
            for name, value in logged_values.items():
                detached = value.detach().float()
                if name not in loss_window_sums:
                    loss_window_sums[name] = detached.clone()
                else:
                    loss_window_sums[name].add_(detached)
            loss_window_microbatches += 1
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            current_lr = next(
                float(group["lr"]) for group in optimizer.param_groups if not group["is_vlm"]
            )
            current_vlm_lr = next(
                float(group["lr"]) for group in optimizer.param_groups if group["is_vlm"]
            )
            lr_window_sum += current_lr
            vlm_lr_window_sum += current_vlm_lr
            if grad_norm is not None:
                grad_norm_value = float(grad_norm)
                grad_norm_window_sum += grad_norm_value
                clipped_grad_norm_window_sum += min(
                    grad_norm_value,
                    max_grad_norm,
                )
            optimizer_steps_in_window += 1
            if global_step % log_every == 0:
                loss_names = sorted(loss_window_sums)
                packed_losses = torch.stack(
                    [loss_window_sums[name] for name in loss_names]
                    + [
                        torch.tensor(
                            float(loss_window_microbatches),
                            device=accelerator.device,
                        )
                    ]
                )
                packed_losses = accelerator.reduce(packed_losses, reduction="sum")
                loss_count = float(packed_losses[-1].item())
                metrics = {
                    name: float(packed_losses[index].item() / loss_count)
                    for index, name in enumerate(loss_names)
                }
                elapsed_steps = max(1, global_step - last_log_step)
                secs_per_step = (time.perf_counter() - log_window_start) / elapsed_steps
                global_samples_in_window = (
                    train_batch_size
                    * accelerator.num_processes
                    * accumulation
                    * optimizer_steps_in_window
                )
                average_lr = lr_window_sum / optimizer_steps_in_window
                average_vlm_lr = vlm_lr_window_sum / optimizer_steps_in_window
                average_grad_norm = (
                    grad_norm_window_sum / optimizer_steps_in_window if max_grad_norm > 0 else None
                )
                average_clipped_grad_norm = (
                    clipped_grad_norm_window_sum / optimizer_steps_in_window
                    if max_grad_norm > 0
                    else None
                )
                fields = [
                    f"total_loss={metrics['loss']:.2f}",
                    f"ce_loss={metrics['ce_loss']:.2f}",
                    f"kd_loss={metrics['kd_loss']:.2f}",
                    f"hidden_loss={metrics['hidden_loss']:.5f}",
                    f"flow_loss={metrics['flow_loss']:.2f}",
                ]
                if "teacher_vector_loss" in metrics:
                    fields.append(f"teacher_vector_loss={metrics['teacher_vector_loss']:.2f}")
                if "teacher_topk_mass" in metrics:
                    fields.append(f"teacher_topk_mass={metrics['teacher_topk_mass']:.4f}")
                if "future_traj_vlm_loss" in metrics:
                    fields.append(f"future_traj_vlm_loss={metrics['future_traj_vlm_loss']:.2f}")
                if "on_policy_kl_loss" in metrics:
                    fields.append(f"on_policy_kl_loss={metrics['on_policy_kl_loss']:.2f}")
                fields.append(f"on_policy_active={metrics['on_policy_active']:.3f}")
                fields.append(f"lingoqa_active={metrics['lingoqa_active']:.3f}")
                fields.append(f"pai_vqa_active={metrics['pai_vqa_active']:.3f}")
                fields.extend(
                    [
                        f"nav_available={metrics['nav_available']:.3f}",
                        f"nav_enabled={metrics['nav_enabled']:.3f}",
                    ]
                )
                if "mixture_requested_nav" in metrics:
                    fields.extend(
                        [
                            f"mixture_requested_basic={metrics['mixture_requested_basic']:.3f}",
                            f"mixture_requested_nav={metrics['mixture_requested_nav']:.3f}",
                            f"mixture_actual_basic={metrics['mixture_actual_basic']:.3f}",
                            f"mixture_actual_nav={metrics['mixture_actual_nav']:.3f}",
                            f"mixture_nav_t0_retries={metrics['mixture_nav_t0_retries']:.3f}",
                            f"mixture_nav_clip_retries={metrics['mixture_nav_clip_retries']:.3f}",
                            f"mixture_nav_fallback={metrics['mixture_nav_fallback']:.3f}",
                        ]
                    )
                fields.extend(
                    [
                        f"vlm_lr={average_vlm_lr:.1e}",
                        f"lr={average_lr:.1e}",
                    ]
                )
                if average_grad_norm is not None and average_clipped_grad_norm is not None:
                    fields.append(f"grad_norm_before_clip={average_grad_norm:.2f}")
                    fields.append(f"grad_norm_after_clip={average_clipped_grad_norm:.2f}")
                fields.extend(
                    [
                        f"global_samples_in_window={global_samples_in_window}",
                        f"secs_per_step={secs_per_step:.1f}",
                    ]
                )
                global_print(f"[epoch={epoch + 1}, step={global_step}] " + ", ".join(fields))
                last_log_step = global_step
                log_window_start = time.perf_counter()
                loss_window_sums.clear()
                loss_window_microbatches = 0
                lr_window_sum = 0.0
                vlm_lr_window_sum = 0.0
                grad_norm_window_sum = 0.0
                clipped_grad_norm_window_sum = 0.0
                optimizer_steps_in_window = 0
            if val_loader is not None and validate_every > 0 and global_step % validate_every == 0:
                metrics = run_validation(
                    module=accelerator.unwrap_model(module),
                    teacher=teacher,
                    loader=val_loader,
                    accelerator=accelerator,
                    teacher_processor=teacher_processor,
                    student_processor=student_processor,
                    online_teacher_config=config.get("online_teacher", {}),
                    loss_temperature=float(config["loss"].get("temperature", 0.5)),
                    compute_minade=bool(validation_config.get("compute_minade", True)),
                    stochastic_samples=int(validation_config.get("stochastic_samples", 6)),
                    top_p=float(validation_config.get("top_p", 0.98)),
                    sampling_temperature=float(validation_config.get("temperature", 0.6)),
                    max_generation_length=int(validation_config.get("max_generation_length", 256)),
                    inference_seed=int(validation_config.get("seed", 42)),
                )
                validation_fields = [
                    f"val_samples={int(metrics['val/samples'])}",
                    f"val_seed={int(validation_config.get('seed', 42))}",
                ]
                stochastic_samples = int(validation_config.get("stochastic_samples", 6))
                minade_key = f"val/minADE@{stochastic_samples}"
                if minade_key in metrics:
                    validation_fields.extend(
                        [
                            f"val_basic_minADE@{stochastic_samples}={metrics[minade_key]:.4f}",
                            f"val_basic_greedy_minADE={metrics['val/greedy_minADE@1']:.4f}",
                        ]
                    )
                nav_minade_key = f"val/nav_minADE@{stochastic_samples}"
                if nav_minade_key in metrics:
                    nav_subset_basic_key = f"val/nav_subset_basic_minADE@{stochastic_samples}"
                    validation_fields.extend(
                        [
                            f"val_nav_samples={int(metrics['val/nav_samples'])}",
                            f"val_nav_subset_basic_minADE@{stochastic_samples}="
                            f"{metrics[nav_subset_basic_key]:.4f}",
                            f"val_nav_minADE@{stochastic_samples}={metrics[nav_minade_key]:.4f}",
                            "val_nav_subset_basic_greedy_minADE="
                            f"{metrics['val/nav_subset_basic_greedy_minADE@1']:.4f}",
                            f"val_nav_greedy_minADE={metrics['val/nav_greedy_minADE@1']:.4f}",
                        ]
                    )
                validation_fields.extend(
                    [
                        f"val_total_loss={metrics['val/loss']:.2f}",
                        f"val_ce_loss={metrics['val/ce_loss']:.2f}",
                        f"val_kd_loss={metrics['val/kd_loss']:.2f}",
                        f"val_flow_loss={metrics['val/flow_loss']:.2f}",
                    ]
                )
                global_print(", ".join(validation_fields))
            if save_every > 0 and global_step % save_every == 0:
                save(epoch, completed_sample_offset)
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            save(epoch, completed_sample_offset)
        resume_sample_offset = 0
        if global_step >= max_steps:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. optimizer.lr=2e-5"
    )
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
