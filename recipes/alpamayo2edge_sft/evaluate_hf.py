# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Alpamayo 2 Edge SFT checkpoints."""

import hashlib
import json
import logging
import re
from copy import deepcopy
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm
from transformers import set_seed
from torch.utils.data import Subset

from alpamayo2edge_sft.evaluation_identity import validation_data_identity
from alpamayo2edge_sft.run_utils import (
    peak_cuda_memory_metrics,
    reset_peak_cuda_memory,
    save_resolved_config,
    tracking_run,
)
from alpamayo2edge_sft.train_hf import training_arguments
from alpamayo2edge_sft.trainer import Alpamayo2EdgeTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_min_ade(pred_xyz: torch.Tensor, target_xyz: torch.Tensor) -> torch.Tensor:
    """Compute per-sample minADE pooled over every trajectory in every set."""
    if (
        pred_xyz.ndim != 5
        or target_xyz.ndim != 3
        or pred_xyz.shape[-1] != 3
        or target_xyz.shape[-1] != 3
        or pred_xyz.shape[0] != target_xyz.shape[0]
        or pred_xyz.shape[-2] != target_xyz.shape[-2]
        or any(size == 0 for size in pred_xyz.shape)
    ):
        raise ValueError("Expected predictions [B, S, K, T, 3] and targets [B, T, 3], nonempty")
    if not torch.isfinite(pred_xyz).all() or not torch.isfinite(target_xyz).all():
        raise ValueError("Predicted and target trajectories must contain only finite values")
    distances = torch.linalg.vector_norm(
        pred_xyz[..., :2].float() - target_xyz[:, None, None, :, :2].float(),
        dim=-1,
    )
    ade = distances.mean(dim=-1)
    return ade.flatten(start_dim=1).min(dim=1).values


def validate_max_eval_steps(value: Any) -> int | None:
    """Return an optional batch limit, shared by loss and trajectory evaluation."""
    if value is None or (type(value) is int and value == -1):
        return None
    if type(value) is not int or value < 1:
        raise ValueError("evaluate.max_eval_steps must be -1, null, or a positive integer")
    return value


def configure_evaluation_output(config: DictConfig) -> None:
    """Give each invocation its own checkpoint-labelled directory."""
    checkpoint = str(config.evaluate.eval_ckpt or config.model.pretrained_model_name_or_path)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", checkpoint.rstrip("/").rsplit("/", 1)[-1])
    digest = hashlib.sha256(checkpoint.encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(config.paths.output_dir) / f"{name}-{digest}" / f"{stamp}-{uuid4().hex[:8]}"
    if torch.distributed.is_initialized():
        shared = [str(output) if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(shared, src=0)
        output = Path(shared[0])
    config.paths.output_dir = str(output)
    config.trainer.output_dir = str(output)
    logger.info("Evaluation output directory: %s", output)


def build_trainer(config: DictConfig) -> Alpamayo2EdgeTrainer:
    """Instantiate the checkpoint, validation dataset, and trainer."""
    max_steps = validate_max_eval_steps(config.evaluate.max_eval_steps)
    if config.evaluate.get("mode", "loss") == "trajectory":
        if config.trainer.deepspeed is not None:
            raise ValueError("Trajectory evaluation requires trainer.deepspeed=null")
        preprocess = config.data.val_dataset.preprocess
        order = list(preprocess.components_order)
        boundary = "prompt" if "prompt" in order else "question"
        if boundary not in order:
            raise ValueError("Trajectory evaluation requires a prompt or question component")
        assistant_components = order[order.index(boundary) + 1 :]
        generate_cot = config.evaluate.get("generate_cot", False)
        if not isinstance(generate_cot, bool):
            raise ValueError("evaluate.generate_cot must be a boolean")
        expected_components = ["cot", "traj_future"] if generate_cot else ["traj_future"]
        if not preprocess.generation_mode or assistant_components != expected_components:
            raise ValueError(
                "Trajectory evaluation requires generation_mode=true and assistant components "
                f"{expected_components} for evaluate.generate_cot={generate_cot}"
            )
    if config.evaluate.eval_ckpt is not None:
        config.model.pretrained_model_name_or_path = config.evaluate.eval_ckpt
        if "stage1_vlm_checkpoint_path" in config.model:
            config.model.stage1_vlm_checkpoint_path = None

    # An SFT export owns its objective. Defaults in the recipe must not replace
    # the weights used during training when evaluating that export.
    model_config = deepcopy(config.model)
    if config.evaluate.eval_ckpt is not None:
        with open_dict(model_config):
            for key in ("loss_weights", "lm_token_loss_weights"):
                model_config.pop(key, None)
    # Register ZeRO-3 before from_pretrained allocates model parameters. The
    # initialized process group also lets loss-eval ranks share one run folder.
    training_args = training_arguments(config)
    configure_evaluation_output(config)
    training_args.output_dir = config.paths.output_dir
    model = instantiate(model_config, _convert_="partial")
    input_identity = validation_data_identity(config.data.val_dataset)
    eval_dataset = instantiate(
        config.data.val_dataset,
        _partial_=True,
        _convert_="partial",
    )(model_config=model.config)
    if validation_data_identity(config.data.val_dataset) != input_identity:
        raise ValueError("Validation data identity changed while loading the dataset")
    if config.evaluate.mode == "loss" and max_steps is not None:
        sample_limit = max_steps * training_args.eval_batch_size * training_args.world_size
        eval_dataset = Subset(eval_dataset, range(min(sample_limit, len(eval_dataset))))
    collator = instantiate(
        config.data.collator,
        _partial_=True,
        _convert_="partial",
    )(model_config=model.config)
    trainer = Alpamayo2EdgeTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=collator.data_processor.processor,
    )
    trainer._evaluation_data_identity = input_identity
    return trainer


def evaluate_trajectory(
    trainer: Alpamayo2EdgeTrainer,
    config: DictConfig,
) -> dict[str, float]:
    """Generate expert trajectories, preserving sample scores and failure coverage.

    If CoT generation misses the trajectory delimiter, the entire affected batch
    is marked unusable, including any earlier candidate chunks. Its exclusion is
    explicit: ``eval_min_ade`` is published only at 100% generation coverage.
    """
    accelerator = trainer.accelerator
    if accelerator.num_processes != 1:
        raise ValueError("Trajectory generation requires one process; use trainer.deepspeed=null")
    for key in ("num_traj_samples", "num_samples_per_forward", "num_traj_sets"):
        value = getattr(config.evaluate, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"evaluate.{key} must be a positive integer, got {value!r}")
    max_steps = validate_max_eval_steps(config.evaluate.max_eval_steps)
    model = accelerator.prepare_model(trainer.model, evaluation_mode=True)
    generation_model = accelerator.unwrap_model(model)
    generation_model.eval()
    dataloader = trainer.get_eval_dataloader()
    total = len(dataloader) if max_steps is None else min(max_steps, len(dataloader))
    if max_steps is not None:
        dataloader = islice(dataloader, max_steps)
    dtype = getattr(torch, config.evaluate.autocast_dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Invalid evaluate.autocast_dtype={config.evaluate.autocast_dtype!r}")
    generate_cot = getattr(config.evaluate, "generate_cot", False)
    sampling_kwargs = {"generate_cot": True} if generate_cot else {}
    if generate_cot:
        sampling_kwargs.update(
            top_p=config.evaluate.top_p,
            temperature=config.evaluate.temperature,
            max_generation_length=config.evaluate.max_generation_length,
        )
    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_sum, metric_count, attempted, failed, batches = 0.0, 0, 0, 0, 0

    def metrics():
        result = {
            "eval_samples": metric_count,
            "eval_attempted_samples": attempted,
            "eval_failed_samples": failed,
            "eval_generation_coverage": metric_count / attempted if attempted else 0.0,
            "eval_batches": batches,
        }
        if metric_count:
            result["eval_min_ade_successful"] = metric_sum / metric_count
            if failed == 0 and metric_count == attempted:
                result["eval_min_ade"] = metric_sum / metric_count
        return result

    def persist(status, error=None):
        current = metrics()
        trainer._trajectory_metrics = current
        summary = {"status": status, **current}
        if error is not None:
            summary["error_type"] = type(error).__name__
            summary["error"] = str(error)
        temporary = output_dir / "trajectory_summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output_dir / "trajectory_summary.json")

    # Flush every batch so interrupted runs retain all completed event scores.
    with (output_dir / "trajectory_samples.jsonl").open("w") as stream:
        try:
            for batch_index, batch in enumerate(tqdm(
                dataloader, total=total, disable=not accelerator.is_main_process,
            )):
                metadata = batch.pop("sample_metadata", None)
                # HF's accelerated loader normally places tensors; explicitly
                # prepare them as well for custom loaders and dtype consistency.
                batch = trainer._prepare_input(batch)
                target = batch["traj_data"]["ego_future_xyz"][:, -1]
                size = target.shape[0]
                rows = metadata or [{} for _ in range(size)]
                if len(rows) != size:
                    raise ValueError("Sample metadata and trajectory batch sizes differ")
                records = [
                    {**row, "sample_index": attempted + offset, "batch_index": batch_index}
                    for offset, row in enumerate(rows)
                ]
                attempted += size
                generation_data: dict[str, Any] = {
                    "tokenized_data": batch["tokenized_data"],
                    **{key: value for key, value in batch["traj_data"].items()
                       if key.startswith("ego_history_")},
                }
                chunks = []
                remaining = config.evaluate.num_traj_samples
                try:
                    with torch.inference_mode(), torch.autocast(
                        target.device.type, dtype=dtype,
                        enabled=target.device.type == "cuda",
                    ):
                        while remaining > 0:
                            count = min(remaining, config.evaluate.num_samples_per_forward)
                            pred_xyz, _, _ = generation_model.sample_trajectories_from_data(
                                data=generation_data, num_traj_samples=count,
                                num_traj_sets=config.evaluate.num_traj_sets, **sampling_kwargs,
                            )
                            expected = (size, config.evaluate.num_traj_sets, count, target.shape[1], 3)
                            if pred_xyz.shape != expected:
                                raise ValueError(f"Sampler returned {tuple(pred_xyz.shape)}; expected {expected}")
                            chunks.append(pred_xyz)
                            remaining -= count
                    scores = compute_min_ade(torch.cat(chunks, dim=2), target).cpu().tolist()
                except ValueError as error:
                    # This exact public-package error is an expected generation
                    # failure. Do not swallow invalid shapes, NaNs, CUDA errors,
                    # tokenizer defects or other programming/data errors.
                    if not generate_cot or not str(error).startswith("No <traj_future_start> token found"):
                        raise
                    failed += size
                    for record in records:
                        record.update(status="batch_missing_trajectory_anchor", min_ade=None, error=str(error))
                    logger.warning("Skipping %d samples in batch %d: %s", size, batch_index, error)
                else:
                    metric_sum += sum(scores)
                    metric_count += len(scores)
                    for record, score in zip(records, scores, strict=True):
                        record.update(status="ok", min_ade=score)
                batches += 1
                for record in records:
                    stream.write(json.dumps(record) + "\n")
                stream.flush()
                persist("in_progress")
            if not attempted:
                raise ValueError("Trajectory evaluation produced no samples; check the validation split")
        except BaseException as error:
            persist("failed", error)
            raise
    persist("completed_with_generation_failures" if failed else "completed")
    result = metrics()
    trainer.log(result)
    logger.info("Trajectory evaluation: %s", result)
    return result


def save_evaluation_results(trainer, config: DictConfig, metrics: dict[str, float]) -> None:
    """Persist scores and the configuration needed to interpret or reproduce them."""
    # Save machine-readable results as well as console/W&B logs, like Trainer's
    # standard evaluation workflows. This also records the exact sampling protocol.
    trainer.save_metrics("eval", metrics)
    if trainer.is_world_process_zero():
        output_dir = Path(trainer.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(config, output_dir / "config.yaml")
        input_identity = validation_data_identity(config.data.val_dataset)
        if input_identity != getattr(trainer, "_evaluation_data_identity", input_identity):
            raise ValueError("Validation data identity changed after the dataset was loaded")
        validation_chunks = config.data.val_dataset.get("chunk_ids")
        protocol = {
            "schema_version": 1,
            "validation_data_identity": input_identity,
            "checkpoint": str(config.model.pretrained_model_name_or_path),
            "seed": config.seed,
            "mode": config.evaluate.mode,
            "max_eval_steps": validate_max_eval_steps(config.evaluate.max_eval_steps),
            "max_eval_steps_unit": "dataloader batches",
            "loss_weights": getattr(trainer.model.config, "loss_weights", None),
            "lm_token_loss_weights": getattr(trainer.model.config, "lm_token_loss_weights", None),
            "validation_chunks": OmegaConf.to_container(validation_chunks)
            if OmegaConf.is_config(validation_chunks) else validation_chunks,
        }
        dataset = getattr(trainer, "eval_dataset", None)
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        sampling_protocol = getattr(dataset, "evaluation_sampling_protocol", None)
        if callable(sampling_protocol):
            protocol["dataset_sampling_protocol"] = sampling_protocol()
        manifest_path = config.data.val_dataset.get("split_manifest")
        if manifest_path is not None:
            manifest_sha256 = hashlib.sha256(Path(manifest_path).expanduser().read_bytes()).hexdigest()
            loaded_sha256 = getattr(dataset, "split_manifest_sha256", manifest_sha256)
            if loaded_sha256 != manifest_sha256:
                raise ValueError("Validation split manifest changed after the dataset was loaded")
            # A path alone cannot distinguish edits made to the manifest in place.
            protocol["validation_split_manifest_sha256"] = manifest_sha256
        if config.evaluate.mode == "trajectory":
            preprocess = config.data.val_dataset.preprocess
            generate_cot = config.evaluate.get("generate_cot", False)
            navigation_input = "nav_instruction" in preprocess.components_order
            conditioning = (
                "model-generated CoT through future-start; no reference CoT"
                if generate_cot
                else "trajectory-only prompt through future-start; no generated CoT"
            )
            if navigation_input:
                conditioning = "navigation instruction; " + conditioning
            protocol.update(
                {
                    # Preprocessing alone does not identify the model inputs:
                    # camera/frame selection and image budgets live on the
                    # dataset, and token truncation/padding on the collator.
                    "validation_dataset": OmegaConf.to_container(
                        config.data.val_dataset, resolve=True, throw_on_missing=True,
                    ),
                    "collator": OmegaConf.to_container(
                        config.data.collator, resolve=True, throw_on_missing=True,
                    ),
                    "autocast_dtype": config.evaluate.autocast_dtype,
                    "per_device_eval_batch_size": config.trainer.per_device_eval_batch_size,
                    "num_traj_samples": config.evaluate.num_traj_samples,
                    "num_traj_sets": config.evaluate.num_traj_sets,
                    "num_samples_per_forward": config.evaluate.num_samples_per_forward,
                    "candidate_count": (
                        config.evaluate.num_traj_samples * config.evaluate.num_traj_sets
                    ),
                    "metric": "minADE: mean XY distance in meters over the full future horizon, "
                    "minimum across all candidates per sample, then mean across samples",
                    "conditioning": conditioning,
                    "generate_cot": generate_cot,
                    "navigation_input": navigation_input,
                    "prompt_format": "nav_cot" if navigation_input and generate_cot else (
                        "nav" if navigation_input else "cot" if generate_cot else "trajectory"
                    ),
                    "native_release_prompt": bool(navigation_input and generate_cot),
                    "log_probability": "unused; sampler returns a zero placeholder, not likelihoods",
                    "per_sample_results": "trajectory_samples.jsonl",
                    "generation_failure_policy": "exclude whole affected batch; report coverage and conditional minADE; omit eval_min_ade if any fail",
                    **({"top_p": config.evaluate.top_p,
                        "temperature": config.evaluate.temperature,
                        "max_generation_length": config.evaluate.max_generation_length}
                       if generate_cot else {}),
                    "preprocess": OmegaConf.to_container(preprocess, resolve=True),
                    "annotations_path": config.data.val_dataset.get("annotations_path"),
                }
            )
            if hasattr(trainer.model, "trajectory_conditioning"):
                action_config = trainer.model.config.expert_config.action_space_cfg
                protocol.update(
                    {
                        "future_steps": action_config["n_waypoints"],
                        "time_step_seconds": action_config["dt"],
                        "diffusion_steps": trainer.model.expert.diffusion.num_inference_steps,
                        "integration_method": trainer.model.expert.diffusion.int_method,
                    }
                )
        (output_dir / "evaluation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


@hydra.main(version_base=None, config_path=None, config_name="config")
def evaluate(config: DictConfig) -> None:
    """Run validation loss or expert-trajectory evaluation."""
    trainer = build_trainer(config)
    # Trainer initialization seeds its own defaults. Apply the requested evaluation
    # seed afterwards so diffusion sampling matches the recorded protocol.
    set_seed(config.seed)
    try:
        with tracking_run(config, trainer):
            reset_peak_cuda_memory()
            if config.evaluate.mode == "loss":
                metrics = trainer.evaluate()
                if trainer.is_world_process_zero():
                    logger.info("Evaluation loss: %.6f", metrics["eval_loss"])
            elif config.evaluate.mode == "trajectory":
                metrics = evaluate_trajectory(trainer, config)
            else:
                raise ValueError(
                    f"evaluate.mode must be 'loss' or 'trajectory', got {config.evaluate.mode!r}"
                )
            memory_metrics = peak_cuda_memory_metrics("eval")
            if memory_metrics:
                trainer.log(memory_metrics)
            metrics.update(memory_metrics)
            save_evaluation_results(trainer, config, metrics)
    except BaseException:
        # Persist aggregate scores and protocol even when an unexpected error
        # stops generation. The JSONL already contains completed event results.
        if hasattr(trainer, "_trajectory_metrics"):
            save_evaluation_results(trainer, config, trainer._trajectory_metrics)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    evaluate()
