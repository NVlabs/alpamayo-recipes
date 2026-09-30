# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate Alpamayo 2 Super SFT checkpoints."""

from itertools import islice
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from tqdm.auto import tqdm
from transformers import set_seed

from alpamayo2_super.common import logging
from alpamayo2_super.common.logging import setup_logging
from alpamayo2_sft.train_hf import training_arguments
from alpamayo2_sft.trainer import Alpamayo2SuperTrainer


setup_logging()
logger = logging.RankedLogger(__name__, rank_zero_only=True)


def compute_min_ade(pred_xyz: torch.Tensor, target_xyz: torch.Tensor) -> torch.Tensor:
    """Compute per-sample minADE over trajectory samples and average over sets."""
    distances = torch.linalg.vector_norm(
        pred_xyz[..., :2] - target_xyz[:, None, None, :, :2],
        dim=-1,
    )
    ade = distances.mean(dim=-1)
    return ade.min(dim=2).values.mean(dim=1)


def build_trainer(config: DictConfig) -> Alpamayo2SuperTrainer:
    """Instantiate the checkpoint, validation dataset, and trainer."""
    if config.evaluate.eval_ckpt is not None:
        config.model.pretrained_model_name_or_path = config.evaluate.eval_ckpt
        config.model.stage1_vlm_checkpoint_path = None

    model = instantiate(config.model, _convert_="partial")
    eval_dataset = instantiate(
        config.data.val_dataset,
        model_config=model.config,
        _convert_="partial",
    )
    collator = instantiate(
        config.data.collator,
        model_config=model.config,
        _convert_="partial",
    )
    training_args = training_arguments(config)
    trainer = Alpamayo2SuperTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    if training_args.deepspeed is not None:
        # Trajectory tokenization uses float32 linear algebra unsupported in bf16.
        trainer.accelerator.state.deepspeed_plugin.hf_ds_config._dtype = torch.float32
    return trainer


def evaluate_trajectory(
    trainer: Alpamayo2SuperTrainer,
    config: DictConfig,
) -> dict[str, float]:
    """Generate expert trajectories and aggregate minADE."""
    accelerator = trainer.accelerator
    if accelerator.num_processes != 1:
        raise ValueError(
            "Trajectory generation requires one process because ZeRO-3 partitioned "
            "embeddings are incompatible with Hugging Face generate()."
        )
    model = accelerator.prepare_model(trainer.model, evaluation_mode=True)
    generation_model = accelerator.unwrap_model(model)
    generation_model.eval()
    dataloader = trainer.get_eval_dataloader()

    max_steps = config.evaluate.max_eval_steps
    if max_steps is not None and max_steps >= 0:
        dataloader = islice(dataloader, max_steps)
        total = max_steps
    else:
        total = len(dataloader)

    metric_sum = 0.0
    metric_count = 0
    dtype = getattr(torch, config.evaluate.autocast_dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"evaluate.autocast_dtype={config.evaluate.autocast_dtype!r} is not a valid torch dtype"
        )
    if config.evaluate.num_traj_samples < 1 or config.evaluate.num_samples_per_forward < 1:
        raise ValueError(
            "evaluate.num_traj_samples and evaluate.num_samples_per_forward must be >= 1, got "
            f"{config.evaluate.num_traj_samples} and {config.evaluate.num_samples_per_forward}"
        )
    for batch in tqdm(
        dataloader,
        total=total,
        disable=not accelerator.is_main_process,
    ):
        generation_data: dict[str, Any] = {
            "tokenized_data": batch["tokenized_data"],
            **batch["traj_data"],
        }
        pred_xyz_chunks = []
        remaining_samples = config.evaluate.num_traj_samples
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            while remaining_samples > 0:
                chunk_size = min(
                    remaining_samples,
                    config.evaluate.num_samples_per_forward,
                )
                pred_xyz, _, _ = generation_model.sample_trajectories_from_data(
                    data=generation_data,
                    num_traj_samples=chunk_size,
                    num_traj_sets=config.evaluate.num_traj_sets,
                    top_p=config.evaluate.top_p,
                    temperature=config.evaluate.temperature,
                    max_generation_length=config.evaluate.max_generation_length,
                )
                pred_xyz_chunks.append(pred_xyz)
                remaining_samples -= chunk_size
        pred_xyz = torch.cat(pred_xyz_chunks, dim=2)
        per_sample = compute_min_ade(
            pred_xyz,
            batch["traj_data"]["ego_future_xyz"][:, -1],
        )
        gathered = accelerator.gather_for_metrics(per_sample)
        if accelerator.is_main_process:
            metric_sum += gathered.float().sum().item()
            metric_count += gathered.numel()

    metrics = {}
    if accelerator.is_main_process and metric_count:
        metrics["eval_min_ade"] = metric_sum / metric_count
        trainer.log(metrics)
        logger.info("Evaluation minADE: %.4f m", metrics["eval_min_ade"])
    return metrics


@hydra.main(version_base=None, config_path=None, config_name="config")
def evaluate(config: DictConfig) -> None:
    """Run validation loss or expert-trajectory evaluation."""
    set_seed(config.seed)
    trainer = build_trainer(config)
    if config.evaluate.mode == "loss":
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            logger.info("Evaluation loss: %.6f", metrics["eval_loss"])
    elif config.evaluate.mode == "trajectory":
        evaluate_trajectory(trainer, config)
    else:
        raise ValueError(
            f"evaluate.mode must be 'loss' or 'trajectory', got {config.evaluate.mode!r}"
        )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    evaluate()
