# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch Alpamayo 2 Edge supervised fine-tuning."""

import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from alpamayo2edge_sft.run_utils import (
    peak_cuda_memory_metrics,
    reset_peak_cuda_memory,
    save_resolved_config,
    tracking_run,
)
from alpamayo2edge_sft.trainer import Alpamayo2EdgeTrainer, TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def training_arguments(config: DictConfig) -> TrainingArguments:
    """Resolve package-relative DeepSpeed paths."""
    trainer_config = OmegaConf.to_container(config.trainer, resolve=True)
    deepspeed_config = trainer_config.get("deepspeed")
    if deepspeed_config is not None and not Path(deepspeed_config).is_absolute():
        trainer_config["deepspeed"] = str(Path(__file__).parent / deepspeed_config)
    return TrainingArguments(**trainer_config)


@hydra.main(version_base=None, config_path=None, config_name="config")
def train(config: DictConfig) -> None:
    """Instantiate the release model, data pipeline, and trainer."""
    set_seed(config.seed)
    training_args = training_arguments(config)
    model = instantiate(config.model, _convert_="partial")
    train_dataset = instantiate(
        config.data.train_dataset,
        _partial_=True,
        _convert_="partial",
    )(model_config=model.config)
    eval_dataset = instantiate(
        config.data.val_dataset,
        _partial_=True,
        _convert_="partial",
    )(model_config=model.config)
    collator = instantiate(
        config.data.collator,
        _partial_=True,
        _convert_="partial",
    )(model_config=model.config)
    callbacks = [
        instantiate(callback, _convert_="partial") for callback in config.callbacks.values()
    ]

    trainer = Alpamayo2EdgeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=collator.data_processor.processor,
        callbacks=callbacks,
    )
    if trainer.is_world_process_zero():
        output_dir = Path(config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Hydra writes its own launch metadata. Its sweep-only fields cannot be
        # resolved during a single run, so persist only the training configuration.
        save_resolved_config(config, output_dir / "config.yaml")

    try:
        with tracking_run(config, trainer):
            global_batch = (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps * training_args.world_size
            )
            if training_args.max_steps > 0:
                budget = (
                    f"{training_args.max_steps} optimizer steps "
                    f"(~{training_args.max_steps * global_batch / len(train_dataset):.2f} "
                    "passes; final-batch padding may affect this estimate)"
                )
            else:
                budget = f"{training_args.num_train_epochs:g} epochs"
            logger.info(
                "Starting Alpamayo 2 Edge %s SFT: %s samples, global batch %s, %s",
                model.training_stage, len(train_dataset), global_batch, budget,
            )
            reset_peak_cuda_memory()
            result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            memory_metrics = peak_cuda_memory_metrics("train")
            if memory_metrics:
                trainer.log(memory_metrics)
            trainer.save_metrics("train", {**result.metrics, **memory_metrics})
            trainer.save_state()
    except Exception:
        # A peer may already be waiting in a ZeRO collective when this rank's
        # data loader fails. Log the original error before distributed cleanup
        # can replace the visible failure with a collective timeout.
        logger.exception("Alpamayo 2 Edge SFT failed on rank %s", training_args.process_index)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
