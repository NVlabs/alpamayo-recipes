# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch Alpamayo 2 Super supervised fine-tuning."""

from copy import deepcopy
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from transformers import set_seed
import wandb

from alpamayo2_super.common import logging
from alpamayo2_super.common.logging import setup_logging
from alpamayo2_sft.trainer import Alpamayo2SuperTrainer, TrainingArguments


setup_logging()
logger = logging.RankedLogger(__name__, rank_zero_only=True)


def init_wandb(config: DictConfig, output_dir: Path) -> None:
    """Initialize W&B, resuming the run id saved in a prior output_dir if present."""
    if config.wandb.key is not None:
        wandb.login(key=config.wandb.key)
    wandb_id_path = output_dir / ".wandb_id"
    if wandb_id_path.is_file():
        wandb_id = wandb_id_path.read_text().strip()
        logger.info("Resuming wandb run with ID %s", wandb_id)
    else:
        wandb_id = wandb.util.generate_id()
        wandb_id_path.write_text(wandb_id)
        logger.info("Starting a new wandb run with ID %s", wandb_id)
    wandb.init(
        id=wandb_id,
        entity=config.wandb.team,
        project=config.wandb.project,
        group=config.wandb.group,
        name=config.wandb.name,
        dir=str(output_dir),
        resume="allow",
    )


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
        model_config=model.config,
        _convert_="partial",
    )
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
    callbacks = [
        instantiate(callback, _convert_="partial") for callback in config.callbacks.values()
    ]

    trainer = Alpamayo2SuperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    if training_args.deepspeed is not None:
        trainer.accelerator.state.deepspeed_plugin.hf_ds_config._dtype = torch.float32
    if trainer.is_world_process_zero():
        output_dir = Path(config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open_dict(config):
            config.hydra = {
                key: HydraConfig.get()[key] for key in HydraConfig.get() if key != "runtime"
            }
        # Keep the API key out of the persisted config; Hydra still records CLI
        # overrides verbatim, so prefer `wandb login` over `wandb.key=...`.
        save_config = deepcopy(config)
        if save_config.get("wandb") is not None and save_config.wandb.get("key") is not None:
            save_config.wandb.key = "***"
        OmegaConf.save(
            config=save_config,
            f=output_dir / "config.yaml",
            resolve=True,
        )
        if config.get("wandb") is not None:
            init_wandb(config, output_dir)

    logger.info("Starting Alpamayo 2 Super %s SFT", model.training_stage)
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
