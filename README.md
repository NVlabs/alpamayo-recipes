# Alpamayo Recipes

A collection of end-to-end Alpamayo recipes for multiple versions (v1, v1.5, and beyond), designed
to help developers quickly build, adapt, and deploy Alpamayo-based applications. This repo
brings together battle-tested workflows across the Alpamayo ecosystem, including post-training
recipes (supervised fine-tuning and reinforcement learning), quantization recipes, etc.
Whether you are experimenting locally or building a full production stack, this repository is
intended as the primary starting point for developers to learn, customize, and extend
Alpamayo for their own use cases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository layout, recipe packaging conventions,
and guidance on adding new recipes for released Alpamayo models.

## Recipes

Each recipe folder contains its own README with installation and training instructions.

| Recipe                                                              | Description                                                                             |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [`recipes/alpamayo1_sft/`](recipes/alpamayo1_sft/README.md)         | Alpamayo 1 supervised fine-tuning (HuggingFace Trainer + DeepSpeed)                     |
| [`recipes/alpamayo1_5_sft/`](recipes/alpamayo1_5_sft/README.md)     | Alpamayo 1.5 SFT (HuggingFace Trainer + DeepSpeed)                                      |
| [`recipes/alpamayo1_x_rl/`](recipes/alpamayo1_x_rl/README.md)       | Alpamayo 1 and 1.5 RL post-training (Cosmos-RL / GRPO)                                  |
| [`recipes/alpamayo1_5_quant/`](recipes/alpamayo1_5_quant/README.md) | Alpamayo 1.5 quantization (Model Optimizer Toolkit / FP8 / NVFP4 + FP8 Mixed Precision) |

## Utility Scripts

| Script                                          | Purpose                                              |
| ----------------------------------------------- | ---------------------------------------------------- |
| `scripts/curate_pai_samples.py`                 | Curate a subset of PAI samples                       |
| `scripts/convert_checkpoint.py`                 | Convert between Alpamayo 1 and 1.5 checkpoints       |
| `scripts/convert_release_config_to_training.py` | Convert a release checkpoint to training format      |
| `scripts/convert_cosmos_rl_checkpoint.py`       | Convert a Cosmos-RL checkpoint to HuggingFace format |
| `scripts/download_pai.py`                       | Download the Physical AI AV dataset from HuggingFace |
