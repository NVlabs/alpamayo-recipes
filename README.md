# Alpamayo Recipes

A collection of end-to-end Alpamayo recipes for multiple versions (v1, v1.5, and beyond), designed to help developers quickly build, adapt, and productionize Alpamayo-based applications. This 
repo brings together battle-tested workflows across the Alpamayo ecosystem, including: Post-training recipes (SFT, RL, and distillation), Auto-labeling and data curation workflows, AlpaGym, etc. 
Whether you are experimenting locally or building a full production stack, this repository is intended as the primary starting point for external developers to learn, customize, and extend Alpamayo for their own use cases.

## Prerequisites

### 1. Install uv

A version of uv is necessary to install the alpamayo related packages.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```
### 2. Install alpamayo (inference models)

All recipes depend on the **Alpamayo** model package. Clone and install it first:

```bash
git clone https://github.com/NVlabs/alpamayo.git
cd alpamayo
uv venv ar1_venv
source ar1_venv/bin/activate
uv sync --active
cd ..
```

Follow any additional setup steps in [alpamayo/README.md](https://github.com/NVlabs/alpamayo/blob/main/README.md).

### 3. Install alpamayo-recipes

```bash
git clone <this-repo-url> alpamayo-recipes
cd alpamayo-recipes
uv sync --active   # automatically installs alpamayo as a dependency
```

## Recipes

Go to each recipe folder for its own README with full setup and training instructions.

| Recipe | Description |
|--------|-------------|
| [`recipes/alpamayo1_sft/`](recipes/alpamayo1_sft/README.md) | Alpamayo-1 supervised fine-tuning (HuggingFace Trainer + DeepSpeed) |
| [`recipes/alpamayo1_5_sft/`](recipes/alpamayo1_5_sft/README.md) | Alpamayo-1.5 SFT *(coming soon)* |
| [`recipes/alpamayo1_x_rl/`](recipes/alpamayo1_x_rl/README.md) | Alpamayo-1.x RL post-training (Cosmos-RL / GRPO) |
| [`recipes/distillation/`](recipes/distillation/README.md) | Distillation *(coming soon)* |
| [`recipes/alpagym/`](recipes/alpagym/README.md) | AlpaGym *(coming soon)* |

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_pai.py` | Download the Physical AI AV dataset from HuggingFace |
| `scripts/curate_pai_samples.py` | Curate a subset of PAI samples |
| `scripts/convert_release_config_to_training.py` | Convert a release checkpoint to training format |
| `scripts/convert_cosmos_rl_checkpoint.py` | Convert a Cosmos-RL checkpoint to HF format |
