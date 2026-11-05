# Alpamayo 1.5 Distillation

This recipe distills the `nvidia/Alpamayo-1.5-10B` teacher into a student built
from a `nvidia/Cosmos-Reason2-2B` VLM and a compact action expert. The student
preserves the teacher's action space, diffusion formulation, and trajectory
tokenization while resizing the expert to consume the 2B VLM's KV cache.
Training combines language, hidden-state, trajectory-flow, and
future-trajectory objectives.

The implementation supports online teacher targets, optional on-policy
rollouts, navigation conditioning, ZeRO-1 optimizer sharding, exact checkpoint
resume, and distributed validation.

## Prerequisites

- Python 3.12
- Ubuntu 22.04
- CUDA 12.8.1 with cuDNN and a compatible NVIDIA driver
- At least 128 NVIDIA H100 GPUs with 80 GB of GPU memory per device are recommended
- Git, which is required to install the pinned Alpamayo dependency
- `uv`
- Access to the gated Alpamayo model and PhysicalAI AV dataset

The reference configuration is intended for multi-GPU training. Model and
dataset loading are not CPU-only workflows, and the full teacher/student setup
generally does not fit on a single GPU.

## Table of contents

1. [Getting started](#getting-started)
    1. [Python environment](#1-python-environment)
    2. [Environment variables](#2-environment-variables)
    3. [Authenticate with Hugging Face](#3-authenticate-with-hugging-face-and-dataset-preparation)
    4. [Prepare navigation labels](#4-prepare-navigation-labels)
2. [Training](#training)
    1. [Settings](#training-settings)
    2. [Run training](#run-training)
    3. [Training modes](#training-modes)
    4. [Navigation mixture](#navigation-mixture)
    5. [Resume training](#resume-training)
    6. [Scheduler integration](#scheduler-integration)
3. [Checkpoints](#checkpoints)
4. [Evaluation](#evaluation)
    1. [Download the gold evaluation metadata](#download-the-gold-evaluation-metadata)
    2. [Student-only trajectory benchmark](#student-only-trajectory-benchmark)
    3. [Trajectory benchmark output](#trajectory-benchmark-output)
5. [Testing](#testing)

## Getting started

### 1. Python environment

Define the workspace containing the `alpamayo-recipes` checkout:

```bash
export YOUR_HOME="/path/to/your/workspace"
export UV_CACHE_DIR="$YOUR_HOME/.cache/uv"
```

Create and synchronize the environment:

```bash
cd "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo_distillation/alpamayo_1_5_distill"
uv venv am15_distill
source am15_distill/bin/activate

# Install torch first, then build flash-attn against it.
uv sync --active --no-install-package flash-attn
MAX_JOBS=4 uv sync --active
```

`MAX_JOBS` limits parallel compilation if building `flash-attn` would otherwise
consume too much memory. The recipe pins the upstream Alpamayo dependency to
revision `36aeb4c5938cbc2eb2aed33b22434773da4ab639`.

### 2. Environment variables

Set these once per session:

```bash
# Required
export ALPAMAYO_DATASET_DIR="/path/to/PhysicalAI-Autonomous-Vehicles"

# Required only when navigation conditioning is enabled
export ALPAMAYO_NAV_DATASET_DIR="/path/to/navigation-labels"

# Required only when LingoQA is enabled; used by the commands below
export LINGOQA_DATASET_DIR="/path/to/LingoQA/scenery"

# Recommended cache locations
export HF_HOME="$YOUR_HOME/.cache/huggingface"
```

- `ALPAMAYO_DATASET_DIR`: local PhysicalAI Autonomous Vehicles dataset root.
- `ALPAMAYO_NAV_DATASET_DIR`: navigation JSONL root; optional when navigation
  is disabled.
- `LINGOQA_DATASET_DIR`: directory containing `train.parquet` and the image
  files referenced by it. This is a shell variable used by the examples; pass
  it through `data.lingoqa.data_root` as shown below.
- `HF_HOME`: Hugging Face model and dataset cache.
- `HF_TOKEN`: optional alternative to `hf auth login`.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`: optional; force cached,
  offline loading.

#### LingoQA data layout

When LingoQA VQA is enabled, download the Scenery training split from the
[official Wayve LingoQA repository](https://github.com/wayveai/LingoQA#download-data-and-annotations)

Download `train.parquet` and `images.zip`, then extract the image archive under
the same data root. For example:

```text
LingoQA/
└── scenery/
    ├── train.parquet
    └── <image directories and JPEGs extracted from images.zip>
```

Set `LINGOQA_DATASET_DIR` to the directory containing `train.parquet` (the
`scenery/` directory in this example). Every value in the parquet `images`
column must resolve relative to this directory, and the parquet must contain
the `images`, `question`, and `answer` columns. With
`data.lingoqa.n_frames` set to `null`, the loader preserves all five Scenery
frames for each example.

Configuration values take precedence over environment variables. In
particular, `data.local_dir`, `data.nav.root`, and `validation.nav.root` can be
set directly with command-line overrides.

### 3. Authenticate with Hugging Face and Dataset preparation

Request access and download the following resources:

- [PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
- [Alpamayo 1.5 10B model](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
- [LingoQA Dataset](https://github.com/wayveai/lingoqa)

The default student VLM is
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
Models may also be mounted locally and selected through configuration
overrides.

To train the model with full functionalities including base-trajectory estimation, navigation trajectory estimation and VQA capability, PhysicalAI Autonomous Vehicles dataset, your customized navigation command and lingoQA dataset is required. For base-trajectory only distillation, PhysicalAI Autonomous Vehicles dataset is sufficient.

### 4. Prepare navigation labels

The navigation-conditioned annotations used to develop this recipe are not
released. To enable navigation conditioning, provide your own route
instructions as JSONL sidecars aligned with clips in the PhysicalAI Autonomous
Vehicles dataset.

Organize the sidecars by cohort, with one file per clip:

```text
navigation-labels/
├── train/
│   ├── <clip-id-1>.jsonl
│   └── <clip-id-2>.jsonl
└── <validation-cohort>/
    ├── <clip-id-3>.jsonl
    └── <clip-id-4>.jsonl
```

Set `data.nav.cohort` and `validation.nav.cohort` to the corresponding
directory names. The baseline uses `train` and `gold644`; replace `gold644`
with the name of your own validation cohort if needed. Also replace
`validation.clip_allowlist_path` with your own clip list, or set it to `null`
to use the configured PhysicalAI AV validation split. File names must use the
PhysicalAI AV clip ID without the optional `clipgt-` prefix.

Each JSONL file is a dense 10 Hz timeline for one clip. Zero-indexed row `i`
must describe `t0_relative_us = i * 100000`; do not omit timestamps that have
no usable instruction. For example:

```json
{"clip_id":"clip-id-1","cohort":"train","hf_split":"train","t0_relative_us":0,"lookup_in_route_timestamp_range":true,"has_future_instruction":true,"nav_command":"TURN_LEFT","nav_distance_m":42.5,"nav_text":"Turn left in 43 m"}
{"clip_id":"clip-id-1","cohort":"train","hf_split":"train","t0_relative_us":100000,"lookup_in_route_timestamp_range":true,"has_future_instruction":true,"nav_command":"TURN_LEFT","nav_distance_m":41.2,"nav_text":"Turn left in 41 m"}
{"clip_id":"clip-id-1","cohort":"train","hf_split":"train","t0_relative_us":200000,"lookup_in_route_timestamp_range":false,"has_future_instruction":false,"nav_command":"UNKNOWN","nav_distance_m":0.0,"nav_text":""}
```

Required fields:

- `clip_id`: PhysicalAI AV clip ID. After removing an optional leading
  `clipgt-` prefix, it must match the JSONL file stem.
- `cohort`: must match the enclosing cohort directory and the configured
  `*.nav.cohort`.
- `hf_split`: must be `train` for the `train` cohort.
- `t0_relative_us`: clip-relative timestamp in microseconds on the 100,000 µs
  grid.
- `lookup_in_route_timestamp_range`: whether this timestamp falls in the valid
  route-label range.
- `has_future_instruction`: whether a future navigation instruction is
  available at this timestamp.
- `nav_command`: command category, such as `TURN_LEFT`, `TURN_RIGHT`, or
  `STRAIGHT`. With `exclude_unknown: true`, `UNKNOWN` rows are not sampled.
- `nav_distance_m`: non-negative distance in meters to the referenced
  maneuver. `max_distance_m` can filter instructions beyond a chosen distance;
  `-1` disables this filter.
- `nav_text`: natural-language route instruction inserted into the model
  prompt. This is the field consumed by the model.

A timestamp is eligible for navigation sampling only when both boolean fields
are true, its command passes the unknown-command filter, and its distance
passes `max_distance_m`. Rows that are not eligible must still remain in the
file to preserve direct line-to-timestamp indexing.

By default, the loader discovers clips by scanning `<root>/<cohort>/*.jsonl`.
For large datasets, `data.nav.clip_index_path` and
`validation.nav.clip_index_path` may point to a prebuilt text or JSONL index.
A text index contains one clip ID per line; a JSONL index contains objects with
a `clip_id` or `name` field.

Finally, point the recipe at the sidecar root:

```bash
export ALPAMAYO_NAV_DATASET_DIR=/path/to/navigation-labels
```

## Training

Training generates teacher responses online and optimizes the student with a
weighted combination of:

- language cross-entropy and teacher-student KL;
- hidden-state projection loss;
- trajectory flow-matching loss;
- optional teacher-vector matching;
- future-trajectory VLM loss; and
- optional on-policy sequence divergence.

### Training settings

Settings are defined in the YAML files under `configs/` and can be overridden
with OmegaConf dot-list arguments. `baseline.yaml` is the generic recipe;
`on_policy_nav_lingoqa.yaml` reproduces the navigation-and-VQA experiment:

- `data`: adjust the dataset sampling strategies, dataloader setups, and different dataset mixture strategies.
- `model` / `teacher`: you can set the teacher model setup in this section, related to model paths, dtype etc.
- `online_teacher.top_k`: default is `-1` for dense full-vocabulary KL, otherwise a positive
  integer for top-k plus residual-tail KL.
- `on_policy`: define when rollout start step start and the ratio between on-policy distillation vs. off-policy distillation, and corresponding hyperparameters.
- `loss`: objective weights and distillation temperature.
- `optimizer`: hyperparameters related to learning rates, weight decay, and ZeRO-1.
- `scheduler`: hyperparameters related to warmup and cosine-decay settings.
- `training`: general training related hyperparameters including output directory, maximum steps, logging and saving steps.
- `validation`: eneral validation related hyperparameters including validation cadence, sampling, navigation, and minADE settings.

### Run training

Choose one of the three canonical commands under [Training modes](#training-modes).
Each example launches one node; change `--nproc-per-node` to the number of GPUs
available on that node.

To use local model checkpoints, append overrides such as:

```text
model.kwargs.teacher_name_or_path=/path/to/Alpamayo-1.5-10B
model.kwargs.student_vlm_name_or_path=/path/to/Cosmos-Reason2-2B
teacher.kwargs.name_or_path=/path/to/Alpamayo-1.5-10B
```

If the teacher VLM is stored separately, also set
`model.kwargs.teacher_vlm_name_or_path` and
`teacher.kwargs.vlm_name_or_path`.

### Training modes

The release exposes three modes through explicit configuration overrides rather
than a cluster-specific launcher. Run the commands from this recipe directory.
Each command uses `configs/on_policy_nav_lingoqa.yaml`; the mode overrides
change only the enabled data streams and leave the model, losses, optimizer,
schedule, checkpointing, and base-trajectory validation unchanged.

| Mode           | Enabled training streams                                                                   | Required datasets                               |
| -------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `base-traj`    | PhysicalAI-AV base trajectories                                                            | PhysicalAI-AV                                   |
| `base-nav`     | Base and navigation-conditioned trajectories                                               | PhysicalAI-AV and navigation sidecars           |
| `base-nav-vqa` | Base trajectories, navigation-conditioned trajectories, LingoQA VQA, and PhysicalAI-AV VQA | PhysicalAI-AV, navigation sidecars, and LingoQA |

#### Base-trajectory distillation only

```bash
torchrun \
  --standalone \
  --nproc-per-node=8 \
  --module alpamayo1_5_distill.train \
  --config configs/on_policy_nav_lingoqa.yaml \
  "data.local_dir=${ALPAMAYO_DATASET_DIR}" \
  data.mixture.enabled=false \
  data.nav.enabled=false \
  data.lingoqa.enabled=false \
  data.pai_vqa.enabled=false \
  validation.nav.enabled=false \
  training.output_dir=./outputs/base-traj
```

This mode disables the explicit navigation mixture, navigation validation,
LingoQA, and PhysicalAI-AV VQA. It does not require `ALPAMAYO_NAV_DATASET_DIR` or
`LINGOQA_DATASET_DIR`.

#### Base and navigation-conditioned trajectory distillation

```bash
torchrun \
  --standalone \
  --nproc-per-node=8 \
  --module alpamayo1_5_distill.train \
  --config configs/on_policy_nav_lingoqa.yaml \
  "data.local_dir=${ALPAMAYO_DATASET_DIR}" \
  "data.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  "validation.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  data.mixture.enabled=true \
  data.nav.enabled=true \
  data.lingoqa.enabled=false \
  data.pai_vqa.enabled=false \
  validation.nav.enabled=true \
  training.output_dir=./outputs/base-nav
```

This mode disables both VQA sources. The default requested trajectory mix is
50% base and 50% navigation-conditioned, controlled by
`data.mixture.basic_ratio` and `data.mixture.nav_ratio`. Because failed
navigation requests may fall back to base samples, logs report both requested
and realized ratios.

#### Base, navigation-conditioned trajectory, and VQA distillation

```bash
torchrun \
  --standalone \
  --nproc-per-node=8 \
  --module alpamayo1_5_distill.train \
  --config configs/on_policy_nav_lingoqa.yaml \
  "data.local_dir=${ALPAMAYO_DATASET_DIR}" \
  "data.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  "validation.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  "data.lingoqa.data_root=${LINGOQA_DATASET_DIR}" \
  data.mixture.enabled=true \
  data.nav.enabled=true \
  data.lingoqa.enabled=true \
  data.pai_vqa.enabled=true \
  validation.nav.enabled=true \
  training.output_dir=./outputs/base-nav-vqa
```

This is the default mode. It adds LingoQA ground-truth VQA and
teacher-supervised PhysicalAI-AV VQA to the base/navigation trajectory mix.
With the supplied release configuration, the sampler targets approximately
80% trajectory and 20% VQA batches. Within VQA, PhysicalAI-AV is selected with
probability `0.3` and LingoQA otherwise.

#### Training smoke test

The smoke test runs the full data path with the same configuration, but uses
two GPUs, stops after 20 steps, starts from scratch, and skips validation:

```bash
torchrun \
  --standalone \
  --nproc-per-node=2 \
  --module alpamayo1_5_distill.train \
  --config configs/on_policy_nav_lingoqa.yaml \
  "data.local_dir=${ALPAMAYO_DATASET_DIR}" \
  "data.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  "validation.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  "data.lingoqa.data_root=${LINGOQA_DATASET_DIR}" \
  training.output_dir=./outputs/smoke \
  training.max_steps=20 \
  training.save_every_steps=10 \
  training.resume_from=null \
  validation.enabled=false
```

### Navigation mixture

The full recipe enables navigation for training and validation. Set
`ALPAMAYO_NAV_DATASET_DIR` and include these overrides in the command for the
selected mode:

```text
data.nav.root=/path/to/navigation-labels
validation.nav.root=/path/to/navigation-labels
```

The requested base/navigation ratio is controlled by:

```yaml
data:
    mixture:
        enabled: true
        basic_ratio: 0.5
        nav_ratio: 0.5
        fallback_nav_to_basic: true
```

Training logs both requested and realized mixture fractions. If a navigation
sample cannot be loaded after the configured retries,
`fallback_nav_to_basic` determines whether it is replaced by a base-trajectory
sample.

### Resume training

Resume from the newest checkpoint under `training.output_dir`:

```text
training.output_dir=/path/to/existing-output
training.resume_from=latest
```

Append these overrides to the command for the selected training mode. Set
`training.resume_from=/path/to/checkpoint-XXXXXXXX` to resume a specific
checkpoint. The seed, world size, dataset length, per-rank batch size, and
gradient accumulation settings must match the saved trainer state. The saved
sampler epoch must also agree with the checkpoint epoch.

### Scheduler integration

The currently reported training results for this recipe were obtained with 128
GPUs, `data.batch_size=1` per GPU, and
`training.gradient_accumulation_steps=1`, giving a global batch size of 128.

For multi-node execution, start this command once on each node with the
appropriate node rank and replace `--standalone` with the distributed
rendezvous arguments:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --module alpamayo1_5_distill.train \
  --config configs/on_policy_nav_lingoqa.yaml \
  "data.local_dir=${ALPAMAYO_DATASET_DIR}" \
  "data.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  "validation.nav.root=${ALPAMAYO_NAV_DATASET_DIR}" \
  "data.lingoqa.data_root=${LINGOQA_DATASET_DIR}" \
  training.output_dir=./outputs/base-nav-vqa
```

Keep production and smoke scheduler files as thin private wrappers around this
command. The smoke wrapper should change only resource allocation,
`training.max_steps`, checkpoint cadence, resume behavior, validation, and its
output directory; it should not maintain a separate training command.

## Checkpoints

Checkpoints are written as `checkpoint-XXXXXXXX` under `training.output_dir`.
Each checkpoint contains:

- `student/`: Hugging Face-compatible student configuration and weights;
- `trainer_state.json`: global step and sampler resume metadata;
- `objective.pt`: learned loss-projection state; and
- optimizer, scheduler, and random-number-generator state.

ZeRO-1 runs additionally store `zero_optimizer.pt` and `zero_scheduler.pt`.

## Evaluation

The public evaluation entry point is `evaluate_trajectory.py`. It is the
student-only trajectory benchmark for the 644-clip gold set, the full PAI
evaluation set, and their navigation-conditioned subsets.
Chain-of-thought (CoT) and LingoQA benchmarks use separate, task-specific
evaluation recipes; use the corresponding recipe for each benchmark.

### Download the gold evaluation metadata

Before running the gold benchmark, manually download
[`1005_7cam_gold_eval_metadb_public.parquet`](https://github.com/NVlabs/alpamayo-recipes/raw/refs/heads/main/data/1005_7cam_gold_eval_metadb_public.parquet)
from the repository's shared `data/` directory and save it under `assets` folder.

### Student-only trajectory benchmark

The commands below evaluate minADE@6 and greedy ADE without loading the
teacher. They use a fixed base timestamp of 5.1 seconds, reset the inference
seed to 42 for every rollout, and assign every selected clip to exactly one
distributed rank without padding or dropping samples. Change
`--nproc-per-node=8` to the number of GPUs available on the node.

`--checkpoint` accepts either `checkpoint-XXXXXXXX` or its `student/`
subdirectory. The explicit VLM and processor paths override paths embedded in
the checkpoint, which is useful when the checkpoint is moved to another
machine.

Evaluate the 644-clip gold set:

```bash
torchrun \
  --standalone \
  --nproc-per-node=8 \
  --module alpamayo1_5_distill.evaluate_trajectory \
  --checkpoint /path/to/checkpoint-XXXXXXXX \
  --dataset-root /path/to/PhysicalAI-Autonomous-Vehicles \
  --student-vlm-path /path/to/Cosmos-Reason2-2B \
  --processor-path /path/to/Cosmos-Reason2-2B \
  --eval-set 644 \
  --output-json /path/to/results/trajectory-644-seed42.json \
  --num-workers 2 \
  --prefetch-factor 2 \
  --log-every 25 \
  --max-generation-length 256
```

Evaluate the full PAI evaluation set:

```bash
torchrun \
  --standalone \
  --nproc-per-node=8 \
  --module alpamayo1_5_distill.evaluate_trajectory \
  --checkpoint /path/to/checkpoint-XXXXXXXX \
  --dataset-root /path/to/PhysicalAI-Autonomous-Vehicles \
  --student-vlm-path /path/to/Cosmos-Reason2-2B \
  --processor-path /path/to/Cosmos-Reason2-2B \
  --eval-set 90k \
  --output-json /path/to/results/trajectory-90k-seed42.json \
  --num-workers 2 \
  --prefetch-factor 2 \
  --log-every 25 \
  --max-generation-length 256
```

Evaluate the 644-clip set and its navigation-eligible subset. The navigation
root must contain a `gold644/` sidecar directory. The canonical benchmark uses
only labels at exactly 5.1 seconds:

```bash
torchrun \
  --standalone \
  --nproc-per-node=8 \
  --module alpamayo1_5_distill.evaluate_trajectory \
  --checkpoint /path/to/checkpoint-XXXXXXXX \
  --dataset-root /path/to/PhysicalAI-Autonomous-Vehicles \
  --student-vlm-path /path/to/Cosmos-Reason2-2B \
  --processor-path /path/to/Cosmos-Reason2-2B \
  --eval-set 644 \
  --output-json /path/to/results/trajectory-644-nav-seed42.json \
  --num-workers 2 \
  --prefetch-factor 2 \
  --log-every 25 \
  --max-generation-length 256 \
  --include-nav-eval \
  --nav-root /path/to/navigation-labels \
  --nav-cohort gold644 \
  --nav-t0-mode exact \
  --nav-max-distance-m -1
```

Evaluate the full PAI evaluation set and its navigation-eligible subset. The
navigation root must contain an `eval/` sidecar directory. For navigation pairs, this mode uses
the valid labeled timestamp closest to 5.1 seconds and accepts the two
validated row cohorts shown below; the full-set base metric remains fixed at
5.1 seconds:

```bash
torchrun \
  --standalone \
  --nproc-per-node=8 \
  --module alpamayo1_5_distill.evaluate_trajectory \
  --checkpoint /path/to/checkpoint-XXXXXXXX \
  --dataset-root /path/to/PhysicalAI-Autonomous-Vehicles \
  --student-vlm-path /path/to/Cosmos-Reason2-2B \
  --processor-path /path/to/Cosmos-Reason2-2B \
  --eval-set 90k \
  --output-json /path/to/results/trajectory-90k-nav-seed42.json \
  --num-workers 2 \
  --prefetch-factor 2 \
  --log-every 25 \
  --max-generation-length 256 \
  --include-nav-eval \
  --nav-root /path/to/navigation-labels \
  --nav-cohort eval \
  --nav-allowed-row-cohorts checkpoint_val_5k,opd_unseen_val_70k \
  --nav-t0-mode closest \
  --nav-max-distance-m -1
```

### Trajectory benchmark output

The output JSON for either base benchmark contains
`mean_minade_at_6_m` and `mean_greedy_ade_m`. Navigation evaluation also
contains:

- `mean_nav_subset_basic_minade_at_6_m` and
  `mean_nav_subset_basic_greedy_ade_m`: unconditioned predictions on the
  navigation-eligible subset; and
- `mean_nav_minade_at_6_m` and `mean_nav_greedy_ade_m`:
  navigation-conditioned predictions on exactly the same samples and
  timestamps.

All displacement metrics are in meters and lower is better. Navigation mode
still evaluates the complete base set; it adds the paired unconditioned and
conditioned measurements for samples with eligible navigation labels. By
default, any failed sample invalidates the result. Do not use
`--allow-partial-results` for benchmark reporting. The command for the full PAI
evaluation set also checks the expected clip count for the pinned dataset
revision; use `--allow-count-mismatch` only for an intentionally different
dataset revision.

## Testing

Run the unit tests from the recipe directory:

```bash
uv run --active pytest
```
