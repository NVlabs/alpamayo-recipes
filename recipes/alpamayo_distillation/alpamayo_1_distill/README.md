# Alpamayo 1 Distillation

This recipe distills the
[`nvidia/Alpamayo-R1-10B`](https://huggingface.co/nvidia/Alpamayo-R1-10B)
teacher into a smaller trajectory-planning student. The supported student is
fixed to [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
plus a compact action expert. Teacher targets are generated online from raw
PhysicalAI Autonomous Vehicles clips; no precomputed teacher-rollout dataset is
required.

The recipe provides:

- off-policy language and trajectory distillation;
- optional on-policy sequence distillation;
- flow-matching trajectory supervision;
- ZeRO-1 optimizer-state sharding;
- versioned, distributed checkpoint resume; and
- distributed student evaluation on the 644-clip gold set or the full PAI
  evaluation set.

This release is for trajectory distillation only. It does not include VQA,
LingoQA, or navigation-conditioned training pipelines.

## Student architecture

Only one student architecture is supported. The implementation validates the
text tower, vision tower, action-expert dimensions, and constructed parameter
count before training or evaluation.

| Component                | Configuration                                                                             |    Parameters |
| ------------------------ | ----------------------------------------------------------------------------------------- | ------------: |
| Vision-language backbone | Cosmos-Reason2-2B; 28 text layers, hidden size 2,048; 24 vision layers, hidden size 1,024 | 2,135,234,560 |
| Action transformer       | 28 layers, hidden size 384, intermediate size 1,536, 16 attention heads, 8 KV heads       |   115,634,560 |
| Action projections       | Per-waypoint input projection and linear output projection                                |       493,442 |
| Complete action path     | Action transformer plus projections                                                       |   116,128,002 |
| Deployed student         | VLM plus complete action path                                                             | 2,251,362,562 |

These are unique PyTorch parameter tensors after the trajectory vocabulary is
added; tied input/output embeddings are counted once. Model-card totals that
count tied embeddings separately therefore use a different convention.

The action expert consumes the corresponding Cosmos VLM layer's KV cache. It
predicts 64 trajectory waypoints at 10 Hz, giving a 6.4-second planning
horizon. The conventional “100M action expert” name refers to the complete
116,128,002-parameter action path shown above.

`--student_vlm_path` and `--student_processor_path` may relocate the supported
Cosmos model to a local mirror. They are not architecture-selection options;
incompatible configurations are rejected.

## Requirements

- Ubuntu 22.04
- Python 3.12
- CUDA 12.8.1 toolkit with cuDNN, `nvcc`, and a compatible NVIDIA driver
- `CUDA_HOME` set to the CUDA toolkit used to compile `flash-attn`
- At least 128 NVIDIA H100 GPUs with 80 GB of GPU memory are recommended
- `git`, `curl`, `ca-certificates`, `build-essential`, `ninja-build`,
  `libhwloc15`, and `libltdl7`
- [`uv`](https://docs.astral.sh/uv/) 0.11.15 or a compatible release
- access to the gated teacher model and PhysicalAI Autonomous Vehicles dataset

The full teacher/student setup is intended for multi-GPU execution. The
reference training configuration used 128 H100 GPUs with a global batch size
of 128.

## Installation

From the `recipes/alpamayo_distillation/alpamayo_1_distill` recipe directory:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install PyTorch first, then compile flash-attn against that installation.
uv sync --active --no-dev --no-install-package flash-attn
MAX_JOBS=4 uv sync --active --no-dev
```

For development and tests, include the optional development group:

```bash
MAX_JOBS=4 uv sync --active --group dev
```

## Download the public assets

Request access to and download the following datasets and checkpoints:

- [Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)
- [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
- [PhysicalAI Autonomous Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)

The teacher also uses configuration and processor assets from the public
[`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
repository.

For Physical AI Autonomous Vehicles dataset, please make sure to use the following revision for reproducing: 2ae73f49ffd2b5db43b404201beb7b92889f7afc

## Dataset contract

`PAI_AV_ROOT` must be a complete local mirror containing at least:

```text
PhysicalAI-Autonomous-Vehicles/
├── features.csv
├── clip_index.parquet
├── metadata/
│   └── sensor_presence.parquet
├── labels/egomotion/
├── camera/camera_cross_left_120fov/
├── camera/camera_front_wide_120fov/
├── camera/camera_cross_right_120fov/
└── camera/camera_front_tele_30fov/
```

Each training example uses four cameras, four frames per camera, and 16 ego
history steps. With `--random_t0`, the timestamp is drawn from the per-clip
window that has complete camera, history, and 6.4-second future coverage.

## Alpackage storage environment variables

`alpamayo1_distill.data.constants` defines where alpackage archives are stored
in S3-compatible storage. Each `ALPAMAYO_<NAME>` environment variable below
overrides the `<NAME>` constant. Export them before launching Python; the values
are read when the module is imported. The training and evaluation commands below
read from the local `PAI_AV_ROOT` mirror and do not need them.

| Variable                          | Purpose                                      | Default                                                    |
| --------------------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| `ALPAMAYO_ALPACKAGES_BUCKET`      | Bucket that stores alpackage archives        | `alpackages`                                               |
| `ALPAMAYO_ALPACKAGE_ENDPOINT_URL` | URL template for an alpackage archive        | `s3://<bucket>/pt/{alpackage_name}_{alpackage_sha256}.tar` |
| `ALPAMAYO_S3_REGION`              | Region of the storage service                | `us-east-1`                                                |
| `ALPAMAYO_S3_ENDPOINT_URL`        | Endpoint of an S3-compatible storage service | Unset; the S3 client's default endpoint is used            |

`<bucket>` is the value of `ALPAMAYO_ALPACKAGES_BUCKET`. A custom
`ALPAMAYO_ALPACKAGE_ENDPOINT_URL` must keep the `{alpackage_name}` and
`{alpackage_sha256}` placeholders. For example:

```bash
export ALPAMAYO_S3_ENDPOINT_URL="https://s3.example.com"
export ALPAMAYO_S3_REGION="us-west-2"
export ALPAMAYO_ALPACKAGES_BUCKET="my-alpackages"
```

## Training

The reference schedule uses 16 nodes with 8 GPUs per node, batch size one per
GPU, no gradient accumulation, and 80,000 optimizer steps. Its global batch
size is therefore 128:

```text
global batch = nodes × GPUs per node × batch per GPU × gradient accumulation
```

Set the distributed rendezvous values on every node. `MASTER_ADDR` must resolve
to the same coordinator, and `RDZV_ID` must be unique among concurrent jobs
sharing that endpoint. Run the rendezvous service only on a trusted network.

```bash
export NNODES="16"
export GPUS_PER_NODE="8"
export MASTER_ADDR="coordinator-host"
export MASTER_PORT="29500"
export RDZV_ID="alpamayo1-distill"
```

Launch the reference recipe with the same command on every node:

```bash
torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  --rdzv-id="$RDZV_ID" \
  -m alpamayo1_distill.train \
  --teacher_ckpt "$TEACHER_CKPT" \
  --teacher_vlm_path "$TEACHER_VLM" \
  --student_vlm_path "$STUDENT_VLM" \
  --student_processor_path "$STUDENT_VLM" \
  --local_dir "$PAI_AV_ROOT" \
  --local_files_only \
  --output_dir "$OUTPUT_DIR" \
  --batch_size_per_gpu 1 \
  --grad_accum_steps 1 \
  --max_steps 80000 \
  --save_steps 4000 \
  --val_every 4000 \
  --lr 2.1e-4 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --min_lr 1e-5 \
  --vlm_lr_scale 0.5 \
  --vlm_min_lr 1e-6 \
  --weight_decay 0.01 \
  --vlm_weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --grad_clip 1.0 \
  --future_traj_weight 2.0 \
  --future_traj_vlm_weight 1.0 \
  --distill_T 0.5 \
  --distill_weight 1.0 \
  --vlm_loss_weight 1.0 \
  --hidden_distill_weight 0.0001 \
  --random_t0 \
  --t0_us 5100000 \
  --t0_samples_per_clip 1 \
  --t0_margin_us 0 \
  --on_policy_start_step 60000 \
  --on_policy_ratio 0.5 \
  --on_policy_divergence forward_kl \
  --on_policy_T 1.0 \
  --on_policy_sample_temperature 0.6 \
  --on_policy_sample_top_p 0.98 \
  --on_policy_kl_weight 1.0 \
  --compute_minade \
  --use_zero1
```

Use a new or empty `OUTPUT_DIR` for a new run. The checkpoint writer refuses to
overwrite an existing step directory.

For a single node, replace the distributed rendezvous arguments with
`--standalone --nproc-per-node="$GPUS_PER_NODE"`. If the world size changes,
adjust `--grad_accum_steps` to preserve the intended global batch size or
retune the learning rate and schedule.

The reference run intentionally sampled random timestamps without a fixed
`--t0_seed`; independent launches are therefore not bitwise reproducible.

Checkpoints are written to `$OUTPUT_DIR/ckpts/step_<N>/`. Each complete
checkpoint contains model weights, `config.json`, `train_args.json`, and
`trainer_state.pt`.

### Resume an interrupted run

Repeat the original training command with the same topology and training
arguments, adding:

```bash
--resume_ckpt "$OUTPUT_DIR/ckpts/step_64000"
```

Resume restores the model, optimizer, scheduler, sampler cursor, and saved
process RNG state. The command rejects changes that would invalidate exact
continuation. To load only model weights into a new run with a fresh optimizer
and schedule, use `--init_ckpt` instead.

## Student trajectory evaluation

Student evaluation does not instantiate the 10B teacher. Export a checkpoint
produced by this recipe:

```bash
export STUDENT_CKPT="$OUTPUT_DIR/ckpts/step_80000"
```

The default protocol uses t0 = 5.1 seconds, seed 42, six stochastic
trajectories, top-p 0.98, and temperature 0.6. It also reports the ADE of one
greedy trajectory. Evaluation currently requires `--batch_size_per_gpu 1`.
Set the GPU count explicitly if evaluation is run in a new shell:

```bash
export GPUS_PER_NODE="8"
```

### Download the gold evaluation metadata

Before running the gold benchmark, manually download
[`1005_7cam_gold_eval_metadb_public.parquet`](https://github.com/NVlabs/alpamayo-recipes/raw/refs/heads/main/data/1005_7cam_gold_eval_metadb_public.parquet)
from the repository's shared `data/` directory and save it under the recipe's `config/` folder.

### Gold 644-clip benchmark

```bash
torchrun --standalone --nproc-per-node="$GPUS_PER_NODE" \
  -m alpamayo1_distill.eval_student \
  --ckpt_dir "$STUDENT_CKPT" \
  --student_vlm_path "$STUDENT_VLM" \
  --student_processor_path "$STUDENT_VLM" \
  --local_dir "$PAI_AV_ROOT" \
  --local_files_only \
  --eval_set gold \
  --t0_us 5100000 \
  --seed 42 \
  --seed_strategy reference \
  --num_traj_samples 6 \
  --top_p 0.98 \
  --temperature 0.6 \
  --output_json "$OUTPUT_DIR/eval-gold.json"
```

### Full PAI evaluation set benchmark

```bash
torchrun --standalone --nproc-per-node="$GPUS_PER_NODE" \
  -m alpamayo1_distill.eval_student \
  --ckpt_dir "$STUDENT_CKPT" \
  --student_vlm_path "$STUDENT_VLM" \
  --student_processor_path "$STUDENT_VLM" \
  --local_dir "$PAI_AV_ROOT" \
  --local_files_only \
  --eval_set full \
  --t0_us 5100000 \
  --seed 42 \
  --seed_strategy reference \
  --num_traj_samples 6 \
  --top_p 0.98 \
  --temperature 0.6 \
  --output_json "$OUTPUT_DIR/eval-full.json"
```

Evaluation shards the selected cohort across ranks without padding or dropping
samples. The rank-zero JSON contains aggregate `stochastic_minade_m` and
`greedy_vlm_single_ade_m`, per-sample records, and any failures. The JSON also
records local checkpoint, model, and dataset paths; inspect or redact it before
publication.

Add `--strict_determinism` when deterministic CUDA kernels are more important
than throughput. Unsupported nondeterministic operations will then fail rather
than silently falling back.

## Command-line verification

After installation, verify that the public entry points import correctly:

```bash
python -m alpamayo1_distill.train --help
python -m alpamayo1_distill.eval_student --help
```
