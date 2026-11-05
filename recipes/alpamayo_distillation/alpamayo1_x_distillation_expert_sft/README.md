# Alpamayo 1.x Expert SFT (stage 2)

Action-expert-only fine-tuning of a distilled Alpamayo student (the stage-1
output of `alpamayo1_5_distill/` or `alpamayo1_distill/`). The VLM is frozen;
the diffusion action expert is trained on ground-truth future trajectories
with the rectified-flow objective, conditioned on the model's own greedily
generated Chain-of-Causation reasoning (self-conditioning — the stage-2
"teacher" is the stage-1 checkpoint itself).

## What is different from stage 1 (and why)

| Aspect    | Stage 1 (distill)                                 | Stage 2 (this recipe)                                                            |
| --------- | ------------------------------------------------- | -------------------------------------------------------------------------------- |
| Objective | CE + logit KD + hidden + flow (+ future-traj VLM) | flow (GT future trajectory) ONLY                                                 |
| Teacher   | Alpamayo-1.5-10B                                  | the stage-1 2B student itself (greedy CoC conditioning)                          |
| Trainable | whole student                                     | action expert + action in/out projections + trajectory tokenizers (`freeze_vlm`) |
| Data      | full train split, random t0                       | curated 3,200-clip subset, fixed t0 = 5.1 s, one sample/clip/epoch               |
| LR        | cosine 2.1e-4                                     | CONSTANT 6.5e-5 (AM1.5) / 5e-5 (AM1)                                             |

## Checkpoint selection

Train the full window, save densely, then select the best checkpoint on your
evaluation panel.

## Prerequisites

- The stage-2/3 container image built from `../alpamayo1_x_distillation_expert_rl/docker/`
  (CUDA 12.8; `constraints.txt` there is a byte-exact freeze of the validated
  environment). 8x NVIDIA H100 80GB recommended.
- The stage-1 distillation package importable as `alpamayo1_5_distill` —
  install the sibling recipe environment first (see
  `../alpamayo_1_5_distill/README.md`); this stage reuses its model, data and
  training utilities and only swaps the model factory, prompt layout and
  teacher-batch cache.
- A stage-1 distilled student checkpoint (passed as BOTH
  `model.kwargs.name_or_path` and `teacher.kwargs.name_or_path` —
  self-conditioning).
- The PhysicalAI-AV dataset local mirror (`ALPAMAYO_DATASET_DIR` or
  `data.local_dir`).
- `PYTHONPATH` must include `recipes/alpamayo_distillation` so that
  `alpamayo1_x_distillation_expert_sft` resolves as a module.

## Launch

```bash
# from the alpamayo-recipes repo root
export PYTHONPATH=$PWD/recipes/alpamayo_distillation:$PYTHONPATH
export ALPAMAYO_DATASET_DIR=/path/to/physical_ai_av_mirror   # or data.local_dir
torchrun --nproc_per_node 8 -m alpamayo1_x_distillation_expert_sft.train \
    --config recipes/alpamayo_distillation/alpamayo1_x_distillation_expert_sft/configs/expert_sft_am15.yaml \
    model.kwargs.name_or_path=/path/to/stage1_student \
    teacher.kwargs.name_or_path=/path/to/stage1_student \
    training.output_dir=./outputs/expert_sft_am15
```

`model.kwargs.name_or_path` and `teacher.kwargs.name_or_path` point at the
SAME stage-1 checkpoint. Effective batch = 1 × 8 GPUs × 8 grad-accum = 64;
one epoch over 3,200 clips = 50 optimizer steps (the full 1,200-step run ≈ 24 epochs).

## Configs

- `configs/expert_sft_am15.yaml` — AM1.5-2B base. Camera-annotated prompts,
  1200 steps, save every 25.
- `configs/expert_sft_am1.yaml` — AM1 (R1) 2B base. 1800 steps, save
  every 50.

## Evaluation

Evaluate saved expert-SFT checkpoints with the evaluator matching the stage-1
model family:

- R1: use the [Alpamayo 1 student evaluator](../alpamayo_1_distill/README.md#student-trajectory-evaluation).
- R1.5: use the [Alpamayo 1.5 trajectory evaluator](../alpamayo_1_5_distill/README.md#evaluation).

Before evaluating the gold set, manually download the
[gold evaluation metadata](https://github.com/NVlabs/alpamayo-recipes/raw/refs/heads/main/data/1005_7cam_gold_eval_metadb_public.parquet)
from the repository's shared `data/` directory. The metadata is not bundled with the
distillation recipes. Follow the
[R1 download instructions](../alpamayo_1_distill/README.md#download-the-gold-evaluation-metadata)
to save it under the R1 recipe's `config/` directory, or the
[R1.5 download instructions](../alpamayo_1_5_distill/README.md#download-the-gold-evaluation-metadata)
to save it under the R1.5 recipe's `assets/` directory.

## Validated results

Each cell reports `minADE@6 / greedy ADE` in meters (lower is better);
percentages are relative to the corresponding 10B teacher.

### AM1.5 mixture-trained student

| Panel        | R1.5-10B teacher | R1.5-2B distilled                   | + expert SFT                       |
| ------------ | ---------------- | ----------------------------------- | ---------------------------------- |
| gold-644     | 0.8280 / 1.5417  | 0.9462 / 1.7250 (+14.28% / +11.89%) | 0.8962 / 1.6541 (+8.24% / +7.29%)  |
| full-PAI-val | 0.8298 / 1.5103  | 0.9108 / 1.5811 (+9.76% / +4.69%)   | 0.9173 / 1.5923 (+10.54% / +5.43%) |
| NAV65        | 0.7347 / 1.3764  | 0.9004 / 1.5492 (+22.55% / +12.55%) | 0.7638 / 1.4062 (+3.96% / +2.17%)  |
| nav-val      | 0.7950 / 1.4611  | 0.8668 / 1.5114 (+9.03% / +3.44%)   | 0.8636 / 1.5138 (+8.63% / +3.61%)  |

### AM1.5 basic-trajectory student

| Panel        | R1.5-10B teacher | R1.5-2B-basic-traj                  | + expert SFT                      |
| ------------ | ---------------- | ----------------------------------- | --------------------------------- |
| gold-644     | 0.8280 / 1.5417  | 0.9196 / 1.7086 (+11.06% / +10.83%) | 0.8774 / 1.6625 (+5.97% / +7.84%) |
| full-PAI-val | 0.8298 / 1.5103  | 0.8978 / 1.5463 (+8.19% / +2.38%)   | 0.8905 / 1.5404 (+7.32% / +1.99%) |

### AM1 student

| Panel        | R1-10B teacher  | R1-2B distilled                    | + expert SFT                       |
| ------------ | --------------- | ---------------------------------- | ---------------------------------- |
| gold-644     | 0.9079 / 1.5806 | 1.0355 / 1.7255 (+14.05% / +9.17%) | 0.9671 / 1.6660 (+6.52% / +5.40%)  |
| full-PAI-val | 0.8500 / 1.6300 | 0.9383 / 1.5843 (+10.39% / -2.80%) | 0.9377 / 1.5739 (+10.32% / -3.44%) |

The expert-SFT champion is the initialization of the expert RL stage —
see `alpamayo1_x_distillation_expert_rl/`.

## Data subsets

`assets/clip_ids_s3_v2.txt` is the curated 3,200-clip training subset used by
both configs.
