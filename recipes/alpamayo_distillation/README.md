# Alpamayo Distillation

This directory contains recipes for distilling open-source Alpamayo models into
smaller student models for autonomous-driving and robotics applications on NVIDIA
DRIVE platforms. The recipes aim to reduce inference cost while preserving trajectory
quality and driving-reasoning capability using publicly available models, datasets,
and infrastructure and NVIDIA DRIVE platform.

The recipe collection supports Alpamayo 1 and Alpamayo 1.5 teacher models. Each model
family has a separate recipe under this shared umbrella. The parent directory defines
the common workflow and evaluation contract, while each child recipe specifies its
dependencies, configurations, launch commands, checkpoints, and results.

## Recipes

- [Alpamayo 1 distillation](alpamayo_1_distill/README.md)
- [Alpamayo 1.5 distillation](alpamayo_1_5_distill/README.md)
- [Expert SFT (stage 2, SFT on the action expert)](alpamayo1_x_distillation_expert_sft/README.md)
- [Expert RL (stage 3, DRaFT on the action expert)](alpamayo1_x_distillation_expert_rl/README.md)

## Shared architecture

The reference setup distills an Alpamayo R1-10B or R1.5-10B teacher into a compact
Qwen-based student with two principal components:

- an approximately 2B-parameter vision-language model (VLM) for perception,
  reasoning, and language generation; and
- a compact action expert for future-trajectory prediction.

The corresponding child recipe must define the exact student model, action-expert
configuration, and parameter count because these may evolve independently for
Alpamayo 1 and Alpamayo 1.5.

## Datasets

These recipes use publicly available datasets for training and evaluation. The
[NVIDIA Physical AI AV dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
provides trajectory-prediction data and, where compatible annotations are available,
navigation-conditioned trajectory examples.
[LingoQA](https://github.com/wayveai/lingoqa) provides question-answer supervision
and evaluation for VQA capability.

For reproducibility, use Physical AI AV dataset revision
[`2ae73f49ffd2b5db43b404201beb7b92889f7afc`](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/tree/2ae73f49ffd2b5db43b404201beb7b92889f7afc).

For reproducibility and training throughput, download the required datasets before
launching a job and configure each recipe to use local dataset paths rather than
fetching data online during training. Please make sure at least 100TB of storage for datasets.

## End-to-end workflow

1. **Prepare data and checkpoints.** Configure the Physical AI AV (PAI) and optional
   VQA datasets, teacher checkpoint, student initialization, sampling policy, and
   model-family-specific inputs.
2. **Run Stage-1 distillation.** Train the student with ground-truth trajectory
   supervision and teacher-provided VLM targets. Supported experiments may include
   cross-entropy and KL objectives, feature-level or multi-stage knowledge
   distillation, future-trajectory auxiliary loss, and an optional on-policy phase.
3. **Optionally run action-expert SFT/RL refinement.** Improve policy and trajectory alignment
   using documented reward functions, representative SFT samples, and explicit
   VLM/action-expert alignment objectives.
4. **Evaluate consistently.** Report trajectory quality on the standardized 644-clip
   split and the full PAI evaluation set. When supported, also report reasoning,
   VQA, and navigation-conditioned trajectory performance.

## Evaluation

Before evaluating the 644-clip gold set, manually download
[`1005_7cam_gold_eval_metadb_public.parquet`](https://github.com/NVlabs/alpamayo-recipes/raw/refs/heads/main/data/1005_7cam_gold_eval_metadb_public.parquet)
from the repository's shared `dataset/` directory. This metadata is not bundled with the
distillation recipes. Follow the download instructions for
[R1](alpamayo_1_distill/README.md#download-the-gold-evaluation-metadata) or
[R1.5](alpamayo_1_5_distill/README.md#download-the-gold-evaluation-metadata)
to place it where the matching evaluator expects it.

For reproducibility, every experiment should record:

- teacher and student checkpoints;
- dataset and evaluation split;
- fixed or random per-clip sampling policy;
- global batch size, learning rate, schedule, and total steps;
- enabled losses and distillation features;
- `minADE@6` (lower is better) and greedy ADE; and
- the resulting checkpoint or export artifact.

The 644-clip split supports rapid iteration, while the full PAI evaluation set is
the primary public-data scorecard. Reasoning benchmarks such as LingoQA and
closed-loop evaluation should be reported only when supported by the corresponding
model and recipe.

## Results

All reported results were achieved with a batch size of 128.

### Alpamayo R1: base trajectory estimation

The following table compares the R1-10B teacher with the R1-2B distilled student
and the student after the two-stage expert pipeline (SFT + DRaFT RL) on the
644-sample validation set and the full PAI evaluation set. These results cover
base trajectory estimation.

Each cell reports `minADE@6 / greedy ADE` in meters (lower is better). Percentages
in parentheses show the change in each error metric relative to the R1-10B teacher;
positive values indicate higher error and negative values indicate lower error.

| Panel        | R1-10B teacher  | R1-2B distilled                    | R1-2B distilled + expert 2-stage   |
| ------------ | --------------- | ---------------------------------- | ---------------------------------- |
| gold-644     | 0.9079 / 1.5806 | 1.0355 / 1.7255 (+14.05% / +9.17%) | 0.9673 / 1.6600 (+6.54% / +5.02%)  |
| full-PAI-val | 0.8500 / 1.6300 | 0.9383 / 1.5843 (+10.39% / -2.80%) | 0.9369 / 1.5691 (+10.22% / -3.74%) |

### Alpamayo R1.5: base trajectory estimation

The following table compares the R1.5-10B teacher with the R1.5-2B student trained
only for base trajectory estimation and the student after the two-stage expert
pipeline (SFT + DRaFT RL) on the same 644-sample validation set and the full
PAI evaluation set.

Each cell reports `minADE@6 / greedy ADE` in meters (lower is better). Percentages
in parentheses show the change in each error metric relative to the R1.5-10B teacher;
positive values indicate higher error and negative values indicate lower error.

| Panel        | R1.5-10B teacher | R1.5-2B-basic-traj                  | R1.5-2B-basic-traj + expert 2-stage |
| ------------ | ---------------- | ----------------------------------- | ----------------------------------- |
| gold-644     | 0.8280 / 1.5417  | 0.9196 / 1.7086 (+11.06% / +10.83%) | 0.8691 / 1.6501 (+4.96% / +7.03%)   |
| full-PAI-val | 0.8298 / 1.5103  | 0.8978 / 1.5463 (+8.19% / +2.38%)   | 0.8875 / 1.5333 (+6.95% / +1.52%)   |

### Alpamayo R1.5: mixed trajectory and VQA training

The following table compares the R1.5-10B teacher with the R1.5-2B distilled student
trained with a mixture of base trajectory estimation, navigation-conditioned
trajectory estimation, and VQA. The `R1.5-2B distilled + expert 2-stage` column
reports the final result after supervised fine-tuning (SFT) and direct reward
fine-tuning (DRaFT) of the action expert.

`NAV65` and `nav-val` are subsets of the original `gold-644` validation set and
the full PAI evaluation set (`full-PAI-val`), respectively. Both subsets are selected
by filtering for samples that satisfy the navigation-command validity restrictions.

For trajectory panels, each cell reports `minADE@6 / greedy ADE` in meters (lower
is better). Percentages in parentheses show the change in each error metric relative
to the R1.5-10B teacher; positive values indicate higher error and negative values
indicate lower error.

The LingoQA and CoT rows report `accuracy (%) / judge score` (higher is better).
The CoT row uses the teacher's REASON CoT metrics and the student's
`cot_gt_judge_acc` and `avg_cot_gt_judge`, respectively.

<table>
  <thead>
    <tr>
      <th>Panel</th>
      <th>R1.5-10B teacher</th>
      <th>R1.5-2B distilled</th>
      <th>R1.5-2B distilled + expert 2-stage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gold-644</td>
      <td>0.8280 / 1.5417</td>
      <td>0.9462 / 1.7250 (+14.28% / +11.89%)</td>
      <td>0.8924 / 1.6366 (+7.78% / +6.16%)</td>
    </tr>
    <tr>
      <td>full-PAI-val</td>
      <td>0.8298 / 1.5103</td>
      <td>0.9108 / 1.5811 (+9.76% / +4.69%)</td>
      <td>0.9035 / 1.5629 (+8.88% / +3.48%)</td>
    </tr>
    <tr>
      <td>NAV65</td>
      <td>0.7347 / 1.3764</td>
      <td>0.9004 / 1.5492 (+22.55% / +12.55%)</td>
      <td>0.7560 / 1.3855 (+2.90% / +0.66%)</td>
    </tr>
    <tr>
      <td>nav-val</td>
      <td>0.7950 / 1.4611</td>
      <td>0.8668 / 1.5114 (+9.03% / +3.44%)</td>
      <td>0.8542 / 1.4890 (+7.45% / +1.91%)</td>
    </tr>
    <tr>
      <td>LingoQA (accuracy / judge)</td>
      <td>69.40% / 0.6481</td>
      <td colspan="2">73.20% / 0.6886</td>
    </tr>
    <tr>
      <td>CoT (accuracy / judge)</td>
      <td>64.94% / 0.6119</td>
      <td colspan="2">64.58% / 0.6111</td>
    </tr>
  </tbody>
</table>
