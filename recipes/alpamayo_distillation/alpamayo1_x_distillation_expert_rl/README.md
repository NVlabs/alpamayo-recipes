# Alpamayo 1.x Expert RL (stage 3)

Direct reward fine-tuning (DRaFT) of the action expert on top of the stage-2
expert-SFT checkpoint (see `../alpamayo1_x_distillation_expert_sft/`): the
soft-min ADE objective is backpropagated through the product 10-step Euler
sampler directly into the action head, while the frozen VLM and a
velocity-field anchor preserve the model's behavior outside the training
stream. A single trainer file, a single node with 8 GPUs, and convergence in
under 200 optimizer steps. Validated on all three
student bases (AM1.5 mixture, AM1.5 basic-trajectory, AM1) — see the results
tables below.

## Package layout

- `draft/train_draft_ade.py` — the whole trainer.
- `draft/conditioning.py` — frozen-VLM conditioning (clip loading, prompt
  construction, KV-cache precompute).
- `draft/scripts/snapshot_to_ckpt.py` — turn the trainer's periodic head
  snapshots into full evaluable checkpoints.
- `draft/configs/make_clip_lists.py` — byte-reproducible clip lists (fixed
  seeds) from the shared s3_v2 asset.
- `docker/` — the validated stage-2/3 container environment
  (`constraints.txt` is a byte-exact freeze).

## Prerequisites

- The container image built from `docker/`; 8x NVIDIA H100 80GB.
- A stage-2 expert-SFT checkpoint (full model directory).
- The PhysicalAI-AV dataset root (`ALPAMAYO_DATASET_DIR`).

## Pipeline

One-time per environment, verify the differentiable sampler against the
product sampler (must print bit-equal), then launch the validated recipe:
train over the **full s3_v2 stream** (the CoC is re-generated with an
epoch-dependent seed on every visit) while anchoring the expert's velocity
field to the frozen init's own Euler-10 paths on a disjoint 256-clip
replay set:

```bash
export ALPAMAYO_DATASET_DIR=/path/to/physical_ai_av_mirror
cd recipes/alpamayo_distillation/alpamayo1_x_distillation_expert_rl

python3 draft/configs/make_clip_lists.py \
    --s3 ../alpamayo1_x_distillation_expert_sft/assets/clip_ids_s3_v2.txt \
    --out-dir ./clip_lists

# One-time per environment: append --parity-check to the command below —
# it verifies the differentiable sampler against the product sampler and
# exits without training (every rank must print bit-equal=True). Then
# launch for real without the flag.
# validated recipe: full-s3_v2 streaming + init-velocity anchor
torchrun --nproc_per_node 8 draft/train_draft_ade.py \
    --init-ckpt <stage2_ckpt> --heldout-clips ./clip_lists/heldout16.txt \
    --stream-clips ../alpamayo1_x_distillation_expert_sft/assets/clip_ids_s3_v2.txt \
    --anchor-beta 1.0 --replay-clips ./clip_lists/replay256.txt \
    --ctx-device cpu \
    --lr 3e-6 --tau 0.2 --steps 100 --save-head-every 25 \
    --out-dir ./outputs/draft_stream_anchor

python3 draft/scripts/snapshot_to_ckpt.py \
    --snapshot ./outputs/draft_stream_anchor/head_step<N> --base <stage2_ckpt> \
    --out ./outputs/draft_sa<N>_full
```

`--ctx-device cpu` is required for streaming runs: it keeps the resident
probe/heldout KV caches in host memory; the default (`cuda`) exhausts an
80GB device within the first optimizer steps.

## Checkpoint selection and expectations

Training converges in <200 optimizer steps and oscillates afterwards —
scan the early-step snapshots on your evaluation panel, ranking by
large-panel minADE@6. The trainer snapshots the head every
`--save-head-every` steps; `snapshot_to_ckpt.py` makes any snapshot
evaluable against any same-family base (the frozen VLM is bit-identical
across the stage-2/3 family).

## Evaluation

Evaluate full checkpoints exported by `draft/scripts/snapshot_to_ckpt.py` with
the evaluator matching the stage-1 model family:

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

| Panel        | R1.5-10B teacher | R1.5-2B distilled                   | + expert SFT                       | + expert RL                       |
| ------------ | ---------------- | ----------------------------------- | ---------------------------------- | --------------------------------- |
| gold-644     | 0.8280 / 1.5417  | 0.9462 / 1.7250 (+14.28% / +11.89%) | 0.8962 / 1.6541 (+8.24% / +7.29%)  | 0.8924 / 1.6366 (+7.78% / +6.16%) |
| full-PAI-val | 0.8298 / 1.5103  | 0.9108 / 1.5811 (+9.76% / +4.69%)   | 0.9173 / 1.5923 (+10.54% / +5.43%) | 0.9035 / 1.5629 (+8.88% / +3.48%) |
| NAV65        | 0.7347 / 1.3764  | 0.9004 / 1.5492 (+22.55% / +12.55%) | 0.7638 / 1.4062 (+3.96% / +2.17%)  | 0.7560 / 1.3855 (+2.90% / +0.66%) |
| nav-val      | 0.7950 / 1.4611  | 0.8668 / 1.5114 (+9.03% / +3.44%)   | 0.8636 / 1.5138 (+8.63% / +3.61%)  | 0.8542 / 1.4890 (+7.45% / +1.91%) |

### AM1.5 basic-trajectory student

| Panel        | R1.5-10B teacher | R1.5-2B-basic-traj                  | + expert SFT                      | + expert RL                       |
| ------------ | ---------------- | ----------------------------------- | --------------------------------- | --------------------------------- |
| gold-644     | 0.8280 / 1.5417  | 0.9196 / 1.7086 (+11.06% / +10.83%) | 0.8774 / 1.6625 (+5.97% / +7.84%) | 0.8691 / 1.6501 (+4.96% / +7.03%) |
| full-PAI-val | 0.8298 / 1.5103  | 0.8978 / 1.5463 (+8.19% / +2.38%)   | 0.8905 / 1.5404 (+7.32% / +1.99%) | 0.8875 / 1.5333 (+6.95% / +1.52%) |

### AM1 student

| Panel        | R1-10B teacher  | R1-2B distilled                    | + expert SFT                       | + expert RL                        |
| ------------ | --------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| gold-644     | 0.9079 / 1.5806 | 1.0355 / 1.7255 (+14.05% / +9.17%) | 0.9671 / 1.6660 (+6.52% / +5.40%)  | 0.9673 / 1.6600 (+6.54% / +5.02%)  |
| full-PAI-val | 0.8500 / 1.6300 | 0.9383 / 1.5843 (+10.39% / −2.80%) | 0.9377 / 1.5739 (+10.32% / −3.44%) | 0.9369 / 1.5691 (+10.22% / −3.74%) |
