# Frozen training environment (expert SFT stage 2 + expert RL stage 3)

Validated for the two expert stages only — the stage-1 distill recipe has
its own uv environment (`alpamayo1_5_distill/pyproject.toml`) and was NOT
validated against this image.

`Dockerfile` bakes the exact environment the expert SFT/RL recipes were
validated in. `constraints.txt` is the pip freeze of that environment,
captured from the training nodes (verified byte-identical across all four).

Why this matters: with a fixed seed, a training trajectory is reproducible
only WITHIN one kernel/library stack. A rebuilt venv with slightly different
wheels produces a different — equally valid — trajectory. If you need
band-level comparability with the released checkpoints, use this image; if
you only need the recipe, any recent stack works but expect your own
oscillation band.

```bash
docker build -f alpamayo1_x_distillation_expert_rl/docker/Dockerfile -t alpamayo-expert:frozen .
docker run --gpus all --shm-size 64g \
    -v $PWD:/workspace/alpamayo1_5-distill \
    -v /path/to/physical_ai_av_mirror:/data/physical_ai_av \
    -e ALPAMAYO_DATASET_DIR=/data/physical_ai_av \
    -it alpamayo-expert:frozen bash
```

After EVERY node (re)start, run the idempotent sanity script before
launching anything:

```bash
EXPECTED_TRAIN_CLIPS=3200 \
EXPERT_CLIP_LIST=/workspace/alpamayo1_5-distill/alpamayo1_x_distillation_expert_sft/assets/clip_ids_s3_v2.txt \
EXPERT_CKPT=/path/to/stage1_student EXPECT_CAMERA=true \
bash /usr/local/bin/on_node_start.sh
```

Every check in that script encodes a real failure mode from recipe
development (silent dataset filtering after a node reprovision, prompt-layout
mismatch, environment drift breaking seed-level reproducibility, forgotten
HF offline toggles).
