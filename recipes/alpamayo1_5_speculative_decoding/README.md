# Alpamayo R1.5 Speculative Decoding

After [Alpamayo 1.5 SFT](../alpamayo1_5_sft/), use your checkpoint as a frozen
target to train a **Domino** or **DFlash2** draft with SpecForge. The recipe reads
PhysicalAI directly, caches greedy CoC tokens, and computes target features online.
Only the draft is trained and saved; no saved VLM features or persistent KV cache.

Targets can be Alpamayo R1.5 10B or Distilled 2B checkpoints, including SFT checkpoints
with `vlm.*` weights. Inputs are 16 images and ego history, **without navigation**.
The target must support CoC generation; VQA-only or trajectory-only SFT may not.

## Installation

Use a separate Python 3.12 environment from SFT:

```bash
cd recipes/alpamayo1_5_speculative_decoding
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync --active --frozen
```

Public Alpamayo and SpecForge sources are pinned in `uv.lock`; no separate checkout
or SGLang server is needed. This standalone package uses public inference modules.
Its dependency overrides use SpecForge's draft/loss modules with PyTorch 2.8 and
Transformers 4.57.1, tested by this recipe's GPU smoke—not the upstream SGLang runtime.

## Data and checkpoint

Reuse your SFT checkpoint, local base-VLM config/tokenizer, and Qwen3-VL processor.
Reuse the PhysicalAI download: `features.csv`, `clip_index.parquet`, four camera ZIPs
per selected chunk, and egomotion ZIPs. Nothing is downloaded implicitly.

Pass your SFT annotation JSON as `--manifest`, or JSONL with one anchor per line:

```json
{"clip_id": "your-clip-id", "t0_relative": 5100000}
```

`t0_relative` (or `t0_us`) is clip-relative microseconds, with `1.6 s < t <= 13.6 s`.
Alpamayo's public loader supplies four frames per camera, camera-major IDs
`0, 1, 2, 6`, plus 16 ego-history samples at 10 Hz. Source answers, navigation,
and future trajectory labels are not target inputs. Keep clip/session splits
disjoint between training and evaluation.

Image preprocessing matches SFT: convert RGB to float32 `[0,1]` before resizing,
with processor rescaling disabled. Regenerate caches made with uint8 resizing;
the target settings distinguish the two input conventions.

## Cache, train and evaluate

Run in Bash after replacing the paths. Each GPU holds a target and draft replica;
use 80 GB GPUs for the 10B target.

```bash
GPUS=8
TARGET=(--checkpoint /path/to/your-sft-checkpoint --base-vlm /path/to/base-vlm --processor /path/to/processor)
DATA_ROOT=/path/to/pai_dataset
TRAIN=/path/to/train_annotations.json
EVAL=/path/to/heldout_annotations.json
CACHE=/path/to/cache/greedy-coc
RUN=/path/to/experiments/domino
METHOD=domino  # Or dflash2; use a separate RUN directory.

torchrun --standalone --nproc_per_node="${GPUS}" -m alpamayo1_5_speculative_decoding.cli cache \
  "${TARGET[@]}" --manifest "${TRAIN}" --data-root "${DATA_ROOT}" --output "${CACHE}"

torchrun --standalone --nproc_per_node="${GPUS}" -m alpamayo1_5_speculative_decoding.cli train \
  "${TARGET[@]}" --manifest "${TRAIN}" --data-root "${DATA_ROOT}" \
  --cache "${CACHE}" --output "${RUN}" --method "${METHOD}" --steps 3000

torchrun --standalone --nproc_per_node="${GPUS}" -m alpamayo1_5_speculative_decoding.cli evaluate \
  "${TARGET[@]}" --manifest "${EVAL}" --data-root "${DATA_ROOT}" \
  --draft "${RUN}/step_3000" --output "${RUN}/eval-final"
```

- Caching is greedy, stopping at `<|traj_future_start|>`. Per-rank JSONLs record
  complete/incomplete answers and reasons; `manifest.json` stores target/data
  settings and counts. Training checks coverage and excludes incomplete answers.
- Defaults: five draft layers, block 16, microbatch 1, global batch 512,
  AdamW `6e-4`, 4% warmup then cosine decay, FP32 draft weights with BF16 autocast.
  SpecForge selects target feature taps from draft depth; `--target-layers` overrides.
- SDPA is the default. For FlashAttention 2, install a compatible build and append
  `--target-attention flash_attention_2` to `TARGET` before caching.
- Video decoding and processing remain serial per rank. Benchmark throughput before
  a large run; faster storage does not remove the target-prefill compute cost.

Rank 0 appends each printed training record to `${RUN}/metrics.jsonl`, including
loss, teacher-forced metrics, scheduler LR, pre-clipping gradient norm, global
samples/second (excluding logging/checkpoint writes), and rank-0 peak allocated GPU
memory in GB. Resumes append; replayed steps after the last checkpoint may appear again.

For optional W&B, install with `uv sync --active --frozen --extra wandb`, then add
`--wandb-project my-project` to the training command. `--wandb-run-name` defaults
to the output directory name. Offline example (using the variables above):

```bash
WANDB_MODE=offline torchrun --standalone --nproc_per_node="${GPUS}" -m alpamayo1_5_speculative_decoding.cli train "${TARGET[@]}" --manifest "${TRAIN}" --data-root "${DATA_ROOT}" --cache "${CACHE}" --output "${RUN}" --method "${METHOD}" --steps 3000 --wandb-project my-project
```

## Resume and outputs

Cache: repeat with `--resume` and the same samples, inputs and GPU/shard count.
Training: repeat with `--resume-from-checkpoint "${RUN}/step_100"`. The checkpoint's
`run.json` must match the model/data settings, method, total steps, global batch,
GPU count and seed; mismatches name the differing key. Data order and anchor-sampling
seeds depend on the absolute sample offset; no per-rank RNG files are needed.
GPU kernels are not forced into deterministic mode, so bitwise equality is not guaranteed.

Checkpoints contain the draft, optimizer/scheduler state and run settings.
Each `step_N.tmp` directory becomes `step_N` after saving; `latest.json` points to it.
To branch from an older checkpoint when newer steps exist, use a fresh output.
Do not run concurrent writers into one directory.

Evaluation always reports greedy agreement and incomplete outputs in
`metrics.json` and per-rank proposal/commit JSONLs; mismatches do not fail the run.
Use a fresh evaluation output and increase `--max-new-tokens` for long held-out CoCs.

The small target identity and fused-prompt/grid hash detect many wiring mismatches,
but do not detect replaced weights at the same path or changed image pixels.
Keep checkpoints and datasets immutable. Existing caches from the earlier
full-hash format need regeneration; they are not silently converted.

The draft uses `<|answer_end|>` as its mask without resizing the target vocabulary;
this token must be absent from prompts and CoCs. This is not a VQA training recipe.
`acceptance_length` is matched draft prefix + 1 before stop/length trimming;
`emitted_tokens_per_verification` counts tokens actually emitted, excluding prefill.
Domino proposes 16 tokens and DFlash2 15 at block size 16.
These PyTorch timings are **not a device speedup benchmark**.
