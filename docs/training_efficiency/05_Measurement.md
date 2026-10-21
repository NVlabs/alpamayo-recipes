## Summary
The core efficiency metric is the GPU-hour cost of a training step (one training iteration, i.e. one optimizer update); the `iteration time` in the trainer log is only a timer covering one part of that step (see item 2). When chasing efficiency problems it is easy to be misled by surface metrics, for example high device GPU utilization with low actual throughput. This chapter covers how to measure and evaluate training efficiency in three parts: efficiency metrics (how to measure accurately), monitoring methods (how to keep detecting anomalies), and problem localization (how to find the bottleneck from a symptom). The item numbers below refer to the sections that follow.

**Efficiency metrics**:
- `severe stall`: the share of training steps in which one rank holds up all the others (arrival-time spread across ranks > 2 s); measures the waste caused by waiting for data (see item 1)
- `wall clock`: compute the complete s/step from log timestamps (see item 2)
- `GPU-hour`: compute it from the mean `wall clock`, never the median (see item 3)
- `MFU / HFU / OFU`: look at actual compute utilization, not device utilization (see item 4)

**Monitoring methods**: continuous recording and regression monitoring during training, plus ways to measure one segment in isolation (items 6–10 are still being added)
- Per-rank telemetry and `severe stall`: record each rank's wait and compute to find the rank that slows everyone down (see item 1)
- CUDA event decomposition: split a training step into `arrival` / `forward` / `backward` / `host residual` (see item 5)
- Regression bands: periodically run a fixed recipe and set ranges for loss, training-step time, MFU and memory; alert when they are exceeded (see item 6)
- Forensics: save the scene when a non-finite gradient appears, and never add a collective to a healthy training step (see item 7)
- Benchmark harness: skip the optimizer and measure the loader and startup time on their own (see item 8)
- Per-component loss and truncation logging: log each loss term separately and keep the identity of truncated samples (see item 9)
- Profiler: offline trace, down to modules and operators (see item 10)

**Problem localization**: when you hit a specific symptom, look at the matching metrics to locate the bottleneck:

| Symptom | What it looks like | Metrics to look at |
|---|---|---|
| Training step slower than the hardware suggests, cause unknown | High device utilization (`sm%`) and normal loss, but the `wall clock` of each training step is longer than the hardware suggests | True `wall clock` / step; `arrival`, `forward`, `backward`, `host residual` (items 2 and 5; down to modules and operators in item 10) |
| One rank slows down all ranks | The mean `wall clock` per training step is far above its median; periodic multi-second training steps, where some ranks are waiting for the batch while others are already waiting for gradient synchronization | `arrival` (i.e. data wait), `severe stall`, `stall gap`, `clean step` (item 1; measure the loader alone in item 8) |
| A fixed extra cost in every training step | The mean `wall clock` per training step is close to its median (no long tail), but both are above the reference configuration: every training step reliably costs a fixed extra amount | `forward`, `backward`, `host residual` (item 5; down to operators in item 10) |
| GPU utilization near 100% but throughput low | Device utilization (`sm%`) near 100%, but `OFU` is low and power draw is close to idle | `MFU`, `HFU`, `OFU`, `HFU / OFU`, power; not device utilization (`sm%`) (item 4) |
| Reported numbers disagree with the actual GPU-hour bill | The logged `iteration time` is below the true `wall clock`; or a cost estimated from the median is below the actual `GPU-hour` bill | True `wall clock` / step, `GPU-hour` (from the mean), compared against `iteration time` (items 2 and 3) |

---

## 1. Per-rank telemetry and `severe stall`: record each rank's wait and compute to find the rank that slows everyone down

**Idea**:

![Left: one stage 2 training run with only job-level iteration time — median 1.254 s, mean 2.688 s, and the gap between them cannot be explained. Right (schematic): per-rank records separate each rank's wait for its batch (orange) from its compute time (blue), so the late-arriving rank 3 and the 2 s arrival spread used to define severe stall are directly visible.](assets/05_per_rank_step_telemetry_with_a_severe_stall_metric.png)

In synchronous data-parallel training, the duration of a training step is set by the slowest rank. A job-level step time alone cannot tell the following apart: whether one rank is slow or all ranks are slow; whether data supply is insufficient (a few ranks receive their batch late, or the whole loader lags) or compute itself is slow (an abnormal or interfered node, or a global slowdown). Each case is handled differently, so the cause has to be located first.
At every training step each rank produces one record, which rank 0 gathers. Four quantities are derived from these records:
- `instrument wall`: for each training step, the sum of a rank's `arrival` (time waiting for the batch) and `arrival-else` (time from receiving the batch to the end of the step, including H2D copy, forward, backward and gradient synchronization), maximized over ranks. This is the total time of the training step on the slowest rank (rank 3 in the figure).
- `severe stall`: a training step in which the spread of `arrival` across ranks (maximum minus minimum, e.g. rank 3 versus rank 0 in the figure) exceeds 2 s, meaning some rank is waiting for data.
- `clean step`: a training step whose maximum `arrival` is at most 0.1 s, meaning the loader caused no visible wait.
- `stall gap`: the mean `instrument wall` over all training steps minus the mean `instrument wall` over `clean step`s, i.e. the extra time caused by stalls.

The stall metric uses a difference rather than an absolute value: if one rank waits 11 s while the others wait only 0.1 s, the loader has a problem regardless of how long compute takes. The analysis should follow the rank that determines the step duration, not an average over ranks, and should report the mean, the median, the severe ratio, and the p95 / p99 of `arrival` together; these four correspond to four different failure shapes.

**Implementation**:

Each record contains: the time this rank waited for its batch (`arrival`); the time from receiving the batch to the end of the training step (including H2D copy, forward, backward and gradient synchronization, called `arrival-else`, stored in the field `fwd_bwd_time_s`); GPU-event timings of forward and backward (see item 5); sample and token counts; and device memory. Rank 0 gathers the records from all ranks and appends them to a file as one JSON-lines row per step.

The analysis looks like this:

```python
# One record per rank per step, gathered by rank 0 and written as JSON-lines; fields:
#   arrival_time_s, fwd_bwd_time_s, forward_time_s, backward_time_s,
#   sample_count, real_tokens, device_used_bytes
rows = [json.loads(l) for l in open("rank_timings.jsonl")]
sel  = rows[20:1000]                                   # drop the compile warm-up steps
wall = [max(x["arrival_time_s"] + x["fwd_bwd_time_s"] for x in r["ranks"]) for r in sel]
sev  = sum(1 for r in sel
           if max(x["arrival_time_s"] for x in r["ranks"])
            - min(x["arrival_time_s"] for x in r["ranks"]) > 2)
```

**Experiment data**: (internal data, for reference)

Setup: 2B model, 16 nodes / 128 GPUs, 1,000-step benchmark; the metric is the `instrument wall` from per-rank telemetry (steps 21~1,000), and severe uses the "arrival spread > 2 s" definition. The table shows three stage 2 configurations of the same model on the same data (a configuration is determined by model size, training stage, parallel topology, per-GPU batch and loader variant; the loader variants are described in chapter 01). Per-rank telemetry alone is enough to tell which kind of problem each one has:

| Configuration | `instrument wall` mean (s) | `instrument wall` median (s) | `severe stall` ratio | `arrival` p95 (s) | Verdict |
|---|---:|---:|---:|---:|---|
| Loader variant A: whole-sample preprocessing, raw image frames dropped after packing | 2.138 | 1.104 | 18.0% | 6.78 | Mean far above median and a high `severe stall` ratio: a long-tail latency problem; improve the loader's supply stability |
| Loader variant B: per-frame streaming preprocessing | 1.576 | 1.556 | 0.0% | 0.29 | Mean close to median, no stalls: loader supply is stable; investigate compute and other overheads next (item 5, CUDA event decomposition) |
| Reference baseline: all loader optimizations off, eager (no `torch.compile`) | 2.472 | 1.524 | 15.3% | 7.03 | Same as variant A: a long-tail latency problem (the stall signature is unchanged under eager) |

Notes:
- Tables in this chapter use the `severe stall` definition (arrival spread across ranks > 2 s). Other material may use an `absolute stall` definition (slowest rank's wait > 5 s); state which one is used.
- Limitation of the difference-based `severe stall`: when all ranks wait for data at the same time, the arrival spread is small and `severe stall` cannot detect it. For example, one experiment had a mean of 7.32 s/step but only 2.3% of training steps were classified as `severe stall`, because all ranks were waiting simultaneously. Read the mean, the median and the arrival spread together; the `absolute stall` definition (slowest rank waits > 5 s) can be used as a complement to cover this case.
- The mean `instrument wall` does not equal the mean true `wall clock`: `instrument wall` excludes the inter-step tail (the time between the previous step's optimizer end and the next step's start, including the wait for the next batch and host-side work), so it is smaller than the true `wall clock` based on trainer-log timestamps by tens of milliseconds (see item 2). In addition, the first ~20 training steps include compile and first-epoch effects and should be dropped from the statistics.

**When to use**: all training jobs; recommended on by default. The cost is one JSON row per training step; afterwards, questions like "where does the time go" can be answered by a single query over existing data.

---

## 2. `wall clock`: compute the complete s/step from log timestamps

**Idea**:

`wall clock` is the complete duration of a training step, including data wait, compute and host tail; the trainer log's `iteration time` covers only the compute part and therefore reads low. Cost accounting and before/after comparisons should use `wall clock`; `iteration time` is for diagnosis.

![Timeline of one training step: the gray bracket is the trainer log's "Iteration time", covering only compute (1.69 s); the blue bracket is the true wall clock, from one log line to the next, which also includes the data wait and host tail at both ends (1.77 s).](assets/05_true_wall_clock_per_step_from_log_timestamps.png)

The trainer prints an `iteration time` for every training step. It measures rank 0's compute: the CUDA event interval from gradient zeroing to the end of the optimizer step. It does **not** include the time spent waiting for the batch before the step, nor the host work after the step (logging, control-flow checks, checkpoint saving). It therefore underestimates the true duration of a training step: in the 2B stage 2 experiment (the shortest training step) by about 13%. The `instrument wall` of item 1 likewise excludes host work after the step and has the same underestimate. `wall clock` takes the time difference between two rank-0 log lines divided by the number of training steps between them. The result is everything a training step spends: the data wait in the figure, compute (forward / backward, optimizer) and the host tail.

**Implementation**:

A long training job is usually a chain of scheduler allocations: when an allocation hits its time limit (or is preempted, or a node fails), the scheduler requeues the job, which resumes from the latest checkpoint (autoresume). Each allocation is called a segment (also a job attempt). Between segments there is queueing, restart and checkpoint-loading time that is not training-step time, so `wall clock` must be computed per segment:
1. Read the rank-0 log of every autoresume segment;
2. Match the lines that carry a timestamp and a training step number (for example `[rank0] <timestamp> ... Step: N/total, ... Iteration time`);
3. Within each segment, keep only the lines inside the chosen window and drop the first 20 lines of that segment (compile warm-up and cache fill after every start or restart);
4. For each segment take the timestamp difference (seconds) and step-number difference (steps) between the last and the first line, then divide the sum of seconds over all segments by the sum of steps to get the training-step time.

Follow these rules when implementing it:
- **Never subtract timestamps across segments**: the gap between segments (queueing, restart, checkpoint loading and warm-up) is not training-step time.
- **Count rerun training steps in both segments**: after a job resumes it retrains from the latest checkpoint. For example, if it is requeued at step 9,400 and resumes from the step-9,000 checkpoint, steps 9,001~9,400 run once in each segment. The time spent on the rerun is real training cost and must be counted; counting both runs makes the computed training-step time reflect the actual training rate.
- **State the statistics window**: the whole run, or the last 1,000 steps. Everything that happens inside the window (checkpointing, evaluation pauses, etc.) is counted; a window that is too short makes the result noisy, so use at least a few hundred training steps.
- **Use the same definition for the baseline and the new result**: when quoting the gain of a change over a baseline, recompute the baseline from the true `wall clock`; for historical data whose logs are gone and cannot be recomputed, state the original definition. If the baseline comes from the trainer log's `iteration time`, the resulting speedup is only a lower bound of the true gain.
- **Regression gates versus cost accounting**: a regression gate (see item 6) may set its band on the median or on `iteration time` for stability, but cost accounting should still use the mean `wall clock`.

The script below is adapted from the implementation used internally and follows the four steps above. Log formats differ between trainers, so adjust the regular expression to your actual logs.

```python
import datetime, glob, re

# Log example: [rank0] 2026-09-09 22:35:12,345 ... Step: 21/1000, ... Iteration time: 1.53s
pat = re.compile(r"\[rank0\].*?(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+).*Step: (\d+)/\d+")

job_dir = "..."        # job directory; each autoresume segment has its own rank-0 log under it
lo, hi = 1, 10**9      # statistics window: the whole run, or e.g. the last 1,000 steps

tot_s = tot_n = 0
for f in sorted(glob.glob(f"{job_dir}/**/rank0.log", recursive=True)):   # one per segment
    rows = [(datetime.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S,%f"), int(m[2]))
            for m in map(pat.search, open(f, errors="replace")) if m and lo <= int(m[2]) <= hi]
    rows = rows[20:]                                        # drop this segment's warm-up
    if len(rows) > 1:
        tot_s += (rows[-1][0] - rows[0][0]).total_seconds() # subtract only within a segment
        tot_n += rows[-1][1] - rows[0][1]
step_time = tot_s / tot_n                                   # sum of seconds / sum of steps (s/step)
```

**Experiment data**: (internal data, for reference)

Two definitions on the same run (2B model, stage 1, 16 nodes, 100k-step reference run): `wall clock` includes the data wait and host tail that `iteration time` leaves out, and is about 0.08 s (about 5%) higher.

| Definition | Mean |
|---|---:|
| trainer `iteration time` | 1.69 s |
| True `wall clock` / training step | 1.77 s |

**When to use**:

`wall clock` applies to any training-step time that serves as the final reported figure, and to comparisons before and after a change or across configurations. These cases require complete timing that does not omit data wait or host tail. The trainer's `iteration time` and per-rank telemetry can be used alongside for diagnosis.

---

## 3. `GPU-hour`: compute it from the mean `wall clock`, never the median

**Idea**:

![On the same scale, two training-step-time bars for a stage 2 run (iteration time as in the trainer log): the median is 1.254 s (blue); the mean is 2.688 s, made of the same blue part plus an orange part, which is the stall time the median discards. A cost estimated from the median is less than half of that estimated from the mean.](assets/05_gpu_hours_from_the_mean_never_the_median.png)

`GPU-hour` is the cost-accounting metric for training. It accumulates per training step and equals the mean `wall clock` of a training step × number of steps × number of GPUs; it lets a mean `wall clock` measured over a short window be turned into the cost of a target training length (for example 100k steps).

The key is to use the **mean** `wall clock`. Stalls raise the mean `wall clock` and the median discards that part: a run with severe stalls can give a good-looking median while its real cost is up to twice the cost the median implies. Cost accounting should therefore use the mean `wall clock`, not the median.

**Implementation**:

The formulas (the training-step time is the mean `wall clock`; the 100,000 in the formulas is the number of training steps used for the cost estimate):

```python
gpu_hours_per_100k = mean_wall_s_per_step * 100_000 * num_gpus / 3600
e2e_days_per_100k  = mean_wall_s_per_step * 100_000 / 86_400
```
State the statistics window and the number of steps behind the mean: a mean over a short window is sensitive to a single stall, so average over at least a few hundred training steps.

**Experiment data**: (internal data, for reference)

Setup: for a 2B model, `GPU-hour` and end-to-end time per 100k steps computed by the formulas:

| Configuration | Mean `wall clock` (s/step) | GPUs | `GPU-hour` (100k steps, computed) | End-to-end (100k steps, computed) |
|---|---:|---:|---:|---:|
| A: before optimization | 2.402 | 128 | 8,541 | 2.78 days |
| B: with optimizations stacked | 0.737 | 128 | 2,619 | 0.85 days |

**When to use**: any estimate of GPU-hours, cost or end-to-end time, and cost comparisons between topologies.

---

## 4. `MFU / HFU / OFU`: look at actual compute utilization, not device utilization

**Idea**:

![Dumbbell chart for three runs: device utilization is 62%~100% while OFU is only 0~14%; the pure-spin run reads about 100% utilization at idle power with OFU about 0%.](assets/05_mfu_hfu_and_ofu_versus_device_utilization.png)

The "utilization" reported by device monitoring (`sm%` in `nvidia-smi`) is **occupancy**: the fraction of the sampling period during which any kernel was resident on the device. It does not reflect how much of the hardware peak the resident work uses; a spin kernel (a kernel that does no useful compute and only occupies the GPU, such as `torch.cuda._sleep`) reads about 100% even at idle power. The following three metrics tie "actual work efficiency" to "hardware peak capability":
- `MFU` (model FLOP utilization): the useful floating-point work the model needs per training step, divided by time and the hardware peak. The work is estimated with the 6N convention (forward + backward cost about 6×N operations per token, N being the parameter count). The hardware peak comes in a sparse and a dense flavor; only an `MFU` computed against the dense peak is comparable with MFU in the published literature.
- `HFU` (hardware FLOP utilization): the same estimate with the 8N convention, which additionally counts the forward recomputed by activation checkpointing, i.e. the work the hardware actually executes.
- `OFU` (operational FLOP utilization): a **measured value, not an estimate**. A side process on each node samples the device monitor's tensor-core activity and clock; OFU = activity × (current clock / max clock).

**Implementation**:

- `MFU` / `HFU`: count the tokens processed per training step, estimate the work as 6N (`MFU`) or 8N (`HFU`) with N the parameter count, then divide by the training-step time and the hardware peak FLOPS. When the model consists of several components (for example a vision encoder, a language model, an expert), estimate each component from its own parameter count and token count and add them up. Some training frameworks already provide such an interface that can be called directly;
- `OFU`: start a side sampling process with `nvidia-smi dmon --gpm-metrics 5`, read tensor-core activity and clock, and compute `OFU` = activity × (current clock / max clock);

Illustrative code:

```python
def flop_utilization(params, tokens_per_step, step_time_s, peak_flops_per_gpu, num_gpus, factor):
    # factor = 6: MFU; factor = 8: HFU (also counts the forward recomputed by activation checkpointing)
    return factor * params * tokens_per_step / (step_time_s * peak_flops_per_gpu * num_gpus)
```

**Experiment data**: (internal data, for reference)

Three runs are compared (2B model, 1 node, `iteration time` definition): spin only (training compute skipped, only spin kernels run), real training (no spin), and real training plus continuous spin (spin kernels run throughout training). Utilization (`sm%`) is not actual compute utilization (`OFU`): spinning pushes the utilization reading close to 100% but `OFU` does not rise, and `iteration time` actually gets slower (about 5% worse on 1 node, about 14% worse on 4 nodes).

| Run | Utilization (`sm%`) | `OFU` | `iteration time` | Power |
|---|---:|---:|---:|---:|
| Spin only (compute skipped) | about 100% | about 0% | n/a | about 123 W (idle) |
| Real training | 62% / 87% (median) | 14.1% | 1.46 s | 415 W |
| Real training + continuous spin | about 98% | 13.5% | 1.54 s | 389 W |

**When to use**: comparing configurations across model sizes or topologies, and judging whether a GPU-bound configuration still has headroom. Use `OFU`, `MFU` and power to decide, never device utilization; spinning should only be used during warm-up.

**Extension: keeping a job from being killed by an idle-GPU reaper (the keep-alive used in Alpamayo training)**

Some clusters terminate jobs whose GPUs show no activity for a long time (an idle-GPU reaper), while the first minutes after a job starts (initialization, checkpoint loading, compilation, ...) naturally have near-zero utilization, so a job can be killed before its first training step. Alpamayo training uses a keep-alive to avoid this: it keeps the GPU "active" during warm-up. It is not needed on clusters without such a policy.
- **How**: during warm-up, spin on a dedicated stream with `torch.cuda._sleep` pulses of about 50 ms; switch to a dormant state at the first training step; stop at the last step.
- **It must go dormant**: a resident spin kernel cannot be preempted by an all-reduce, so continuous spinning slows the training step, and the cost grows with node count. Keep-alive only satisfies the reaper's check and must not be used as evidence of efficiency.

Validation (2B model, 1 node × 8 GPUs): with keep-alive on, the warm-up utilization reading rises from 16% (median 0%) to 88% (median 100%), power is unchanged, and the steady-state `iteration time` is unchanged. This comparison only validates the mechanism and cost of keep-alive; it does not show a performance gain.

| Configuration | Warm-up `sm%` mean / median | Warm-up power | Steady-state `iteration time` mean (s) |
|---|---:|---:|---:|
| keep-alive on (default) | 88% / 100% | 132 W | 1.600 |
| keep-alive off (reference) | 16% / 0% | 135 W | 1.593 |

---

## 5. CUDA event decomposition: split a training step into `arrival` / `forward` / `backward` / `host residual`

**Idea**:

![Three training-step timelines on the same scale (seconds): with telemetry alone, the slower configuration is one 1.449 s block of arrival-else; CUDA events split it into forward 0.835, backward 0.478 and host residual 0.110 s, versus 0.553, 0.321 and 0.090 s for the normal configuration. An orange marker spans the +0.28 s difference in forward, attributed to launch gaps caused by insufficient CPU thread resources, while host residual is essentially unchanged.](assets/05_cuda_event_decomposition_of_the_step.png)

The telemetry of item 1 splits a training step into the wait (`arrival`) and `arrival-else`. When `arrival-else` itself gets longer, the next question is where the time goes: forward compute, backward plus gradient all-reduce, or the host work in between.

Approach: with per-rank telemetry enabled, the trainer wraps forward and backward with CUDA events (`torch.cuda.Event(enable_timing=True)`, timestamps taken by the GPU on its own stream). The time between two events **includes the gaps where the GPU waits for the CPU to submit the next kernel**, i.e. event interval = kernel time + launch gaps.

On the rank that determines the step duration, a training step then splits into four segments: `arrival` (loader telemetry), `forward` and `backward` (events), and `host residual` (`arrival-else` − forward − backward, i.e. data movement, the optimizer's host-side cost, and the Python code between stages). How to read it: a longer `host residual` means the host code between stages got slower; a longer `forward` or `backward` while kernels and data have not changed can only be launch gaps, i.e. the CPU thread that submits kernels to the GPU is being interfered with.

**Implementation**:

The data comes from the per-rank records of item 1, and one query is enough (illustrative):

```python
# For each step take the rank with the largest fwd_bwd_time_s, read its forward / backward, compute the residual
for r in rows[20:1000]:
    x = max(r["ranks"], key=lambda e: e["fwd_bwd_time_s"])
    residual = x["fwd_bwd_time_s"] - x["forward_time_s"] - x["backward_time_s"]
# report the median of each segment
```

Place one pair of events around forward and one pair around `loss.backward()`. Follow these rules when implementing it:
- **An event interval contains both kernel time and launch gaps**: so "forward got slower" does not mean "the model got slower". Fix the kernels first (bit-identical inputs, same compilation); any remaining difference is then a launch gap.
- **Backward absorbs synchronization waits**: a rank waiting in a blocking gradient collective for a stuck peer records that wait as backward time (for example, in a configuration with 4 workers the `backward` median was 0.920 s versus 0.398 s in the normal configuration). Only decompose configurations whose `arrival` curve is normal.
- **The event overhead is negligible, but it exists only when per-rank telemetry is on**: it must be enabled at benchmark time; when it is not, the heavier offline alternative is a profiler trace (item 10).

**Experiment data**: (internal data, for reference)

Background: while optimizing the data loader, we found that besides `arrival` (waiting for the batch), `arrival-else` also got longer (in the configuration using per-frame streaming preprocessing, `arrival-else` per training step increased markedly), and CUDA events were needed to find out which segment grew.

Purpose: find which segment holds the bottleneck from the decomposition, and thereby decide the optimization direction.

Setup: 2B model, stage 2, 16 nodes, 1,000-step benchmark; each segment is the median on the rank that determines the step duration (per-rank telemetry, steps 21~1,000), in seconds. The two loader variants are the same as in item 1:

| Configuration | `arrival-else` | `forward` | `backward` | `host residual` |
|---|---:|---:|---:|---:|
| Loader variant A: whole-sample preprocessing, raw image frames dropped after packing | 1.017 | 0.553 | 0.321 | 0.090 |
| Loader variant B: per-frame streaming preprocessing | 1.449 | 0.835 | 0.478 | 0.110 |

Notes:
- Each segment in the table is a median, so the segments do not necessarily add up to `arrival-else` (for example in the first row 1.017 − 0.553 − 0.321 ≠ 0.090).
- Variant B's `arrival-else` is about 0.43 s longer than variant A's, while `host residual` changes by only about 0.02 s, so almost all the added cost falls inside the GPU-timed intervals. `forward` grows by about **0.28 s** (0.553 → 0.835) even though the batches are bit-identical and the compiled kernels are the same; our inference is a forward launch-gap problem: the trainer's CPU thread is not getting enough CPU resources.
- A CPU-partitioning fix on the data-loading side supports this: in a 10-worker configuration, giving the trainer dedicated physical cores lowered the `forward` median from 0.766 s to 0.589 s. The two preprocessing recipes differ, so this is not a strict controlled comparison.

**When to use**: when the per-rank telemetry of item 1 shows a normal `arrival` (no obvious data-wait problem) but the training step is still slow and `arrival-else` is long, use CUDA events to split `arrival-else` into `forward`, `backward` and `host residual`, locate the segment that holds the bottleneck, and then decide the optimization direction. Before forming a hypothesis about a training-step-time regression, decompose first and hypothesize afterwards.

---

## 6. Regression bands: periodically run a fixed recipe and set ranges for loss, training-step time, MFU and memory; alert when they are exceeded

To be added.

---

## 7. Forensics: save the scene when a non-finite gradient appears, and never add a collective to a healthy training step

To be added.

---

## 8. Benchmark harness: skip the optimizer and measure the loader and startup time on their own

To be added.

---

## 9. Per-component loss and truncation logging: log each loss term separately and keep the identity of truncated samples

To be added.

---

## 10. Profiler: offline trace, down to modules and operators

To be added.
