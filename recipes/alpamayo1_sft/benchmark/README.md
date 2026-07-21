# Alpamayo 1 SFT Performance Benchmarks

This benchmark measures the effect of the data-loading and compute optimizations in the Alpamayo 1
Stage-1 SFT recipe. It runs the same incremental optimization matrix both without and with
chain-of-causality (CoC) reasoning labels, and records wall-clock training step intervals after
warmup.

## Prerequisites

Follow the installation, dataset, and checkpoint preparation instructions in the
[Alpamayo 1 SFT README](../README.md). The benchmark requires:

- an Alpamayo 1 SFT Python environment;
- an Alpamayo-R1 checkpoint;
- the required PAI dataset chunks, including `reasoning/ood_reasoning.parquet` and
  `clip_index_reasoning_mini.parquet`; and
- one or more CUDA GPUs.

The default launch uses eight GPUs. Set `--nproc-per-node` to the number available on the host.

## Run the Benchmark

From the repository root, pass the required inputs directly:

```bash
python recipes/alpamayo1_sft/benchmark/run_benchmarks.py \
  --python recipes/alpamayo1_sft/a1_sft/bin/python \
  --checkpoint /path/to/Alpamayo-R1 \
  --pai-dir /path/to/pai_dataset \
  --nproc-per-node 8
```

When the recipe environment is already active, omit `--python`. The paths can instead be supplied
with environment variables:

```bash
export ALPAMAYO_BENCH_PYTHON=/path/to/recipe/a1_sft/bin/python
export ALPAMAYO_R1_CHECKPOINT=/path/to/Alpamayo-R1
export ALPAMAYO_PAI_DIR=/path/to/pai_dataset
python recipes/alpamayo1_sft/benchmark/run_benchmarks.py
```

By default, each case runs for 20 steps and the report averages steps 5 through 20. Use
`--max-steps`, `--stable-start-step`, and `--stable-end-step` to change that window. To run a
subset or reuse completed results:

```bash
# Run only the CoC cases for two configurations.
python recipes/alpamayo1_sft/benchmark/run_benchmarks.py \
  --checkpoint /path/to/checkpoint \
  --pai-dir /path/to/pai_dataset \
  --only-group coc \
  --only-variant baseline zip_collate_cache

# Keep existing per-case JSON results and run only missing cases.
python recipes/alpamayo1_sft/benchmark/run_benchmarks.py \
  --checkpoint /path/to/checkpoint \
  --pai-dir /path/to/pai_dataset \
  --skip-existing
```

Run `python recipes/alpamayo1_sft/benchmark/run_benchmarks.py --help` for all options.

## Benchmark Matrix

Each variant runs for both the `no_coc` and `coc` groups. The variants are cumulative and run in
this order:

| Variant | Additional settings |
| --- | --- |
| `baseline` | No data-loader workers, recipe caches, TF32, or cuDNN benchmarking |
| `dataloader_workers` | 8 persistent workers with a prefetch factor of 4 |
| `zip_collate_cache` | ZIP archive and collate-processor caches |
| `tf32_cudnn_benchmark` | TF32 and cuDNN algorithm benchmarking |

## Read the Results

Generated files are written under `recipes/alpamayo1_sft/benchmark/`:

- `performance_optimization_report.md` contains the comparison table and run metadata.
- `results/summary.json` contains all raw results and calculated summary rows.
- `results/<group>_<variant>.json` contains step timings for one case.
- `results/<group>_<variant>.log` contains the training output.
- `results/<group>_<variant>.command.json` records the exact launch command.

The primary metric is **stable average step time**; lower is better. **Speedup vs group baseline**
is the `baseline` step time for the same CoC group divided by the current variant's step time.
**Incremental speedup** compares the current variant with the preceding variant in that group. For
example, `1.20x` means the current configuration completed steps 20% faster. Compare runs only when
the hardware, data, checkpoint, step window, and other training settings are identical.

The timing interval includes data loading, collation, forward and backward passes, and the optimizer
update. Short runs can be noisy, so increase `--max-steps` and select a later stable window when
collecting results for decisions.
