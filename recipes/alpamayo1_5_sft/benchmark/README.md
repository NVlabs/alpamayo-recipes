# Alpamayo 1.5 SFT Performance Benchmarks

This benchmark measures the effect of the data-loading and compute optimizations in the
Alpamayo 1.5 Stage-1 navigation SFT recipe. It runs four incremental configurations and records
the wall-clock interval between training steps after warmup.

## Prerequisites

Follow the installation, PAI dataset, and checkpoint preparation instructions in the
[Alpamayo 1.5 SFT README](../README.md). The benchmark requires:

- an activated Alpamayo 1.5 SFT environment;
- an A1-format Alpamayo 1.5 checkpoint;
- the PAI dataset chunks used by the navigation samples;
- a navigation annotations JSON file; and
- one or more CUDA GPUs.

The default launch uses eight GPUs. Set `--nproc-per-node` to the number available on the host.

## Run the Benchmark

From the repository root, pass the required inputs directly:

```bash
python recipes/alpamayo1_5_sft/benchmark/run_benchmarks.py \
  --checkpoint /path/to/Alpamayo-1.5-10B-A1-format \
  --pai-dir /path/to/pai_dataset \
  --nav-annotations /path/to/nav_demo_samples.json \
  --nproc-per-node 8
```

The paths can instead be supplied with environment variables:

```bash
export ALPAMAYO15_SFT_CHECKPOINT=/path/to/Alpamayo-1.5-10B-A1-format
export ALPAMAYO_PAI_DIR=/path/to/pai_dataset
export ALPAMAYO15_NAV_ANNOTATIONS=/path/to/nav_demo_samples.json
python recipes/alpamayo1_5_sft/benchmark/run_benchmarks.py
```

By default, each variant runs for 20 steps and the report averages steps 5 through 20. Use
`--max-steps`, `--stable-start-step`, and `--stable-end-step` to change that window. To run a
subset or reuse completed results:

```bash
# Run one configuration.
python recipes/alpamayo1_5_sft/benchmark/run_benchmarks.py \
  --checkpoint /path/to/checkpoint \
  --pai-dir /path/to/pai_dataset \
  --nav-annotations /path/to/nav_demo_samples.json \
  --only zip_collate_cache

# Keep existing per-variant JSON results and run only missing configurations.
python recipes/alpamayo1_5_sft/benchmark/run_benchmarks.py \
  --checkpoint /path/to/checkpoint \
  --pai-dir /path/to/pai_dataset \
  --nav-annotations /path/to/nav_demo_samples.json \
  --skip-existing
```

Run `python recipes/alpamayo1_5_sft/benchmark/run_benchmarks.py --help` for all options.

## Benchmark Variants

The variants are cumulative and run in this order:

| Variant | Additional settings |
| --- | --- |
| `baseline` | No data-loader workers, recipe caches, TF32, or cuDNN benchmarking |
| `dataloader_workers` | 8 persistent workers with a prefetch factor of 4 |
| `zip_collate_cache` | ZIP archive and collate-processor caches |
| `tf32_cudnn_benchmark` | TF32 and cuDNN algorithm benchmarking |

## Read the Results

Generated files are written under `recipes/alpamayo1_5_sft/benchmark/`:

- `performance_optimization_report.md` contains the comparison table and run metadata.
- `results/summary.json` contains all raw results and calculated summary rows.
- `results/<variant>.json` contains step timings for one variant.
- `results/<variant>.log` contains the training output.
- `results/<variant>.command.json` records the exact launch command.

The primary metric is **stable average step time**; lower is better. **Speedup vs baseline** is
the baseline step time divided by the current variant's step time. **Incremental speedup** compares
the current variant with the preceding row. For example, `1.20x` means the current configuration
completed steps 20% faster. Compare runs only when the hardware, data, checkpoint, step window,
and other training settings are identical.

The timing interval includes data loading, collation, forward and backward passes, and the optimizer
update. Short runs can be noisy, so increase `--max-steps` and select a later stable window when
collecting results for decisions.
