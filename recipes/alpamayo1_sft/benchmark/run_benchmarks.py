# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Alpamayo-1 SFT data-pipeline benchmarks with and without CoC."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


RECIPE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RECIPE_DIR.parents[1]
RESULTS_DIR = RECIPE_DIR / "benchmark" / "results"
REPORT_PATH = RECIPE_DIR / "benchmark" / "performance_optimization_report.md"
DEFAULT_DEEPSPEED = RECIPE_DIR / "configs" / "deepspeed" / "zero2.json"
BENCH_CHUNKS = "[214,224,276,317,420,727,728,968,982,1519,1657,1984,2277,2368,2372,2447,2599,2634,2868]"


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


@dataclass(frozen=True)
class TaskGroup:
    name: str
    description: str
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: tuple[str, ...]


TASK_GROUPS = (
    TaskGroup(
        name="no_coc",
        description="Default trajectory SFT without chain-of-causality reasoning labels.",
        overrides=(),
    ),
    TaskGroup(
        name="coc",
        description="Trajectory SFT with CoC reasoning labels enabled.",
        overrides=(
            "data.train_dataset.vla_preprocess_args.components_order=[image,traj_history,prompt,cot,traj_future]",
            "data.train_dataset.vla_preprocess_args.components_prompt=[cot,traj_future]",
            "data.train_dataset.vla_preprocess_args.label_components=[cot,traj_future]",
            "data.val_dataset.vla_preprocess_args.components_order=[image,traj_history,prompt,cot,traj_future]",
            "data.val_dataset.vla_preprocess_args.components_prompt=[cot,traj_future]",
            "data.val_dataset.vla_preprocess_args.label_components=[cot,traj_future]",
            "+data.train_dataset.reasoning_metadata=reasoning/ood_reasoning.parquet",
            "+data.val_dataset.reasoning_metadata=reasoning/ood_reasoning.parquet",
            "+data.train_dataset.clip_index_metadata=clip_index_reasoning_mini.parquet",
            "+data.val_dataset.clip_index_metadata=clip_index_reasoning_mini.parquet",
            "data.train_dataset.use_default_keyframe=false",
            "data.val_dataset.use_default_keyframe=false",
        ),
    ),
)


VARIANTS = (
    Variant(
        name="baseline",
        description="Baseline without dataloader workers or recipe performance caches.",
        overrides=(
            "trainer.dataloader_num_workers=0",
            "trainer.dataloader_persistent_workers=false",
            "trainer.dataloader_prefetch_factor=null",
            "performance.zip_cache=false",
            "performance.collate_cache=false",
            "performance.tf32=false",
            "performance.cudnn_benchmark=false",
        ),
    ),
    Variant(
        name="dataloader_workers",
        description=(
            "Enable dataloader_num_workers=8, dataloader_persistent_workers=true, "
            "and dataloader_prefetch_factor=4."
        ),
        overrides=(
            "trainer.dataloader_num_workers=8",
            "trainer.dataloader_persistent_workers=true",
            "trainer.dataloader_prefetch_factor=4",
            "performance.zip_cache=false",
            "performance.collate_cache=false",
            "performance.tf32=false",
            "performance.cudnn_benchmark=false",
        ),
    ),
    Variant(
        name="zip_collate_cache",
        description="Add zip_cache=true and collate_cache=true on top of dataloader workers.",
        overrides=(
            "trainer.dataloader_num_workers=8",
            "trainer.dataloader_persistent_workers=true",
            "trainer.dataloader_prefetch_factor=4",
            "performance.zip_cache=true",
            "performance.collate_cache=true",
            "performance.tf32=false",
            "performance.cudnn_benchmark=false",
        ),
    ),
    Variant(
        name="tf32_cudnn_benchmark",
        description="Add tf32=true and cudnn_benchmark=true on top of data-pipeline caches.",
        overrides=(
            "trainer.dataloader_num_workers=8",
            "trainer.dataloader_persistent_workers=true",
            "trainer.dataloader_prefetch_factor=4",
            "performance.zip_cache=true",
            "performance.collate_cache=true",
            "performance.tf32=true",
            "performance.cudnn_benchmark=true",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_checkpoint = env_path("ALPAMAYO_R1_CHECKPOINT")
    default_pai_dir = env_path("ALPAMAYO_PAI_DIR")
    default_python = env_path("ALPAMAYO_BENCH_PYTHON") or Path(sys.executable)
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python,
        help=(
            "Python executable from the uv environment. Defaults to the current "
            "interpreter, or ALPAMAYO_BENCH_PYTHON when set."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        required=default_checkpoint is None,
        help="Alpamayo-R1 checkpoint. Can also be set via ALPAMAYO_R1_CHECKPOINT.",
    )
    parser.add_argument(
        "--pai-dir",
        type=Path,
        default=default_pai_dir,
        required=default_pai_dir is None,
        help="PAI dataset root. Can also be set via ALPAMAYO_PAI_DIR.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--stable-start-step", type=int, default=5)
    parser.add_argument("--stable-end-step", type=int, default=20)
    parser.add_argument("--master-port", type=int, default=29625)
    parser.add_argument("--only-group", choices=[group.name for group in TASK_GROUPS], nargs="*")
    parser.add_argument("--only-variant", choices=[variant.name for variant in VARIANTS], nargs="*")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    required_paths = {
        "python": args.python,
        "checkpoint": args.checkpoint,
        "PAI dataset": args.pai_dir,
        "DeepSpeed config": DEFAULT_DEEPSPEED,
        "reasoning metadata": args.pai_dir / "reasoning" / "ood_reasoning.parquet",
        "reasoning clip index": args.pai_dir / "clip_index_reasoning_mini.parquet",
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required benchmark inputs:\n" + "\n".join(missing))


def base_overrides(
    args: argparse.Namespace,
    task_group: TaskGroup,
    variant: Variant,
    step_json: Path,
) -> list[str]:
    run_name = f"{task_group.name}_{variant.name}"
    output_dir = RESULTS_DIR / "outputs" / run_name
    return [
        "--config-path",
        "pkg://alpamayo1_sft/configs",
        "--config-name",
        "sft_stage1",
        f"model.checkpoint_path={args.checkpoint}",
        f"data.train_dataset.local_dir={args.pai_dir}",
        f"data.val_dataset.local_dir={args.pai_dir}",
        f"data.train_dataset.chunk_ids={BENCH_CHUNKS}",
        "data.val_dataset.chunk_ids=[2868]",
        f"paths.output_dir={output_dir}",
        f"trainer.output_dir={output_dir}",
        f"trainer.deepspeed={DEFAULT_DEEPSPEED}",
        f"+trainer.max_steps={args.max_steps}",
        "+trainer.save_strategy=no",
        "+trainer.eval_strategy=no",
        "trainer.logging_steps=1",
        "trainer.report_to=none",
        "+callbacks.step_timer._target_=alpamayo1_sft.performance.step_time_callback.StepTimeCallback",
        f"+callbacks.step_timer.output_path={step_json}",
        f"+callbacks.step_timer.stable_start_step={args.stable_start_step}",
        f"+callbacks.step_timer.stable_end_step={args.stable_end_step}",
    ]


def run_case(args: argparse.Namespace, task_group: TaskGroup, variant: Variant) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"{task_group.name}_{variant.name}"
    step_json = RESULTS_DIR / f"{run_name}.json"
    run_log = RESULTS_DIR / f"{run_name}.log"
    command_json = RESULTS_DIR / f"{run_name}.command.json"

    if args.skip_existing and step_json.exists():
        with step_json.open(encoding="utf-8") as f:
            result = json.load(f)
        result.update(
            {
                "task_group": task_group.name,
                "variant": variant.name,
                "description": variant.description,
                "skipped": True,
            }
        )
        return result

    command = [
        str(args.python),
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_port",
        str(args.master_port),
        "-m",
        "alpamayo1_sft.train_hf",
        *base_overrides(args, task_group, variant, step_json),
        *task_group.overrides,
        *variant.overrides,
    ]
    command_json.write_text(
        json.dumps({"task_group": task_group.name, "variant": variant.name, "command": command}, indent=2),
        encoding="utf-8",
    )

    env = os.environ.copy()
    public_pythonpath = f"{REPO_DIR / 'recipes'}:{REPO_DIR / 'src'}"
    env["PYTHONPATH"] = (
        public_pythonpath if not env.get("PYTHONPATH") else f"{public_pythonpath}:{env['PYTHONPATH']}"
    )
    env.update(
        {
            "HYDRA_FULL_ERROR": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
        }
    )

    with run_log.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=RECIPE_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"{run_name} failed with exit code {completed.returncode}. See {run_log}"
        )
    if not step_json.exists():
        raise FileNotFoundError(f"{run_name} did not produce step timing file: {step_json}")

    with step_json.open(encoding="utf-8") as f:
        result = json.load(f)
    result.update(
        {
            "task_group": task_group.name,
            "variant": variant.name,
            "description": variant.description,
            "log_path": str(run_log),
            "command_path": str(command_json),
        }
    )
    return result


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        result["task_group"]: result["summary"]["stable_avg_step_seconds"]
        for result in results
        if result["variant"] == "baseline"
    }
    previous_by_group: dict[str, float | None] = {}
    rows = []
    for result in results:
        avg = result["summary"]["stable_avg_step_seconds"]
        group = result["task_group"]
        records = [
            record
            for record in result["records"]
            if "step_seconds" in record
            and result["summary"]["stable_start_step"]
            <= record["step"]
            <= result["summary"]["stable_end_step"]
        ]
        step_times = [float(record["step_seconds"]) for record in records]
        previous = previous_by_group.get(group)
        baseline = baselines.get(group)
        rows.append(
            {
                "task_group": group,
                "variant": result["variant"],
                "avg_step_seconds": avg,
                "samples": len(step_times),
                "min_step_seconds": min(step_times) if step_times else None,
                "max_step_seconds": max(step_times) if step_times else None,
                "speedup_vs_group_baseline": baseline / avg if baseline and avg else None,
                "incremental_speedup": previous / avg if previous and avg else None,
            }
        )
        previous_by_group[group] = avg
    return rows


def format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def write_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    rows = summarize(results)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    gpu_name = "unknown"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if smi.returncode == 0 and smi.stdout.splitlines():
            gpu_name = smi.stdout.splitlines()[0].strip()
    except OSError:
        pass

    lines = [
        "# Alpamayo-1 SFT CoC / No-CoC Performance Optimization Report",
        "",
        "## Summary",
        "",
        (
            "This benchmark compares four incremental optimization settings for "
            "`recipes/alpamayo1_sft` using the Stage-1 SFT recipe, with CoC disabled "
            "and enabled. The reported metric is the mean wall-clock step interval "
            f"from step {args.stable_start_step} to {args.stable_end_step}, after "
            "initial warmup."
        ),
        "",
        "| CoC | Variant | Stable avg step time (s) | Samples | Min (s) | Max (s) | Speedup vs group baseline | Incremental speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {group} | {variant} | {avg} | {samples} | {min_time} | {max_time} | {speedup} | {inc} |".format(
                group=row["task_group"],
                variant=row["variant"],
                avg=format_seconds(row["avg_step_seconds"]),
                samples=row["samples"],
                min_time=format_seconds(row["min_step_seconds"]),
                max_time=format_seconds(row["max_step_seconds"]),
                speedup=format_ratio(row["speedup_vs_group_baseline"]),
                inc=format_ratio(row["incremental_speedup"]),
            )
        )

    lines.extend(
        [
            "",
            "## Benchmark Matrix",
            "",
            "| Variant | Settings |",
            "| --- | --- |",
        ]
    )
    for variant in VARIANTS:
        lines.append(f"| `{variant.name}` | {variant.description} |")

    lines.extend(
        [
            "",
            "## CoC Settings",
            "",
            "- `no_coc`: Uses the default `vla_processor/default.yaml` and supervises `traj_future`.",
            "- `coc`: Adds `cot` to the inputs and labels, and sets `reasoning/ood_reasoning.parquet`, `clip_index_reasoning_mini.parquet`, and `use_default_keyframe=false`.",
            "",
            "## Methodology",
            "",
            f"- Generated at: {generated_at}",
            f"- Host: `{socket.gethostname()}`",
            f"- OS: `{platform.platform()}`",
            f"- GPU: `{args.nproc_per_node} x {gpu_name}`",
            f"- Python / uv environment: `{args.python}`",
            f"- Checkpoint: `{args.checkpoint}`",
            f"- PAI data root: `{args.pai_dir}`",
            "- Training entry: `python -m torch.distributed.run -m alpamayo1_sft.train_hf`",
            "- Hydra config: `sft_stage1`",
            f"- Max steps per run: `{args.max_steps}`",
            f"- Stable window: steps `{args.stable_start_step}-{args.stable_end_step}`",
            "- Timing source: `StepTimeCallback` records intervals between consecutive `on_step_end` callbacks, with CUDA synchronization before timestamps. This includes data loading, collation, forward/backward, and optimizer update time.",
            "- Checkpoint saving, evaluation, W&B, and external reporting are disabled for benchmark runs.",
            "",
            "## Result Artifacts",
            "",
        ]
    )
    for result in results:
        run_name = f"{result['task_group']}_{result['variant']}"
        lines.append(
            f"- `{run_name}`: `{RESULTS_DIR / (run_name + '.json')}`, "
            f"`{result.get('log_path', RESULTS_DIR / (run_name + '.log'))}`"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_paths(args)
    selected_groups = [
        group for group in TASK_GROUPS if not args.only_group or group.name in args.only_group
    ]
    selected_variants = [
        variant for variant in VARIANTS if not args.only_variant or variant.name in args.only_variant
    ]

    results = []
    for group in selected_groups:
        for variant in selected_variants:
            print(f"Running {group.name}/{variant.name}...", flush=True)
            results.append(run_case(args, group, variant))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_path = RESULTS_DIR / "summary.json"
    aggregate_path.write_text(
        json.dumps({"results": results, "summary": summarize(results)}, indent=2),
        encoding="utf-8",
    )
    write_report(args, results)
    print(f"Wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
