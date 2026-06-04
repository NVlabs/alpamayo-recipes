# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""minADE evaluation for Alpamayo 1.5 over one or more PhysicalAI-AV clips.

For each (clip_id, t0_us) pair the script:
  1. Loads the clip from PhysicalAI-AV (ego-motion + four camera streams).
  2. Runs Alpamayo 1.5 VLM-rollout inference to produce K trajectory candidates.
  3. Computes minADE against the ground-truth future ego trajectory.

Clips can be specified via ``--clip_id`` / ``--t0_us``, a JSON ``--annotations``
file, or left unset to use a built-in example clip.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alpamayo.metrics.distance_metrics import compute_minade
from alpamayo1_5_eval.load_datasets import SingleClipDatasetConfig, SingleClipDatasetLoader
from alpamayo1_5_eval.model import AlpamayoModelConfig, AlpamayoSingleClipModel

_DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
_DEFAULT_T0_US = 5_100_000


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert tensors and numpy arrays to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    return obj


def _print_trajectory(pred_xyz: torch.Tensor, traj_index: int = 0) -> None:
    """Print one predicted trajectory candidate to stdout.

    Accepts common shapes: ``(B, N, K, T, 3)``, ``(B, K, T, 3)``,
    ``(K, T, 3)``, or ``(T, 3)``.
    """
    pred = pred_xyz.detach().float().cpu()
    if pred.ndim == 5:
        traj = pred[0, 0, traj_index]
    elif pred.ndim == 4:
        traj = pred[0, traj_index]
    elif pred.ndim == 3:
        traj = pred[traj_index]
    elif pred.ndim == 2:
        traj = pred
    else:
        raise ValueError(f"Unsupported pred_xyz shape: {tuple(pred.shape)}")

    print("\n========== Predicted Trajectory ==========")
    print(f"trajectory_index: {traj_index}")
    print(f"num_points: {traj.shape[0]}")
    for i, point in enumerate(traj):
        x, y = float(point[0]), float(point[1])
        z = float(point[2]) if point.shape[0] > 2 else 0.0
        print(f"  [{i:02d}] x={x:8.3f}  y={y:8.3f}  z={z:8.3f}")


def _reshape_pred_for_minade(pred_xyz: torch.Tensor) -> torch.Tensor:
    """Ensure ``pred_xyz`` is ``(B, N, K, T, 3)`` for :func:`compute_minade`.

    The model may return ``(B, N, K, T, 3)``, ``(B, K, T, 3)``, or
    ``(K, T, 3)``; missing leading dimensions are expanded.
    """
    if pred_xyz.ndim == 5:
        return pred_xyz
    if pred_xyz.ndim == 4:
        return pred_xyz.unsqueeze(1)
    if pred_xyz.ndim == 3:
        return pred_xyz.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Cannot reshape pred_xyz with ndim={pred_xyz.ndim} to (B, N, K, T, 3)")


def _reshape_gt_for_minade(gt_future_xyz: torch.Tensor) -> torch.Tensor:
    """Ensure ``gt_future_xyz`` is ``(B, T, 3)`` for :func:`compute_minade`.

    The dataset loader returns ``(1, 1, T, 3)``; the extra group dim is squeezed.
    """
    if gt_future_xyz.ndim == 4:
        return gt_future_xyz.squeeze(1)
    if gt_future_xyz.ndim == 3:
        return gt_future_xyz
    raise ValueError(
        f"Cannot reshape gt_future_xyz with ndim={gt_future_xyz.ndim} to (B, T, 3)"
    )


def _all_ade_per_sample(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> list[float]:
    """Return per-candidate ADE list (XY plane) for logging.

    Args:
        pred_xyz: ``(B, N, K, T, 3)``
        gt_xyz: ``(B, T, 3)``

    Returns:
        List of K float values — ADE for each trajectory candidate.
    """
    diff = pred_xyz - gt_xyz[:, None, None, :, :]
    l2 = diff[..., :2].norm(dim=-1)
    ade = l2.mean(dim=-1)
    return ade[0, 0].tolist()


def _parse_clip_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve the list of ``{"clip_id", "t0_us", ...}`` dicts to evaluate.

    Priority: ``--annotations`` > ``--clip_id`` > built-in default.
    """
    if args.annotations is not None:
        path = Path(args.annotations)
        with path.open(encoding="utf-8") as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            raise ValueError(
                f"--annotations file must contain a JSON list, got {type(specs).__name__}"
            )
        for i, s in enumerate(specs):
            if "clip_id" not in s or "t0_us" not in s:
                raise ValueError(
                    f"Entry {i} in annotations file is missing 'clip_id' or 't0_us'."
                )
        return specs

    if args.clip_id:
        clip_ids: list[str] = args.clip_id
        t0_values: list[int] = args.t0_us if args.t0_us else [_DEFAULT_T0_US]
        if len(t0_values) == 1:
            t0_values = t0_values * len(clip_ids)
        if len(t0_values) != len(clip_ids):
            raise ValueError(
                f"--clip_id has {len(clip_ids)} entries but --t0_us has "
                f"{len(t0_values)}. Pass one --t0_us (broadcast) or one per clip."
            )
        return [{"clip_id": cid, "t0_us": t0} for cid, t0 in zip(clip_ids, t0_values)]

    return [{"clip_id": _DEFAULT_CLIP_ID, "t0_us": _DEFAULT_T0_US}]


def _evaluate_one_clip(
    spec: dict[str, Any],
    model: AlpamayoSingleClipModel,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run the full eval pipeline for a single ``(clip_id, t0_us)`` pair.

    Args:
        spec: Dict with at minimum ``"clip_id"`` and ``"t0_us"`` keys.
              Any extra fields are preserved verbatim in the returned result.
        model: Pre-loaded model (shared across all clips).
        args: Parsed CLI arguments for inference parameters.

    Returns:
        Result dict with ``"clip_id"``, ``"t0_us"``, ``"minADE"``,
        ``"all_ADE"``, ``"cot"``, and any extra fields from *spec*.
    """
    clip_id: str = spec["clip_id"]
    t0_us: int = spec["t0_us"]

    dataset_config = SingleClipDatasetConfig(
        clip_id=clip_id,
        t0_us=t0_us,
        maybe_stream=not args.no_stream,
        dataset_local_dir=args.dataset_local_dir,
        dataset_revision=args.dataset_revision,
    )
    data = SingleClipDatasetLoader(dataset_config).load()

    infer_outputs = model.infer(
        data=data,
        top_p=args.top_p,
        temperature=args.temperature,
        num_traj_samples=args.num_traj_samples,
        max_generation_length=args.max_generation_length,
        seed=args.seed,
    )

    pred_xyz = _reshape_pred_for_minade(infer_outputs["pred_xyz"])
    gt_xyz = _reshape_gt_for_minade(data["ego_future_xyz"]).to(pred_xyz.device)

    minade_metrics = compute_minade(pred_xyz, gt_xyz)
    min_ade = float(minade_metrics["min_ade"].mean().item())
    all_ade = _all_ade_per_sample(pred_xyz, gt_xyz)

    extra = infer_outputs.get("extra", {})
    cot_text = None
    if isinstance(extra, dict) and "cot" in extra:
        try:
            raw = extra["cot"]
            cot_text = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, str) else [raw]
        except Exception:
            cot_text = [str(extra["cot"])]

    result: dict[str, Any] = {
        **{k: v for k, v in spec.items() if k not in ("clip_id", "t0_us")},
        "clip_id": clip_id,
        "t0_us": t0_us,
        "num_traj_samples": args.num_traj_samples,
        "minADE": min_ade,
        "all_ADE": all_ade,
        "cot": cot_text,
    }

    print(f"\n  clip_id:  {clip_id}")
    print(f"  t0_us:    {t0_us}")
    print(f"  minADE:   {min_ade:.6f} m")
    print(f"  all_ADE:  {all_ade}")
    _print_trajectory(infer_outputs["pred_xyz"], traj_index=0)
    if cot_text is not None:
        print("\n  Chain-of-Thought:")
        for i, cot in enumerate(cot_text):
            print(f"  [{i}] {cot}")

    return result


class MinADEEvaluator:
    """Orchestrate minADE evaluation over one or more PhysicalAI-AV clips.

    Clips are evaluated sequentially on a single GPU.  The model is loaded
    once and reused across all clips, so the overhead is O(1) in model
    loading rather than O(N).

    Args:
        args: Parsed CLI arguments from :func:`build_parser`.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> dict[str, Any]:
        """Execute the full evaluation pipeline.

        Returns:
            A dict with ``"results"`` (list, one entry per clip) and
            ``"summary"`` (aggregate statistics).
        """
        specs = _parse_clip_specs(self.args)
        n = len(specs)

        print(f"[1/3] Loading model (shared across {n} clip(s))...")
        model_config = AlpamayoModelConfig(
            model_name=self.args.model_name,
            vlm_name_or_path=self.args.vlm_name_or_path,
            device=self.args.device,
            dtype=self.args.dtype,
            attn_implementation=self.args.attn_implementation,
        )
        model = AlpamayoSingleClipModel(model_config)

        print(f"[2/3] Evaluating {n} clip(s)...")
        results: list[dict[str, Any]] = []
        for i, spec in enumerate(specs):
            print(f"\n--- Clip {i + 1}/{n} ---")
            results.append(_evaluate_one_clip(spec, model, self.args))

        print("\n[3/3] Aggregating results...")
        summary = self._summarize(results)
        self._print_summary(summary, n)
        self._save(results, summary)

        return {"results": results, "summary": summary}

    @staticmethod
    def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate statistics over all clip results."""
        min_ades = [r["minADE"] for r in results]
        summary: dict[str, Any] = {
            "total_clips": len(results),
            "mean_minADE@6.4s": statistics.mean(min_ades),
        }
        if len(min_ades) > 1:
            summary["std_minADE@6.4s"] = statistics.stdev(min_ades)
            summary["min_minADE@6.4s"] = min(min_ades)
            summary["max_minADE@6.4s"] = max(min_ades)
        return summary

    @staticmethod
    def _print_summary(summary: dict[str, Any], n: int) -> None:
        print("\n========== Aggregate Summary ==========")
        print(f"total_clips:         {n}")
        print(f"mean_minADE@6.4s:    {summary['mean_minADE@6.4s']:.6f} m")
        if "std_minADE@6.4s" in summary:
            print(f"std_minADE@6.4s:     {summary['std_minADE@6.4s']:.6f} m")
            print(f"min_minADE@6.4s:     {summary['min_minADE@6.4s']:.6f} m")
            print(f"max_minADE@6.4s:     {summary['max_minADE@6.4s']:.6f} m")

    def _save(self, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
        save_path = Path(self.args.output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output = _make_json_serializable({"summary": summary, "results": results})
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {save_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="minADE evaluation for Alpamayo 1.5 over one or more clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    clip_group = parser.add_argument_group(
        "Clip specification",
        "Use --annotations OR --clip_id/--t0_us.  "
        "If neither is given, a built-in example clip is used.",
    )
    clip_group.add_argument(
        "--annotations",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file containing a list of clip specs, each with "
            "'clip_id' and 't0_us' keys.  Extra keys are preserved in the output."
        ),
    )
    clip_group.add_argument(
        "--clip_id",
        type=str,
        nargs="+",
        default=None,
        metavar="UUID",
        help="One or more PhysicalAI-AV clip UUIDs.  Pair with --t0_us.",
    )
    clip_group.add_argument(
        "--t0_us",
        type=int,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Evaluation timestamp(s) in microseconds.  "
            "Provide one value (broadcast to all clips) or one per --clip_id."
        ),
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model_name",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help=(
            "HuggingFace Hub model ID (default) or local path to a checkpoint directory. "
            "Both the released A1.5 format (model_type='alpamayo1_5') and the A1-format "
            "converted checkpoint are accepted; the config is auto-patched when needed."
        ),
    )
    model_group.add_argument(
        "--vlm_name_or_path",
        type=str,
        default=None,
        metavar="PATH_OR_ID",
        help=(
            "HuggingFace Hub model ID or local path for the VLM backbone processor "
            "(e.g. 'Qwen/Qwen3-VL-8B-Instruct').  When omitted the path is read from "
            "the loaded model's config.  Override this to avoid network access when the "
            "model config points to a HuggingFace Hub ID."
        ),
    )
    model_group.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "sdpa", "flash_attention_2", "eager"],
        help="Attention backend override.  Use 'sdpa' if flash-attn is unavailable.",
    )

    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument(
        "--num_traj_samples",
        type=int,
        default=1,
        help="Number of trajectory candidates to sample per clip (K).",
    )
    infer_group.add_argument("--top_p", type=float, default=0.98, help="Nucleus sampling probability.")
    infer_group.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    infer_group.add_argument(
        "--max_generation_length", type=int, default=256, help="Maximum tokens to generate."
    )
    infer_group.add_argument("--seed", type=int, default=42, help="CUDA random seed.")

    io_group = parser.add_argument_group("Dataset / output")
    io_group.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable dataset streaming; requires a full local dataset download.",
    )
    io_group.add_argument(
        "--dataset_local_dir",
        type=str,
        default=None,
        help=(
            "Path to a locally downloaded copy of the PhysicalAI-AV dataset. "
            "When provided, data files are read from disk instead of being downloaded. "
            "Set --dataset_revision as well to skip the HuggingFace list_repo_refs call."
        ),
    )
    io_group.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help=(
            "HuggingFace dataset revision (branch, tag, or commit hash). "
            "Passing any non-empty value (e.g. 'main') avoids a network round-trip "
            "to resolve the default branch commit hash."
        ),
    )
    io_group.add_argument(
        "--output",
        type=str,
        default="outputs/minade_results.json",
        help="Path to write the JSON result file.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    MinADEEvaluator(args).run()


if __name__ == "__main__":
    main()
