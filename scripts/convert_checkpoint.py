#!/usr/bin/env python3
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

r"""Convert checkpoints between Alpamayo 1 and Alpamayo 1.5 formats.

The model weights are architecturally identical -- only config.json metadata
(model_type, architectures, Hydra _target_ paths) differs between the two
release packages.

Subcommands
-----------
to-a15   Convert an A1 checkpoint for inference with alpamayo_1.5_release.
to-a1    Convert an A1.5 checkpoint for SFT with alpamayo_r1_release.

Both subcommands symlink the weight files and write a converted config.json.

Usage
-----
    # A1 -> A1.5 (for inference with alpamayo_1.5_release)
    python scripts/convert_checkpoint.py to-a15 \
        --input /path/to/Alpamayo-R1-10B \
        --output /path/to/converted-for-a15

    # A1.5 -> A1 (for SFT with alpamayo_r1_release)
    python scripts/convert_checkpoint.py to-a1 \
        --input /path/to/Alpamayo-1.5-10B \
        --output /path/to/converted-for-sft
"""

import argparse
import copy
import json
from pathlib import Path

from alpamayo.utils.checkpoint_utils import prepare_output_dir, remap_targets, setup_checkpoint_output

# _target_ prefix remapping: A1 -> A1.5
_A1_TO_A15 = {
    "alpamayo_r1.models.action_in_proj.": "alpamayo1_5.models.action_in_proj.",
    "alpamayo_r1.models.delta_tokenizer.": "alpamayo1_5.models.delta_tokenizer.",
    "alpamayo_r1.action_space.": "alpamayo1_5.action_space.",
    "alpamayo_r1.diffusion.": "alpamayo1_5.diffusion.",
}

# Inverse: A1.5 -> A1
_A15_TO_A1 = {v: k for k, v in _A1_TO_A15.items()}

_COPY_NAMES = {"generation_config.json", "tokenizer_config.json", "tokenizer.json"}


def _convert_config(
    config: dict,
    model_type: str,
    architectures: list[str],
    target_table: dict[str, str],
) -> dict:
    """Return a converted copy of *config* with remapped metadata."""
    out = copy.deepcopy(config)
    out["model_type"] = model_type
    out["architectures"] = architectures
    remap_targets(out, target_table)
    return out


def _run(args: argparse.Namespace) -> None:
    """Shared entry point for both subcommands."""
    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    if input_dir == output_dir:
        raise ValueError(
            f"--input and --output resolve to the same path ({input_dir}); "
            "this would destroy the source weights. Choose a distinct --output."
        )

    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {input_dir}")

    with open(config_path) as f:
        original = json.load(f)

    converted = _convert_config(
        original,
        model_type=args._model_type,
        architectures=args._architectures,
        target_table=args._target_table,
    )

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    setup_checkpoint_output(input_dir, output_dir, copy_names=_COPY_NAMES)

    with open(output_dir / "config.json", "w") as f:
        json.dump(converted, f, indent=2)
        f.write("\n")

    print(f"Converted: {input_dir}")
    print(f"      ->   {output_dir}")
    print(f"  model_type:    {original.get('model_type')!r} -> {converted['model_type']!r}")
    print(f"  architectures: {original.get('architectures')} -> {converted['architectures']}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between Alpamayo 1 and Alpamayo 1.5 formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- to-a15 --
    p15 = sub.add_parser("to-a15", help="Convert A1 checkpoint -> A1.5 inference format")
    p15.add_argument("--input", type=Path, required=True, help="A1 checkpoint directory")
    p15.add_argument("--output", type=Path, required=True, help="Output directory")
    p15.add_argument(
        "--overwrite",
        action="store_true",
        help="If --output exists and is non-empty, delete its contents before writing.",
    )
    p15.set_defaults(
        _model_type="alpamayo1_5",
        _architectures=["Alpamayo1_5"],
        _target_table=_A1_TO_A15,
    )

    # -- to-a1 --
    p1 = sub.add_parser("to-a1", help="Convert A1.5 checkpoint -> A1 SFT format")
    p1.add_argument("--input", type=Path, required=True, help="A1.5 checkpoint directory")
    p1.add_argument("--output", type=Path, required=True, help="Output directory")
    p1.add_argument(
        "--overwrite",
        action="store_true",
        help="If --output exists and is non-empty, delete its contents before writing.",
    )
    p1.set_defaults(
        _model_type="alpamayo_r1",
        _architectures=["AlpamayoR1"],
        _target_table=_A15_TO_A1,
    )

    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()