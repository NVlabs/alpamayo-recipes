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

"""Portable torchrun entry points. No scheduler or machine-specific settings."""

import argparse
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("cache", "train", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument(
            "--base-vlm", type=Path, required=True, help="Local VLM config/tokenizer assets"
        )
        p.add_argument(
            "--processor", type=Path, required=True, help="Local Qwen3-VL processor assets"
        )
        p.add_argument(
            "--manifest",
            type=Path,
            required=True,
            help="SFT annotation JSON or JSONL clip/timestamp selection",
        )
        p.add_argument(
            "--data-root", type=Path, required=True, help="Local PhysicalAI dataset directory"
        )
        p.add_argument(
            "--data-source",
            choices=("physicalai", "extracted"),
            default="physicalai",
            help="Read PhysicalAI videos directly (default), or an existing extracted manifest",
        )
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--target-attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
        p.add_argument("--limit", type=positive_int)
        p.add_argument("--max-new-tokens", type=positive_int, default=96)
        p.add_argument("--seed", type=int, default=42)
        if command == "cache":
            p.add_argument(
                "--resume", action="store_true", help="Continue the existing cache in --output"
            )
        if command == "train":
            p.add_argument("--cache", type=Path, required=True)
            p.add_argument("--method", choices=("domino", "dflash2"), required=True)
            p.add_argument("--steps", type=positive_int, default=3000)
            p.add_argument("--global-batch-size", type=positive_int, default=512)
            p.add_argument("--learning-rate", type=float, default=6e-4)
            p.add_argument("--save-every", type=positive_int, default=100)
            p.add_argument(
                "--resume-from-checkpoint",
                dest="resume",
                type=Path,
                help="A complete step_N directory",
            )
            p.add_argument("--block-size", type=positive_int, default=16)
            p.add_argument("--draft-layers", type=positive_int, default=5)
            p.add_argument("--target-layers", type=int, nargs="+")
            p.add_argument("--num-anchors", type=positive_int, default=64)
            p.add_argument(
                "--wandb-project", help="Opt in to W&B logging (requires the wandb extra)"
            )
            p.add_argument("--wandb-run-name", help="W&B run name (default: output directory name)")
        if command == "evaluate":
            p.add_argument("--draft", type=Path, required=True, help="A complete step_N directory")
    return root


def main():
    args = parser().parse_args()
    if args.command == "cache":
        from .cache import run
    elif args.command == "train":
        from .train import run
    else:
        from .evaluate import run
    run(args)


if __name__ == "__main__":
    main()
