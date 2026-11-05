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

"""Actual speculative acceptance and exact greedy-output parity, not a speedup benchmark."""

import json
import time

import torch

from .contracts import (
    generation_issue,
    load_samples,
    sample_key,
)
from .data import InputData
from .decode import speculative_decode
from .draft import DRAFT_CLASSES
from .runtime import Runtime
from .target import Target, greedy_decode


def summarize(records):
    rounds = [r for row in records for r in row["rounds"]]
    n = len(rounds)

    def mean(key):
        return sum(r[key] for r in rounds) / n if n else None

    exact = sum(row["greedy_exact_match"] for row in records)
    complete = sum(row["complete"] and row["greedy_complete"] for row in records)
    return {
        "samples": len(records),
        "verification_calls": n,
        "acceptance_length": mean("acceptance_length"),
        "emitted_tokens_per_verification": mean("emitted_tokens"),
        "draft_tokens_per_proposal": mean("proposed_tokens"),
        "output_tokens": sum(len(row["generated_ids"]) for row in records),
        "greedy_exact_matches": exact,
        "greedy_mismatches": len(records) - exact,
        "complete_samples": complete,
        "greedy_parity_status": (
            "incomplete"
            if complete != len(records)
            else "exact"
            if exact == len(records)
            else "mismatch"
        ),
        "status": "completed",
        "metric_note": (
            "acceptance_length is matching draft prefix + 1 before stop/length trimming; "
            "emitted tokens exclude prefill; metrics include all attempted samples"
        ),
        "timing_note": (
            "Unoptimized PyTorch, includes target prefill; excludes separate AR parity checks. "
            "Not a speedup benchmark."
        ),
        "sum_clip_seconds": sum(row["seconds"] for row in records),
    }


def run(args):
    samples = load_samples(args.manifest, args.limit, source=args.data_source)
    run_contract = json.loads((args.draft / "run.json").read_text())
    with Runtime(args.output, args.seed) as runtime:
        if runtime.rank == 0:
            if (args.output / "metrics.json").exists() or any(
                args.output.glob("predictions_rank*.jsonl")
            ):
                raise ValueError("Evaluation output exists; choose a fresh --output")

        runtime.barrier()
        data = InputData(args)
        # Held-out selections may differ, but preprocessing must match training.
        for key in data.contract.keys() | run_contract["data"].keys():
            if data.contract.get(key) != run_contract["data"].get(key):
                raise ValueError(f"Evaluation data differs from training: {key}")
        target = Target(args, runtime)
        # A longer held-out CoC may need a larger evaluation budget.
        for key in target.contract.keys() | run_contract["target"].keys():
            if key != "max_new_tokens" and target.contract.get(key) != run_contract["target"].get(
                key
            ):
                raise ValueError(f"Evaluation target differs from training: {key}")
        method = run_contract["method"]
        draft = (
            DRAFT_CLASSES[method]
            .from_pretrained(
                args.draft,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(runtime.device)
            .eval()
        )
        output_path = args.output / f"predictions_rank{runtime.rank:04d}.jsonl"
        with output_path.open("x") as stream:
            for row in samples[runtime.rank :: runtime.world_size]:
                inputs, _ = target.prepare(row, data)
                torch.cuda.synchronize()
                started = time.monotonic()
                ids, rounds = speculative_decode(target, draft, inputs)
                torch.cuda.synchronize()
                seconds = time.monotonic() - started
                reference = greedy_decode(target, inputs)
                issue = generation_issue(ids, target.contract)
                greedy_issue = generation_issue(reference, target.contract)
                result = {
                    "key": sample_key(row),
                    "generated_ids": ids,
                    "greedy_ids": reference,
                    "text": target.tokenizer.decode(ids, skip_special_tokens=False),
                    "greedy_text": target.tokenizer.decode(reference, skip_special_tokens=False),
                    "greedy_exact_match": ids == reference,
                    "complete": issue is None,
                    "incomplete_reason": issue,
                    "greedy_complete": greedy_issue is None,
                    "greedy_incomplete_reason": greedy_issue,
                    "rounds": rounds,
                    "seconds": seconds,
                }
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(
                    f"eval rank={runtime.rank} key={result['key']} "
                    f"parity={result['greedy_exact_match']}",
                    flush=True,
                )
        runtime.barrier()

        if runtime.rank == 0:
            records = []
            for rank in range(runtime.world_size):
                with (args.output / f"predictions_rank{rank:04d}.jsonl").open() as stream:
                    records.extend(json.loads(line) for line in stream)
            if len(records) != len(samples) or {row["key"] for row in records} != {
                sample_key(row) for row in samples
            }:
                raise ValueError("Evaluation coverage mismatch")
            metrics = summarize(records)
            metrics.update(
                method=method,
                target=target.contract,
                data=data.contract,
                draft=str(args.draft),
            )
            (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    from .cli import main

    main()
