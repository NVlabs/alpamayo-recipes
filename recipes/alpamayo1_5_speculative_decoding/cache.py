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

"""Greedy CoC caching: one resumable JSONL shard per torchrun rank."""

import json

from .contracts import (
    digest,
    generation_issue,
    load_samples,
    read_cache,
    sample_key,
    validate_cached,
)


def load_cache(directory):
    metadata = json.loads((directory / "manifest.json").read_text())
    rows = {}
    for path in sorted(directory.glob("tokens_rank*.jsonl")):
        for key, row in read_cache(path).items():
            if key in rows:
                raise ValueError(f"Duplicate sample across cache shards: {key}")
            validate_cached(row, metadata["target"])
            rows[key] = row
    return metadata, rows


def cache_counts(rows):
    complete = sum(row["complete"] for row in rows.values())
    return {"attempted": len(rows), "complete": complete, "incomplete": len(rows) - complete}


def run(args):
    from .data import InputData
    from .runtime import Runtime
    from .target import Target, greedy_decode

    samples = load_samples(args.manifest, args.limit, source=args.data_source)
    with Runtime(args.output, args.seed) as runtime:
        data = InputData(args)
        target = Target(args, runtime)
        manifest = {
            "data": data.contract,
            "target": target.contract,
            "samples": len(samples),
            "shards": runtime.world_size,
            "keys_sha256": digest([sample_key(row) for row in samples]),
        }
        if runtime.rank == 0:
            path = args.output / "manifest.json"
            if path.exists():
                if not args.resume:
                    raise ValueError("Cache output exists; use --resume or a fresh --output")
                previous = json.loads(path.read_text())
                if any(previous.get(key) != value for key, value in manifest.items()):
                    raise ValueError(
                        "Resume requires the same samples, target, data and shard count"
                    )
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        runtime.barrier()
        output = args.output / f"tokens_rank{runtime.rank:04d}.jsonl"
        done = read_cache(output, repair_tail=args.resume)
        assigned = samples[runtime.rank :: runtime.world_size]
        if set(done) - {sample_key(row) for row in assigned}:
            raise ValueError("Cache shard contains samples not assigned to this rank")
        for row in done.values():
            validate_cached(row, target.contract)
        with output.open("a") as stream:
            for index, row in enumerate(assigned):
                key = sample_key(row)
                if key in done:
                    continue
                inputs, fingerprint = target.prepare(row, data)
                ids = greedy_decode(target, inputs)
                issue = generation_issue(ids, target.contract)
                cached = {
                    "key": key,
                    "input_hash": fingerprint,
                    "gen_ids": ids,
                    "text": target.tokenizer.decode(ids, skip_special_tokens=False),
                    "complete": issue is None,
                    "incomplete_reason": issue,
                }
                stream.write(json.dumps(cached) + "\n")
                stream.flush()
                if (index + 1) % 100 == 0 or index + 1 == len(assigned):
                    print(
                        f"cache rank={runtime.rank} completed={index + 1}/{len(assigned)}",
                        flush=True,
                    )
        runtime.barrier()
        if runtime.rank == 0:
            _, rows = load_cache(args.output)
            if set(rows) != {sample_key(row) for row in samples}:
                raise ValueError("Final cache coverage mismatch")
            manifest["counts"] = cache_counts(rows)
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"CoC cache complete: {manifest['counts']}", flush=True)


if __name__ == "__main__":
    from .cli import main

    main()
