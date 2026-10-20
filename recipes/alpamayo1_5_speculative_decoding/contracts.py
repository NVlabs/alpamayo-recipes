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

"""Small, dependency-free data and persistence contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SPECFORGE_REVISION = "bc5d8451aa5e7d30aa971dba93d1b7d0c61b83c1"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_key(row: dict) -> str:
    name = row.get("name")
    timestamp = row.get("t0_us")
    if not isinstance(name, str) or not name:
        raise ValueError("Each sample needs a nonempty 'name'")
    if type(timestamp) is not int or timestamp < 0:
        raise ValueError(f"{name}: t0_us must be a nonnegative integer")
    return f"{name}@{timestamp}"


def image_paths(row: dict) -> list[str]:
    """Read extracted camera-major images; never use an assistant answer as input."""
    paths = row.get("images")
    if paths is None:
        paths = [
            item["image"]
            for message in row.get("messages", [])
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for item in message["content"]
            if item.get("type") == "image"
        ]
    if not isinstance(paths, list) or len(paths) != 16:
        raise ValueError(f"{sample_key(row)}: require 16 camera-major images (4 x 4)")
    for path in paths:
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError("Image paths must be relative to --data-root without '..'")
    return paths


def sample_inputs(row: dict) -> dict:
    """The exact source fields used by the no-navigation CoC prompt."""
    value = {"key": sample_key(row), "images": image_paths(row)}
    for field in (
        "past_pos_x",
        "past_pos_y",
        "past_pos_z",
        "past_qx",
        "past_qy",
        "past_qz",
        "past_qw",
    ):
        numbers = row.get(field)
        if not isinstance(numbers, list) or len(numbers) != 16:
            raise ValueError(f"{value['key']}: {field} must have 16 history values")
        if any(type(x) not in (float, int) or not math.isfinite(x) for x in numbers):
            raise ValueError(f"{value['key']}: invalid history values in {field}")
        value[field] = numbers
    for quaternion in zip(*(value[f"past_q{x}"] for x in "xyzw"), strict=True):
        if sum(x * x for x in quaternion) < 1e-12:
            raise ValueError(f"{value['key']}: zero history quaternion")
    return value


def physicalai_anchor(entry: dict) -> dict:
    """SFT's t0_relative is clip-relative microseconds; never guess from t0."""
    clip = entry.get("clip_id", entry.get("name"))
    if not isinstance(clip, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip):
        raise ValueError(f"Invalid clip ID: {clip!r}")
    if "name" in entry and entry["name"] != clip:
        raise ValueError(f"Conflicting clip IDs: {clip}")
    times = [entry[key] for key in ("t0_relative", "t0_us") if key in entry]
    if not times or any(type(t) is not int or not 1_600_000 < t <= 13_600_000 for t in times):
        raise ValueError(
            f"{clip}: require t0_relative or t0_us in integer microseconds, 1.6s < t <= 13.6s"
        )
    if len(set(times)) != 1:
        raise ValueError(f"{clip}: conflicting relative timestamps")
    # Text, navigation, future trajectory and human answers are deliberately excluded.
    return {"name": clip, "t0_us": times[0]}


def load_samples(path: Path, limit: int | None = None, *, source="extracted") -> list[dict]:
    rows, seen = [], set()
    with path.open() as stream:
        if path.suffix == ".json":
            entries = json.load(stream)
            if not isinstance(entries, list):
                raise ValueError("Annotation JSON must be a list")
        else:
            entries = (json.loads(line) for line in stream if line.strip())
        for number, row in enumerate(entries, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a sample object")
            if source == "physicalai":
                row = physicalai_anchor(row)
            elif source == "extracted":
                sample_inputs(row)
            else:
                raise ValueError(f"Unknown data source: {source}")
            key = sample_key(row)
            if key in seen:
                raise ValueError(f"{path}:{number}: duplicate {key}")
            seen.add(key)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def read_cache(path: Path, *, repair_tail: bool = False) -> dict[str, dict]:
    """Reject corrupt/duplicate records; a writer may discard an interrupted final line."""
    records = {}
    if not path.exists():
        return records
    mode = "rb+" if repair_tail else "rb"
    with path.open(mode) as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Incomplete final line: {path}")
                stream.truncate(offset)
                break
            row = json.loads(line)
            key = row["key"]
            if key in records:
                raise ValueError(f"Duplicate cache record: {key}")
            records[key] = row
    return records


def generation_issue(ids: list[int], contract: dict) -> str | None:
    """Classify incomplete output; invalid IDs or impossible stop layouts still fail."""
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(x) is not int or not 0 <= x < contract["vocab_size"] for x in ids)
    ):
        raise ValueError("Invalid generated token IDs")
    stop, cot_end = contract["stop_id"], contract["cot_end_id"]
    budget = contract.get("max_new_tokens")
    if stop in ids[:-1] or (budget is not None and len(ids) > budget):
        raise ValueError("Invalid generated stop position or token-budget overrun")
    if ids[-1] == stop:
        return None if cot_end in ids[:-1] else "missing_cot_end"
    if len(ids) == budget:
        return "max_new_tokens"
    raise ValueError("Incomplete generation stopped before its token budget")


def validate_cached(row: dict, contract: dict) -> None:
    """Incomplete output is recorded, not treated as a failed caching job."""
    issue = generation_issue(row.get("gen_ids"), contract)
    if row.get("complete") is not (issue is None) or row.get("incomplete_reason") != issue:
        raise ValueError(f"{row.get('key')}: inconsistent generation completion metadata")


def training_samples(samples: list[dict], cache: dict) -> list[dict]:
    """Preserve selection order and never silently train on truncated target answers."""
    missing = {sample_key(row) for row in samples} - set(cache)
    if missing:
        raise ValueError(f"{len(missing)} training samples are absent from the cache")
    eligible = [row for row in samples if cache[sample_key(row)]["complete"]]
    if not eligible:
        raise ValueError("No complete CoCs remain for training")
    return eligible


def check_training_output(output: Path, resume: Path | None) -> int:
    """Fail before loading weights if this run would branch into existing checkpoints."""
    if resume is not None and (not resume.is_dir() or not re.fullmatch(r"step_\d+", resume.name)):
        raise ValueError("Resume requires a step_N checkpoint directory")
    start = int(resume.name.split("_")[1]) if resume else 0
    for path in sorted(output.glob("step_*")):
        match = re.fullmatch(r"step_(\d+)", path.name)
        if match and int(match[1]) > start:
            raise FileExistsError(
                f"Later checkpoint already exists: {path}; resume the latest step "
                "or branch into a fresh --output"
            )
    return start


def check_resume_settings(previous: dict, current: dict) -> None:
    """Keep the resumed sample stream, schedule and model setup unchanged."""
    for key in ("method", "steps", "global_batch_size", "world_size", "seed", "target", "data"):
        if previous.get(key) != current[key]:
            raise ValueError(f"Resume settings differ: {key}")


def accumulation_steps(global_batch: int, world_size: int) -> int:
    if world_size < 1 or global_batch < world_size or global_batch % world_size:
        raise ValueError("--global-batch-size must be a positive multiple of the GPU count")
    return global_batch // world_size


def mixture_weight(method: str, step: int, total_steps: int) -> float:
    """Domino base-loss decay; DFlash2 selector warmup/ramp, per optimizer step."""
    if method == "domino":
        return max(0.0, 1.0 - step / max(1, int(total_steps * 0.5)))
    warmup = int(total_steps * 0.04)
    return min(1.0, max(0.0, (step - warmup + 1) / max(1, int(total_steps * 0.04))))


def trim_committed(tokens: list[int], remaining: int, stop: int) -> list[int]:
    tokens = tokens[:remaining]
    if stop in tokens:
        tokens = tokens[: tokens.index(stop) + 1]
    return tokens
