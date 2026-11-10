# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit a supplied whole-recording split; never assign or drop recordings."""

import argparse
import json
from pathlib import Path

from alpamayo2edge_sft.driving_splits import audit_split, load_split_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("once", "tartan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-time-gap-seconds", type=float, default=86400,
                        help="Buffer between complete recording intervals; at least 8.4 seconds.")
    parser.add_argument("--min-location-gap-m", type=float, default=250)
    parser.add_argument("--temporal-only", action="store_true",
                        help="Explicitly accept a time-only audit; geography remains a separate claim.")
    args = parser.parse_args(argv)
    if args.manifest.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the input manifest")
    try:
        manifest, digest = load_split_manifest(args.manifest, args.dataset, return_sha256=True)
        report = audit_split(manifest, args.min_time_gap_seconds, args.min_location_gap_m,
                             args.temporal_only)
        report["manifest_sha256"] = digest
    except (OSError, ValueError) as error:
        report = {"schema_version": 1, "dataset": args.dataset, "status": "invalid",
                  "passed": False, "full_validation_passed": False, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
