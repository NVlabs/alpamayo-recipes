# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write or check the model bundle manifest.

    python -m alpamayo1_x_edgellm.bundle <bundle> [--check]

The bundle is one directory: the target checkpoint (``config.json``, one ``model*.safetensors``
and its index when sharded; it includes the action expert), ``domino-drafter.safetensors``,
``fp8-scales.json``, ``domino-vocab.json`` and ``action-noise.bin``. ``MANIFEST.json`` records the
size and SHA-256 of every file; the exporters and the runtime check it before reading the bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alpamayo1_x_edgellm.core import Bundle, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="Model bundle directory.")
    parser.add_argument("--check", action="store_true", help="Verify the existing MANIFEST.json instead of writing it.")
    args = parser.parse_args()
    if args.check:
        bundle = Bundle(args.bundle)
        print(f"PASS {bundle.directory}: manifest {bundle.manifest_sha256}")
    else:
        path = write_manifest(args.bundle)
        print(path)


if __name__ == "__main__":
    main()
