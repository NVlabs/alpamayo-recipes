# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and verify the pinned Cosmos base-VLM metadata snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VLM_METADATA_REPO_ID = "nvidia/Cosmos-Reason2-2B"
VLM_METADATA_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"  # the HF commit fixes the file contents
VLM_METADATA_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def verify_metadata(root: Path) -> None:
    missing = [name for name in VLM_METADATA_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing VLM metadata files in {root}: {missing}")
    config = json.loads((root / "config.json").read_text())
    json.loads((root / "tokenizer.json").read_text())
    if config.get("model_type") != "qwen3_vl":
        raise RuntimeError(f"Unexpected VLM model type: {config.get('model_type')!r}")


def download_metadata(output_dir: Path) -> None:
    """Download the pinned files with the Hugging Face credential (``hf auth login``, or ``HF_TOKEN``)."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError as error:
        raise RuntimeError("huggingface_hub is not installed (README step 0)") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=VLM_METADATA_REPO_ID,
            revision=VLM_METADATA_REVISION,
            allow_patterns=list(VLM_METADATA_FILES),
            local_dir=output_dir,
        )
    except GatedRepoError as error:
        raise RuntimeError(
            f"No access to the gated {VLM_METADATA_REPO_ID}: run `hf auth login` (or set HF_TOKEN) "
            "with a Hugging Face account that has accepted its license, then rerun this command."
        ) from error
    verify_metadata(output_dir)
    print(
        "PASS pinned VLM metadata: "
        f"{VLM_METADATA_REPO_ID}@{VLM_METADATA_REVISION}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    download_metadata(args.output_dir)


if __name__ == "__main__":
    main()
