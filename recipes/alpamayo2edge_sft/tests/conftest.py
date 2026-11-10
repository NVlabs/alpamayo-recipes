# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit inference-package test doubles; never used by training."""

import logging
import sys
from types import ModuleType


def pytest_addoption(parser):
    parser.addoption(
        "--require-deepspeed-contracts",
        action="store_true",
        help="Fail if the pinned DeepSpeed installation required by contract tests is unavailable.",
    )
    parser.addoption(
        "--edge-release-stub",
        action="store_true",
        help="Test recipe behavior using API doubles, without validating the inference release.",
    )


def pytest_report_header(config):
    if config.getoption("--edge-release-stub"):
        return "Edge release API doubles enabled: real release compatibility is NOT tested"


def pytest_configure(config):
    if config.getoption("--require-deepspeed-contracts"):
        from importlib import import_module, metadata

        import pytest

        from alpamayo2edge_sft.deepspeed_clipping import _SUPPORTED_DEEPSPEED_VERSION

        try:
            installed = metadata.version("deepspeed")
        except metadata.PackageNotFoundError as error:
            raise pytest.UsageError("DeepSpeed contract tests require an installed DeepSpeed package") from error
        if installed != _SUPPORTED_DEEPSPEED_VERSION:
            raise pytest.UsageError(
                f"DeepSpeed contract tests require {_SUPPORTED_DEEPSPEED_VERSION}, got {installed}"
            )
        try:
            import_module("deepspeed")
        except Exception as error:
            raise pytest.UsageError(f"Cannot import the required DeepSpeed package: {error}") from error
    if not config.getoption("--edge-release-stub"):
        return
    import torch

    names = [
        "alpamayo2_edge",
        "alpamayo2_edge.models",
        "alpamayo2_edge.models.alpamayo2_edge",
        "alpamayo2_edge.config",
        "alpamayo2_edge.helper",
        "alpamayo2_edge.chat_template",
        "alpamayo2_edge.common",
        "alpamayo2_edge.common.logging",
    ]
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)

    class Alpamayo2Edge(torch.nn.Module):
        """Only the nn.Module behavior required by recipe unit tests."""

    def unavailable(*args, **kwargs):
        raise AssertionError("A release API must be replaced with a test-specific fixture")

    sys.modules["alpamayo2_edge.models.alpamayo2_edge"].Alpamayo2Edge = Alpamayo2Edge
    cfg = sys.modules["alpamayo2_edge.config"]
    cfg.Alpamayo2EdgeConfig = type("Alpamayo2EdgeConfig", (), {})
    cfg.build_alpamayo2_edge_tokenizer = unavailable
    cfg.resolve_checkpoint_name_or_path = unavailable
    sys.modules["alpamayo2_edge.helper"].get_processor = unavailable
    sys.modules["alpamayo2_edge.chat_template"].build_conversation = unavailable
    log = sys.modules["alpamayo2_edge.common.logging"]
    log.RankedLogger = lambda name, **kwargs: logging.getLogger(name)
    log.setup_logging = lambda: None
