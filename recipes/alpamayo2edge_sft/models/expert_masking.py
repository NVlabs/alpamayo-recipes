# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instance-local mask construction for a noncausal expert training forward."""
from contextlib import contextmanager


@contextmanager
def noncausal_expert_mask(config, enabled=True):
    """Let HF construct a bidirectional mask without persisting config changes.

    The caller must also pass ``is_causal=False`` to the attention kernels.
    HF mask construction reads the config separately from that kernel argument.
    """
    if not enabled:
        yield
        return
    missing = object()
    previous = vars(config).get("is_causal", missing)
    config.is_causal = False
    try:
        yield
    finally:
        if previous is missing:
            del config.is_causal
        else:
            config.is_causal = previous
