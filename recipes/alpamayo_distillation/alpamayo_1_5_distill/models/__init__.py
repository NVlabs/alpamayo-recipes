# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distillation model and configuration modules."""

from alpamayo1_5_distill.models.configuration_distill import (
    DistillAlpamayo1_5Config,
)
from alpamayo1_5_distill.models.modeling_distill import DistillAlpamayo1_5

__all__ = ["DistillAlpamayo1_5", "DistillAlpamayo1_5Config"]
