# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training extensions for the Alpamayo 2 Edge inference model."""

__all__ = ["TrainableAlpamayo2Edge"]


def __getattr__(name):
    if name == "TrainableAlpamayo2Edge":
        from alpamayo2edge_sft.models.trainable_alpamayo2_edge import TrainableAlpamayo2Edge

        return TrainableAlpamayo2Edge
    raise AttributeError(name)
