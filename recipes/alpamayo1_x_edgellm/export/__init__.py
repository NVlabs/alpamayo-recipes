# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX exporters for the Alpamayo 2B DRIVE Thor recipe, one per TensorRT engine.

Each writes ``model.onnx`` + ``weights.bin`` straight from the checkpoints, in FP8 (default, with
the bundle's ``fp8-scales.json``) or FP16 (``--precision fp16``):

- ``visual``: vision encoder, newest frame of four cameras;
- ``prefill``: full-attention language prefill (3006 rows, publishes the paged K/V pool; FP8:
  AlpamayoFusedGateUpPlugin);
- ``verify``: speculative verifier ``q17`` and stop-token appender ``q1`` (Edge-LLM ``AttentionPlugin``, tree mode);
- ``domino``: Domino drafter round (16 proposals);
- ``action``: five-step action expert, six candidates (Edge-LLM ``AttentionPlugin``, tree mode).
"""
