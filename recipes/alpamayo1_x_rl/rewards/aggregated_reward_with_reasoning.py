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


"""Aggregated reward with chain-of-thought (CoT) scoring for Alpamayo Cosmos-RL."""

from __future__ import annotations

from typing import Any

_REQUIRED_REWARD_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
    "reasoning_weight",
]


def _get_reward_cfg(config: object | None) -> dict[str, float]:
    """Extract reward parameters from Cosmos TOML [custom.alpamayo.reward]."""
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError) as e:
        raise ValueError(
            "Reward config not found in TOML. "
            f"Required keys under [custom.alpamayo.reward]: {_REQUIRED_REWARD_KEYS}"
        ) from e

    missing = [k for k in _REQUIRED_REWARD_KEYS if k not in reward_cfg]
    if missing:
        raise ValueError(f"Missing key(s) in [custom.alpamayo.reward]: {missing}")

    return {k: float(reward_cfg[k]) for k in _REQUIRED_REWARD_KEYS}


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Aggregate traj, comfort, and CoT reasoning into one scalar reward.

    Trajectory and comfort match :func:`alpamayo1_x_rl.rewards.aggregated_reward.compute_reward`.
    Additionally parses CoT from ``to_be_evaluated``, grades it against ground-truth
    CoT when both are present, and adds the weighted reasoning component (or a
    penalty when CoT is missing after decode).
    """
    from alpamayo_r1.models.token_utils import extract_between_special_tokens
    from alpamayo1_x_rl.rewards.comfort_reward import compute_comfort
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.utils.trajectory_decode import decode_rollout_trajectory

    w = _get_reward_cfg(config)

    gt_fut_xyz = reference["ego_future_xyz"]
    predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
        to_be_evaluated,
        reference["ego_history_xyz"],
        reference["ego_history_rot"],
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        model_config=model_config,
    )

    l2_dist = calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0])

    comfort_dict_t = compute_comfort(
        predicted_fut_xyz[:, None, None, ...],
        predicted_fut_rot[:, None, None, ...],
    )
    comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))
    comfort_score = comfort_score - 1.0

    pred_cot = extract_between_special_tokens([to_be_evaluated], token="cot")[0]
    gt_cot = reference.get("cot", "")

    logger.debug(f"[compute_reward] Pred_cot: {pred_cot}")
    logger.debug(f"[compute_reward] GT_cot: {gt_cot}")

    from alpamayo1_x_rl.utils.light_weight_reasoning_grading_model import (
        get_reasoning_grader_from_config,
    )

    reasoning_score = -1.0
    if pred_cot and gt_cot:
        grader = get_reasoning_grader_from_config(config)
        raw_score = float(grader.score(pred_cot, gt_cot).item())
        reasoning_score = raw_score - 1.0

    # Continuous reward: each component contributes independently, no hard gates.
    ade_threshold = 3.0
    reasoning_threshold = -0.4
    pred_cot_decoded = bool(pred_cot and len(pred_cot.strip()) > 0)

    if pred_cot_decoded and reasoning_score > reasoning_threshold and l2_dist < ade_threshold:
        final_reward = (
            -w["traj_l2_weight"] * (l2_dist / ade_threshold)
            + w["comfort_weight"] * comfort_score
            + w["reasoning_weight"] * (reasoning_score / reasoning_threshold)
        )
    else:
        final_reward = -1.0

    logger.debug(
        f"[compute_reward] l2={l2_dist:.3f} reasoning={reasoning_score:.3f} "
        f"cot_decoded={pred_cot_decoded} final={final_reward:.4f}"
    )
    reward_dict: dict[str, float] = {
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        "reasoning_score": float(reasoning_score),
        "reward": float(final_reward),
    }

    return reward_dict["reward"], reward_dict
