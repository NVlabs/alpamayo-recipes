from __future__ import annotations

import importlib.util
import logging
import sys
import types
from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = (
    ROOT
    / "recipes"
    / "alpamayo1_x_rl"
    / "rewards"
    / "aggregated_reward_with_reasoning.py"
)


class _Trajectory:
    """Support the reward's indexing without requiring a tensor runtime."""

    def __getitem__(self, key: object) -> _Trajectory:
        return self


@pytest.fixture
def reward_case(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Load the real aggregation function with only its external inputs stubbed."""
    case = types.SimpleNamespace(raw_score=1.0, ade=1.0)
    case.extract_cot = Mock(return_value=["prediction"])
    case.grader = Mock()
    case.grader.score.side_effect = lambda *_: types.SimpleNamespace(
        item=lambda: case.raw_score
    )
    case.get_grader = Mock(return_value=case.grader)

    dependencies = {
        "alpamayo_r1.models.token_utils": {
            "extract_between_special_tokens": case.extract_cot,
        },
        "alpamayo1_x_rl.rewards.comfort_reward": {
            "compute_comfort": lambda *_: {"comfort": 0.75},
        },
        "alpamayo1_x_rl.rewards.traj_reward": {
            "calculate_ade": lambda *_: case.ade,
        },
        "alpamayo1_x_rl.utils.trajectory_decode": {
            "decode_rollout_trajectory": lambda *_, **__: (
                _Trajectory(),
                _Trajectory(),
            ),
        },
        "alpamayo1_x_rl.utils.light_weight_reasoning_grading_model": {
            "get_reasoning_grader_from_config": case.get_grader,
        },
        "cosmos_rl.utils.logging": {"logger": logging.getLogger(__name__)},
    }
    for name, attributes in dependencies.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_reasoning_reward_test", REWARD_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    case.config = types.SimpleNamespace(
        custom={
            "alpamayo": {
                "reward": {
                    "traj_l2_weight": 0.2,
                    "comfort_weight": 0.0,
                    "reasoning_weight": 0.3,
                },
            },
        },
    )
    case.reference = {
        "ego_future_xyz": _Trajectory(),
        "ego_history_xyz": _Trajectory(),
        "ego_history_rot": _Trajectory(),
        "cot": "reference",
    }

    def compute() -> tuple[float, dict[str, float]]:
        return module.compute_reward(
            "rollout",
            case.reference,
            tokenizer=None,
            traj_tokenizer=None,
            config=case.config,
            model_config=None,
        )

    case.compute = compute
    return case


def test_better_reasoning_increases_reward(reward_case: types.SimpleNamespace) -> None:
    rewards = []
    for raw_score in (0.61, 0.7, 0.8, 0.9, 1.0):
        reward_case.raw_score = raw_score
        reward, metrics = reward_case.compute()
        rewards.append(reward)
        assert metrics == pytest.approx(
            {
                "traj_L2": 1.0,
                "comfort_reward": -0.25,
                "reasoning_score": raw_score - 1.0,
                "reward": reward,
            }
        )

    assert rewards == pytest.approx(
        [-0.3591666667, -0.2916666667, -0.2166666667, -0.1416666667, -0.0666666667]
    )
    assert all(later > earlier for earlier, later in pairwise(rewards))


@pytest.mark.parametrize(
    ("raw_score", "ade"),
    [(0.0, 1.0), (0.59, 1.0), (0.6, 1.0), (1.0, 3.0), (1.0, 3.01)],
)
def test_failed_gates_return_fixed_penalty(
    reward_case: types.SimpleNamespace,
    raw_score: float,
    ade: float,
) -> None:
    reward_case.raw_score = raw_score
    reward_case.ade = ade

    reward, metrics = reward_case.compute()

    assert reward == metrics["reward"] == -1.0
    assert metrics["reasoning_score"] == pytest.approx(raw_score - 1.0)
    assert metrics["traj_L2"] == ade


@pytest.mark.parametrize(
    ("pred_cot", "reference_cot"),
    [
        pytest.param(None, {"cot": "reference"}, id="missing-prediction"),
        pytest.param("", {"cot": "reference"}, id="empty-prediction"),
        pytest.param(" \n\t", {"cot": "reference"}, id="whitespace-prediction"),
        pytest.param("prediction", {}, id="missing-reference"),
        pytest.param("prediction", {"cot": None}, id="null-reference"),
        pytest.param("prediction", {"cot": ""}, id="empty-reference"),
        pytest.param("prediction", {"cot": " \n\t"}, id="whitespace-reference"),
    ],
)
def test_unavailable_reasoning_returns_penalty_without_loading_grader(
    reward_case: types.SimpleNamespace,
    pred_cot: str | None,
    reference_cot: dict[str, str | None],
) -> None:
    reward_case.extract_cot.return_value = [pred_cot]
    del reward_case.reference["cot"]
    reward_case.reference.update(reference_cot)

    reward, metrics = reward_case.compute()

    assert reward == metrics["reward"] == -1.0
    assert metrics["reasoning_score"] == -1.0
    reward_case.get_grader.assert_not_called()
    reward_case.grader.score.assert_not_called()
