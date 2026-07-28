from __future__ import annotations

import importlib.util
import json
import logging as stdlib_logging
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAI_UTILS_PATH = ROOT / "src" / "alpamayo" / "data" / "pai_utils.py"


class _ReasoningFrame:
    columns = {"events"}

    def __init__(self, events_by_clip: dict[str, str]) -> None:
        self._events_by_clip = events_by_clip

    def __getitem__(self, key: str) -> dict[str, str]:
        assert key == "events"
        return self._events_by_clip


def _load_pai_utils(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    numpy = types.ModuleType("numpy")
    numpy.int64 = int
    numpy.isscalar = lambda value: isinstance(
        value, (str, bytes, int, float, bool, type(None))
    )
    numpy.array = lambda values, dtype=None: list(values)
    numpy.asarray = lambda values, dtype=None: list(values)

    pandas = types.ModuleType("pandas")
    pandas.DataFrame = _ReasoningFrame
    pandas.Series = dict
    pandas.isna = lambda value: value is None

    egomotion = types.ModuleType("physical_ai_av.egomotion")
    video = types.ModuleType("physical_ai_av.video")
    physical_ai_av = types.ModuleType("physical_ai_av")
    physical_ai_av.__path__ = []
    physical_ai_av.egomotion = egomotion
    physical_ai_av.video = video
    physical_ai_av_dataset = types.ModuleType("physical_ai_av.dataset")
    physical_ai_av_dataset.Features = object

    alpamayo_r1 = types.ModuleType("alpamayo_r1")
    alpamayo_r1.__path__ = []
    common = types.ModuleType("alpamayo_r1.common")
    ranked_logging = types.ModuleType("alpamayo_r1.common.logging")
    ranked_logging.RankedLogger = (
        lambda name, rank_zero_only=False: stdlib_logging.getLogger(name)
    )
    common.logging = ranked_logging
    alpamayo_r1.common = common

    for name, module in {
        "numpy": numpy,
        "pandas": pandas,
        "physical_ai_av": physical_ai_av,
        "physical_ai_av.egomotion": egomotion,
        "physical_ai_av.video": video,
        "physical_ai_av.dataset": physical_ai_av_dataset,
        "alpamayo_r1": alpamayo_r1,
        "alpamayo_r1.common": common,
        "alpamayo_r1.common.logging": ranked_logging,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_pai_utils_reasoning_test", PAI_UTILS_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("event_fields", "expected"),
    [
        pytest.param({"coc": "current"}, "current", id="current-coc"),
        pytest.param({"cot": "legacy"}, "legacy", id="legacy-cot"),
        pytest.param(
            {"coc": "current", "cot": "legacy"},
            "current",
            id="current-coc-takes-precedence",
        ),
        pytest.param(
            {"coc": "", "cot": "legacy"},
            "",
            id="empty-coc-does-not-fall-back",
        ),
        pytest.param(
            {"coc": None, "cot": "legacy"},
            "None",
            id="null-coc-does-not-fall-back",
        ),
        pytest.param({}, "", id="missing-remains-empty"),
    ],
)
def test_reasoning_event_text_supports_current_and_legacy_keys(
    monkeypatch: pytest.MonkeyPatch,
    event_fields: dict[str, str | None],
    expected: str,
) -> None:
    pai_utils = _load_pai_utils(monkeypatch)
    event = {"event_start_timestamp": 1_000_000, **event_fields}
    frame = _ReasoningFrame({"clip-id": json.dumps([event])})

    parsed = pai_utils.PhysicalAIAVDatasetLocalInterface._read_reasoning_data(frame)

    assert parsed["clip-id"]["cot"] == [expected]
