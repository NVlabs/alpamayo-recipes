# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free validation shared by PAI readers and preparation commands."""

import json
import re
from numbers import Integral

EDGE_MIN_T0_US = 2_000_000
EDGE_MAX_T0_US = 13_600_000
LEGACY_MIN_T0_US = 1_600_001


def parse_chunk_ids(value):
    """Parse nonnegative IDs or a half-open range, rejecting empty selections."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
        if match:
            chunks = list(range(int(match[1]), int(match[2])))
        else:
            try:
                chunks = [int(item) for item in text.split()]
            except ValueError as exc:
                raise ValueError("chunk_ids must be integers or a half-open range such as 0-3") from exc
    elif isinstance(value, Integral) and not isinstance(value, bool):
        chunks = [int(value)]
    else:
        try:
            chunks = list(value)
        except TypeError as exc:
            raise ValueError("chunk_ids must be integer IDs or a range string") from exc
    if not chunks or any(isinstance(item, bool) or not isinstance(item, Integral) or item < 0 for item in chunks):
        raise ValueError("chunk_ids must select at least one nonnegative integer chunk")
    return list(dict.fromkeys(int(item) for item in chunks))


def reasoning_text(event):
    """Use nonblank CoC text, falling back to legacy CoT; never stringify null."""
    for key in ("coc", "cot"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def events_have_reasoning(events, min_t0_us=LEGACY_MIN_T0_US, max_t0_us=EDGE_MAX_T0_US):
    """Whether any event has usable text and an integer timestamp in the window."""
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except (ValueError, TypeError):
            return False
    if events is None or isinstance(events, dict) or not hasattr(events, "__iter__"):
        return False
    for event in events:
        if not isinstance(event, dict):
            continue
        timestamp = event.get("event_start_timestamp")
        if (isinstance(timestamp, Integral) and not isinstance(timestamp, bool)
                and min_t0_us <= timestamp <= max_t0_us and reasoning_text(event)):
            return True
    return False


_LATERAL_DIRECTION = r"(?:(?:slight|sharp) )?(?:left|right)"
_DIRECTION = rf"{_LATERAL_DIRECTION}|straight"
_ACTION = (
    rf"Turn (?:{_LATERAL_DIRECTION})|Continue(?: (?:{_DIRECTION}|at fork))?"
    rf"|Keep (?:{_DIRECTION})|Merge(?: (?:{_DIRECTION}))?"
    rf"|Take the ramp(?: (?:{_DIRECTION}))?|Make a U-turn"
    r"|Enter roundabout|Exit (?:roundabout|rotary)|At roundabout, take exit [1-9][0-9]*"
    r"|Continue around the roundabout and take the next exit"
    r"|Head [A-Za-z ]+|Depart|Arrive at destination"
)
_NAVIGATION = re.compile(
    rf"(?:{_ACTION})(?: onto [^\n]+?)?(?: \([^\n()]+\))?(?: in (?P<distance>[0-9]+)m)?"
)


def validate_navigation(text, context="navigation"):
    """Check route-instruction syntax and the 250 m horizon, not route semantics."""
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError(f"{context}: navigation must be a nonblank string without surrounding whitespace")
    match = _NAVIGATION.fullmatch(text)
    if match is None or (match['distance'] is not None and int(match['distance']) > 250):
        raise ValueError(
            f"{context}: unsupported navigation format {text!r}; use 'Continue straight', "
            "a maneuver such as 'Turn left in 25m' (integer distance 0-250 m, "
            "optional 'onto <road>'), or an active maneuver without a distance. "
            "Review alignment with the recorded event; this check does not infer intent."
        )
    # A road name must not absorb an invalid trailing distance suffix.
    suffix = re.search(r" in ([^ ]+)m$", text)
    if suffix and (not suffix[1].isdigit() or int(suffix[1]) > 250):
        raise ValueError(f"{context}: navigation distance must be an integer between 0 and 250 m")
    return text
