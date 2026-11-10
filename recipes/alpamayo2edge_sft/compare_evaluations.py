# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare completed trajectory evaluations using a paired cluster bootstrap.

Use ``--means-only`` for descriptive paired means without session metadata or an
interval. Supply repeated ``--candidate NAME DIR [DIR ...]`` groups to compare
all checkpoints on one shared cohort, or ``--sft`` for a single pair.
``--drop-failed-events`` restricts every supplied checkpoint and seed to events
that generated successfully everywhere; its report lists exclusions. This
conditional comparison does not measure performance on failed events. Separate
pairwise invocations can retain different events and cannot establish a shared
cohort for checkpoint selection.

Each input directory must contain the unmodified ``evaluation_protocol.json``,
``trajectory_samples.jsonl``, and ``trajectory_summary.json`` emitted by
``evaluate_hf``. Events are identified by clip ID, requested timestamp, and actual
camera anchor. Every run must contain exactly the same attempted events and
all checkpoints must have exactly the same evaluation seeds.

Supply ``--session-map`` with a JSON object mapping clip IDs to driving-session
IDs, or ``--session-metadata`` with a parquet containing a session column.
Without session metadata, explicitly opt into ``--assume-independent-clips``
for a clip bootstrap conditional on clip independence. Correlations between clips
from the same drive may underestimate uncertainty; this does not establish
drive-level generalization. There is no automatic fallback to clip independence.
Repeated seeds are averaged within each event before whole sessions (or clips in
the explicit clip mode) are resampled; each bootstrap mean weights events equally,
including when clusters contain different numbers of events. The interval describes
cluster sampling uncertainty conditional on the evaluated checkpoints and seeds.
It does not correct for choosing the best checkpoint on the same validation data
or account for variation across training runs or the full distribution of
generation randomness.
Versioned protocols also check the saved dataset inputs, collator, precision and
batch size. Legacy unversioned protocols remain readable, but the report marks
that their omitted input settings cannot be checked.
"""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from alpamayo2edge_sft.evaluation_identity import comparable_protocol


_IDENTITY_FIELDS = {"checkpoint", "seed", "loss_weights", "lm_token_loss_weights"}
_REQUIRED_PROTOCOL = {
    "checkpoint", "seed", "mode", "max_eval_steps", "validation_chunks",
    "num_traj_samples", "num_traj_sets", "num_samples_per_forward", "candidate_count",
    "metric", "conditioning", "generate_cot", "navigation_input", "preprocess",
    "annotations_path", "generation_failure_policy", "per_sample_results",
}
_EVENT_FIELDS = ("clip_id", "requested_t0_us", "t0_us")
_VERSIONED_PROTOCOL_FIELDS = {
    "validation_dataset", "collator", "autocast_dtype", "per_device_eval_batch_size",
}
_CLIP_INDEPENDENCE_WARNING = (
    "Assumes independent clips. Correlations between clips from the same drive may "
    "cause this interval to underestimate uncertainty; it does not establish "
    "drive-level generalization."
)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return number


def _read_json(text: str, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text, object_pairs_hook=_object, parse_constant=_invalid_constant,
            parse_float=_json_float,
        )
    except ValueError as error:
        raise ValueError(f"{context}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a JSON object")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    return _read_json(path.read_text(), str(path))


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _finite_score(value: Any, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{context} must be a finite nonnegative number")
    return float(value)


def load_clip_sessions(
    path: str | Path, clip_ids: Sequence[str],
) -> tuple[dict[str, str], str]:
    """Read used clips from a parquet index with one unambiguous session column.

    The clip ID may be a ``clip_id`` column or the dataframe index. Supported
    session columns are ``session_id``, ``session``, and ``drive_id``. If multiple
    such columns exist, their values must agree for every evaluated clip.
    Missing or null sessions for unused clips do not prevent a comparison.
    """
    import pandas as pd

    frame = pd.read_parquet(path)
    if not frame.columns.is_unique:
        raise ValueError(f"{path}: clip index has duplicate column names")
    if "clip_id" in frame.columns:
        frame = frame.set_index("clip_id")
    if isinstance(frame.index, pd.MultiIndex):
        raise ValueError(f"{path}: clip index requires a single clip_id index or column")
    columns = [key for key in ("session_id", "session", "drive_id") if key in frame.columns]
    if not columns:
        raise ValueError(
            f"{path}: no session_id, session, or drive_id column; "
            "provide --session-map, --session-metadata with a session_id, session, "
            "or drive_id column, --assume-independent-clips for an interval conditional "
            "on clip independence, or --means-only for descriptive means without an interval"
        )
    used = set(clip_ids)
    selected = frame.loc[frame.index.isin(used), columns]
    mapping = {}
    for clip, row in selected.iterrows():
        session = _nonempty_string(row[columns[0]], f"{path}: session for clip {clip!r}")
        for column in columns[1:]:
            other = _nonempty_string(row[column], f"{path}: {column} for clip {clip!r}")
            if session != other:
                raise ValueError(f"{path}: conflicting session columns for clip {clip!r}")
        if clip in mapping and mapping[clip] != session:
            raise ValueError(f"{path}: contradictory duplicate clip ID {clip!r}")
        mapping[clip] = session
    if used - mapping.keys():
        raise ValueError(f"{path}: missing sessions for {len(used - mapping.keys())} used clips")
    return mapping, columns[0]


def _load_run(directory: str | Path, *, drop_failed_events: bool = False) -> dict[str, Any]:
    directory = Path(directory)
    protocol = _load_object(directory / "evaluation_protocol.json")
    missing = _REQUIRED_PROTOCOL - protocol.keys()
    if missing:
        raise ValueError(f"{directory}: missing protocol fields: {sorted(missing)}")
    if "schema_version" in protocol:
        if type(protocol["schema_version"]) is not int or protocol["schema_version"] != 1:
            raise ValueError(f"{directory}: unsupported protocol schema_version")
        missing = _VERSIONED_PROTOCOL_FIELDS - protocol.keys()
        if missing:
            raise ValueError(f"{directory}: missing versioned protocol fields: {sorted(missing)}")
        for field in ("validation_dataset", "collator"):
            if not isinstance(protocol[field], dict) or not protocol[field]:
                raise ValueError(f"{directory}: protocol {field} must be a nonempty object")
        _nonempty_string(protocol["autocast_dtype"], f"{directory}: autocast_dtype")
        batch_size = protocol["per_device_eval_batch_size"]
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError(f"{directory}: per_device_eval_batch_size must be a positive integer")
    if protocol["mode"] != "trajectory":
        raise ValueError(f"{directory}: requires trajectory evaluation")
    if type(protocol["seed"]) is not int or protocol["seed"] < 0:
        raise ValueError(f"{directory}: protocol seed must be a nonnegative integer")
    _nonempty_string(protocol["checkpoint"], "protocol checkpoint")
    if protocol["per_sample_results"] != "trajectory_samples.jsonl":
        raise ValueError(f"{directory}: unexpected per_sample_results filename")
    for key in ("num_traj_samples", "num_traj_sets", "num_samples_per_forward"):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f"{directory}: protocol {key} must be a positive integer")
    if protocol["candidate_count"] != protocol["num_traj_samples"] * protocol["num_traj_sets"]:
        raise ValueError(f"{directory}: candidate_count does not match the sampling protocol")
    summary = _load_object(directory / "trajectory_summary.json")
    allowed_statuses = {"completed"}
    if drop_failed_events:
        allowed_statuses.add("completed_with_generation_failures")
    if summary.get("status") not in allowed_statuses:
        raise ValueError(f"{directory}: evaluation summary must have status {sorted(allowed_statuses)}")
    records = {}
    failed_events = set()
    with (directory / "trajectory_samples.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            context = f"{directory}/trajectory_samples.jsonl:{line_number}"
            row = _read_json(line, context)
            failed = row.get("status") == "batch_missing_trajectory_anchor"
            if row.get("status") != "ok" and not (drop_failed_events and failed):
                raise ValueError(f"{context}: failure or missing status; require every row to be ok "
                                 "unless explicitly dropping known trajectory-anchor failures")
            clip_id = _nonempty_string(row.get("clip_id"), f"{context}: clip_id")
            for field in _EVENT_FIELDS[1:]:
                if type(row.get(field)) is not int:
                    raise ValueError(f"{context}: {field} must be an integer")
            key = (clip_id, row["requested_t0_us"], row["t0_us"])
            if key in records:
                raise ValueError(f"{context}: duplicate event")
            if failed:
                if "min_ade" not in row or row["min_ade"] is not None:
                    raise ValueError(f"{context}: a failed event must have null min_ade")
                score = None
                failed_events.add(key)
            else:
                score = _finite_score(row.get("min_ade"), f"{context}: min_ade")
            session = row.get("session_id")
            if "session_id" in row:
                _nonempty_string(session, f"{context}: session_id")
            records[key] = (score, session)
    if not records:
        raise ValueError(f"{directory}: no evaluation events")
    successful = len(records) - len(failed_events)
    expected_counts = {"eval_samples": successful, "eval_attempted_samples": len(records),
                       "eval_failed_samples": len(failed_events)}
    for field, expected in expected_counts.items():
        if type(summary.get(field)) is not int or summary[field] != expected:
            raise ValueError(f"{directory}: summary {field} does not match the event count")
    coverage = summary.get("eval_generation_coverage")
    if (type(coverage) not in (int, float) or not math.isfinite(coverage)
            or not math.isclose(coverage, successful / len(records), rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f"{directory}: summary generation coverage does not match the event scores")
    expected_status = "completed_with_generation_failures" if failed_events else "completed"
    if summary["status"] != expected_status:
        raise ValueError(f"{directory}: summary status does not match the generation failures")
    mean_field = "eval_min_ade_successful" if failed_events else "eval_min_ade"
    if failed_events and "eval_min_ade" in summary:
        raise ValueError(f"{directory}: failures must not publish an unconditional eval_min_ade")
    if successful:
        try:
            observed_mean = math.fsum(row[0] for row in records.values() if row[0] is not None) / successful
        except OverflowError as error:
            raise ValueError(f"{directory}: event scores overflow floating-point aggregation") from error
        summary_mean = _finite_score(summary.get(mean_field), f"{directory}: {mean_field}")
        if not math.isclose(observed_mean, summary_mean, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{directory}: summary minADE does not match the event scores")
    elif mean_field in summary:
        raise ValueError(f"{directory}: no successful events for the reported mean")
    return {"directory": str(directory), "protocol": protocol, "records": records,
            "failed_events": failed_events, "summary": summary}


def _load_arm(
    directories: Sequence[str | Path], name: str, *, drop_failed_events: bool = False,
) -> dict[int, dict[str, Any]]:
    if isinstance(directories, (str, Path)) or not directories:
        raise ValueError(f"{name} must contain one or more run directories")
    runs = {}
    checkpoint = None
    for directory in directories:
        run = _load_run(directory, drop_failed_events=drop_failed_events)
        protocol = run["protocol"]
        seed = protocol["seed"]
        if seed in runs:
            raise ValueError(f"{name}: duplicate evaluation seed {seed}")
        if checkpoint is not None and checkpoint != protocol["checkpoint"]:
            raise ValueError(f"{name}: all repeated seeds must evaluate the same checkpoint")
        checkpoint = protocol["checkpoint"]
        runs[seed] = run
    return runs


def _sessions(
    runs: list[dict[str, Any]],
    events: list[tuple[str, int, int]],
    session_mapping: Mapping[str, str] | str | Path | None,
) -> tuple[list[str], str]:
    if isinstance(session_mapping, (str, Path)):
        session_mapping = _load_object(Path(session_mapping))
    if session_mapping is not None and not isinstance(session_mapping, Mapping):
        raise ValueError("session_mapping must be a clip-to-session mapping or JSON file")
    if session_mapping is not None:
        for clip, session in session_mapping.items():
            _nonempty_string(clip, "session mapping clip ID")
            _nonempty_string(session, "session mapping session ID")
    clip_sessions = {}
    for run in runs:
        for event in events:
            clip = event[0]
            recorded = run["records"][event][1]
            mapped = session_mapping.get(clip) if session_mapping is not None else None
            if session_mapping is not None and mapped is None:
                raise ValueError(f"Session mapping is missing clip {clip!r}")
            if mapped is not None and recorded is not None and mapped != recorded:
                raise ValueError(f"Session mapping disagrees with metadata for clip {clip!r}")
            session = mapped if mapped is not None else recorded
            if session is None:
                raise ValueError(
                    "Driving-session IDs are required: supply --session-map, "
                    "--session-metadata with a session_id, session, or drive_id column, "
                    "--assume-independent-clips for an interval conditional on clip independence, "
                    "or --means-only for descriptive means without an interval; "
                    "clip independence is never assumed automatically"
                )
            if clip in clip_sessions and clip_sessions[clip] != session:
                raise ValueError(f"Inconsistent session metadata for clip {clip!r}")
            clip_sessions[clip] = session
    source = "explicit_clip_to_session_mapping" if session_mapping is not None else "sample_metadata"
    return [clip_sessions[event[0]] for event in events], source


def _validate_comparison_options(
    *, session_mapping, clip_index, replications, seed, confidence,
    drop_failed_events, means_only, assume_independent_clips,
) -> None:
    if type(replications) is not int or replications < 2:
        raise ValueError("replications must be an integer of at least two")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if session_mapping is not None and clip_index is not None:
        raise ValueError("Provide only one of session_mapping and clip_index")
    if any(type(option) is not bool for option in (drop_failed_events, means_only, assume_independent_clips)):
        raise ValueError("drop_failed_events, means_only and assume_independent_clips must be booleans")
    if assume_independent_clips and (means_only or session_mapping is not None or clip_index is not None):
        raise ValueError("assume_independent_clips is mutually exclusive with means_only and session metadata")
    if means_only and (session_mapping is not None or clip_index is not None):
        raise ValueError("means_only does not use session metadata; omit session_mapping and clip_index")


def _prepare_comparison(
    arms: Mapping[str, dict[int, dict[str, Any]]], *,
    session_mapping, clip_index, drop_failed_events, means_only, assume_independent_clips,
) -> dict[str, Any]:
    """Validate every participant before selecting one shared successful cohort."""
    baseline = arms["baseline"]
    for name, arm in arms.items():
        if baseline.keys() != arm.keys():
            raise ValueError(f"Baseline and {name} evaluation seed sets do not match")
    seeds = sorted(baseline)
    runs = [arm[s] for arm in arms.values() for s in seeds]
    reference_protocol = comparable_protocol(runs[0]["protocol"], _IDENTITY_FIELDS)
    reference_events = runs[0]["records"].keys()
    for run in runs[1:]:
        protocol = comparable_protocol(run["protocol"], _IDENTITY_FIELDS)
        # Compare serialized JSON to distinguish true from 1, and 1 from 1.0.
        if json.dumps(protocol, sort_keys=True) != json.dumps(reference_protocol, sort_keys=True):
            changed = sorted(
                key for key in protocol.keys() | reference_protocol.keys()
                if json.dumps(protocol.get(key), sort_keys=True)
                != json.dumps(reference_protocol.get(key), sort_keys=True)
                or (key in protocol) != (key in reference_protocol)
            )
            raise ValueError(f"{run['directory']}: protocol differences in {changed}")
        if run["records"].keys() != reference_events:
            missing = len(reference_events - run["records"].keys())
            extra = len(run["records"].keys() - reference_events)
            raise ValueError(
                f"{run['directory']}: unmatched events ({missing} missing, {extra} extra)"
            )
    excluded_events = set().union(*(run["failed_events"] for run in runs))
    events = sorted(reference_events - excluded_events)
    if not events:
        raise ValueError("No common successful events remain after excluding generation failures")
    clusters, source, cluster_names, cluster_scope = None, None, [], None
    if assume_independent_clips:
        clusters = [event[0] for event in events]
        cluster_names = sorted(set(clusters))
        cluster_scope = "clip"
        if len(cluster_names) < 2:
            raise ValueError("At least two distinct clips are required for a clip bootstrap")
    elif not means_only:
        session_column = None
        if clip_index is not None:
            session_mapping, session_column = load_clip_sessions(clip_index, [e[0] for e in events])
        clusters, source = _sessions(runs, events, session_mapping)
        if session_column is not None:
            source = f"clip_index:{session_column}"
        cluster_names = sorted(set(clusters))
        cluster_scope = "driving_session"
        if len(cluster_names) < 2:
            raise ValueError("At least two driving sessions are required for a session bootstrap")
    event_selection = {
        "policy": "exclude_union_of_generation_failures" if drop_failed_events else "require_all_successful",
        "attempted_events_per_run": len(reference_events),
        "included_events": len(events),
        "excluded_events": len(excluded_events),
        "excluded_event_keys": [dict(zip(_EVENT_FIELDS, event)) for event in sorted(excluded_events)],
        "comparison_coverage": len(events) / len(reference_events),
        "per_run": [
            {"arm": arm, "seed": eval_seed, "directory": run["directory"],
             "attempted_events": len(run["records"]),
             "successful_events": len(run["records"]) - len(run["failed_events"]),
             "failed_events": len(run["failed_events"]),
             "generation_coverage": run["summary"]["eval_generation_coverage"]}
            for arm, arm_runs in arms.items()
            for eval_seed, run in sorted(arm_runs.items())
        ],
        "estimand": ("Paired minADE conditional on successful generation in every arm and seed; "
                     "excluded events have unknown trajectory error. This is not full-dataset quality."
                     if excluded_events else "Paired minADE over every attempted event."),
    }
    return {
        "seeds": seeds, "reference_protocol": reference_protocol,
        "attempted_events": sorted(reference_events), "events": events,
        "excluded_events": excluded_events, "clusters": clusters, "cluster_scope": cluster_scope,
        "source": source, "cluster_names": cluster_names,
        "event_selection": event_selection,
    }


def _compare_pair(
    baseline, sft, comparison, *, replications, seed, confidence, means_only,
) -> dict[str, Any]:
    """Score a pair on the already validated cohort shared by every participant."""
    seeds = comparison["seeds"]
    events = comparison["events"]
    excluded_events = comparison["excluded_events"]
    clusters = comparison["clusters"]
    cluster_scope = comparison["cluster_scope"]
    clip_mode = cluster_scope == "clip"
    source = comparison["source"]
    cluster_names = comparison["cluster_names"]
    reference_protocol = comparison["reference_protocol"]
    event_selection = comparison["event_selection"]
    lower = upper = counts = None
    base_values = np.array([[baseline[s]["records"][e][0] for e in events] for s in seeds])
    sft_values = np.array([[sft[s]["records"][e][0] for e in events] for s in seeds])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            base_event_means = base_values.mean(axis=0)
            sft_event_means = sft_values.mean(axis=0)
            event_deltas = (sft_values - base_values).mean(axis=0)
            base_mean = float(base_event_means.mean())
            sft_mean = float(sft_event_means.mean())
            delta = float(event_deltas.mean())
            if not means_only:
                cluster_indices = {name: index for index, name in enumerate(cluster_names)}
                indices = np.array([cluster_indices[name] for name in clusters])
                counts = np.bincount(indices)
                totals = np.bincount(indices, weights=event_deltas)
                # bincount's C accumulation does not honor NumPy's error state.
                if not np.isfinite(totals).all():
                    raise FloatingPointError("nonfinite cluster totals")
                rng = np.random.default_rng(seed)
                bootstrap = np.empty(replications)
                # Bound temporary allocations independently of replication count.
                batch_size = max(1, min(1024, 1_000_000 // len(cluster_names)))
                for start in range(0, replications, batch_size):
                    stop = min(start + batch_size, replications)
                    sampled = rng.integers(
                        0, len(cluster_names), size=(stop - start, len(cluster_names)),
                    )
                    bootstrap[start:stop] = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
                tail = (1 - confidence) / 2
                lower, upper = np.quantile(bootstrap, [tail, 1 - tail])
            statistics = [base_mean, sft_mean, delta] + ([] if means_only else [lower, upper])
            if not np.isfinite(statistics).all():
                raise FloatingPointError("nonfinite comparison statistics")
    except FloatingPointError as error:
        raise ValueError("Event scores overflow floating-point comparison arithmetic") from error
    if means_only:
        conclusion = "descriptive_only"
        interpretation = "Paired descriptive means only; no confidence interval or significance claim."
    elif upper < 0:
        conclusion = "evidence_of_lower_sft_min_ade"
        interpretation = "The interval is below zero: evidence of lower SFT minADE."
    elif lower > 0:
        conclusion = "evidence_of_higher_sft_min_ade"
        interpretation = "The interval is above zero: evidence of higher SFT minADE."
    else:
        conclusion = "no_clear_evidence_of_improvement"
        interpretation = (
            "The interval includes zero: no clear evidence of improvement; "
            "this does not establish equivalence or no effect."
        )
    if clip_mode:
        conclusion += "_conditional_on_clip_independence"
        interpretation = "Conditional on independent clips: " + interpretation
    if excluded_events:
        interpretation += " Applies only to the common successful events; failed generations are not scored."
    identity_limitations = []
    identity = reference_protocol.get("validation_data_identity")
    if identity is None:
        identity_limitations.append(
            "No content identity was recorded; matching paths cannot detect edited annotations "
            "or establish the dataset revision."
        )
    elif identity["dataset"]["kind"] == "unverified_local":
        identity_limitations.append(
            "Dataset source has no pinned download manifest; local paths are compared, "
            "but underlying camera/trajectory file contents are not verified."
        )
    else:
        identity_limitations.append(
            "The pinned download manifest identifies the claimed upstream revision; "
            "local camera/trajectory archives are not rehashed."
        )
    uncertainty_scope = "Not estimated; descriptive comparison only."
    if not means_only:
        uncertainty_scope = (
            "Clip sampling conditional on the evaluated checkpoints and seed set. "
            + _CLIP_INDEPENDENCE_WARNING if clip_mode else
            "Session sampling conditional on the evaluated checkpoints and seed set; assumes "
            "independent driving sessions."
        )
        uncertainty_scope += (
            " The interval does not correct for choosing the best checkpoint on the same "
            "validation data or account for variation across training runs or the full "
            "distribution of generation randomness."
        )
    return {
        "metric": "minADE",
        "units": "meters",
        "baseline_mean_min_ade": base_mean,
        "sft_mean_min_ade": sft_mean,
        "delta_sft_minus_baseline": delta,
        "confidence_interval": None if means_only else {
            "method": ("paired clip cluster bootstrap percentile" if clip_mode else
                       "paired driving-session cluster bootstrap percentile"),
            "confidence": confidence, "lower": float(lower), "upper": float(upper),
        },
        "bootstrap_seed": None if means_only else seed,
        "bootstrap_replications": None if means_only else replications,
        "evaluation_seeds": seeds,
        "counts": {
            "seeds": len(seeds), "events": len(events), "clips": len({e[0] for e in events}),
            "sessions": len(cluster_names) if cluster_scope == "driving_session" else None,
            "paired_event_seed_observations": len(events) * len(seeds),
            "events_per_session": ({"min": int(counts.min()), "max": int(counts.max())}
                                   if cluster_scope == "driving_session" else None),
            **({"clusters": len(cluster_names),
                "events_per_cluster": {"min": int(counts.min()), "max": int(counts.max())}}
               if clip_mode else {}),
        },
        "cluster_scope": cluster_scope,
        "event_selection": event_selection,
        "warnings": ([_CLIP_INDEPENDENCE_WARNING] if clip_mode else [])
                    + ([event_selection["estimand"]] if excluded_events else []),
        "session_mapping_source": source,
        "estimator": "equal event weights; paired seed deltas averaged within each event",
        "uncertainty_scope": uncertainty_scope,
        "conclusion": conclusion,
        "interpretation": interpretation,
        "baseline_checkpoint": baseline[seeds[0]]["protocol"]["checkpoint"],
        "sft_checkpoint": sft[seeds[0]]["protocol"]["checkpoint"],
        "baseline_runs": [baseline[s]["directory"] for s in seeds],
        "sft_runs": [sft[s]["directory"] for s in seeds],
        "matched_protocol": reference_protocol,
        "protocol_fields_allowed_to_differ": sorted(_IDENTITY_FIELDS),
        "protocol_comparability": {
            "schema_version": reference_protocol.get("schema_version"),
            "scope": (
                "recorded_inputs_and_sampling_settings"
                if "schema_version" in reference_protocol else "legacy_recorded_settings_only"
            ),
            "limitations": identity_limitations + (
                [] if "schema_version" in reference_protocol else [
                    "Legacy protocols omit camera selection, frame offsets, image pixel budgets, "
                    "other validation dataset inputs, collator settings, autocast dtype and "
                    "evaluation batch size. Matching their saved fields does not certify "
                    "equality of these omitted settings."
                ]
            ),
        },
    }


def compare_evaluations(
    baseline_dirs: Sequence[str | Path],
    sft_dirs: Sequence[str | Path],
    *,
    session_mapping: Mapping[str, str] | str | Path | None = None,
    clip_index: str | Path | None = None,
    replications: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    drop_failed_events: bool = False,
    means_only: bool = False,
    assume_independent_clips: bool = False,
) -> dict[str, Any]:
    """Compare one checkpoint pair across repeated seeds.

    Optional failure exclusion is local to this pair. Use compare_checkpoint_set
    when choosing among candidates so all pairs share the same retained events.
    Negative SFT-minus-baseline minADE is better. Means-only mode provides no
    confidence or significance claim. Opt-in clip intervals assume independent
    clips and do not establish drive-level generalization.
    """
    _validate_comparison_options(
        session_mapping=session_mapping, clip_index=clip_index, replications=replications,
        seed=seed, confidence=confidence, drop_failed_events=drop_failed_events, means_only=means_only,
        assume_independent_clips=assume_independent_clips,
    )
    baseline = _load_arm(baseline_dirs, "baseline", drop_failed_events=drop_failed_events)
    sft = _load_arm(sft_dirs, "SFT", drop_failed_events=drop_failed_events)
    comparison = _prepare_comparison(
        {"baseline": baseline, "sft": sft}, session_mapping=session_mapping,
        clip_index=clip_index, drop_failed_events=drop_failed_events, means_only=means_only,
        assume_independent_clips=assume_independent_clips,
    )
    return _compare_pair(
        baseline, sft, comparison, replications=replications, seed=seed,
        confidence=confidence, means_only=means_only,
    )


def _cohort_sha256(events: Sequence[tuple[str, int, int]]) -> str:
    """Hash compact UTF-8 JSON of sorted [clip, requested time, anchor] triples."""
    encoded = json.dumps(sorted(events), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_checkpoint_set(
    baseline_dirs: Sequence[str | Path],
    candidates: Mapping[str, Sequence[str | Path]],
    *,
    session_mapping: Mapping[str, str] | str | Path | None = None,
    clip_index: str | Path | None = None,
    replications: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
    drop_failed_events: bool = False,
    means_only: bool = False,
    assume_independent_clips: bool = False,
) -> dict[str, Any]:
    """Compare all candidates with step zero on one cohort across every seed.

    Candidate labels identify groups of repeated-seed evaluations of one distinct
    checkpoint. All attempted event sets, protocols and seed sets must match.
    Optional known-failure exclusion is global to the supplied checkpoint set.
    The report makes no automatic selection or simultaneous significance claim.
    Opt-in clip intervals assume independent clips and do not establish
    drive-level generalization.
    """
    _validate_comparison_options(
        session_mapping=session_mapping, clip_index=clip_index, replications=replications,
        seed=seed, confidence=confidence, drop_failed_events=drop_failed_events, means_only=means_only,
        assume_independent_clips=assume_independent_clips,
    )
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a nonempty mapping of labels to run directories")
    for name in candidates:
        _nonempty_string(name, "candidate label")
        if name == "baseline":
            raise ValueError("candidate label 'baseline' is reserved")
    baseline = _load_arm(baseline_dirs, "baseline", drop_failed_events=drop_failed_events)
    arms = {"baseline": baseline}
    checkpoints = {next(iter(baseline.values()))["protocol"]["checkpoint"]}
    for name in sorted(candidates):
        arm = _load_arm(candidates[name], name, drop_failed_events=drop_failed_events)
        checkpoint = next(iter(arm.values()))["protocol"]["checkpoint"]
        if checkpoint in checkpoints:
            raise ValueError(f"{name}: duplicate checkpoint identity {checkpoint!r}")
        checkpoints.add(checkpoint)
        arms[name] = arm
    comparison = _prepare_comparison(
        arms, session_mapping=session_mapping, clip_index=clip_index,
        drop_failed_events=drop_failed_events, means_only=means_only,
        assume_independent_clips=assume_independent_clips,
    )
    selection = comparison["event_selection"]
    selection.update(
        scope="all_supplied_checkpoints_and_seeds",
        cohort_sha256=_cohort_sha256(comparison["events"]),
        attempted_cohort_sha256=_cohort_sha256(comparison["attempted_events"]),
    )
    for row in selection["per_run"]:
        row["checkpoint"] = arms[row["arm"]][row["seed"]]["protocol"]["checkpoint"]
    reports = {
        name: _compare_pair(
            baseline, arms[name], comparison, replications=replications, seed=seed,
            confidence=confidence, means_only=means_only,
        )
        for name in sorted(candidates)
    }
    first = next(iter(reports.values()))
    guidance = (
        "Step zero remains eligible. No automatic winner is selected. "
        "Use the shared-cohort scores together with each run's generation coverage; "
        "excluded-event quality is unknown. Confirm selected-checkpoint quality on "
        "a separate untouched test set. Adding or removing a candidate requires "
        "recomputing the shared cohort for the entire set."
    )
    interval_warning = (
        "Confidence intervals are unadjusted per-comparison intervals, not simultaneous "
        "intervals or evidence that a selected checkpoint is best among all candidates."
    )
    if not means_only:
        for report in reports.values():
            report["uncertainty_scope"] += " " + interval_warning
    return {
        "comparison_scope": "checkpoint_set",
        "metric": "minADE", "units": "meters",
        "baseline_checkpoint": first["baseline_checkpoint"],
        "baseline_runs": first["baseline_runs"],
        "baseline_mean_min_ade": first["baseline_mean_min_ade"],
        "evaluation_seeds": comparison["seeds"],
        "event_selection": selection,
        "comparisons": reports,
        "selection_guidance": guidance,
        "warnings": first["warnings"] + ([] if means_only else [interval_warning]),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True, help="Baseline run directories")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--sft", nargs="+", help="Run directories for one SFT checkpoint")
    targets.add_argument(
        "--candidate", action="append", nargs="+", metavar="NAME_OR_DIR",
        help="Candidate label followed by its repeated-seed run directories; repeat for every checkpoint",
    )
    session_source = parser.add_mutually_exclusive_group()
    session_source.add_argument(
        "--session-map", type=Path, help="JSON object: clip ID -> driving-session ID",
    )
    session_source.add_argument(
        "--clip-index", "--session-metadata", type=Path,
        help="Per-clip parquet metadata with a session_id, session, or drive_id column",
    )
    session_source.add_argument(
        "--means-only", action="store_true",
        help="Report descriptive paired means without session metadata or a confidence interval",
    )
    session_source.add_argument(
        "--assume-independent-clips", action="store_true",
        help="Bootstrap whole clips assuming independence; same-drive correlations may underestimate uncertainty",
    )
    parser.add_argument(
        "--drop-failed-events", action="store_true",
        help="Exclude the union of recorded trajectory-anchor failures across all arms and seeds; report conditional means and coverage",
    )
    parser.add_argument("--replications", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, help="Write report JSON; also printed to stdout")
    args = parser.parse_args(argv)
    try:
        options = dict(
            session_mapping=args.session_map, clip_index=args.clip_index,
            replications=args.replications, seed=args.seed, confidence=args.confidence,
            drop_failed_events=args.drop_failed_events, means_only=args.means_only,
            assume_independent_clips=args.assume_independent_clips,
        )
        if args.candidate is None:
            result = compare_evaluations(args.baseline, args.sft, **options)
        else:
            candidates = {}
            for group in args.candidate:
                if len(group) < 2:
                    raise ValueError("--candidate requires NAME and at least one run directory")
                name, *directories = group
                if name in candidates:
                    raise ValueError(f"Duplicate candidate label {name!r}")
                candidates[name] = directories
            result = compare_checkpoint_set(args.baseline, candidates, **options)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
