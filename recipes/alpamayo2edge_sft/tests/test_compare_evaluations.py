# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired evaluation contracts tested without loading a checkpoint or dataset."""

import hashlib
import json

import numpy as np
import pytest

from alpamayo2edge_sft.compare_evaluations import (
    compare_checkpoint_set,
    compare_evaluations,
    load_clip_sessions,
    main,
)


def _write_run(tmp_path, arm, seed, rows, **protocol_updates):
    directory = tmp_path / f"{arm}-{seed}"
    directory.mkdir()
    protocol = {
        "checkpoint": arm,
        "seed": seed,
        "mode": "trajectory",
        "max_eval_steps": -1,
        "max_eval_steps_unit": "dataloader batches",
        "validation_chunks": [0],
        "num_traj_samples": 6,
        "num_traj_sets": 1,
        "num_samples_per_forward": 2,
        "candidate_count": 6,
        "metric": "minADE: mean XY distance in meters",
        "conditioning": "navigation; generated CoT",
        "generate_cot": True,
        "navigation_input": True,
        "preprocess": {"components_order": ["nav_instruction", "prompt", "cot", "traj_future"]},
        "annotations_path": "annotations.jsonl",
        "generation_failure_policy": "exclude whole affected batch",
        "per_sample_results": "trajectory_samples.jsonl",
        "future_steps": 64,
        "time_step_seconds": 0.1,
        "diffusion_steps": 10,
        "temperature": 0.6,
        **protocol_updates,
    }
    (directory / "evaluation_protocol.json").write_text(json.dumps(protocol))
    (directory / "trajectory_samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (directory / "trajectory_summary.json").write_text(json.dumps({
        "status": "completed",
        "eval_samples": len(rows),
        "eval_attempted_samples": len(rows),
        "eval_failed_samples": 0,
        "eval_generation_coverage": 1.0,
        "eval_min_ade": sum(row["min_ade"] for row in rows) / len(rows),
    }))
    return directory


def _rows(scores, sessions=("session-a", "session-a", "session-b")):
    return [
        {"clip_id": f"clip-{i}", "requested_t0_us": 6_000_000, "t0_us": 6_000_010,
         "min_ade": score, "status": "ok", "session_id": sessions[i]}
        for i, score in enumerate(scores)
    ]


def _update_json(path, **updates):
    path.write_text(json.dumps({**json.loads(path.read_text()), **updates}))


def _replace_rows(directory, rows):
    (directory / "trajectory_samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_repeated_seeds_pair_by_event_and_bootstrap_whole_sessions(tmp_path):
    # Sessions have unequal event counts. Averaging session means would yield
    # -2 rather than the correct event-weighted -5/3.
    baseline, sft = [], []
    for seed, base, tuned in [(7, [4, 4, 4], [4, 4, 2]), (11, [6, 6, 6], [4, 4, 2])]:
        baseline.append(_write_run(tmp_path, "base", seed, _rows(base)))
        sft.append(_write_run(tmp_path, "sft", seed, list(reversed(_rows(tuned)))))
    result = compare_evaluations(
        baseline, list(reversed(sft)), seed=13, replications=500, confidence=0.8,
    )
    assert result["baseline_mean_min_ade"] == 5
    assert result["sft_mean_min_ade"] == pytest.approx(10 / 3)
    assert result["delta_sft_minus_baseline"] == pytest.approx(-5 / 3)
    assert result["counts"] == {
        "seeds": 2, "events": 3, "clips": 3, "sessions": 2,
        "paired_event_seed_observations": 6, "events_per_session": {"min": 1, "max": 2},
    }
    # Independent oracle: sample either session, keeping all of its events.
    rng = np.random.default_rng(13)
    draws = rng.integers(0, 2, size=(500, 2))
    session_events = [np.array([-1.0, -1.0]), np.array([-3.0])]
    expected = [
        np.concatenate([session_events[index] for index in draw]).mean() for draw in draws
    ]
    low, high = np.quantile(expected, [0.1, 0.9])
    assert result["confidence_interval"]["lower"] == pytest.approx(low)
    assert result["confidence_interval"]["upper"] == pytest.approx(high)
    assert result["conclusion"] == "evidence_of_lower_sft_min_ade"
    assert compare_evaluations(baseline, sft, seed=13, replications=500, confidence=0.8) == result


def test_seeds_are_averaged_before_resampling_sessions(tmp_path):
    baseline, sft = [], []
    for seed, scores in [(7, [3, 3, 3]), (11, [5, 5, 5])]:
        baseline.append(_write_run(tmp_path, "base", seed, _rows([4, 4, 4])))
        sft.append(_write_run(tmp_path, "sft", seed, _rows(scores)))
    result = compare_evaluations(baseline, sft, replications=200)
    assert result["delta_sft_minus_baseline"] == 0
    assert result["confidence_interval"]["lower"] == 0
    assert result["confidence_interval"]["upper"] == 0
    assert result["conclusion"] == "no_clear_evidence_of_improvement"


def test_clip_bootstrap_pairs_seeds_and_resamples_all_timestamps_within_clips(tmp_path):
    clips = ("a", "a", "a", "b", "c")

    def rows(scores):
        return [
            {"clip_id": clip, "requested_t0_us": 6_000_000 + index * 1_000_000,
             "t0_us": 6_000_010 + index * 1_000_000, "min_ade": score, "status": "ok"}
            for index, (clip, score) in enumerate(zip(clips, scores))
        ]

    baseline, sft = [], []
    for seed, base_score in ((7, 10), (11, 12)):
        baseline.append(_write_run(tmp_path, "base", seed, rows([base_score] * 5)))
        sft.append(_write_run(tmp_path, "sft", seed, list(reversed(rows([10, 8, 6, 4, 2])))))
    with pytest.raises(ValueError, match="Driving-session IDs are required"):
        compare_evaluations(baseline, sft, replications=20)
    result = compare_evaluations(
        baseline, list(reversed(sft)), assume_independent_clips=True,
        seed=13, replications=500, confidence=.8,
    )
    assert result["baseline_mean_min_ade"] == 11
    assert result["sft_mean_min_ade"] == 6
    assert result["delta_sft_minus_baseline"] == -5
    # Average the two seed deltas first, then resample intact clips. Unequal
    # event counts distinguish this estimator from averaging three clip means.
    clip_deltas = [np.array([-1., -3., -5.]), np.array([-7.]), np.array([-9.])]
    draws = np.random.default_rng(13).integers(0, 3, size=(500, 3))
    expected = [np.concatenate([clip_deltas[i] for i in draw]).mean() for draw in draws]
    interval = result["confidence_interval"]
    assert interval["method"] == "paired clip cluster bootstrap percentile"
    assert interval["lower"] == pytest.approx(np.quantile(expected, .1))
    assert interval["upper"] == pytest.approx(np.quantile(expected, .9))
    assert result["counts"] == {
        "seeds": 2, "events": 5, "clips": 3, "sessions": None,
        "paired_event_seed_observations": 10, "events_per_session": None,
        "clusters": 3, "events_per_cluster": {"min": 1, "max": 3},
    }
    assert result["cluster_scope"] == "clip"
    assert result["session_mapping_source"] is None
    assert result["conclusion"] == "evidence_of_lower_sft_min_ade_conditional_on_clip_independence"
    assert result["interpretation"].startswith("Conditional on independent clips:")
    assert "Assumes independent clips" in result["uncertainty_scope"]
    assert "same drive" in result["warnings"][0] and "underestimate uncertainty" in result["warnings"][0]
    assert "does not establish drive-level generalization" in result["uncertainty_scope"]
    assert "same validation data" in result["uncertainty_scope"]
    assert "variation across training runs" in result["uncertainty_scope"]


def test_clip_bootstrap_rejects_one_clip_with_multiple_events(tmp_path):
    rows = _rows([1, 2, 3])
    for index, row in enumerate(rows):
        row.update(clip_id="one-clip", requested_t0_us=6_000_000 + index, t0_us=6_000_010 + index)
        row.pop("session_id")
    baseline = _write_run(tmp_path, "base", 7, rows)
    sft = _write_run(tmp_path, "sft", 7, rows)
    with pytest.raises(ValueError, match="At least two distinct clips"):
        compare_evaluations([baseline], [sft], assume_independent_clips=True)


@pytest.mark.parametrize("other", [{"means_only": True}, {"session_mapping": {}}, {"clip_index": "unused.parquet"}])
@pytest.mark.parametrize("checkpoint_set", [False, True])
def test_clip_assumption_rejects_other_comparison_modes(paired_runs, other, checkpoint_set):
    baseline, sft = paired_runs
    compare = compare_checkpoint_set if checkpoint_set else compare_evaluations
    target = {"candidate": [sft]} if checkpoint_set else [sft]
    with pytest.raises(ValueError, match="assume_independent_clips is mutually exclusive"):
        compare([baseline], target, assume_independent_clips=True, **other)


@pytest.mark.parametrize("other", [
    ["--means-only"], ["--session-map", "unused.json"],
    ["--session-metadata", "unused.parquet"], ["--clip-index", "unused.parquet"],
])
def test_cli_clip_assumption_rejects_other_comparison_modes(paired_runs, other):
    baseline, sft = paired_runs
    with pytest.raises(SystemExit) as error:
        main(["--baseline", str(baseline), "--sft", str(sft), "--assume-independent-clips", *other])
    assert error.value.code == 2


def test_finite_scores_cannot_overflow_bootstrap_into_a_nan_interval(tmp_path):
    # Each run's sum and mean are finite, but sampling the large session twice
    # would overflow its combined total before division by the event count.
    base = _write_run(tmp_path, "base", 7, _rows([0, 0, 0]))
    sft = _write_run(tmp_path, "sft", 7, _rows([6e307, 6e307, 0]))
    with pytest.raises(ValueError, match="overflow floating-point comparison"):
        compare_evaluations([base], [sft], replications=200)


@pytest.fixture
def paired_runs(tmp_path):
    return (
        _write_run(tmp_path, "base", 7, _rows([4, 4, 4])),
        _write_run(tmp_path, "sft", 7, _rows([3, 3, 6])),
    )


def test_interval_spanning_zero_is_not_evidence_of_improvement(paired_runs):
    base, sft = paired_runs
    result = compare_evaluations([base], [sft], replications=200)
    assert result["confidence_interval"]["lower"] < 0 < result["confidence_interval"]["upper"]
    assert result["conclusion"] == "no_clear_evidence_of_improvement"
    assert "does not establish equivalence" in result["interpretation"]


@pytest.mark.parametrize("damage", ["duplicate", "unmatched", "failure", "nan", "inf", "null",
                                  "missing_timestamp", "string_timestamp"])
def test_rejects_invalid_sample_rows(paired_runs, damage):
    base, sft = paired_runs
    rows = _rows([3, 3, 6])
    if damage == "duplicate":
        rows.append(rows[0])
    elif damage == "unmatched":
        rows[0]["clip_id"] = "different-clip"
    elif damage == "failure":
        rows[0]["status"] = "batch_missing_trajectory_anchor"
    elif damage in ("nan", "inf", "null"):
        rows[0]["min_ade"] = {"nan": float("nan"), "inf": float("inf"), "null": None}[damage]
    elif damage == "missing_timestamp":
        del rows[0]["requested_t0_us"]
    else:
        rows[0]["t0_us"] = str(rows[0]["t0_us"])
    _replace_rows(sft, rows)
    with pytest.raises(ValueError):
        compare_evaluations([base], [sft], replications=20)


@pytest.mark.parametrize("updates", [
    {"num_traj_samples": 8, "candidate_count": 8}, {"diffusion_steps": 20},
    {"temperature": 0.7}, {"validation_chunks": [1]}, {"annotations_path": "other.jsonl"},
    {"preprocess": {"components_order": ["prompt", "traj_future"]}},
    {"future_steps": 32}, {"new_sampling_option": True},
])
def test_rejects_protocol_differences(paired_runs, updates):
    base, sft = paired_runs
    _update_json(sft / "evaluation_protocol.json", **updates)
    with pytest.raises(ValueError, match="protocol differences"):
        compare_evaluations([base], [sft], replications=20)


@pytest.mark.parametrize("change", ["same", "missing", "variant", "offsets", "target_step"])
def test_dataset_sampling_protocol_must_match(paired_runs, change):
    base, sft = paired_runs
    sampling = {
        "adapter_variant": "once_causal_images_interpolated_lidar_poses_v1",
        "images": {"requested_offsets_seconds": [-2., -1., -.5, -.1, 0.]},
        "trajectory": {"time_step_seconds": .1},
    }
    _update_json(base / "evaluation_protocol.json", dataset_sampling_protocol=sampling)
    if change == "variant":
        sampling["adapter_variant"] = "different"
    elif change == "offsets":
        sampling["images"]["requested_offsets_seconds"] = [-2., -1., -.5, 0.]
    elif change == "target_step":
        sampling["trajectory"]["time_step_seconds"] = .2
    if change != "missing":
        _update_json(sft / "evaluation_protocol.json", dataset_sampling_protocol=sampling)
    if change == "same":
        result = compare_evaluations([base], [sft], replications=20)
        assert result["matched_protocol"]["dataset_sampling_protocol"] == sampling
    else:
        with pytest.raises(ValueError, match="protocol differences"):
            compare_evaluations([base], [sft], replications=20)


def test_objective_weights_may_differ_between_checkpoints(paired_runs):
    base, sft = paired_runs
    _update_json(base / "evaluation_protocol.json", loss_weights={"action": 1})
    _update_json(sft / "evaluation_protocol.json", loss_weights={"action": 2})
    compare_evaluations([base], [sft], replications=20)


def _versioned_fields():
    return {
        "schema_version": 1,
        "validation_dataset": {
            "camera_features": ["camera_front_wide_120fov"],
            "frame_offsets": [-0.1, 0.0],
        },
        "collator": {"max_length": 8192, "pad_to_fixed_length": False},
        "autocast_dtype": "bfloat16",
        "per_device_eval_batch_size": 1,
    }


def test_legacy_protocol_comparability_is_explicitly_limited(paired_runs):
    base, sft = paired_runs
    result = compare_evaluations([base], [sft], replications=20)
    comparability = result["protocol_comparability"]
    assert comparability["schema_version"] is None
    assert comparability["scope"] == "legacy_recorded_settings_only"
    assert any("does not certify equality" in limitation for limitation in comparability["limitations"])
    assert any("No content identity" in limitation for limitation in comparability["limitations"])


def test_versioned_protocol_checks_include_input_settings(paired_runs):
    base, sft = paired_runs
    for directory in (base, sft):
        _update_json(directory / "evaluation_protocol.json", **_versioned_fields())
    result = compare_evaluations([base], [sft], replications=20)
    assert result["protocol_comparability"] == {
        "schema_version": 1, "scope": "recorded_inputs_and_sampling_settings",
        "limitations": [
            "No content identity was recorded; matching paths cannot detect edited annotations "
            "or establish the dataset revision."
        ],
    }


@pytest.mark.parametrize("missing", [
    "validation_dataset", "collator", "autocast_dtype", "per_device_eval_batch_size",
])
def test_versioned_protocol_rejects_missing_input_settings(paired_runs, missing):
    base, sft = paired_runs
    fields = _versioned_fields()
    del fields[missing]
    for directory in (base, sft):
        _update_json(directory / "evaluation_protocol.json", **fields)
    with pytest.raises(ValueError, match="missing versioned protocol fields"):
        compare_evaluations([base], [sft], replications=20)


@pytest.mark.parametrize("version", [0, 2, None, True, "1"])
def test_protocol_rejects_unsupported_schema_version(paired_runs, version):
    base, sft = paired_runs
    for directory in (base, sft):
        _update_json(directory / "evaluation_protocol.json", schema_version=version)
    with pytest.raises(ValueError, match="unsupported protocol schema_version"):
        compare_evaluations([base], [sft], replications=20)


def test_cannot_mix_legacy_and_versioned_protocols(paired_runs):
    base, sft = paired_runs
    _update_json(sft / "evaluation_protocol.json", **_versioned_fields())
    with pytest.raises(ValueError, match="protocol differences"):
        compare_evaluations([base], [sft], replications=20)


@pytest.mark.parametrize("updates", [
    {"status": "in_progress"}, {"status": "failed"},
    {"status": "completed_with_generation_failures"}, {"eval_failed_samples": 1},
    {"eval_generation_coverage": 0.5}, {"eval_samples": 2}, {"eval_attempted_samples": 4},
    {"eval_min_ade": 100.0},
])
def test_rejects_partial_runs_and_inconsistent_summaries(paired_runs, updates):
    base, sft = paired_runs
    _update_json(sft / "trajectory_summary.json", **updates)
    with pytest.raises(ValueError):
        compare_evaluations([base], [sft], replications=20)


def test_rejects_missing_or_repeated_seeds(paired_runs):
    base, sft = paired_runs
    with pytest.raises(ValueError, match="duplicate evaluation seed"):
        compare_evaluations([base, base], [sft], replications=20)
    _update_json(sft / "evaluation_protocol.json", seed=8)
    with pytest.raises(ValueError, match="seed sets"):
        compare_evaluations([base], [sft], replications=20)


def test_requires_the_same_events_across_repeated_seeds(tmp_path, paired_runs):
    base, sft = paired_runs
    shorter = _rows([4, 4])
    second_base = _write_run(tmp_path, "base", 8, shorter)
    second_sft = _write_run(tmp_path, "sft", 8, shorter)
    with pytest.raises(ValueError, match="unmatched events"):
        compare_evaluations([base, second_base], [sft, second_sft], replications=20)


def test_explicit_mapping_is_required_when_session_metadata_is_absent(paired_runs, tmp_path):
    base, sft = paired_runs
    for directory, scores in [(base, [4, 4, 4]), (sft, [3, 3, 6])]:
        rows = _rows(scores)
        for row in rows:
            del row["session_id"]
        _replace_rows(directory, rows)
    with pytest.raises(ValueError, match="Driving-session IDs are required") as error:
        compare_evaluations([base], [sft], replications=20)
    assert "--session-map" in str(error.value)
    assert "--session-metadata with a session_id, session, or drive_id column" in str(error.value)
    assert "--means-only" in str(error.value)
    assert "--assume-independent-clips" in str(error.value)
    assert "sample record" not in str(error.value)
    mapping = {"clip-0": "session-a", "clip-1": "session-a", "clip-2": "session-b"}
    path = tmp_path / "sessions.json"
    path.write_text(json.dumps(mapping))
    result = compare_evaluations([base], [sft], session_mapping=path, replications=20)
    assert result["counts"]["sessions"] == 2
    assert result["session_mapping_source"] == "explicit_clip_to_session_mapping"
    assert result["cluster_scope"] == "driving_session"
    assert "conditional" in result["uncertainty_scope"]
    assert "does not correct for choosing the best checkpoint on the same validation data" in result["uncertainty_scope"]
    assert "variation across training runs" in result["uncertainty_scope"]


@pytest.mark.parametrize("mapping", [
    {"clip-0": "session-a"},
    {"clip-0": "different", "clip-1": "session-a", "clip-2": "session-b"},
    {"clip-0": "session-a", "clip-1": "", "clip-2": "session-b"},
])
def test_rejects_incomplete_conflicting_or_empty_session_mapping(paired_runs, mapping):
    base, sft = paired_runs
    with pytest.raises(ValueError):
        compare_evaluations([base], [sft], session_mapping=mapping, replications=20)


def test_rejects_a_single_session(tmp_path):
    rows = _rows([1, 2, 3], sessions=("same",) * 3)
    base = _write_run(tmp_path, "base", 7, rows)
    sft = _write_run(tmp_path, "sft", 7, rows)
    with pytest.raises(ValueError, match="At least two driving sessions"):
        compare_evaluations([base], [sft], replications=20)


def test_cli_emits_the_reproducible_report(paired_runs, tmp_path, capsys):
    base, sft = paired_runs
    output = tmp_path / "report.json"
    main(["--baseline", str(base), "--sft", str(sft), "--replications", "200",
          "--seed", "17", "--output", str(output)])
    report = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert report["bootstrap_seed"] == 17
    assert report["bootstrap_replications"] == 200
    assert report["evaluation_seeds"] == [7]


@pytest.mark.parametrize("session_column", ["session_id", "session", "drive_id"])
@pytest.mark.parametrize("use_index", [False, True])
def test_parquet_mapping_accepts_clip_column_or_index(tmp_path, session_column, use_index):
    import pandas as pd

    frame = pd.DataFrame({
        "clip_id": ["clip-0", "clip-1", "clip-1", "unused"],
        session_column: ["a", "b", "b", None],
    })
    if use_index:
        frame = frame.set_index("clip_id")
    path = tmp_path / "clip_index.parquet"
    frame.to_parquet(path)
    mapping, column = load_clip_sessions(path, ["clip-0", "clip-1"])
    assert mapping == {"clip-0": "a", "clip-1": "b"}
    assert column == session_column


@pytest.mark.parametrize("damage", [
    "no_session_column", "null_session", "missing_clip", "duplicate_conflict", "column_conflict",
])
def test_parquet_mapping_rejects_invalid_used_sessions(tmp_path, damage):
    import pandas as pd

    frame = pd.DataFrame({"clip_id": ["clip-0", "clip-1"], "session_id": ["a", "b"]})
    if damage == "no_session_column":
        frame = frame.drop(columns="session_id")
    elif damage == "null_session":
        frame.loc[0, "session_id"] = None
    elif damage == "missing_clip":
        frame.loc[0, "clip_id"] = "unused"
    elif damage == "duplicate_conflict":
        frame = pd.concat([frame, pd.DataFrame({"clip_id": ["clip-0"], "session_id": ["c"]})])
    else:
        frame["drive_id"] = ["different", "b"]
    path = tmp_path / "clip_index.parquet"
    frame.to_parquet(path)
    with pytest.raises(ValueError) as error:
        load_clip_sessions(path, ["clip-0", "clip-1"])
    if damage == "no_session_column":
        assert "--session-map" in str(error.value)
        assert "--session-metadata with a session_id, session, or drive_id column" in str(error.value)
        assert "--means-only" in str(error.value)
        assert "--assume-independent-clips" in str(error.value)


def test_comparison_uses_parquet_sessions(paired_runs, tmp_path):
    import pandas as pd

    base, sft = paired_runs
    path = tmp_path / "clip_index.parquet"
    pd.DataFrame({
        "clip_id": ["clip-0", "clip-1", "clip-2"],
        "session_id": ["session-a", "session-a", "session-b"],
    }).to_parquet(path)
    result = compare_evaluations([base], [sft], clip_index=path, replications=20)
    assert result["session_mapping_source"] == "clip_index:session_id"
    assert result["counts"]["sessions"] == 2
    with pytest.raises(ValueError, match="only one"):
        compare_evaluations([base], [sft], clip_index=path, session_mapping={})


def test_cli_rejects_multiple_session_sources(paired_runs):
    base, sft = paired_runs
    with pytest.raises(SystemExit) as error:
        main(["--baseline", str(base), "--sft", str(sft),
              "--session-map", "map.json", "--clip-index", "index.parquet"])
    assert error.value.code == 2


def _mark_generation_failures(directory, failed_indices):
    rows = [json.loads(line) for line in (directory / "trajectory_samples.jsonl").read_text().splitlines()]
    for index in failed_indices:
        rows[index].update(status="batch_missing_trajectory_anchor", min_ade=None)
    _replace_rows(directory, rows)
    scores = [row["min_ade"] for row in rows if row["status"] == "ok"]
    summary = {
        "status": "completed_with_generation_failures" if failed_indices else "completed",
        "eval_samples": len(scores), "eval_attempted_samples": len(rows),
        "eval_failed_samples": len(rows) - len(scores),
        "eval_generation_coverage": len(scores) / len(rows),
    }
    if scores:
        summary["eval_min_ade_successful" if failed_indices else "eval_min_ade"] = sum(scores) / len(scores)
    (directory / "trajectory_summary.json").write_text(json.dumps(summary))


def test_opt_in_drops_union_of_failures_across_both_arms_and_all_seeds(tmp_path):
    sessions = ("a", "a", "b", "c")
    baseline, sft = [], []
    for seed, base_scores, sft_scores in [(7, [100, 200, 4, 6], [7, 9, 3, 5]),
                                         (11, [300, 400, 6, 8], [8, 10, 5, 7])]:
        baseline.append(_write_run(tmp_path, "base", seed, _rows(base_scores, sessions)))
        sft.append(_write_run(tmp_path, "sft", seed, _rows(sft_scores, sessions)))
    _mark_generation_failures(baseline[0], [0])
    _mark_generation_failures(sft[1], [1])
    with pytest.raises(ValueError, match="status"):
        compare_evaluations(baseline, sft, replications=20)
    report = compare_evaluations(baseline, sft, drop_failed_events=True, replications=100)
    assert report["baseline_mean_min_ade"] == 6
    assert report["sft_mean_min_ade"] == 5
    assert report["delta_sft_minus_baseline"] == -1
    assert report["confidence_interval"]["lower"] == report["confidence_interval"]["upper"] == -1
    selection = report["event_selection"]
    assert selection["attempted_events_per_run"] == 4
    assert selection["included_events"] == selection["excluded_events"] == 2
    assert selection["comparison_coverage"] == 0.5
    assert [row["clip_id"] for row in selection["excluded_event_keys"]] == ["clip-0", "clip-1"]
    assert [row["generation_coverage"] for row in selection["per_run"]] == [0.75, 1.0, 1.0, 0.75]
    assert [row["failed_events"] for row in selection["per_run"]] == [1, 0, 0, 1]
    assert "conditional on successful generation" in report["warnings"][0]
    assert "failed generations are not scored" in report["interpretation"]
    assert report["counts"]["events"] == 2 and report["counts"]["sessions"] == 2


@pytest.mark.parametrize("damage", [
    "in_progress", "interrupted", "failed", "completed_status", "attempted_count",
    "successful_count", "failed_count", "coverage", "successful_mean", "missing_mean",
    "unconditional_mean", "missing_row", "unknown_failure", "nonnull_failed_score",
    "missing_failed_score",
])
def test_failure_opt_in_rejects_incomplete_or_inconsistent_runs(paired_runs, damage):
    base, sft = paired_runs
    _mark_generation_failures(sft, [0])
    summary_path = sft / "trajectory_summary.json"
    if damage in {"in_progress", "interrupted", "failed"}:
        _update_json(summary_path, status=damage)
    elif damage == "completed_status":
        _update_json(summary_path, status="completed")
    elif damage.endswith("_count"):
        field = {"attempted_count": "eval_attempted_samples", "successful_count": "eval_samples",
                 "failed_count": "eval_failed_samples"}[damage]
        _update_json(summary_path, **{field: 99})
    elif damage == "coverage":
        _update_json(summary_path, eval_generation_coverage=1.0)
    elif damage == "successful_mean":
        _update_json(summary_path, eval_min_ade_successful=1000)
    elif damage == "missing_mean":
        summary = json.loads(summary_path.read_text())
        del summary["eval_min_ade_successful"]
        summary_path.write_text(json.dumps(summary))
    elif damage == "unconditional_mean":
        _update_json(summary_path, eval_min_ade=4.5)
    else:
        rows = [json.loads(line) for line in (sft / "trajectory_samples.jsonl").read_text().splitlines()]
        if damage == "missing_row":
            rows.pop()
        elif damage == "unknown_failure":
            rows[0]["status"] = "out_of_memory"
        elif damage == "nonnull_failed_score":
            rows[0]["min_ade"] = 0.0
        else:
            del rows[0]["min_ade"]
        _replace_rows(sft, rows)
    with pytest.raises(ValueError):
        compare_evaluations([base], [sft], drop_failed_events=True, means_only=True)


def test_failure_opt_in_still_requires_identical_attempted_sets_for_every_seed(tmp_path, paired_runs):
    base, sft = paired_runs
    _mark_generation_failures(sft, [0])
    second_base = _write_run(tmp_path, "base", 8, _rows([4, 4]))
    second_sft = _write_run(tmp_path, "sft", 8, _rows([3, 3]))
    with pytest.raises(ValueError, match="unmatched events"):
        compare_evaluations([base, second_base], [sft, second_sft],
                            drop_failed_events=True, means_only=True)


@pytest.mark.parametrize("all_in_one_run", [False, True])
def test_failure_opt_in_requires_a_nonempty_common_successful_set(paired_runs, all_in_one_run):
    base, sft = paired_runs
    _mark_generation_failures(base, [0, 1, 2] if all_in_one_run else [0, 1])
    if not all_in_one_run:
        _mark_generation_failures(sft, [2])
    with pytest.raises(ValueError, match="No common successful events"):
        compare_evaluations([base], [sft], drop_failed_events=True, means_only=True)


def test_means_only_needs_no_session_ids_and_makes_no_uncertainty_claim(paired_runs):
    base, sft = paired_runs
    for directory, scores in [(base, [4, 4, 4]), (sft, [3, 3, 6])]:
        rows = _rows(scores)
        for row in rows:
            del row["session_id"]
        _replace_rows(directory, rows)
    report = compare_evaluations([base], [sft], means_only=True)
    assert report["baseline_mean_min_ade"] == report["sft_mean_min_ade"] == 4
    assert report["delta_sft_minus_baseline"] == 0
    assert report["confidence_interval"] is None
    assert report["bootstrap_seed"] is report["bootstrap_replications"] is None
    assert report["counts"]["sessions"] is report["counts"]["events_per_session"] is None
    assert report["cluster_scope"] is report["session_mapping_source"] is None
    assert report["conclusion"] == "descriptive_only"
    assert "no confidence interval or significance claim" in report["interpretation"]
    assert report["event_selection"]["comparison_coverage"] == 1


def test_means_only_does_not_claim_inference_for_a_single_session(tmp_path):
    base = _write_run(tmp_path, "base", 42, _rows([4, 4, 4], sessions=("same",) * 3))
    sft = _write_run(tmp_path, "sft", 42, _rows([3, 3, 3], sessions=("same",) * 3))
    report = compare_evaluations([base], [sft], means_only=True)
    assert report["delta_sft_minus_baseline"] == -1
    assert report["conclusion"] == "descriptive_only" and report["confidence_interval"] is None
    with pytest.raises(ValueError, match="does not use session metadata"):
        compare_evaluations([base], [sft], means_only=True, session_mapping={})


def test_cli_combines_explicit_recovery_and_means_only(paired_runs, capsys):
    base, sft = paired_runs
    _mark_generation_failures(sft, [0])
    main(["--baseline", str(base), "--sft", str(sft), "--means-only", "--drop-failed-events"])
    report = json.loads(capsys.readouterr().out)
    assert report["confidence_interval"] is None
    assert report["delta_sft_minus_baseline"] == 0.5
    assert report["event_selection"]["excluded_events"] == 1
    assert report["event_selection"]["comparison_coverage"] == pytest.approx(2 / 3)
    with pytest.raises(SystemExit):
        main(["--baseline", str(base), "--sft", str(sft), "--means-only", "--session-map", "sessions.json"])


@pytest.fixture
def checkpoint_runs(tmp_path):
    sessions = ("a", "a", "b", "b", "c", "c", "d")
    scores = {
        "base": ([100, 200, 300, 400, 4, 6, 10], [200, 300, 400, 500, 6, 8, 12]),
        "epoch1": ([1, 2, 3, 4, 3, 5, 9], [2, 3, 4, 5, 5, 7, 11]),
        "epoch2": ([2, 3, 4, 5, 4, 8, 14], [3, 4, 5, 6, 6, 10, 16]),
        "epoch3": ([3, 4, 5, 6, 6, 8, 12], [4, 5, 6, 7, 8, 10, 14]),
    }
    arms = {
        arm: [_write_run(tmp_path, arm, seed, _rows(values, sessions))
              for seed, values in zip((7, 11), per_seed)]
        for arm, per_seed in scores.items()
    }
    return arms.pop("base"), arms


def _fail_distinct_checkpoint_events(baseline, candidates):
    _mark_generation_failures(baseline[0], [0])
    _mark_generation_failures(candidates["epoch1"][1], [1])
    _mark_generation_failures(candidates["epoch2"][0], [2])
    _mark_generation_failures(candidates["epoch3"][1], [3])


def test_checkpoint_set_uses_one_failure_union_and_bootstrap_cohort(checkpoint_runs):
    baseline, candidates = checkpoint_runs
    _fail_distinct_checkpoint_events(baseline, candidates)
    with pytest.raises(ValueError, match="status"):
        compare_checkpoint_set(baseline, candidates, replications=20)
    result = compare_checkpoint_set(
        baseline, candidates, drop_failed_events=True, replications=200, seed=13, confidence=.8,
    )
    assert result["comparison_scope"] == "checkpoint_set"
    assert result["baseline_checkpoint"] == "base"
    assert result["evaluation_seeds"] == [7, 11]
    assert result["baseline_mean_min_ade"] == pytest.approx(23 / 3)
    assert set(result["comparisons"]) == {"epoch1", "epoch2", "epoch3"}
    selection = result["event_selection"]
    assert selection["scope"] == "all_supplied_checkpoints_and_seeds"
    assert selection["attempted_events_per_run"] == 7
    assert selection["included_events"] == 3 and selection["excluded_events"] == 4
    assert selection["comparison_coverage"] == pytest.approx(3 / 7)
    assert [row["clip_id"] for row in selection["excluded_event_keys"]] == [f"clip-{i}" for i in range(4)]
    events = [[f"clip-{i}", 6_000_000, 6_000_010] for i in range(4, 7)]
    expected_hash = hashlib.sha256(json.dumps(events, separators=(",", ":")).encode()).hexdigest()
    assert selection["cohort_sha256"] == expected_hash
    assert selection["cohort_sha256"] != selection["attempted_cohort_sha256"]
    coverage = {(row["arm"], row["seed"]): row for row in selection["per_run"]}
    assert len(coverage) == 8
    failed_runs = {("baseline", 7), ("epoch1", 11), ("epoch2", 7), ("epoch3", 11)}
    for key, row in coverage.items():
        assert row["failed_events"] == int(key in failed_runs)
        assert row["generation_coverage"] == pytest.approx(6 / 7 if key in failed_runs else 1)
        assert row["checkpoint"] == ("base" if key[0] == "baseline" else key[0])
    for label, delta in (("epoch1", -1), ("epoch2", 2), ("epoch3", 2)):
        comparison = result["comparisons"][label]
        assert comparison["event_selection"] == selection
        assert comparison["baseline_mean_min_ade"] == pytest.approx(23 / 3)
        assert comparison["delta_sft_minus_baseline"] == pytest.approx(delta)
        assert comparison["counts"]["events"] == 3 and comparison["counts"]["sessions"] == 2
        assert "failed generations are not scored" in comparison["interpretation"]
    # Independent bootstrap oracle for epoch2: retain both events from session c
    # together and the one event from d, preserving unequal cluster sizes.
    session_deltas = [np.array([0., 2.]), np.array([4.])]
    draws = np.random.default_rng(13).integers(0, 2, size=(200, 2))
    expected = [np.concatenate([session_deltas[i] for i in draw]).mean() for draw in draws]
    interval = result["comparisons"]["epoch2"]["confidence_interval"]
    assert interval["lower"] == pytest.approx(np.quantile(expected, .1))
    assert interval["upper"] == pytest.approx(np.quantile(expected, .9))


def test_checkpoint_set_is_invariant_to_candidate_seed_and_row_order(checkpoint_runs):
    baseline, candidates = checkpoint_runs
    _fail_distinct_checkpoint_events(baseline, candidates)
    first = compare_checkpoint_set(baseline, candidates, drop_failed_events=True, replications=40)
    for directory in baseline + [p for paths in candidates.values() for p in paths]:
        rows = [json.loads(line) for line in (directory / "trajectory_samples.jsonl").read_text().splitlines()]
        _replace_rows(directory, list(reversed(rows)))
    reordered = {label: list(reversed(candidates[label])) for label in reversed(candidates)}
    second = compare_checkpoint_set(list(reversed(baseline)), reordered,
                                    drop_failed_events=True, replications=40)
    assert first == second


def test_checkpoint_set_without_failures_preserves_existing_pair_statistics(checkpoint_runs):
    baseline, candidates = checkpoint_runs
    result = compare_checkpoint_set(baseline, candidates, replications=40)
    assert result["event_selection"]["included_events"] == 7
    assert result["event_selection"]["excluded_events"] == 0
    assert result["event_selection"]["cohort_sha256"] == result["event_selection"]["attempted_cohort_sha256"]
    for label, directories in candidates.items():
        pair = compare_evaluations(baseline, directories, replications=40)
        for key in ("baseline_mean_min_ade", "sft_mean_min_ade", "delta_sft_minus_baseline",
                    "confidence_interval", "counts", "conclusion", "evaluation_seeds"):
            assert result["comparisons"][label][key] == pair[key]
        scope = result["comparisons"][label]["uncertainty_scope"]
        assert "does not correct for choosing the best checkpoint on the same validation data" in scope
        assert "variation across training runs" in scope
        assert "unadjusted per-comparison intervals" in scope


@pytest.mark.parametrize("damage", ["protocol", "seeds", "events", "incomplete", "unknown_failure", "session"])
def test_checkpoint_set_validates_every_candidate_before_excluding_events(checkpoint_runs, damage):
    baseline, candidates = checkpoint_runs
    _mark_generation_failures(candidates["epoch1"][0], [0])
    directory = candidates["epoch3"][1]
    if damage == "protocol":
        _update_json(directory / "evaluation_protocol.json", temperature=.9)
    elif damage == "seeds":
        candidates["epoch3"].pop()
    elif damage == "incomplete":
        _update_json(directory / "trajectory_summary.json", status="in_progress")
    else:
        rows = [json.loads(line) for line in (directory / "trajectory_samples.jsonl").read_text().splitlines()]
        if damage == "events":
            rows.pop(0)  # Even an event excluded by another candidate must exist.
        elif damage == "unknown_failure":
            rows[0].update(status="out_of_memory", min_ade=None)
        else:
            rows[-1]["session_id"] = "contradictory-session"
        _replace_rows(directory, rows)
        if damage == "events":
            _mark_generation_failures(directory, [])  # Keep the shorter run internally consistent.
    with pytest.raises(ValueError):
        compare_checkpoint_set(baseline, candidates, drop_failed_events=True,
                               means_only=damage != "session", replications=20)


def test_checkpoint_set_rejects_an_empty_global_cohort(checkpoint_runs):
    baseline, candidates = checkpoint_runs
    _mark_generation_failures(baseline[0], [0])
    _mark_generation_failures(candidates["epoch1"][0], [1, 2])
    _mark_generation_failures(candidates["epoch2"][0], [3, 4])
    _mark_generation_failures(candidates["epoch3"][0], [5, 6])
    with pytest.raises(ValueError, match="No common successful events"):
        compare_checkpoint_set(baseline, candidates, drop_failed_events=True, means_only=True)


@pytest.mark.parametrize("invalid", ["empty", "not_mapping", "blank_label", "reserved_label", "duplicate_checkpoint", "baseline_checkpoint"])
def test_checkpoint_set_rejects_ambiguous_candidate_groups(checkpoint_runs, invalid):
    baseline, candidates = checkpoint_runs
    paths = candidates["epoch1"]
    bad = {
        "empty": {}, "not_mapping": [paths], "blank_label": {" ": paths},
        "reserved_label": {"baseline": paths},
        "duplicate_checkpoint": {"one": paths, "two": paths},
        "baseline_checkpoint": {"same_as_initial": baseline},
    }[invalid]
    with pytest.raises(ValueError):
        compare_checkpoint_set(baseline, bad, means_only=True)


@pytest.mark.parametrize("mode", ["--means-only", "--assume-independent-clips"])
def test_cli_compares_all_candidate_groups_on_one_cohort(checkpoint_runs, tmp_path, capsys, mode):
    baseline, candidates = checkpoint_runs
    for directory in baseline + [path for paths in candidates.values() for path in paths]:
        rows = [json.loads(line) for line in (directory / "trajectory_samples.jsonl").read_text().splitlines()]
        for row in rows:
            row.pop("session_id")
        _replace_rows(directory, rows)
    _fail_distinct_checkpoint_events(baseline, candidates)
    output = tmp_path / "comparison.json"
    arguments = ["--baseline", *map(str, baseline), mode, "--drop-failed-events"]
    for label, directories in candidates.items():
        arguments.extend(["--candidate", label, *map(str, directories)])
    main([*arguments, "--output", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert set(report["comparisons"]) == set(candidates)
    assert report["event_selection"]["included_events"] == 3
    for comparison in report["comparisons"].values():
        if mode == "--means-only":
            assert comparison["conclusion"] == "descriptive_only"
            assert comparison["confidence_interval"] is None
        else:
            assert comparison["conclusion"].endswith("_conditional_on_clip_independence")
            assert comparison["confidence_interval"]["method"] == "paired clip cluster bootstrap percentile"
            assert comparison["cluster_scope"] == "clip"
            assert comparison["counts"]["clusters"] == 3
            assert comparison["counts"]["sessions"] is None
            assert "Assumes independent clips" in comparison["uncertainty_scope"]
            assert "unadjusted per-comparison intervals" in comparison["uncertainty_scope"]
        assert comparison["event_selection"] == report["event_selection"]
    if mode == "--assume-independent-clips":
        assert any("Assumes independent clips" in warning for warning in report["warnings"])
        # The common surviving events are clips 4, 5 and 6 in every candidate.
        clip_deltas = np.array([0., 2., 4.])
        draws = np.random.default_rng(0).integers(0, 3, size=(10_000, 3))
        expected = clip_deltas[draws].mean(axis=1)
        interval = report["comparisons"]["epoch2"]["confidence_interval"]
        assert interval["lower"] == pytest.approx(np.quantile(expected, .025))
        assert interval["upper"] == pytest.approx(np.quantile(expected, .975))


@pytest.mark.parametrize("invalid", ["both_modes", "duplicate_label", "no_directories", "no_mode"])
def test_cli_rejects_ambiguous_checkpoint_grouping(paired_runs, invalid):
    base, sft = paired_runs
    args = {
        "both_modes": ["--sft", str(sft), "--candidate", "one", str(sft)],
        "duplicate_label": ["--candidate", "one", str(sft), "--candidate", "one", str(sft)],
        "no_directories": ["--candidate", "one"],
        "no_mode": [],
    }[invalid]
    with pytest.raises(SystemExit) as error:
        main(["--baseline", str(base), "--means-only", *args])
    assert error.value.code == 2
