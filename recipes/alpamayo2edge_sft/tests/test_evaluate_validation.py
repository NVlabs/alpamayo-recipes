# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject malformed trajectory evaluation inputs before reporting misleading metrics."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import alpamayo2edge_sft.evaluate_hf as evaluate_module
from alpamayo2edge_sft.evaluate_hf import (
    compute_min_ade,
    evaluate_trajectory,
    save_evaluation_results,
)


@pytest.mark.parametrize(
    ("prediction_shape", "target_shape"),
    [
        ((1, 2, 4, 3), (1, 4, 3)),
        ((1, 1, 2, 4, 3), (1, 1, 4, 3)),
        ((2, 1, 2, 4, 3), (1, 4, 3)),
        ((1, 1, 2, 4, 3), (1, 1, 3)),
        ((1, 1, 2, 4, 2), (1, 4, 3)),
        ((1, 1, 2, 4, 3), (1, 4, 2)),
        ((0, 1, 2, 4, 3), (0, 4, 3)),
        ((1, 0, 2, 4, 3), (1, 4, 3)),
        ((1, 1, 0, 4, 3), (1, 4, 3)),
        ((1, 1, 2, 0, 3), (1, 0, 3)),
    ],
    ids=[
        "missing-set-axis",
        "extra-target-axis",
        "broadcastable-batch-mismatch",
        "broadcastable-time-mismatch",
        "missing-predicted-z",
        "missing-target-z",
        "empty-batch",
        "empty-sets",
        "empty-candidates",
        "empty-time",
    ],
)
def test_min_ade_rejects_malformed_or_empty_trajectories(prediction_shape, target_shape):
    with pytest.raises(ValueError):
        compute_min_ade(torch.zeros(prediction_shape), torch.zeros(target_shape))


@pytest.mark.parametrize("corrupt_target", [False, True], ids=["prediction", "target"])
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("coordinate", [0, 2], ids=["x", "z"])
def test_min_ade_rejects_nonfinite_coordinates(corrupt_target, invalid_value, coordinate):
    predictions = torch.zeros(1, 1, 2, 4, 3)
    target = torch.zeros(1, 4, 3)
    if corrupt_target:
        target[0, 0, coordinate] = invalid_value
    else:
        predictions[0, 0, 1, 0, coordinate] = invalid_value

    with pytest.raises(ValueError):
        compute_min_ade(predictions, target)


def test_min_ade_uses_xy_distance_for_each_batch_row():
    target = torch.zeros(2, 4, 3)
    predictions = torch.zeros(2, 1, 2, 4, 3)
    predictions[0, :, 0, :, :2] = torch.tensor([3.0, 4.0])
    predictions[0, :, 1, :, 0] = 8.0
    predictions[1, :, 0, :, 0] = 7.0
    predictions[1, :, 1, :, 1] = 2.0
    predictions[..., 2] = 1000.0

    torch.testing.assert_close(compute_min_ade(predictions, target), torch.tensor([5.0, 2.0]))


@pytest.fixture
def evaluation_case(monkeypatch, tmp_path):
    """Exercise evaluator input checks on CPU without loading the inference model."""
    monkeypatch.setattr(
        "alpamayo2edge_sft.evaluate_hf.torch.autocast",
        lambda *args, **kwargs: nullcontext(),
    )
    batch = {
        "tokenized_data": {"input_ids": torch.ones(1, 2, dtype=torch.long)},
        "traj_data": {
            "ego_history_xyz": torch.zeros(1, 1, 2, 3),
            "ego_history_rot": torch.eye(3).repeat(1, 1, 2, 1, 1),
            "ego_future_xyz": torch.zeros(1, 1, 4, 3),
        },
    }
    accelerator = SimpleNamespace(
        num_processes=1,
        is_main_process=True,
        device=torch.device("cpu"),
        prepare_model=lambda model, evaluation_mode: model,
        unwrap_model=lambda model: model,
        gather_for_metrics=lambda values: values,
    )
    model = SimpleNamespace(eval=lambda: None)
    trainer = SimpleNamespace(
        accelerator=accelerator,
        model=model,
        args=SimpleNamespace(deepspeed=None, output_dir=str(tmp_path)),
        _prepare_input=lambda batch: batch,
        is_deepspeed_enabled=False,
        get_eval_dataloader=lambda: [batch],
        log=lambda metrics: pytest.fail("Invalid sampler output must not be logged as metrics"),
    )
    config = SimpleNamespace(
        evaluate=SimpleNamespace(
            max_eval_steps=-1,
            autocast_dtype="bfloat16",
            num_traj_samples=3,
            num_samples_per_forward=2,
            num_traj_sets=1,
            top_p=0.98,
            temperature=0.6,
            max_generation_length=256,
        ),
        trainer=SimpleNamespace(deepspeed=None),
    )
    return trainer, config


@pytest.mark.parametrize("num_sets", [0, -1])
def test_trajectory_evaluation_rejects_nonpositive_set_count_before_sampling(
    evaluation_case, num_sets
):
    trainer, config = evaluation_case
    config.evaluate.num_traj_sets = num_sets
    trainer.model.sample_trajectories_from_data = lambda **kwargs: pytest.fail(
        "Invalid set counts must be rejected before sampling"
    )

    with pytest.raises(ValueError):
        evaluate_trajectory(trainer, config)


@pytest.mark.parametrize(
    "returned_shape",
    [(1, 2, 2, 4, 3), (1, 1, 1, 4, 3), (1, 1, 3, 4, 3)],
    ids=["wrong-set-count", "too-few-candidates", "too-many-candidates"],
)
def test_trajectory_evaluation_rejects_sampler_counts_that_differ_from_request(
    evaluation_case, returned_shape
):
    trainer, config = evaluation_case

    def sample(**kwargs):
        assert kwargs["num_traj_sets"] == 1
        assert kwargs["num_traj_samples"] == 2
        return torch.zeros(returned_shape), torch.empty(0), torch.empty(0)

    trainer.model.sample_trajectories_from_data = sample
    with pytest.raises(ValueError):
        evaluate_trajectory(trainer, config)


def test_trajectory_evaluation_checks_short_final_candidate_chunk(evaluation_case):
    trainer, config = evaluation_case
    calls = []

    def sample(**kwargs):
        calls.append(kwargs["num_traj_samples"])
        # The second call requests only one candidate. Returning two must fail,
        # rather than silently evaluating best-of-four under a best-of-three label.
        return torch.zeros(1, 1, 2, 4, 3), torch.empty(0), torch.empty(0)

    trainer.model.sample_trajectories_from_data = sample
    with pytest.raises(ValueError):
        evaluate_trajectory(trainer, config)
    assert calls == [2, 1]


@pytest.fixture
def evaluation_config(tmp_path):
    data_dir = tmp_path / "pai"
    data_dir.mkdir()
    # Serialization hashes these bytes; the tests never parse a parquet table.
    (data_dir / "clip_index.parquet").write_bytes(b"fixed clip selection\n")
    directory = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        return compose(
            config_name="sft_eval_trajectory",
            overrides=[
                f"model.pretrained_model_name_or_path={tmp_path}/stage2/checkpoint-100",
                f"data.train_dataset.local_dir={tmp_path}/pai",
                f"data.val_dataset.local_dir={tmp_path}/pai",
                f"paths.output_dir={tmp_path}/evaluation",
            ],
        )


@pytest.mark.parametrize("navigation_input", [False, True])
@pytest.mark.parametrize("generate_cot", [False, True])
@pytest.mark.parametrize("max_eval_steps, expected_limit", [(None, None), (-1, None), (3, 3)])
def test_save_evaluation_persists_scores_config_and_sampling_protocol(
    evaluation_config, navigation_input, generate_cot, max_eval_steps, expected_limit,
):
    config = evaluation_config
    config.evaluate.max_eval_steps = max_eval_steps
    OmegaConf.update(config, "evaluate.generate_cot", generate_cot, force_add=True)
    components = ["image", "traj_history"]
    if navigation_input:
        components.append("nav_instruction")
    components.append("prompt")
    if generate_cot:
        components.append("cot")
    components.append("traj_future")
    config.data.val_dataset.preprocess.components_order = components
    output_dir = Path(config.paths.output_dir)
    metrics = {"eval_min_ade": 0.75, "eval_samples": 100}

    class Trainer:
        args = SimpleNamespace(output_dir=str(output_dir))
        model = SimpleNamespace(
            config=SimpleNamespace(
                expert_config=SimpleNamespace(action_space_cfg={"n_waypoints": 64, "dt": 0.1}),
            ),
            expert=SimpleNamespace(
                diffusion=SimpleNamespace(num_inference_steps=10, int_method="euler")
            ),
        )

        def save_metrics(self, split, saved_metrics):
            assert split == "eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "eval_results.json").write_text(json.dumps(saved_metrics))

        def is_world_process_zero(self):
            return True

    trainer = Trainer()
    from alpamayo2edge_sft.models import TrainableAlpamayo2Edge

    trainer.model.trajectory_conditioning = TrainableAlpamayo2Edge.trajectory_conditioning
    save_evaluation_results(trainer, config, metrics)

    assert json.loads((output_dir / "eval_results.json").read_text()) == metrics
    saved_config = OmegaConf.load(output_dir / "config.yaml")
    assert saved_config.seed == 42
    assert saved_config.evaluate.max_eval_steps == max_eval_steps
    assert saved_config.model.pretrained_model_name_or_path == (
        config.model.pretrained_model_name_or_path
    )
    assert saved_config.data.val_dataset.preprocess.generation_mode is True
    assert saved_config.data.val_dataset.chunk_ids == config.data.val_dataset.chunk_ids

    protocol = json.loads((output_dir / "evaluation_protocol.json").read_text())
    assert protocol["schema_version"] == 1
    assert "dataset_sampling_protocol" not in protocol  # No ONCE metadata on PAI.
    assert protocol["validation_dataset"] == json.loads(json.dumps(
        OmegaConf.to_container(config.data.val_dataset, resolve=True)
    ))
    assert protocol["validation_dataset"]["camera_features"] == list(config.camera_features)
    assert protocol["validation_dataset"]["frame_offsets"] == list(config.frame_offsets)
    assert protocol["validation_dataset"]["front_frame_offsets"] == list(config.front_frame_offsets)
    assert protocol["validation_dataset"]["camera_pixel_bounds"] == {
        str(key): list(value) for key, value in config.camera_pixel_bounds.items()
    }
    assert protocol["collator"] == OmegaConf.to_container(config.data.collator, resolve=True)
    assert protocol["autocast_dtype"] == config.evaluate.autocast_dtype
    assert protocol["per_device_eval_batch_size"] == config.trainer.per_device_eval_batch_size
    assert protocol["checkpoint"] == config.model.pretrained_model_name_or_path
    assert protocol["seed"] == 42
    assert protocol["mode"] == "trajectory"
    assert protocol["validation_chunks"] == config.data.val_dataset.chunk_ids
    assert protocol["candidate_count"] == protocol["num_traj_samples"] == 6
    assert protocol["num_traj_sets"] == 1
    assert protocol["num_samples_per_forward"] == 2
    assert protocol["future_steps"] == 64
    assert protocol["time_step_seconds"] == 0.1
    assert protocol["diffusion_steps"] == 10
    assert protocol["integration_method"] == "euler"
    assert "meters" in protocol["metric"]
    expected_conditioning = "no reference CoT" if generate_cot else "no generated CoT"
    assert expected_conditioning in protocol["conditioning"]
    assert protocol["generate_cot"] is generate_cot
    assert protocol["navigation_input"] is navigation_input
    assert protocol["preprocess"]["components_order"] == components
    assert protocol["max_eval_steps"] == expected_limit
    assert protocol["native_release_prompt"] is (navigation_input and generate_cot)
    assert "zero placeholder" in protocol["log_probability"]
    if generate_cot:
        assert protocol["temperature"] == config.evaluate.temperature
        assert protocol["max_generation_length"] == config.evaluate.max_generation_length
    else:
        assert "temperature" not in protocol
        assert "top_p" not in protocol
        assert "max_generation_length" not in protocol


@pytest.mark.parametrize("mode", ["loss", "trajectory"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_save_evaluation_records_dataset_sampling_through_subsets(evaluation_config, mode, wrapped):
    from torch.utils.data import Subset

    expected = {"adapter_variant": "test", "images": {"requested_offsets_seconds": [-.5, 0.]}}
    dataset = SimpleNamespace(evaluation_sampling_protocol=lambda: expected)
    if wrapped:
        dataset = Subset(Subset(dataset, [0]), [0])
    config = evaluation_config
    config.evaluate.mode = mode
    output_dir = Path(config.paths.output_dir)
    trainer = SimpleNamespace(
        eval_dataset=dataset,
        args=SimpleNamespace(output_dir=str(output_dir)),
        model=SimpleNamespace(config=SimpleNamespace()),
        is_world_process_zero=lambda: True,
        save_metrics=lambda *args: None,
    )
    save_evaluation_results(trainer, config, {})
    protocol = json.loads((output_dir / "evaluation_protocol.json").read_text())
    assert protocol["mode"] == mode
    assert protocol["dataset_sampling_protocol"] == expected


@pytest.mark.parametrize("changed_setting", [
    "camera_features", "frame_offsets", "front_frame_offsets", "camera_pixel_bounds",
    "collator", "autocast_dtype", "per_device_eval_batch_size",
    "max_eval_steps", "unlimited_max_eval_steps", "split_manifest",
    "annotations_contents", "dataset_revision", "relocated_identical_inputs",
])
def test_saved_protocol_checks_settings_in_paired_comparison(
    evaluation_config, tmp_path, changed_setting,
):
    from alpamayo2edge_sft.compare_evaluations import compare_evaluations

    directories = []
    manifest = tmp_path / "split.json"
    for arm in ("base", "sft"):
        config = OmegaConf.create(OmegaConf.to_container(evaluation_config, resolve=True))
        config.model.pretrained_model_name_or_path = arm
        if changed_setting == "split_manifest":
            del config.data.val_dataset.chunk_ids
            OmegaConf.update(config, "data.val_dataset.split_manifest", str(manifest), force_add=True)
            manifest.write_text(json.dumps({"train": [arm], "val": ["held-out"]}))
        if arm == "sft":
            if changed_setting == "camera_features":
                config.data.val_dataset.camera_features = ["camera_front_wide_120fov"]
            elif changed_setting == "frame_offsets":
                config.data.val_dataset.frame_offsets = [-0.1, 0.0]
            elif changed_setting == "front_frame_offsets":
                config.data.val_dataset.front_frame_offsets = [-1.0, 0.0]
            elif changed_setting == "camera_pixel_bounds":
                config.data.val_dataset.camera_pixel_bounds[0] = [40960, 49152]
            elif changed_setting == "collator":
                config.data.collator.max_length = 4096
            elif changed_setting == "autocast_dtype":
                config.evaluate.autocast_dtype = "float32"
            elif changed_setting == "max_eval_steps":
                config.evaluate.max_eval_steps = 2
            elif changed_setting == "unlimited_max_eval_steps":
                config.evaluate.max_eval_steps = None
            elif changed_setting == "per_device_eval_batch_size":
                config.trainer.per_device_eval_batch_size = 2
        if changed_setting in {"annotations_contents", "dataset_revision", "relocated_identical_inputs"}:
            data_dir = tmp_path / (f"data-{arm}" if changed_setting == "relocated_identical_inputs" else "data")
            data_dir.mkdir(exist_ok=True)
            # Relocation preserves the default fixture's metadata contents.
            (data_dir / "clip_index.parquet").write_bytes(b"fixed clip selection\n")
            annotation = data_dir / "annotations.json"
            annotation.write_text(json.dumps([{"cot": arm if changed_setting == "annotations_contents" else "Yield."}]))
            revision = "b" if changed_setting == "dataset_revision" and arm == "sft" else "a"
            (data_dir / "download_manifest.json").write_text(json.dumps({
                "repo_id": "example/driving", "resolved_revision": revision * 40,
            }))
            config.data.val_dataset.local_dir = str(data_dir)
            OmegaConf.update(config, "data.val_dataset.annotations_path", str(annotation), force_add=True)
        directory = tmp_path / arm
        directory.mkdir()
        directories.append(directory)
        rows = [
            {"clip_id": f"clip-{index}", "requested_t0_us": 6_000_000,
             "t0_us": 6_000_010, "status": "ok", "min_ade": 1.0,
             "session_id": f"session-{index}"}
            for index in range(2)
        ]
        (directory / "trajectory_samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        metrics = {
            "eval_samples": 2, "eval_attempted_samples": 2, "eval_failed_samples": 0,
            "eval_generation_coverage": 1.0, "eval_min_ade": 1.0,
        }
        (directory / "trajectory_summary.json").write_text(json.dumps({
            "status": "completed", **metrics,
        }))
        trainer = SimpleNamespace(
            args=SimpleNamespace(output_dir=str(directory)),
            model=SimpleNamespace(config=SimpleNamespace()),
            is_world_process_zero=lambda: True,
            save_metrics=lambda *args: None,
        )
        save_evaluation_results(trainer, config, metrics)
    if changed_setting in {"unlimited_max_eval_steps", "relocated_identical_inputs"}:
        comparison = compare_evaluations([directories[0]], [directories[1]], replications=20)
        assert comparison["matched_protocol"]["max_eval_steps"] is None
        assert comparison["delta_sft_minus_baseline"] == 0.0
    else:
        with pytest.raises(ValueError, match="protocol differences"):
            compare_evaluations([directories[0]], [directories[1]], replications=20)


def test_evaluation_sampling_uses_config_seed_after_trainer_initialization(monkeypatch):
    config = OmegaConf.create({"seed": 123, "evaluate": {"mode": "trajectory"}})
    trainer = SimpleNamespace(args=SimpleNamespace(report_to=[]))
    sampled = []
    saved = []
    metrics = {"eval_min_ade": 0.5, "eval_samples": 1}

    def build_trainer(received_config):
        assert received_config is config
        # Hugging Face Trainer initialization resets the RNG to its own seed.
        torch.manual_seed(42)
        return trainer

    def evaluate_trajectory(received_trainer, received_config):
        assert received_trainer is trainer
        assert received_config is config
        sampled.append(torch.rand(4))
        return metrics

    monkeypatch.setattr(evaluate_module, "build_trainer", build_trainer)
    monkeypatch.setattr(evaluate_module, "evaluate_trajectory", evaluate_trajectory)
    monkeypatch.setattr(
        evaluate_module,
        "save_evaluation_results",
        lambda *args: saved.append(args),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    expected = torch.rand(4, generator=torch.Generator().manual_seed(123))

    with torch.random.fork_rng(devices=[]):
        evaluate_module.evaluate.__wrapped__(config)

    assert len(sampled) == 1
    torch.testing.assert_close(sampled[0], expected)
    assert saved == [(trainer, config, metrics)]


@pytest.mark.parametrize(
    ("generate_cot", "assistant_components", "generation_mode"),
    [(True, ["traj_future"], True), (False, ["cot", "traj_future"], True),
     (True, ["cot", "traj_future"], False)],
)
def test_generation_config_mismatch_rejected_before_loading_checkpoint(
    monkeypatch, evaluation_config, generate_cot, assistant_components, generation_mode
):
    config = evaluation_config
    OmegaConf.update(config, "evaluate.generate_cot", generate_cot, force_add=True)
    config.data.val_dataset.preprocess.components_order = [
        "image", "traj_history", "prompt", *assistant_components
    ]
    config.data.val_dataset.preprocess.generation_mode = generation_mode
    monkeypatch.setattr(evaluate_module, "instantiate", lambda *args, **kwargs: pytest.fail(
        "A mismatched evaluation config must fail before loading the checkpoint"
    ))
    with pytest.raises(ValueError, match="generation_mode=true and assistant components"):
        evaluate_module.build_trainer(config)


@pytest.mark.parametrize("value", [0, -2, 1.5, True, "2"])
def test_both_evaluation_modes_reject_invalid_batch_limits(value):
    with pytest.raises(ValueError, match="max_eval_steps"):
        evaluate_module.validate_max_eval_steps(value)


def test_trajectory_evaluation_prepares_all_inputs_before_sampling(evaluation_case):
    trainer, config = evaluation_case
    original = trainer.get_eval_dataloader()[0]
    prepared = {"tokenized_data": {"input_ids": torch.tensor([[3, 4]])},
                "traj_data": {key: value.to("meta") for key, value in original["traj_data"].items()}}
    prepared["tokenized_data"]["input_ids"] = prepared["tokenized_data"]["input_ids"].to("meta")
    preparation = []
    def prepare(batch):
        preparation.append(batch)
        return prepared
    def sample(data, **kwargs):
        assert data["tokenized_data"]["input_ids"].device.type == "meta"
        assert data["ego_history_xyz"].device.type == "meta"
        assert data["ego_history_xyz"].dtype == torch.float32
        raise RuntimeError("device preparation confirmed")
    trainer._prepare_input = prepare
    trainer.model.sample_trajectories_from_data = sample
    with pytest.raises(RuntimeError, match="device preparation confirmed"):
        evaluate_trajectory(trainer, config)
    assert preparation == [original]


def test_expected_cot_failures_preserve_rows_and_do_not_publish_biased_min_ade(evaluation_case):
    trainer, config = evaluation_case
    config.evaluate.generate_cot = True
    template = trainer.get_eval_dataloader()[0]
    batches = [{**template, "sample_metadata": [{"clip_id": f"clip-{i}", "requested_t0_us": 2000000}]}
               for i in range(3)]
    trainer.get_eval_dataloader = lambda: batches
    trainer.log = lambda metrics: None
    calls = []
    def sample(**kwargs):
        calls.append(kwargs["num_traj_samples"])
        if len(calls) == 3:
            raise ValueError("No <traj_future_start> token found in generated rows [0].")
        return torch.zeros(1, 1, kwargs["num_traj_samples"], 4, 3), None, None
    trainer.model.sample_trajectories_from_data = sample
    metrics = evaluate_trajectory(trainer, config)
    assert metrics["eval_attempted_samples"] == 3
    assert metrics["eval_samples"] == 2
    assert metrics["eval_failed_samples"] == 1
    assert metrics["eval_generation_coverage"] == pytest.approx(2 / 3)
    assert metrics["eval_min_ade_successful"] == 0
    assert "eval_min_ade" not in metrics
    rows = [json.loads(line) for line in (Path(trainer.args.output_dir) / "trajectory_samples.jsonl").read_text().splitlines()]
    assert [row["clip_id"] for row in rows] == ["clip-0", "clip-1", "clip-2"]
    assert [row["status"] for row in rows] == ["ok", "batch_missing_trajectory_anchor", "ok"]
    assert rows[1]["min_ade"] is None
    summary = json.loads((Path(trainer.args.output_dir) / "trajectory_summary.json").read_text())
    assert summary["status"] == "completed_with_generation_failures"


def test_all_cot_failures_report_zero_coverage(evaluation_case):
    trainer, config = evaluation_case
    config.evaluate.generate_cot = True
    trainer.log = lambda metrics: None
    def sample(**kwargs):
        raise ValueError("No <traj_future_start> token found in generated rows [0].")
    trainer.model.sample_trajectories_from_data = sample
    metrics = evaluate_trajectory(trainer, config)
    assert metrics["eval_generation_coverage"] == 0
    assert metrics["eval_failed_samples"] == 1
    assert "eval_min_ade" not in metrics and "eval_min_ade_successful" not in metrics


def test_unexpected_error_retains_completed_samples_and_propagates(evaluation_case):
    trainer, config = evaluation_case
    config.evaluate.generate_cot = True
    batch = trainer.get_eval_dataloader()[0]
    trainer.get_eval_dataloader = lambda: [batch, batch]
    count = 0
    def sample(**kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise ValueError("Sampler bug")
        return torch.zeros(1, 1, kwargs["num_traj_samples"], 4, 3), None, None
    trainer.model.sample_trajectories_from_data = sample
    with pytest.raises(ValueError, match="Sampler bug"):
        evaluate_trajectory(trainer, config)
    summary = json.loads((Path(trainer.args.output_dir) / "trajectory_summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["eval_samples"] == 1
    assert "eval_min_ade" not in summary
    assert len((Path(trainer.args.output_dir) / "trajectory_samples.jsonl").read_text().splitlines()) == 1


def test_max_eval_steps_limits_trajectory_batches(evaluation_case):
    trainer, config = evaluation_case
    config.evaluate.max_eval_steps = 1
    batch = trainer.get_eval_dataloader()[0]
    trainer.get_eval_dataloader = lambda: [batch] * 3
    trainer.log = lambda metrics: None
    trainer.model.sample_trajectories_from_data = lambda **kw: (
        torch.zeros(1, 1, kw["num_traj_samples"], 4, 3), None, None
    )
    metrics = evaluate_trajectory(trainer, config)
    assert metrics["eval_batches"] == metrics["eval_samples"] == 1


def test_loss_eval_preserves_export_objective_and_honors_batch_limit(monkeypatch, evaluation_config):
    config = evaluation_config
    config.evaluate.mode = "loss"
    config.evaluate.eval_ckpt = "/export/checkpoint-500"
    config.evaluate.max_eval_steps = 2
    captured = {}
    model = SimpleNamespace(config=SimpleNamespace(loss_weights={"others": 3.0}))
    collator = SimpleNamespace(data_processor=SimpleNamespace(processor=object()))
    def instantiate(spec, **kwargs):
        if "pretrained_model_name_or_path" in spec:
            captured["model_spec"] = OmegaConf.to_container(spec)
            return model
        if "local_dir" in spec:
            return lambda **kw: list(range(100))
        return lambda **kw: collator
    args = SimpleNamespace(eval_batch_size=3, world_size=2, deepspeed=None)
    monkeypatch.setattr(evaluate_module, "instantiate", instantiate)
    monkeypatch.setattr(evaluate_module, "training_arguments", lambda _: args)
    monkeypatch.setattr(evaluate_module, "Alpamayo2EdgeTrainer", lambda **kw: SimpleNamespace(**kw))
    result = evaluate_module.build_trainer(config)
    assert len(result.eval_dataset) == 2 * 3 * 2
    assert "loss_weights" not in captured["model_spec"]
    assert "lm_token_loss_weights" not in captured["model_spec"]
    assert captured["model_spec"]["pretrained_model_name_or_path"] == "/export/checkpoint-500"
    assert result.model.config.loss_weights == {"others": 3.0}


def test_evaluation_output_is_unique_and_checkpoint_labelled(evaluation_config):
    original = str(evaluation_config.paths.output_dir)
    evaluate_module.configure_evaluation_output(evaluation_config)
    first = Path(evaluation_config.paths.output_dir)
    evaluation_config.paths.output_dir = original
    evaluate_module.configure_evaluation_output(evaluation_config)
    second = Path(evaluation_config.paths.output_dir)
    assert first != second
    assert first.parent == second.parent
    assert first.parent.name.startswith("checkpoint-100-")
    assert second.is_relative_to(Path(original))


def test_manifest_protocol_rejects_changes_after_dataset_construction(evaluation_config, tmp_path):
    import hashlib

    manifest = tmp_path / "split.json"
    manifest.write_text('{"train": ["one"], "val": ["two"]}')
    loaded_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config = evaluation_config
    del config.data.val_dataset.chunk_ids
    OmegaConf.update(config, "data.val_dataset.split_manifest", str(manifest), force_add=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path / "evaluation")),
        model=SimpleNamespace(config=SimpleNamespace()),
        eval_dataset=SimpleNamespace(split_manifest_sha256=loaded_sha256),
        is_world_process_zero=lambda: True,
        save_metrics=lambda *args: None,
    )
    save_evaluation_results(trainer, config, {})
    protocol = json.loads((tmp_path / "evaluation/evaluation_protocol.json").read_text())
    assert protocol["validation_chunks"] is None
    assert protocol["validation_split_manifest_sha256"] == loaded_sha256
    manifest.write_text('{"train": ["two"], "val": ["one"]}')
    with pytest.raises(ValueError, match="manifest changed"):
        save_evaluation_results(trainer, config, {})


def test_annotations_cannot_change_after_loading(evaluation_config, tmp_path):
    from alpamayo2edge_sft.evaluation_identity import validation_data_identity

    annotations = tmp_path / "annotations.json"
    annotations.write_text('[{"cot": "Yield."}]')
    config = evaluation_config
    OmegaConf.update(config, "data.val_dataset.annotations_path", str(annotations), force_add=True)
    trainer = SimpleNamespace(
        args=SimpleNamespace(output_dir=str(tmp_path / "evaluation")),
        model=SimpleNamespace(config=SimpleNamespace()),
        is_world_process_zero=lambda: True,
        save_metrics=lambda *args: None,
        _evaluation_data_identity=validation_data_identity(config.data.val_dataset),
    )
    annotations.write_text('[{"cot": "Proceed."}]')
    with pytest.raises(ValueError, match="identity changed"):
        save_evaluation_results(trainer, config, {})
    assert not (tmp_path / "evaluation/evaluation_protocol.json").exists()
