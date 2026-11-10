# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for selected-frame public PAI loading."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from alpamayo2edge_sft import dataset as dataset_module
from alpamayo2edge_sft.dataset import (
    CAMERA_NAMES_TO_INDICES,
    PhysicalAIAnnotatedDataset,
    PhysicalAIDataset,
    sample_linear_poses,
)
from scipy.spatial.transform import Rotation


class _DatasetInterface:
    reasoning_db = None

    def __init__(self, **kwargs):
        self.requested_times = {}

    def get_all_clip_ids(self):
        return ["clip"]

    def get_clip_feature(self, clip_id, feature):
        assert clip_id == "clip"
        if feature == "egomotion":
            timestamps = np.arange(0, 20_000_001, 10_000)
            xyz = np.zeros((len(timestamps), 3))
            xyz[:, 0] = timestamps / 1_000_000
            rotation = Rotation.from_euler("z", np.full((len(timestamps), 1), 90), degrees=True)
            return SimpleNamespace(
                timestamps=timestamps,
                values=SimpleNamespace(pose=SimpleNamespace(translation=xyz, rotation=rotation)),
            )

        def decode(timestamps):
            self.requested_times[feature] = timestamps.copy()
            frames = np.full((len(timestamps), 2, 3, 3), CAMERA_NAMES_TO_INDICES[feature], np.uint8)
            return frames, timestamps

        return SimpleNamespace(decode_images_from_timestamps=decode, close=lambda: None)


class _ReaderLifecycleInterface(_DatasetInterface):
    failure = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.readers = []

    def get_clip_feature(self, clip_id, feature):
        if feature == "egomotion" and self.failure == "egomotion":
            raise ValueError("egomotion failed")
        original = super().get_clip_feature(clip_id, feature)
        if feature == "egomotion":
            return original
        reader = SimpleNamespace(close_count=0)
        self.readers.append(reader)

        def decode(timestamps):
            assert reader.close_count == 0, "A closed reader was reused"
            if self.failure == "decode":
                raise ValueError("decode failed")
            frames, actual = original.decode_images_from_timestamps(timestamps)
            if self.failure == "timestamps":
                actual = actual.astype(np.float64)
            return frames, actual

        def close():
            reader.close_count += 1

        reader.decode_images_from_timestamps = decode
        reader.close = close
        return reader


def test_camera_readers_close_once_and_are_fresh_for_each_draw(monkeypatch):
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", _ReaderLifecycleInterface)
    dataset = PhysicalAIDataset(local_dir="unused", num_history_steps=16, use_default_keyframe=True)
    first = dataset[0]
    second = dataset[0]
    readers = dataset.dataset.readers
    assert len(readers) == 2 * len(dataset.camera_features)
    assert len({id(reader) for reader in readers}) == len(readers)
    assert [reader.close_count for reader in readers] == [1] * len(readers)
    torch.testing.assert_close(first["image_frames"], second["image_frames"])
    torch.testing.assert_close(first["ego_future_xyz"], second["ego_future_xyz"])


@pytest.mark.parametrize("failure", ["decode", "timestamps", "egomotion"])
def test_camera_readers_close_once_when_sample_loading_fails(monkeypatch, failure):
    class FailingInterface(_ReaderLifecycleInterface):
        pass
    FailingInterface.failure = failure
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", FailingInterface)
    dataset = PhysicalAIDataset(local_dir="unused", num_history_steps=16, use_default_keyframe=True)
    with pytest.raises(ValueError):
        dataset._load_sample("clip", dataset.DEFAULT_T0_US)
    readers = dataset.dataset.readers
    expected = len(dataset.camera_features) if failure == "egomotion" else 1
    assert len(readers) == expected
    assert [reader.close_count for reader in readers] == [1] * expected


@pytest.mark.parametrize("history_tokens,expected_poses", [(45, 16), (57, 20)])
def test_dataset_selects_camera_times_and_derives_history_from_checkpoint(
    monkeypatch,
    history_tokens,
    expected_poses,
):
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", _DatasetInterface)
    dataset = PhysicalAIDataset(
        local_dir="unused",
        model_config=SimpleNamespace(tokens_per_history_traj=history_tokens),
        use_default_keyframe=True,
        camera_pixel_bounds={0: [81920, 98304], 1: [163840, 196608]},
    )
    sample = dataset[0]

    assert sample["image_frames"].shape == (22, 1, 3, 2, 3)
    assert (
        sample["camera_indices"].tolist()
        == [0] * 3 + [1] * 5 + [2] * 3 + [3] * 3 + [5] * 3 + [6] * 5
    )
    assert sample["absolute_timestamps"].dtype == torch.int64
    assert sample["camera_pixel_bounds"] == {0: (81920, 98304), 1: (163840, 196608)}
    assert sample["image_frames"].dtype == torch.uint8
    np.testing.assert_array_equal(
        dataset.dataset.requested_times["camera_front_wide_120fov"],
        [3_100_000, 4_100_000, 4_600_000, 5_000_000, 5_100_000],
    )
    np.testing.assert_array_equal(
        dataset.dataset.requested_times["camera_cross_left_120fov"],
        [4_100_000, 5_000_000, 5_100_000],
    )
    assert sample["ego_history_xyz"].shape == (1, expected_poses, 3)
    assert sample["ego_future_xyz"].shape == (1, 64, 3)
    torch.testing.assert_close(sample["ego_history_xyz"][0, -1], torch.zeros(3))
    torch.testing.assert_close(sample["ego_future_xyz"][0, 0], torch.tensor([0.0, -0.1, 0.0]))
    torch.testing.assert_close(sample["ego_history_rot"][0, -1], torch.eye(3))


def test_dataset_rejects_history_length_that_cannot_fill_checkpoint_tokens(monkeypatch):
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", _DatasetInterface)
    with pytest.raises(ValueError, match="does not match"):
        PhysicalAIDataset(
            local_dir="unused",
            model_config=SimpleNamespace(tokens_per_history_traj=57),
            num_history_steps=16,
        )


@pytest.mark.parametrize("offsets", [[-1, -0.1], [0, -1, 0], []])
def test_dataset_rejects_invalid_frame_sampling(monkeypatch, offsets):
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", _DatasetInterface)
    with pytest.raises(ValueError, match="Frame offsets"):
        PhysicalAIDataset(local_dir="unused", num_history_steps=16, frame_offsets=offsets)


@pytest.mark.parametrize(
    "bounds",
    [
        [],
        {"1": [81920, 98304]},
        {7: [81920, 98304]},
        {1: [98304]},
        {1: [0, 98304]},
        {1: [98304, 81920]},
        {1: [81920.0, 98304]},
        {1: "81920,98304"},
    ],
)
def test_dataset_rejects_invalid_camera_pixel_bounds(monkeypatch, bounds):
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", _DatasetInterface)
    expected_error = TypeError if isinstance(bounds, list) else ValueError
    with pytest.raises(expected_error, match="camera_pixel_bounds"):
        PhysicalAIDataset(local_dir="unused", num_history_steps=16, camera_pixel_bounds=bounds)


def test_pose_anchor_is_latest_captured_image_across_all_cameras(monkeypatch):
    class DelayedCameras(_DatasetInterface):
        def get_clip_feature(self, clip_id, feature):
            result = super().get_clip_feature(clip_id, feature)
            if feature == "egomotion":
                result.values.pose.translation[:, 0] = (result.timestamps / 1_000_000) ** 2
                return result

            def decode(timestamps):
                frames, actual = result.decode_images_from_timestamps(timestamps)
                delay = 20_000 if feature == "camera_front_wide_120fov" else 40_000
                return frames, actual - delay

            return SimpleNamespace(decode_images_from_timestamps=decode, close=lambda: None)

    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", DelayedCameras)
    sample = PhysicalAIDataset(
        local_dir="unused", num_history_steps=16, use_default_keyframe=True
    )[0]
    assert sample["requested_t0_us"] == 5_100_000
    assert sample["t0_us"] == 5_080_000
    assert sample["absolute_timestamps"][-1] == 5_060_000
    expected_displacement = 5.18**2 - 5.08**2
    torch.testing.assert_close(
        sample["ego_future_xyz"][0, 0],
        torch.tensor([0.0, -expected_displacement, 0.0]),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(sample["ego_history_xyz"][0, -1], torch.zeros(3))


def test_boundary_event_accepts_recorded_frame_before_nominal_clip_start(monkeypatch):
    class BoundaryCameras(_DatasetInterface):
        def get_clip_feature(self, clip_id, feature):
            result = super().get_clip_feature(clip_id, feature)
            if feature == "egomotion":
                return result

            def decode(timestamps):
                frames, actual = result.decode_images_from_timestamps(timestamps)
                actual = actual - 13_070
                actual[timestamps == 0] = -33_150
                return frames, actual

            return SimpleNamespace(decode_images_from_timestamps=decode, close=lambda: None)

    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", BoundaryCameras)
    dataset = PhysicalAIDataset(local_dir="unused", num_history_steps=20)
    sample = dataset._load_sample("clip", 2_000_000)
    assert sample["absolute_timestamps"].min().item() == -33_150
    assert sample["t0_us"] == 1_986_930
    assert sample["image_frames"].shape[0] == 22
    assert sample["ego_history_xyz"].shape == (1, 20, 3)
    assert sample["ego_future_xyz"].shape == (1, 64, 3)
    torch.testing.assert_close(sample["ego_history_xyz"][0, -1], torch.zeros(3))


@pytest.mark.parametrize("event_us,capture_offset_us", [(2_000_000, -200_000), (13_600_000, 10_000)])
def test_captured_pose_window_still_rejects_out_of_clip_targets(
    monkeypatch, event_us, capture_offset_us,
):
    class ShiftedCameras(_DatasetInterface):
        def get_clip_feature(self, clip_id, feature):
            result = super().get_clip_feature(clip_id, feature)
            if feature == "egomotion":
                return result

            def decode(timestamps):
                frames, actual = result.decode_images_from_timestamps(timestamps)
                return frames, actual + capture_offset_us

            return SimpleNamespace(decode_images_from_timestamps=decode, close=lambda: None)

    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", ShiftedCameras)
    dataset = PhysicalAIDataset(local_dir="unused", num_history_steps=20)
    with pytest.raises(ValueError, match="outside the 20-second PAI clip"):
        dataset._load_sample("clip", event_us)


def test_pose_sampling_uses_linear_translation_and_normalized_quaternions():
    quaternions = Rotation.from_euler("z", [[0], [120], [120]], degrees=True).as_quat()
    quaternions[1] *= -1  # Equivalent rotation with the opposite quaternion sign.
    source = SimpleNamespace(
        timestamps=np.array([0, 1_000_000, 2_000_000]),
        values=SimpleNamespace(
            pose=SimpleNamespace(
                translation=np.array([[10.0, 0, 0], [20.0, 0, 0], [40.0, 0, 0]]),
                rotation=Rotation.from_quat(quaternions),
            )
        ),
    )
    positions, rotations = sample_linear_poses(source, 0, np.array([0, 250_000]))
    np.testing.assert_allclose(positions[1] - positions[0], [2.5, 0, 0], atol=1e-6)
    expected = 0.75 * quaternions[0] - 0.25 * quaternions[1]
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(
        rotations[1].as_matrix(), Rotation.from_quat(expected).as_matrix(), atol=1e-7
    )
    # SLERP would yield 30 degrees; normalized linear interpolation differs here.
    assert abs(rotations[1].as_euler("xyz", degrees=True)[-1] - 30) > 1


def test_pose_sampling_rejects_extrapolation():
    source = _DatasetInterface().get_clip_feature("clip", "egomotion")
    with pytest.raises(ValueError, match="interpolation range"):
        sample_linear_poses(source, 20_000_000, np.array([100_000]))


def test_capture_timestamps_cannot_be_silently_truncated_to_integers(monkeypatch):
    class FloatCaptureTimes(_DatasetInterface):
        def get_clip_feature(self, clip_id, feature):
            result = super().get_clip_feature(clip_id, feature)
            if feature == "egomotion":
                return result

            def decode(timestamps):
                frames, times = result.decode_images_from_timestamps(timestamps)
                return frames, times.astype(float)

            return SimpleNamespace(decode_images_from_timestamps=decode, close=lambda: None)

    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", FloatCaptureTimes)
    with pytest.raises(ValueError, match="integer timestamps"):
        PhysicalAIDataset(local_dir="unused", num_history_steps=16, use_default_keyframe=True)[0]


@pytest.fixture
def annotated_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", _DatasetInterface)

    def create(rows, **kwargs):
        path = tmp_path / "annotations.json"
        path.write_text(json.dumps(rows))
        options = {"local_dir": "unused", "num_history_steps": 20, **kwargs}
        return PhysicalAIAnnotatedDataset(annotations_path=str(path), **options)

    return create


def test_annotated_dataset_preserves_repeated_events_and_text_before_preprocessing(
    annotated_dataset, monkeypatch,
):
    seen = []

    def preprocess(*, data):
        seen.append((data["requested_t0_us"], list(data["nav_text"]), data["cot"]))
        return {"text": data["nav_text"][0] + " " + data["cot"]}

    monkeypatch.setattr(
        dataset_module, "instantiate",
        lambda config, _partial_: lambda model_config: preprocess,
    )
    dataset = annotated_dataset(
        [
            {"clip_id": "clip", "t0_relative": 5_100_000,
             "nav_text": "Continue straight", "cot": "The lane is clear."},
            {"clip_id": "clip", "t0_relative": 6_200_000,
             "nav_text": ["Turn left"], "cot": "Yield to the oncoming vehicle."},
        ],
        preprocess={"_target_": "unused"}, require_navigation=True, require_cot=True,
    )
    assert len(dataset) == 2
    assert dataset.clip_ids == ["clip", "clip"]
    first, second = dataset[0], dataset[1]
    assert seen == [
        (5_100_000, ["Continue straight"], "The lane is clear."),
        (6_200_000, ["Turn left"], "Yield to the oncoming vehicle."),
    ]
    assert second["tokenized_data"]["text"] == "Turn left Yield to the oncoming vehicle."
    assert first["t0_us"] == 5_100_000 and second["t0_us"] == 6_200_000
    assert first["image_frames"].shape == second["image_frames"].shape == (22, 1, 3, 2, 3)
    assert first["ego_history_xyz"].shape == (1, 20, 3)
    assert first["ego_future_xyz"].shape == (1, 64, 3)
    np.testing.assert_array_equal(
        dataset.dataset.requested_times["camera_front_wide_120fov"],
        [4_200_000, 5_200_000, 5_700_000, 6_100_000, 6_200_000],
    )
    first["nav_text"][0] = "Changed by caller"
    assert dataset[0]["nav_text"] == ["Continue straight"]


@pytest.mark.parametrize(
    "labels,options",
    [
        ({"nav_text": "Continue straight"}, {"require_navigation": True}),
        ({"cot": "The lane ahead is clear"}, {"require_cot": True}),
        ({}, {}),
    ],
)
def test_annotated_dataset_allows_only_the_enabled_text_labels(
    annotated_dataset, labels, options,
):
    dataset = annotated_dataset(
        [{"clip_id": "clip", "t0_relative": 5_100_000, **labels}], **options,
    )
    sample = dataset[0]
    assert ("nav_text" in sample) == ("nav_text" in labels)
    assert ("cot" in sample) == ("cot" in labels)


def test_annotated_dataset_filters_by_selected_clip_ids_and_warns(annotated_dataset, caplog):
    dataset = annotated_dataset(
        [{"clip_id": clip, "t0_relative": 5_100_000} for clip in ["missing", "clip"]],
        chunk_ids=[7],
    )
    assert len(dataset) == 1
    assert dataset[0]["clip_id"] == "clip"
    assert "filtered out 1/2 annotations" in caplog.text
    assert "chunk_ids=[7]" in caplog.text


def test_annotated_dataset_rejects_empty_selection(annotated_dataset):
    with pytest.raises(ValueError, match="no annotated samples remain"):
        annotated_dataset([{"clip_id": "missing", "t0_relative": 5_100_000}])


@pytest.mark.parametrize("rows", [[], {}, [None], ["clip"]])
def test_annotated_dataset_rejects_invalid_json_shapes(annotated_dataset, rows):
    with pytest.raises(ValueError, match="JSON list|JSON object"):
        annotated_dataset(rows)


@pytest.mark.parametrize("clip_id", [None, "", "  ", 5, ["clip"]])
def test_annotated_dataset_requires_nonempty_clip_id(annotated_dataset, clip_id):
    with pytest.raises(ValueError, match="clip_id must be a nonempty string"):
        annotated_dataset([{"clip_id": clip_id, "t0_relative": 5_100_000}])


@pytest.mark.parametrize("timestamp", [None, True, 5_100_000.0, "5100000", []])
def test_annotated_dataset_rejects_noninteger_timestamps(annotated_dataset, timestamp):
    with pytest.raises(ValueError, match="integer clip-relative microseconds"):
        annotated_dataset([{"clip_id": "clip", "t0_relative": timestamp}])


@pytest.mark.parametrize("timestamp", [-1, 0, 1_999_999, 13_600_001, 29_708_785_000])
def test_annotated_dataset_checks_safe_requested_horizon_before_preprocessing(
    annotated_dataset, monkeypatch, timestamp,
):
    def unexpected_processor(*args, **kwargs):
        pytest.fail("Invalid annotation horizons must fail before processor construction")

    monkeypatch.setattr(dataset_module, "instantiate", unexpected_processor)
    with pytest.raises(ValueError, match="history, camera frames, and future"):
        annotated_dataset(
            [{"clip_id": "clip", "t0_relative": timestamp}],
            preprocess={"_target_": "unused"},
        )


@pytest.mark.parametrize("timestamp", [2_000_000, 13_600_000])
def test_annotated_dataset_accepts_safe_horizon_boundaries(annotated_dataset, timestamp):
    sample = annotated_dataset([{"clip_id": "clip", "t0_relative": timestamp}])[0]
    assert sample["t0_us"] == timestamp


def test_annotated_dataset_horizon_respects_custom_history(annotated_dataset):
    with pytest.raises(ValueError, match="3100000"):
        annotated_dataset(
            [{"clip_id": "clip", "t0_relative": 3_000_000}], num_history_steps=32,
        )


@pytest.mark.parametrize("nav", [None, "", "   ", [], ["left", "right"], [1], {}])
def test_annotated_dataset_rejects_invalid_navigation(annotated_dataset, nav):
    with pytest.raises(
        ValueError, match="nav_text must be a nonempty string or single-string list"
    ):
        annotated_dataset([{"clip_id": "clip", "t0_relative": 5_100_000, "nav_text": nav}])


@pytest.mark.parametrize("cot", [None, "", "   ", [], 5])
def test_annotated_dataset_rejects_invalid_reasoning(annotated_dataset, cot):
    with pytest.raises(ValueError, match="cot must be a nonempty string"):
        annotated_dataset([{"clip_id": "clip", "t0_relative": 5_100_000, "cot": cot}])


@pytest.mark.parametrize(
    "option,label", [("require_navigation", "nav_text"), ("require_cot", "cot")]
)
def test_annotated_dataset_requires_enabled_labels(annotated_dataset, option, label):
    with pytest.raises(ValueError, match=f"{label} is required"):
        annotated_dataset([{"clip_id": "clip", "t0_relative": 5_100_000}], **{option: True})


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"use_default_keyframe": True}, "use_default_keyframe=false"),
        ({"reasoning_metadata": "reasoning.parquet"}, "reasoning_metadata"),
    ],
)
def test_annotated_dataset_rejects_competing_event_sources(annotated_dataset, kwargs, message):
    with pytest.raises(ValueError, match=message):
        annotated_dataset([{"clip_id": "clip", "t0_relative": 5_100_000}], **kwargs)


def test_base_dataset_keeps_keyframe_reasoning_and_preprocessing_behavior(monkeypatch):
    class ReasoningInterface(_DatasetInterface):
        reasoning_db = {"clip": {}}

        def get_clip_key_frame(self, clip_id):
            return 6_100_000

        def get_reasoning_data(self, clip_id, requested_t0_us):
            assert clip_id == "clip" and requested_t0_us == 6_100_000
            return {"cot": "The original event label"}

    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", ReasoningInterface)
    monkeypatch.setattr(
        dataset_module, "instantiate",
        lambda config, _partial_: lambda model_config: lambda data: {"cot": data["cot"]},
    )
    sample = PhysicalAIDataset(
        local_dir="unused", num_history_steps=20, preprocess={"_target_": "unused"},
    )[0]
    assert sample["requested_t0_us"] == 6_100_000
    assert sample["cot"] == sample["tokenized_data"]["cot"] == "The original event label"


def test_random_keyframes_repeat_by_seed_epoch_and_index_across_workers(monkeypatch):
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", _DatasetInterface)
    first = PhysicalAIDataset(local_dir="unused", num_history_steps=20, random_keyframes=True, data_seed=7)
    second = PhysicalAIDataset(local_dir="unused", num_history_steps=20, random_keyframes=True, data_seed=7)
    initial = [first._random_t0_us(index) for index in range(20)]
    assert initial == [second._random_t0_us(index) for index in range(20)]
    assert all(2_000_000 <= value <= 13_600_000 and value % 100_000 == 0 for value in initial)
    # Persistent worker copies hold the same shared-memory epoch tensor.
    import copy
    worker_copy = copy.copy(first)
    first.set_epoch(3)
    assert int(worker_copy._epoch) == 3
    later = [first._random_t0_us(index) for index in range(20)]
    assert later != initial
    assert later == [worker_copy._random_t0_us(index) for index in range(20)]
    second.set_epoch(3)
    assert later == [second._random_t0_us(index) for index in range(20)]
    first.set_epoch(0)
    assert initial == [first._random_t0_us(index) for index in range(20)]


def test_reasoning_metadata_indexes_all_event_times(monkeypatch):
    class ReasoningInterface(_DatasetInterface):
        reasoning_db = {"clip": {"event_t0s": [2_000_000, 6_100_000], "cot": ["Wait.", "Proceed."]}}

        def get_reasoning_data(self, clip_id, requested_t0_us):
            entry = self.reasoning_db[clip_id]
            return {"cot": entry["cot"][entry["event_t0s"].index(requested_t0_us)]}

    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", ReasoningInterface)
    dataset = PhysicalAIDataset(local_dir="unused", num_history_steps=20, reasoning_metadata="labels.parquet")
    assert len(dataset) == 2
    assert [dataset[index]["requested_t0_us"] for index in range(2)] == [2_000_000, 6_100_000]
    assert [dataset[index]["cot"] for index in range(2)] == ["Wait.", "Proceed."]
    with pytest.raises(ValueError, match="annotated event times"):
        PhysicalAIDataset(local_dir="unused", num_history_steps=20, reasoning_metadata="labels.parquet", use_default_keyframe=True)


def test_official_split_passed_to_public_reader(monkeypatch):
    recorded = {}
    def factory(**kwargs):
        recorded.update(kwargs)
        return _DatasetInterface()
    monkeypatch.setattr(dataset_module, "_create_public_dataset_interface", factory)
    PhysicalAIDataset(local_dir="unused", num_history_steps=20, split="val")
    assert recorded["split"] == "val" and recorded["inclusive_start"] is True


@pytest.mark.parametrize("text", ["Turn left in 594m", "Go left somehow", "Turn left in 2.5m"])
def test_annotated_dataset_rejects_navigation_outside_release_format(annotated_dataset, text):
    row = {"clip_id": "clip", "t0_relative": 5_100_000, "nav_text": text}
    with pytest.raises(ValueError, match="navigation"):
        annotated_dataset([row])
    # Partners can deliberately train a different documented input format.
    dataset = annotated_dataset([row], validate_navigation_format=False)
    assert dataset[0]["nav_text"] == [text]
