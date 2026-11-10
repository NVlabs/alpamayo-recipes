# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw-fixture checks for partner-data axes, clocks, manifests and public samples."""

import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from alpamayo2edge_sft.driving_datasets import ONCEDataset, TartanDriveDataset, load_split_manifest


def _manifest(tmp_path, dataset_name, **changes):
    document = {"schema_version": 1, "dataset": dataset_name, "train": ["a"], "val": ["b"]}
    document.update(changes)
    supplied = document.get("recordings", {})
    document["recordings"] = {
        name: {"start_time_us": 0 if name == "a" else 86_440_000_000,
               "end_time_us": 40_000_000 if name == "a" else 86_480_000_000,
               "time_reference": "unix", **supplied.get(name, {})}
        for name in ("a", "b")
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(document))
    return path


def _image(value):
    stream = io.BytesIO()
    Image.fromarray(np.full((6, 8, 3), value % 256, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def _tar(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def _once(tmp_path, *, tar=False, missing=(), times=None, yaw=0.):
    times = np.arange(0, 16_001, 500) if times is None else np.asarray(times)
    # ONCE forward is -Y. Rotate the entire sequence to exercise anchor rotation.
    rotation = Rotation.from_euler("z", yaw, degrees=True)
    positions = rotation.apply(np.stack([np.zeros(len(times)), -times / 1000., np.zeros(len(times))], axis=1))
    frames = [{"frame_id": str(int(t)), "sequence_id": "a",
               "pose": [*rotation.as_quat(), *position]}
              for t, position in zip(times, positions)]
    if tar:
        _tar(tmp_path / "val_anno/val_infos.tar", {"data/a/a.json": json.dumps({"frames": frames}).encode()})
        _tar(tmp_path / "val_camera/val_cam01.tar", {
            f"./data/a/cam01/{int(t)}.jpg": _image(i) for i, t in enumerate(times) if i not in missing})
    else:
        folder = tmp_path / "data/a"
        (folder / "cam01").mkdir(parents=True)
        (folder / "a.json").write_text(json.dumps({"frames": frames}))
        for i, t in enumerate(times):
            if i not in missing:
                (folder / f"cam01/{int(t)}.jpg").write_bytes(_image(i))
    return tmp_path


def _tartan(tmp_path, *, tar=False, duplicate=False, conflicting=False, duration=16., camera_clock=True):
    folder = tmp_path / "a"
    (folder / "super_odom").mkdir(parents=True)
    (folder / "tartanvo_odom").mkdir()
    times = np.arange(0, duration + .001, .05)
    poses = np.zeros((len(times), 7))
    poses[:, 0] = times
    poses[:, 6] = 1
    if duplicate:
        times = np.insert(times, 41, times[40])
        poses = np.insert(poses, 41, poses[40], axis=0)
        if conflicting:
            poses[41, 0] += 1
    np.savetxt(folder / "super_odom/timestamps.txt", times)
    np.save(folder / "super_odom/odometry.npy", poses)
    images = np.arange(0, duration + .001, .1)
    if duplicate:
        images = np.insert(images, 21, images[20])
    np.savetxt(folder / "tartanvo_odom/timestamps.txt", images)
    entries = {f"image_left_color/{i:06d}.png": _image(i) for i in range(len(images))}
    if camera_clock:
        # These are the synthetic camera's acquisition ticks, independently
        # generated rather than copied from the optionally corrupted VO clock.
        capture_times = np.arange(len(images)) / 10.
        stream = io.StringIO()
        np.savetxt(stream, capture_times)
        entries["image_left_color/timestamps.txt"] = stream.getvalue().encode()
    if tar:
        _tar(folder / "image_left_color.tar", entries)
    else:
        (folder / "image_left_color").mkdir()
        for name, data in entries.items():
            (folder / name).write_bytes(data)
    return tmp_path


@pytest.mark.parametrize("dataset,adapter,fixture", [("once", ONCEDataset, _once), ("tartan", TartanDriveDataset, _tartan)])
@pytest.mark.parametrize("tar", [False, True])
def test_raw_data_produces_public_sample_contract(tmp_path, dataset, adapter, fixture, tar):
    fixture(tmp_path, tar=tar)
    manifest = _manifest(tmp_path, dataset, recordings={"a": {"source_partition": "val"}})
    data = adapter(str(tmp_path), str(manifest), model_config={"tokens_per_history_traj": 57})
    sample = data[0]
    assert sample["image_frames"].shape == (5, 1, 3, 6, 8)
    assert sample["image_frames"].dtype == torch.uint8
    assert sample["ego_history_xyz"].shape == (1, 20, 3)
    assert sample["ego_future_xyz"].shape == (1, 64, 3)
    assert sample["ego_history_rot"].shape == (1, 20, 3, 3)
    assert sample["ego_future_rot"].shape == (1, 64, 3, 3)
    torch.testing.assert_close(sample["ego_history_xyz"][0, -1], torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(sample["ego_future_xyz"][0, :, 0], torch.arange(1, 65) / 10., atol=2e-6, rtol=0)
    torch.testing.assert_close(sample["ego_future_xyz"][0, :, 1:], torch.zeros(64, 2), atol=2e-6, rtol=0)
    torch.testing.assert_close(sample["ego_future_rot"], torch.eye(3).expand(1, 64, 3, 3))
    assert sample["camera_indices"].tolist() == [1] * 5
    assert (sample["absolute_timestamps"].numpy() <= sample["t0_us"] + data.frame_offsets_us).all()
    assert sample["absolute_timestamps"][-1].item() == sample["t0_us"]
    assert sample["relative_timestamps"][-1].item() == 0
    assert "nav_text" not in sample and "cot" not in sample
    assert all(value == "a" for value in data.clip_ids)
    assert not list(tmp_path.glob("**/cache"))  # Reading does not write inside the package/data.


@pytest.mark.parametrize("tar", [False, True])
def test_once_default_slots_repeat_capture_with_its_actual_timestamp(tmp_path, tar):
    _once(tmp_path, tar=tar)
    data = ONCEDataset(str(tmp_path), str(_manifest(tmp_path, "once")), num_history_steps=20)
    sample = data[0]
    assert sample["t0_us"] == 2_000_000
    assert data.frame_offsets_us.tolist() == [-2_000_000, -1_000_000, -500_000, -100_000, 0]
    assert sample["absolute_timestamps"].tolist() == [0, 1_000_000, 1_500_000, 1_500_000, 2_000_000]
    assert sample["relative_timestamps"].tolist() == [-2., -1., -.5, -.5, 0.]
    assert len(torch.unique(sample["absolute_timestamps"])) == 4
    assert torch.equal(sample["image_frames"][2], sample["image_frames"][3])
    # The .1 s future target lies between measured .5 s poses.
    assert sample["ego_future_xyz"][0, 0, 0].item() == pytest.approx(.1, abs=2e-6)


def test_once_sampling_protocol_uses_resolved_instance_settings(tmp_path):
    _once(tmp_path)
    data = ONCEDataset(
        str(tmp_path), str(_manifest(tmp_path, "once")), num_history_steps=6,
        num_future_steps=10, time_step=.2, frame_offsets=[-1., -.5, 0.],
        sample_stride_seconds=3., max_pose_gap_seconds=.75, max_image_age_seconds=.25,
    )
    protocol = json.loads(json.dumps(data.evaluation_sampling_protocol()))
    assert protocol["adapter_variant"] == "once_causal_images_interpolated_lidar_poses_v1"
    assert protocol["source"]["nominal_camera_hz"] == 2
    assert protocol["source"]["nominal_pose_hz"] == 2
    assert "actual intervals may vary" in protocol["source"]["rate_basis"]
    assert protocol["images"]["requested_offsets_seconds"] == [-1., -.5, 0.]
    assert protocol["images"]["max_age_from_request_seconds"] == .25
    assert protocol["images"]["duplicate_frames"] == "retained"
    assert protocol["trajectory"]["source_origin"] == "LiDAR sensor center"
    assert "NLERP" in protocol["trajectory"]["rotation_interpolation"]
    assert protocol["trajectory"]["time_step_seconds"] == .2
    assert protocol["trajectory"]["history_steps"] == 6
    assert protocol["trajectory"]["future_steps"] == 10
    assert protocol["trajectory"]["max_source_gap_seconds"] == .75
    assert protocol["minimum_anchor_stride_seconds"] == 3.
    assert data[0]["image_frames"].shape[0] == 3
    assert data[0]["ego_history_xyz"].shape[1] == 6
    assert data[0]["ego_future_xyz"].shape[1] == 10


def test_once_rotated_sequence_keeps_forward_body_targets(tmp_path):
    _once(tmp_path, yaw=70.)
    data = ONCEDataset(str(tmp_path), str(_manifest(tmp_path, "once")), num_history_steps=20)
    sample = data[1]
    torch.testing.assert_close(sample["ego_future_xyz"][0, :, 0], torch.arange(1, 65) / 10., atol=2e-6, rtol=0)
    torch.testing.assert_close(sample["ego_future_xyz"][0, :, 1:], torch.zeros(64, 2), atol=2e-6, rtol=0)


def test_tartan_duplicate_clocks_preserve_original_image_index(tmp_path):
    _tartan(tmp_path, duplicate=True, camera_clock=False)
    with pytest.warns(RuntimeWarning, match="unverified"):
        data = TartanDriveDataset(
            str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20,
            allow_unverified_image_timestamps=True,
            legacy_image_timestamp_evidence="Synthetic fixture explicitly aligns image indices with VO rows",
        )
    assert data.clock_cleanup["a"] == {"identical_pose_duplicates_removed": 1, "repeated_image_timestamp_rows_removed": 1}
    sample = data[1]
    # 4 seconds is image row 41 after insertion of the duplicate timestamp.
    assert sample["t0_us"] == 4_000_000
    assert sample["image_frames"][-1, 0, 0, 0, 0] == 41


def test_tartan_conflicting_duplicate_pose_is_rejected(tmp_path):
    _tartan(tmp_path, duplicate=True, conflicting=True)
    with pytest.raises(ValueError, match="conflicting"):
        TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20)


@pytest.mark.parametrize("changes,match", [
    ({"val": ["a"]}, "overlap"), ({"train": ["a", "a"]}, "duplicate"),
    ({"train": []}, "nonempty"), ({"train": ["../a"]}, "path-component"),
    ({"dataset": "invalid"}, "dataset"), ({"schema_version": True}, "schema_version"),
])
def test_manifest_rejects_ambiguous_or_overlapping_splits(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        load_split_manifest(_manifest(tmp_path, "once", **changes), "once")


def test_once_selection_is_independent_of_official_bundle(tmp_path):
    _once(tmp_path, tar=True)
    manifest = _manifest(tmp_path, "once", recordings={"a": {"source_partition": "val"}})
    data = ONCEDataset(str(tmp_path), str(manifest), split="train", num_history_steps=20)
    assert data.selected_recordings == ["a"]


def test_missing_requested_recording_is_error_not_fallback(tmp_path):
    _once(tmp_path)
    with pytest.raises(ValueError, match="exactly one annotation"):
        ONCEDataset(str(tmp_path), str(_manifest(tmp_path, "once")), split="val", num_history_steps=20)


def test_pose_gaps_exclude_crossing_windows(tmp_path):
    _once(tmp_path, times=np.r_[np.arange(0, 6001, 500), np.arange(9000, 20_001, 500)])
    data = ONCEDataset(str(tmp_path), str(_manifest(tmp_path, "once")), num_history_steps=20)
    for _, anchor, _ in data._samples:
        time = data._samples[0][0]["image_times"][anchor]
        assert time - 1_900_000 >= 9_000_000 or time + 6_400_000 <= 6_000_000


def test_decode_error_is_not_silently_replaced(tmp_path):
    _once(tmp_path)
    data = ONCEDataset(str(tmp_path), str(_manifest(tmp_path, "once")), num_history_steps=20)
    (tmp_path / "data/a/cam01/0.jpg").write_bytes(b"invalid")
    with pytest.raises(OSError):
        data[0]


def test_unknown_options_and_wrong_history_do_not_silently_pass(tmp_path):
    manifest = str(_manifest(tmp_path, "once"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        ONCEDataset(str(tmp_path), manifest, num_history_steps=20, random_keyframes=True)
    with pytest.raises(ValueError, match="history tokens"):
        ONCEDataset(str(tmp_path), manifest, num_history_steps=20, model_config={"tokens_per_history_traj": 45})


@pytest.mark.parametrize("option,value", [("max_samples", 0), ("data_seed", -1), ("time_step", 0),
                                         ("frame_offsets", [-1, .1]), ("num_future_steps", True)])
def test_invalid_sampling_arguments_fail_early(tmp_path, option, value):
    with pytest.raises(ValueError):
        ONCEDataset(str(tmp_path), str(_manifest(tmp_path, "once")), num_history_steps=20, **{option: value})


def test_limit_is_opt_in_and_seeded(tmp_path):
    _tartan(tmp_path)
    arguments = dict(data_root=str(tmp_path), split_manifest=str(_manifest(tmp_path, "tartan")), num_history_steps=20)
    full = TartanDriveDataset(**arguments)
    limited = TartanDriveDataset(**arguments, max_samples=2, data_seed=99)
    repeated = TartanDriveDataset(**arguments, max_samples=2, data_seed=99)
    assert len(full) > len(limited) == 2
    assert [limited[i]["t0_us"] for i in range(2)] == [repeated[i]["t0_us"] for i in range(2)]


def test_optional_tar_cache_outside_dataset_and_reader_is_picklable(tmp_path):
    import pickle
    _once(tmp_path, tar=True)
    manifest = _manifest(tmp_path, "once")
    data = ONCEDataset(str(tmp_path), str(manifest), num_history_steps=20, cache_dir=str(tmp_path / "index"))
    assert len(list((tmp_path / "index").glob("*.json"))) == 1
    restored = pickle.loads(pickle.dumps(data))
    torch.testing.assert_close(data[0]["image_frames"], restored[0]["image_frames"])


def test_epoch_timestamps_keep_subsecond_trajectory_precision(tmp_path):
    _tartan(tmp_path)
    for name in ("super_odom", "tartanvo_odom", "image_left_color"):
        path = tmp_path / "a" / name / "timestamps.txt"
        np.savetxt(path, np.loadtxt(path) + 1_600_000_000)
    manifest = _manifest(tmp_path, "tartan", recordings={"a": {
        "start_time_us": 1_600_000_000_000_000, "end_time_us": 1_600_000_040_000_000}})
    data = TartanDriveDataset(str(tmp_path), str(manifest), num_history_steps=20)
    sample = data[0]
    assert sample["t0_us"] == 1_600_000_002_000_000
    torch.testing.assert_close(sample["ego_future_xyz"][0, :, 0], torch.arange(1, 65) / 10., atol=2e-6, rtol=0)


def test_once_pose_order_and_norm_are_validated(tmp_path):
    _once(tmp_path)
    path = tmp_path / "data/a/a.json"
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    changed["frames"][5]["frame_id"] = changed["frames"][4]["frame_id"]
    path.write_text(json.dumps(changed))
    arguments = dict(data_root=str(tmp_path), split_manifest=str(_manifest(tmp_path, "once")), num_history_steps=20)
    with pytest.raises(ValueError, match="increasing"):
        ONCEDataset(**arguments)
    original["frames"][5]["pose"][3] = 2.
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="normalized"):
        ONCEDataset(**arguments)


def test_tartan_requires_numeric_image_indices(tmp_path):
    _tartan(tmp_path)
    (tmp_path / "a/image_left_color/000000.png").rename(tmp_path / "a/image_left_color/front.png")
    with pytest.raises(ValueError, match="zero-based decimal indices"):
        TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20)


def test_dataset_does_not_have_legacy_256_window_cap(tmp_path):
    _tartan(tmp_path, duration=40.)
    data = TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20,
                             sample_stride_seconds=.1)
    assert len(data) > 256


def test_manifest_hash_and_raw_interval_are_retained(tmp_path):
    import hashlib
    _tartan(tmp_path)
    manifest = _manifest(tmp_path, "tartan")
    data = TartanDriveDataset(str(tmp_path), str(manifest), num_history_steps=20)
    assert data.split_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert data.source_intervals["a"] == {"start_time_us": 0, "end_time_us": 16_000_000}


@pytest.mark.parametrize("metadata", [{"start_time_us": 1}, {"end_time_us": 15_999_999}])
def test_manifest_cannot_understate_raw_recording_interval(tmp_path, metadata):
    _tartan(tmp_path)
    manifest = _manifest(tmp_path, "tartan", recordings={"a": metadata})
    with pytest.raises(ValueError, match="does not cover raw image/pose clocks"):
        TartanDriveDataset(str(tmp_path), str(manifest), num_history_steps=20)


@pytest.mark.parametrize("tar", [False, True])
def test_tartan_requires_camera_clock_without_explicit_legacy_opt_in(tmp_path, tar):
    _tartan(tmp_path, tar=tar, camera_clock=False)
    with pytest.raises(ValueError, match="timestamps.txt"):
        TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20)


@pytest.mark.parametrize("evidence", [None, "", " "])
def test_tartan_legacy_clock_requires_evidence(tmp_path, evidence):
    _tartan(tmp_path, camera_clock=False)
    with pytest.raises(ValueError):
        TartanDriveDataset(
            str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20,
            allow_unverified_image_timestamps=True, legacy_image_timestamp_evidence=evidence,
        )


def test_tartan_legacy_clock_is_warned_and_identified_in_protocol(tmp_path):
    _tartan(tmp_path, camera_clock=False)
    evidence = "Explicitly reviewed the synthetic fixture's VO/image row association"
    with pytest.warns(RuntimeWarning, match="unverified"):
        data = TartanDriveDataset(
            str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20,
            allow_unverified_image_timestamps=True, legacy_image_timestamp_evidence=evidence,
        )
    source = data.image_timestamp_sources["a"]
    assert source["mode"] == "unverified_tartanvo_index"
    assert source["clock_sha256"] == hashlib.sha256(
        (tmp_path / "a/tartanvo_odom/timestamps.txt").read_bytes()).hexdigest()
    protocol = data.evaluation_sampling_protocol()
    assert protocol["contains_unverified_timing"] is True
    assert protocol["recordings"]["a"] == source
    assert evidence in json.dumps(protocol)


@pytest.mark.parametrize("tar", [False, True])
@pytest.mark.parametrize("vo_change", ["leading_row", "middle_row", "equal_count_shift", "missing"])
def test_tartan_camera_clock_is_independent_of_vo_clock(tmp_path, tar, vo_change):
    _tartan(tmp_path, tar=tar)
    manifest = str(_manifest(tmp_path, "tartan"))
    expected = TartanDriveDataset(str(tmp_path), manifest, num_history_steps=20)[0]
    vo_path = tmp_path / "a/tartanvo_odom/timestamps.txt"
    clock = np.loadtxt(vo_path)
    if vo_change == "missing":
        vo_path.unlink()
    elif vo_change == "leading_row":
        np.savetxt(vo_path, np.r_[0., clock + .1])
    elif vo_change == "middle_row":
        np.savetxt(vo_path, np.insert(clock, 10, .95))
    else:
        np.savetxt(vo_path, clock + .3)
    data = TartanDriveDataset(str(tmp_path), manifest, num_history_steps=20)
    actual = data[0]
    assert actual["t0_us"] == expected["t0_us"] == 2_000_000
    assert actual["image_frames"][-1, 0, 0, 0, 0].item() == 20
    for key in ("image_frames", "absolute_timestamps", "relative_timestamps", "ego_future_xyz"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    protocol = data.evaluation_sampling_protocol()
    assert protocol["contains_unverified_timing"] is False
    source = protocol["recordings"]["a"]
    assert source == data.image_timestamp_sources["a"]
    assert source["mode"] == "camera_modality_timestamps"
    assert len(source["clock_sha256"]) == len(source["image_index_sha256"]) == 64
    assert "tartanvo_odom" not in source["source"]


@pytest.mark.parametrize("prefix", ["", "./", "image_left_color/"])
def test_tartan_camera_clock_accepts_native_archive_layouts(tmp_path, prefix):
    _tartan(tmp_path, tar=True)
    path = tmp_path / "a/image_left_color.tar"
    with tarfile.open(path, "r:") as archive:
        entries = {prefix + member.name.removeprefix("image_left_color/"): archive.extractfile(member).read()
                   for member in archive if member.isfile()}
    _tar(path, entries)
    data = TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20)
    assert data[0]["t0_us"] == 2_000_000
    assert data[0]["image_frames"][-1, 0, 0, 0, 0].item() == 20


@pytest.mark.parametrize("damage", ["missing_image", "renumbered", "extra_image", "short_clock", "long_clock",
                                     "duplicate_clock", "descending_clock", "invalid_clock"])
def test_tartan_camera_mapping_rejects_ambiguous_associations_without_legacy_fallback(tmp_path, damage):
    _tartan(tmp_path)
    images = tmp_path / "a/image_left_color"
    clock_path = images / "timestamps.txt"
    clock = np.loadtxt(clock_path)
    if damage == "missing_image":
        (images / "000010.png").unlink()
    elif damage == "renumbered":
        (images / "000010.png").rename(images / "000999.png")
    elif damage == "extra_image":
        (images / "000161.png").write_bytes(_image(161))
    elif damage == "short_clock":
        np.savetxt(clock_path, clock[:-1])
    elif damage == "long_clock":
        np.savetxt(clock_path, np.r_[clock, 16.1])
    elif damage == "duplicate_clock":
        clock[10] = clock[9]
        np.savetxt(clock_path, clock)
    elif damage == "descending_clock":
        np.savetxt(clock_path, clock[::-1])
    else:
        clock_path.write_text("not a camera timestamp\n")
    # Even explicit legacy permission must not hide a malformed camera source.
    with pytest.raises(ValueError):
        TartanDriveDataset(
            str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20,
            allow_unverified_image_timestamps=True,
            legacy_image_timestamp_evidence="Legacy mode is only permitted if the camera clock is absent",
        )


def test_tartan_duplicate_camera_clock_members_are_rejected(tmp_path):
    _tartan(tmp_path, tar=True)
    path = tmp_path / "a/image_left_color.tar"
    with tarfile.open(path, "r:") as archive:
        entries = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
    entries["./image_left_color/timestamps.txt"] = entries["image_left_color/timestamps.txt"]
    _tar(path, entries)
    with pytest.raises(ValueError):
        TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20)


@pytest.mark.parametrize("legacy", [False, True])
def test_tartan_sampling_protocol_rejects_a_clock_changed_after_loading(tmp_path, legacy):
    _tartan(tmp_path, camera_clock=not legacy)
    arguments = dict(data_root=str(tmp_path), split_manifest=str(_manifest(tmp_path, "tartan")), num_history_steps=20)
    if legacy:
        with pytest.warns(RuntimeWarning, match="unverified"):
            data = TartanDriveDataset(**arguments, allow_unverified_image_timestamps=True,
                                     legacy_image_timestamp_evidence="Synthetic VO clock association reviewed")
    else:
        data = TartanDriveDataset(**arguments)
    data.evaluation_sampling_protocol()
    modality = "tartanvo_odom" if legacy else "image_left_color"
    path = tmp_path / "a" / modality / "timestamps.txt"
    np.savetxt(path, np.loadtxt(path) + .01)
    with pytest.raises(ValueError, match="changed"):
        data.evaluation_sampling_protocol()



def test_tartan_cached_archive_preserves_camera_clock_through_pickling(tmp_path, monkeypatch):
    import pickle

    _tartan(tmp_path, tar=True)
    cache = tmp_path / "external-index-cache"
    arguments = dict(data_root=str(tmp_path), split_manifest=str(_manifest(tmp_path, "tartan")),
                     num_history_steps=20, cache_dir=str(cache))
    initial = TartanDriveDataset(**arguments)
    expected_sample = initial[0]
    expected_protocol = initial.evaluation_sampling_protocol()
    assert expected_protocol["recordings"]["a"]["mode"] == "camera_modality_timestamps"
    cached_files = list(cache.glob("*.json"))
    assert len(cached_files) == 1
    original_cache = cached_files[0].read_bytes()

    def unexpected_archive_scan(*args, **kwargs):
        pytest.fail("A cached image/clock index must not rescan the archive")

    monkeypatch.setattr(tarfile, "open", unexpected_archive_scan)
    cached = TartanDriveDataset(**arguments)
    restored = pickle.loads(pickle.dumps(cached))
    for dataset in (cached, restored):
        assert dataset.evaluation_sampling_protocol() == expected_protocol
        sample = dataset[0]
        assert sample["t0_us"] == expected_sample["t0_us"]
        for key in ("image_frames", "absolute_timestamps", "relative_timestamps", "ego_future_xyz"):
            torch.testing.assert_close(sample[key], expected_sample[key], rtol=0, atol=0)
    assert cached_files[0].read_bytes() == original_cache


@pytest.mark.parametrize("tar", [False, True])
def test_tartan_rejects_conflicting_root_and_image_sibling_clocks(tmp_path, tar):
    _tartan(tmp_path, tar=tar)
    alternative = io.StringIO()
    np.savetxt(alternative, np.arange(161) / 10. + .3)
    if tar:
        path = tmp_path / "a/image_left_color.tar"
        with tarfile.open(path, "r:") as archive:
            entries = {member.name: archive.extractfile(member).read() for member in archive if member.isfile()}
        entries["timestamps.txt"] = alternative.getvalue().encode()
        _tar(path, entries)
    else:
        images = tmp_path / "a/image_left_color"
        nested = images / "frames"
        nested.mkdir()
        for path in list(images.glob("*.png")) + [images / "timestamps.txt"]:
            path.rename(nested / path.name)
        (images / "timestamps.txt").write_text(alternative.getvalue())
    with pytest.raises(ValueError, match="Ambiguous camera timestamps"):
        TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20)



@pytest.mark.parametrize("mutation", ["append", "replace"])
def test_tartan_protocol_rejects_changed_archive_even_when_clock_payload_is_identical(tmp_path, mutation):
    _tartan(tmp_path, tar=True)
    path = tmp_path / "a/image_left_color.tar"
    data = TartanDriveDataset(str(tmp_path), str(_manifest(tmp_path, "tartan")), num_history_steps=20)
    loaded_clock_sha = data.evaluation_sampling_protocol()["recordings"]["a"]["clock_sha256"]
    if mutation == "append":
        with path.open("ab") as stream:
            stream.write(b"x")
    else:
        replacement = path.with_suffix(".replacement.tar")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    # Content identity of the clock alone cannot protect cached tar offsets or
    # associations after the archive itself is modified/replaced.
    with tarfile.open(path, "r:") as archive:
        clock = archive.extractfile("image_left_color/timestamps.txt").read()
    assert hashlib.sha256(clock).hexdigest() == loaded_clock_sha
    with pytest.raises(ValueError, match="Image archive changed after"):
        data.evaluation_sampling_protocol()



def test_tartan_archive_cache_does_not_reuse_offsets_when_size_and_mtime_are_preserved(tmp_path):
    import os

    _tartan(tmp_path, tar=True)
    path = tmp_path / "a/image_left_color.tar"
    cache = tmp_path / "external-index-cache"
    arguments = dict(data_root=str(tmp_path), split_manifest=str(_manifest(tmp_path, "tartan")),
                     num_history_steps=20, cache_dir=str(cache))
    original = TartanDriveDataset(**arguments)
    expected_sample = original[0]
    expected_protocol = original.evaluation_sampling_protocol()
    assert len(list(cache.glob("*.json"))) == 1
    before = path.stat()
    with tarfile.open(path, "r:") as archive:
        entries = [(member.name, archive.extractfile(member).read()) for member in archive if member.isfile()]
    replacement = path.with_suffix(".reordered.tar")
    _tar(replacement, dict(reversed(entries)))
    assert replacement.stat().st_size == before.st_size
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    after = path.stat()
    assert after.st_size == before.st_size and after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ino != before.st_ino

    # Filename/size/mtime are unchanged, but reversing member order changes every
    # image and clock byte offset. A stale cache would decode the wrong payload.
    fresh = TartanDriveDataset(**arguments)
    sample = fresh[0]
    assert fresh.evaluation_sampling_protocol() == expected_protocol
    assert sample["t0_us"] == expected_sample["t0_us"]
    for key in ("image_frames", "absolute_timestamps", "relative_timestamps", "ego_future_xyz"):
        torch.testing.assert_close(sample[key], expected_sample[key], rtol=0, atol=0)
    assert len(list(cache.glob("*.json"))) == 2


@pytest.mark.parametrize("dataset,adapter", [("once", ONCEDataset), ("tartan", TartanDriveDataset)])
@pytest.mark.parametrize("split", ["train", "val"])
def test_dataset_split_audit_cannot_be_bypassed_by_skipping_the_cli(tmp_path, dataset, adapter, split):
    # There are deliberately no image/pose files: unsafe separation must fail
    # before either train or evaluation attempts to open the selected data.
    manifest = _manifest(tmp_path, dataset, recordings={"b": {
        "start_time_us": 46_000_000, "end_time_us": 86_000_000,
    }})
    with pytest.raises(ValueError, match="split audit failed"):
        adapter(str(tmp_path), str(manifest), split=split, num_history_steps=20)


@pytest.mark.parametrize("dataset,adapter,fixture", [("once", ONCEDataset, _once), ("tartan", TartanDriveDataset, _tartan)])
def test_dataset_split_audit_default_is_temporal_only_and_uses_exact_loaded_manifest(tmp_path, dataset, adapter, fixture):
    fixture(tmp_path)
    manifest = _manifest(tmp_path, dataset)
    data = adapter(str(tmp_path), str(manifest), num_history_steps=20)
    report = data.split_audit_report
    assert report["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["policy_source"] == "runtime_configuration"
    assert report["requested_context_span_seconds"] == pytest.approx(8.4)
    assert report["status"] == "passed_temporal_only" and report["passed"]
    assert report["temporal_passed"] and not report["spatial_passed"]
    assert not report["full_validation_passed"]
    assert report["min_time_gap_seconds"] == 86400
    assert len(report["pairs"]) == 1
    assert report["policy"]["min_time_gap_seconds"] == 86400
    metadata = data.evaluation_sampling_protocol()["split_audit"]
    for key in ("status", "manifest_sha256", "policy_source", "requested_context_span_seconds",
                "temporal_passed", "spatial_passed", "full_validation_passed", "policy"):
        assert metadata[key] == report[key]
    # Caller changes must not alter the policy captured by the loaded dataset.
    metadata["policy"]["min_time_gap_seconds"] = 8.4
    assert data.evaluation_sampling_protocol()["split_audit"]["policy"]["min_time_gap_seconds"] == 86400
    assert report["policy"]["min_time_gap_seconds"] == 86400


@pytest.mark.parametrize("dataset,adapter,fixture", [("once", ONCEDataset, _once), ("tartan", TartanDriveDataset, _tartan)])
@pytest.mark.parametrize("geography", ["verified", "missing", "nearby"])
def test_dataset_full_split_audit_enforces_spatial_evidence(tmp_path, dataset, adapter, fixture, geography):
    fixture(tmp_path)
    shared = {"spatial_frame": "EPSG:26917", "spatial_units": "metres",
              "spatial_frame_verified": True,
              "spatial_frame_evidence": "Synthetic routes share verified projected coordinates"}
    records = {"a": {**shared, "positions_xy_m": [[0, 0], [1, 0]]},
               "b": {**shared, "positions_xy_m": [[0, 1000], [1, 1000]]}}
    if geography == "missing":
        del records["b"]["positions_xy_m"]
    elif geography == "nearby":
        records["b"]["positions_xy_m"] = [[0, 1], [1, 1]]
    manifest = _manifest(tmp_path, dataset, recordings=records)
    arguments = dict(data_root=str(tmp_path), split_manifest=str(manifest), num_history_steps=20,
                     split_audit={"temporal_only": False})
    if geography != "verified":
        with pytest.raises(ValueError, match="split audit failed"):
            adapter(**arguments)
    else:
        data = adapter(**arguments)
        assert data.split_audit_report["status"] == "passed"
        assert data.split_audit_report["full_validation_passed"]
        assert data.evaluation_sampling_protocol()["split_audit"]["spatial_passed"]


@pytest.mark.parametrize("dataset,adapter,fixture", [("once", ONCEDataset, _once), ("tartan", TartanDriveDataset, _tartan)])
def test_dataset_sampling_protocol_rejects_changed_split_manifest_bytes(tmp_path, dataset, adapter, fixture):
    fixture(tmp_path)
    manifest = _manifest(tmp_path, dataset)
    data = adapter(str(tmp_path), str(manifest), num_history_steps=20)
    data.evaluation_sampling_protocol()
    # A formatting-only edit still changes the exact manifest audited at load.
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        data.evaluation_sampling_protocol()


@pytest.mark.parametrize("adapter,dataset", [(ONCEDataset, "once"), (TartanDriveDataset, "tartan")])
@pytest.mark.parametrize("policy", [False, True, [], "disabled", {"enabled": False}, {"unknown": 1},
                                    {"temporal_only": "true"}, {"min_time_gap_seconds": 0},
                                    {"min_time_gap_seconds": float("nan")}, {"min_location_gap_m": 0}])
def test_dataset_rejects_malformed_or_disabled_split_audit_before_data_loading(tmp_path, adapter, dataset, policy):
    with pytest.raises(ValueError):
        adapter(str(tmp_path), str(_manifest(tmp_path, dataset)), num_history_steps=20, split_audit=policy)


@pytest.mark.parametrize("context", ["short", "long_history", "long_images"])
def test_dataset_audit_guard_covers_requested_context_with_an_absolute_floor(tmp_path, context):
    _tartan(tmp_path, duration=40.)
    arguments = dict(data_root=str(tmp_path), split_manifest=str(_manifest(tmp_path, "tartan")),
                     num_history_steps=20)
    if context == "short":
        arguments.update(num_history_steps=2, num_future_steps=1, frame_offsets=[-.1, 0.])
        span, minimum = .2, 8.4
    elif context == "long_history":
        arguments["num_history_steps"] = 101
        span = minimum = 16.4
    else:
        arguments["frame_offsets"] = [-10., 0.]
        span = minimum = 16.4
    with pytest.raises(ValueError):
        TartanDriveDataset(**arguments, split_audit={"min_time_gap_seconds": minimum - .000001})
    data = TartanDriveDataset(**arguments, split_audit={"min_time_gap_seconds": minimum})
    assert data.split_audit_report["requested_context_span_seconds"] == pytest.approx(span)
    assert data.split_audit_report["policy"]["min_time_gap_seconds"] == minimum
    assert len(data) > 0
