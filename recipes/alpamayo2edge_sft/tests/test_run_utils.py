# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check real tracking boundaries without contacting W&B or loading a model."""

import json
from types import SimpleNamespace

import pytest
from alpamayo2edge_sft import run_utils
from omegaconf import OmegaConf


@pytest.fixture
def tracking_backend(monkeypatch):
    calls = {"init": [], "login": [], "finish": []}

    def init(**kwargs):
        calls["init"].append(kwargs)
        return SimpleNamespace(
            id=kwargs["id"],
            project=kwargs["project"],
            entity=kwargs.get("entity", "test-team"),
            settings=SimpleNamespace(mode="online"),
            url=f"https://wandb.ai/test-team/{kwargs['project']}/runs/{kwargs['id']}",
            finish=lambda **options: calls["finish"].append(options),
        )

    monkeypatch.setattr(
        run_utils,
        "wandb",
        SimpleNamespace(
            init=init,
            login=lambda **kwargs: calls["login"].append(kwargs),
            util=SimpleNamespace(generate_id=lambda: "generated-id"),
        ),
    )
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "sft-test")
    return calls


def trainer(output_dir, *, report_to=None, main=True, resume=None):
    return SimpleNamespace(
        args=SimpleNamespace(
            output_dir=str(output_dir),
            report_to=["wandb"] if report_to is None else report_to,
            run_name="stage1",
            resume_from_checkpoint=resume,
        ),
        is_world_process_zero=lambda: main,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_tracking_persists_link_and_finishes_success_or_failure(tmp_path, tracking_backend, fail):
    config = OmegaConf.create({"seed": 42})

    def run():
        with run_utils.tracking_run(config, trainer(tmp_path)):
            metadata = json.loads((tmp_path / "wandb_run.json").read_text())
            assert metadata["url"].endswith("/sft-test/runs/generated-id")
            assert (tmp_path / ".wandb_id").read_text().strip() == metadata["id"]
            if fail:
                raise RuntimeError("training failed")

    if fail:
        with pytest.raises(RuntimeError, match="training failed"):
            run()
    else:
        run()
    assert tracking_backend["finish"] == [{"exit_code": int(fail)}]
    assert tracking_backend["init"][0]["name"] == "stage1"
    assert tracking_backend["login"] == []


@pytest.mark.parametrize("report_to,main", [([], True), (["wandb"], False)])
def test_disabled_tracking_and_other_ranks_do_not_start_run(
    tmp_path, tracking_backend, report_to, main
):
    config = OmegaConf.create({"wandb": {"project": "legacy-project"}})
    with run_utils.tracking_run(config, trainer(tmp_path, report_to=report_to, main=main)):
        pass
    assert tracking_backend["init"] == []
    assert tracking_backend["finish"] == []
    assert not (tmp_path / ".wandb_id").exists()


def test_tracking_resumes_saved_id_and_honors_legacy_settings(tmp_path, tracking_backend):
    (tmp_path / ".wandb_id").write_text("prior-id\n")
    config = OmegaConf.create(
        {"wandb": {"team": "legacy-team", "project": "legacy-project", "group": "sft"}}
    )
    with run_utils.tracking_run(config, trainer(tmp_path, resume="checkpoint-500")):
        pass
    options = tracking_backend["init"][0]
    assert options["id"] == "prior-id"
    assert options["resume"] == "allow"
    assert options["entity"] == "legacy-team"
    assert options["project"] == "legacy-project"
    assert options["group"] == "sft"


def test_persisted_config_redacts_key_before_resolving_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("EDGE_TEST_UNSET_KEY", raising=False)
    config = OmegaConf.create(
        {"seed": 123, "wandb": {"key": "${oc.env:EDGE_TEST_UNSET_KEY}", "project": "sft"}}
    )
    run_utils.save_resolved_config(config, tmp_path / "config.yaml")
    saved = OmegaConf.load(tmp_path / "config.yaml")
    assert saved.wandb.key == "***"
    assert saved.wandb.project == "sft"
    assert saved.seed == 123
    # Saving must not mutate the authentication config used by the current run.
    assert OmegaConf.to_container(config, resolve=False)["wandb"]["key"].startswith("${oc.env:")


def test_tracking_does_not_resume_merely_because_output_directory_is_reused(tmp_path, tracking_backend):
    (tmp_path / ".wandb_id").write_text("old-id\n")
    with run_utils.tracking_run(OmegaConf.create({}), trainer(tmp_path)):
        pass
    options = tracking_backend["init"][0]
    assert options["id"] == "generated-id"
    assert options["resume"] == "never"


def test_explicit_environment_id_does_not_implicitly_resume(tmp_path, tracking_backend, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "requested-id")
    with run_utils.tracking_run(OmegaConf.create({}), trainer(tmp_path)):
        pass
    assert tracking_backend["init"][0]["id"] == "requested-id"
    assert tracking_backend["init"][0]["resume"] == "never"
