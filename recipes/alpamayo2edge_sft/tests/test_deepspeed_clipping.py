# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instance-local clipping precision and the actual Trainer installation boundary."""

import ast
from importlib import metadata
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from alpamayo2edge_sft import deepspeed_clipping as module


class FakeZero:
    def __init__(self, *, offload=False):
        self.dtype = torch.bfloat16
        self.gradient_accumulation_dtype = torch.float32
        self.loss_scale = 1.0
        self.clip_grad = 1.0
        self.offload_optimizer = offload
        self.fp32_partitioned_groups_flat = [torch.tensor([1.0, 2.0])]
        self.calls = []

    def unscale_and_clip_grads(self, sub_group_id, total_norm):
        self.calls.append((sub_group_id, total_norm))
        return "delegated"


def test_installation_idempotence_and_other_optimizer_unchanged():
    first, second = FakeZero(), FakeZero()
    original_class_method = FakeZero.unscale_and_clip_grads
    assert module._install_fp32_clip_input(first)
    installed = first.unscale_and_clip_grads
    assert not module._install_fp32_clip_input(first)
    assert first.unscale_and_clip_grads == installed
    assert not hasattr(first, module._MARKER)
    assert second.unscale_and_clip_grads.__func__ is original_class_method
    norm = torch.tensor(7.691895075583647, dtype=torch.float64)
    assert first.unscale_and_clip_grads(0, norm) == "delegated"
    assert len(first.calls) == 1 and first.calls[0][0] == 0
    assert first.calls[0][1] is norm
    assert norm.dtype == torch.float64


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_norm_device_and_offload_configuration_preserved(offload, dtype):
    optimizer = FakeZero(offload=offload)
    norm = torch.tensor(0.5, dtype=dtype, device="cpu")
    module._install_fp32_clip_input(optimizer)
    optimizer.unscale_and_clip_grads(0, norm)
    actual = optimizer.calls[0][1]
    assert actual.device == norm.device and actual.dtype == norm.dtype
    assert optimizer.offload_optimizer == offload
    assert actual is norm


def test_disabled_clipping_delegates_without_cast():
    optimizer = FakeZero()
    optimizer.clip_grad = 0.0
    module._install_fp32_clip_input(optimizer)
    norm = torch.tensor(4.0, dtype=torch.float64)
    optimizer.unscale_and_clip_grads(0, norm)
    assert optimizer.calls[0][1] is norm


def test_installed_nvtx_decorator_preserves_profiling_and_engine_norm():
    try:
        distribution = metadata.distribution("deepspeed")
    except metadata.PackageNotFoundError:
        pytest.skip("DeepSpeed is not installed")
    if distribution.version != module._SUPPORTED_DEEPSPEED_VERSION:
        pytest.skip("NVTX fixture requires the supported DeepSpeed version")
    path = Path(distribution.locate_file("deepspeed/utils/nvtx.py"))
    tree = ast.parse(path.read_text())
    decorator = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "instrument_w_nvtx")
    ranges = []
    namespace = {
        "enable_nvtx": True, "is_compiling": lambda: False,
        "get_accelerator": lambda: None,
        "_range_push": lambda accelerator, name: ranges.append("push"),
        "_range_pop": lambda accelerator: ranges.append("pop"),
    }
    code = ast.fix_missing_locations(ast.Module(body=[decorator], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)

    class ProfiledZero(FakeZero):
        unscale_and_clip_grads = namespace["instrument_w_nvtx"](FakeZero.unscale_and_clip_grads)

    optimizer = ProfiledZero()
    assert module._install_fp32_clip_input(optimizer)
    assert optimizer.unscale_and_clip_grads(0, torch.tensor(3., dtype=torch.float64)) == "delegated"
    assert ranges == ["push", "pop"]
    assert optimizer.calls[0][1].dtype == torch.float64
    assert not module._install_fp32_clip_input(optimizer)


def test_rejects_incompatible_positional_call(monkeypatch):
    class IncompatibleZero(FakeZero):
        def unscale_and_clip_grads(self, *, sub_group_id, total_norm):
            raise AssertionError("Must reject before calling the optimizer")

    optimizer = IncompatibleZero()
    monkeypatch.setattr(module, "zero_norm_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(module, "ZeroFp32GlobalNorm", lambda *args: lambda: torch.tensor(1.))
    optimizer._get_norm_groups = lambda: []
    with pytest.raises(RuntimeError, match="Unsupported DeepSpeed clipping signature"):
        module._install_fp32_clip_input(optimizer, object(), "fsdp_fp32")
    assert not hasattr(optimizer, module._MARKER)


def test_non_deepspeed_model_does_not_import_deepspeed():
    before = set(sys.modules)
    assert not module.configure_joint_deepspeed_clipping(torch.nn.Linear(2, 2))
    assert not any(name.startswith("deepspeed") for name in set(sys.modules) - before)


@pytest.mark.parametrize("attribute,value", [
    ("dtype", torch.float16),
    ("gradient_accumulation_dtype", torch.bfloat16),
    ("loss_scale", 2.0),
])
def test_rejects_unsupported_runtime_precision(attribute, value, monkeypatch):
    optimizer = FakeZero()
    setattr(optimizer, attribute, value)
    optimizer._get_norm_groups = lambda: []
    monkeypatch.setattr(module, "zero_norm_unsupported_reason", lambda *args: None)
    monkeypatch.setattr(module, "ZeroFp32GlobalNorm", lambda *args: lambda: torch.tensor(1.))
    with pytest.raises(ValueError):
        module._install_fp32_clip_input(optimizer, object(), "fsdp_fp32")
    assert not hasattr(optimizer, module._MARKER)


def test_delegate_retains_nonfinite_behavior():
    optimizer = FakeZero()
    module._install_fp32_clip_input(optimizer)
    optimizer.unscale_and_clip_grads(0, torch.tensor(float("nan"), dtype=torch.float64))
    assert torch.isnan(optimizer.calls[0][1])


def test_actual_original_method_formula_below_at_above_threshold():
    # Compile the installed method without initializing CUDA-dependent runtime
    # imports. An absent or different DeepSpeed installation is not this fixture.
    try:
        distribution = metadata.distribution("deepspeed")
    except metadata.PackageNotFoundError:
        pytest.skip("DeepSpeed is not installed")
    if distribution.version != module._SUPPORTED_DEEPSPEED_VERSION:
        pytest.skip("Formula fixture requires the supported DeepSpeed version")
    path = Path(distribution.locate_file("deepspeed/runtime/zero/stage3.py"))
    if not path.is_file():
        pytest.skip("Installed DeepSpeed source is unavailable")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "DeepSpeedZeroOptimizer_Stage3")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "unscale_and_clip_grads")
    method.decorator_list = []
    namespace = {"torch": torch}
    code = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)

    class ExactZero(FakeZero):
        unscale_and_clip_grads = namespace["unscale_and_clip_grads"]

    # Engine policy must call this exact installed method with its original
    # FP64 norm, including thresholds where a downcast can alter arithmetic.
    for threshold in (.25, 1.0, 3.5):
        for norm_value in (.3, 1., 7.691895075583647):
            original, adapted = ExactZero(), ExactZero()
            for optimizer in (original, adapted):
                optimizer.clip_grad = threshold
                optimizer.fp32_partitioned_groups_flat[0].grad = torch.tensor([1.5, -2.0])
            norm = torch.tensor(norm_value, dtype=torch.float64)
            original.unscale_and_clip_grads(0, norm)
            original_method = adapted.unscale_and_clip_grads
            module._install_fp32_clip_input(adapted, norm_algorithm="engine")
            assert adapted.unscale_and_clip_grads == original_method
            adapted.unscale_and_clip_grads(0, norm)
            assert torch.equal(original.fp32_partitioned_groups_flat[0].grad,
                               adapted.fp32_partitioned_groups_flat[0].grad)


def test_joint_stage3_engine_attaches_once_after_creation(monkeypatch):
    class FakeEngine:
        def __init__(self, stage="joint", zero_stage=3):
            self.module = SimpleNamespace(training_stage=stage)
            self.optimizer = FakeZero(offload=True)
            self.zero_stage = zero_stage

        def zero_optimization_stage(self):
            return self.zero_stage

    engine_module = ModuleType("deepspeed.runtime.engine")
    engine_module.DeepSpeedEngine = FakeEngine
    stage_module = ModuleType("deepspeed.runtime.zero.stage3")
    stage_module.DeepSpeedZeroOptimizer_Stage3 = FakeZero
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.engine", engine_module)
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.zero.stage3", stage_module)
    version_calls = []
    monkeypatch.setattr(module, "version", lambda name: version_calls.append(name) or "0.19.1")
    assert not module.configure_joint_deepspeed_clipping(FakeEngine("vlm"))
    stage2 = FakeEngine(zero_stage=2)
    assert module.configure_joint_deepspeed_clipping(stage2)
    assert getattr(stage2.optimizer, module._POLICY_MARKER)["resolved"] == "engine"
    assert not hasattr(stage2.optimizer, module._MARKER)
    assert not module.configure_joint_deepspeed_clipping(stage2)
    with pytest.raises(ValueError, match="ZeRO-3"):
        module.configure_joint_deepspeed_clipping(FakeEngine(zero_stage=2), "fsdp_fp32")
    engine = FakeEngine()
    assert module.configure_joint_deepspeed_clipping(engine)
    assert not module.configure_joint_deepspeed_clipping(engine)
    assert version_calls == ["deepspeed"]


def test_training_step_adapter_runs_before_backward_and_bypasses_non_ds():
    # Execute the actual recipe override with a minimal superclass to check
    # argument forwarding and installation before its backward entry point.
    path = Path(module.__file__).with_name("trainer.py")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "Alpamayo2EdgeTrainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "training_step")
    candidate = ast.ClassDef(name="Candidate", bases=[ast.Name(id="Base", ctx=ast.Load())],
                             keywords=[], body=[method], decorator_list=[])
    events = []

    class Base:
        def training_step(self, model, inputs, num_items_in_batch=None):
            events.append(("backward", model, inputs, num_items_in_batch))
            return "loss"

    namespace = {
        "Base": Base,
        "configure_joint_deepspeed_reduction": lambda model: events.append(("reduce", model)),
        "configure_joint_deepspeed_clipping": lambda model, **kwargs: events.append(("clip", model, kwargs)),
    }
    code = ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    trainer = namespace["Candidate"]()
    trainer.is_deepspeed_enabled = True
    trainer.args = SimpleNamespace(gradient_norm_algorithm="fsdp_fp32")
    engine = SimpleNamespace(optimizer=SimpleNamespace())
    assert trainer.training_step(engine, {"x": 1}, 64) == "loss"
    assert events == [
        ("reduce", engine), ("clip", engine, {"norm_algorithm": "fsdp_fp32"}), ("backward", engine, {"x": 1}, 64)
    ]
    events.clear()
    trainer.is_deepspeed_enabled = False
    trainer.training_step("plain", {}, None)
    assert events == [("backward", "plain", {}, None)]
