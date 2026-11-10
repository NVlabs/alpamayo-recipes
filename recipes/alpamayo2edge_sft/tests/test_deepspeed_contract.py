# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU engine/source contracts for the installed, pinned DeepSpeed ZeRO-3 implementation.

Real engine steps cover accumulation, norm consumption, clipping, and AdamW
updates. Source hashes additionally guard gradient ownership and padding. A
source change must be reviewed before updating these contracts. No CUDA required.
"""
import ast
from copy import deepcopy
from datetime import timedelta
from functools import partial
import hashlib
import importlib
from importlib import metadata
from pathlib import Path

import pytest
import torch

from alpamayo2edge_sft import deepspeed_clipping
from alpamayo2edge_sft.global_norm import ZeroFp32GlobalNorm

try:
    DISTRIBUTION = metadata.distribution("deepspeed")
except metadata.PackageNotFoundError:
    pytest.skip("DeepSpeed is not installed", allow_module_level=True)
if DISTRIBUTION.version != "0.19.1":
    pytest.skip("Contracts require DeepSpeed 0.19.1", allow_module_level=True)


def installed_method(filename, classname, name):
    source = Path(DISTRIBUTION.locate_file("deepspeed/runtime/zero/" + filename))
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    return source, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize("filename,classname,name,digest", [
    ("stage3.py", "DeepSpeedZeroOptimizer_Stage3", "step", "bbbd705ae47c4fc044d007891a17ace3d63b3b5a31e8c22c793c76c315882225"),
    ("stage3.py", "DeepSpeedZeroOptimizer_Stage3", "_get_norm_groups", "8268ce6a923c8a94d2474b06c544508be81683c7221d0c84b6b66a95cdf01974"),
    ("stage3.py", "DeepSpeedZeroOptimizer_Stage3", "partition_grads", "9cfbad53e3438aa871757a5ce63a6e050b8245abb71db8b954685e8683f37119"),
    ("stage3.py", "DeepSpeedZeroOptimizer_Stage3", "unscale_and_clip_grads", "5ff52558f0cc958018570d4281b78fbd49980b9e6cde40eeae49487157b30664"),
    ("partition_parameters.py", "Init", "_partition_param", "243f0f30c6e36bf8818b687e9938d374480d3f2d5e45820ad864d499bbbf8bbf"),
])
def test_installed_zero_gradient_ownership_and_step_contract(filename, classname, name, digest):
    _, method = installed_method(filename, classname, name)
    actual = hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()
    assert actual == digest, "Review the changed DeepSpeed norm/partition/step contract"


@pytest.fixture
def cpu_zero_engine(monkeypatch, tmp_path):
    """Initialize the real CPU/Gloo engine without building optional C++ ops.

    Only the shared-memory communication accelerator is disabled: collectives,
    parameter/gradient ownership, backward, and the complete ZeRO step stay real.
    This is a single-rank lifecycle test, not GPU/distributed numerical coverage.
    """
    import deepspeed
    from deepspeed.accelerator import get_accelerator, set_accelerator
    from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
    from deepspeed.utils import groups

    owns_group = not torch.distributed.is_initialized()
    if not owns_group:
        assert torch.distributed.get_world_size() == 1
        assert torch.distributed.get_backend() == "gloo", "CPU fixture requires a single-rank Gloo group"
    previous_accelerator = get_accelerator()
    set_accelerator(CPU_Accelerator())
    backend = importlib.import_module("deepspeed.comm.torch")
    communication = importlib.import_module("deepspeed.comm.comm")
    monkeypatch.setattr(backend, "build_shm_op", lambda: None)
    # DeepSpeed retains these after torch destroys the underlying process group.
    # Restore their previous values at fixture teardown, including when a caller
    # supplied an existing compatible process group.
    monkeypatch.setattr(communication, "cdb", None)
    monkeypatch.setattr(groups, "_WORLD_GROUP", None)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    engines = []
    try:
        if owns_group:
            torch.distributed.init_process_group(
                "gloo", init_method=(tmp_path / "gloo-rendezvous").as_uri(),
                rank=0, world_size=1, timeout=timedelta(seconds=30),
            )

        def create(threshold):
            model = torch.nn.Linear(4, 2).to(torch.bfloat16)
            with torch.no_grad():
                model.weight.copy_(torch.tensor([
                    [.25, -.5, .125, .75], [-.25, .5, -.75, .125],
                ]))
                model.bias.copy_(torch.tensor([.125, -.25]))
            model.training_stage = "joint"
            reference = deepcopy(model)
            # Two subgroups exercise the real step's global-norm consumption
            # across separate FP32 master gradients and optimizer updates.
            optimizer = torch.optim.AdamW(
                [{"params": [parameter]} for parameter in model.parameters()],
                lr=.03, betas=(.8, .9), eps=1e-4, weight_decay=.01, foreach=False,
            )
            config = {
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 2,
                "bf16": {"enabled": True},
                "data_types": {"grad_accum_dtype": "fp32"},
                "communication_data_type": "fp32",
                "gradient_clipping": threshold,
                "zero_allow_untested_optimizer": True,
                "zero_optimization": {
                    "stage": 3, "overlap_comm": False, "reduce_scatter": True,
                    "contiguous_gradients": True, "stage3_param_persistence_threshold": 0,
                },
                "steps_per_print": 1_000_000,
            }
            engine, _, _, _ = deepspeed.initialize(
                model=model, optimizer=optimizer, config=config, dist_init_required=False,
            )
            engines.append(engine)
            return engine, reference

        yield create
    finally:
        for engine in engines:
            engine.destroy()
        if torch.distributed.is_initialized():
            if owns_group:
                torch.distributed.destroy_process_group()
            elif groups._WORLD_GROUP is not None:
                torch.distributed.destroy_process_group(groups._WORLD_GROUP)
        set_accelerator(previous_accelerator)


@pytest.mark.parametrize("policy,threshold", [
    ("engine", 1.), ("auto", 3.5), ("fsdp_fp32", 1.), ("fsdp_fp32", .25),
])
def test_initialized_zero_step_matches_torch_adamw(cpu_zero_engine, monkeypatch, policy, threshold):
    """Execute installed ZeRO-3.step via engine.step, including its real lifecycle.

    CPU testing opts into the actual DTensor norm calculator with a CPU mesh;
    it does not replace the calculator, norm consumer, clipping, or step body.
    Torch supplies an independent BF16-backward/FP32-accumulation reference.
    """
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

    engine, reference = cpu_zero_engine(threshold)
    zero = engine.optimizer
    assert isinstance(zero, DeepSpeedZeroOptimizer_Stage3)
    assert zero.step.__func__ is DeepSpeedZeroOptimizer_Stage3.step
    assert zero.dtype == torch.bfloat16
    assert zero.gradient_accumulation_dtype == torch.float32
    assert zero.loss_scale == 1.
    assert not zero.offload_optimizer and not zero.swap_optimizer
    assert len(zero.fp32_partitioned_groups_flat) == 2
    assert all(master.dtype == torch.float32 for master in zero.fp32_partitioned_groups_flat)

    original_norm, original_clip = zero._get_norm_groups, zero.unscale_and_clip_grads
    if policy == "fsdp_fp32":
        supported_reason = deepspeed_clipping.zero_norm_unsupported_reason

        def allow_cpu_mesh(model, optimizer):
            reason = supported_reason(model, optimizer)
            assert reason == "the matched ZeRO norm requires GPU-resident optimizer gradients"
            return None

        monkeypatch.setattr(deepspeed_clipping, "zero_norm_unsupported_reason", allow_cpu_mesh)
        monkeypatch.setattr(deepspeed_clipping, "ZeroFp32GlobalNorm",
                            partial(ZeroFp32GlobalNorm, device_type="cpu"))
    assert deepspeed_clipping.configure_joint_deepspeed_clipping(engine, policy)
    assert not deepspeed_clipping.configure_joint_deepspeed_clipping(engine, policy)
    installed_policy = getattr(zero, deepspeed_clipping._POLICY_MARKER)
    assert installed_policy["requested"] == policy
    assert installed_policy["resolved"] == ("fsdp_fp32" if policy == "fsdp_fp32" else "engine")
    if policy == "auto":
        assert installed_policy["fallback_reason"] == "the matched ZeRO norm requires GPU-resident optimizer gradients"
    else:
        assert installed_policy["fallback_reason"] is None
    if policy != "fsdp_fp32":
        assert zero._get_norm_groups == original_norm
        assert zero.unscale_and_clip_grads == original_clip

    reference_parameters = list(reference.parameters())
    masters = [torch.nn.Parameter(parameter.detach().float().clone()) for parameter in reference_parameters]
    reference_optimizer = torch.optim.AdamW(
        [{"params": [master]} for master in masters],
        lr=.03, betas=(.8, .9), eps=1e-4, weight_decay=.01, foreach=False,
    )
    observed_gradients = []

    def observe_optimizer_input(optimizer, args, kwargs):
        # Observe the actual post-clipping FP32 gradients without modifying the
        # Adam implementation or any ZeRO step/norm/clip method.
        observed_gradients.extend(
            parameter.grad.detach().clone()
            for group in optimizer.param_groups for parameter in group["params"]
        )

    hook = zero.optimizer.register_step_pre_hook(observe_optimizer_input)
    tolerance = {"rtol": 0, "atol": 0} if policy == "fsdp_fp32" else {"rtol": 1e-6, "atol": 1e-8}
    try:
        for step, scale in enumerate((.001, 4.)):
            observed_gradients.clear()
            reference_optimizer.zero_grad(set_to_none=True)
            for parameter, master in zip(reference_parameters, masters, strict=True):
                parameter.data.copy_(master.detach().to(torch.bfloat16))
            for microbatch, values in enumerate(([1., -.75, .5, 2.], [-.5, 1.25, 2., -.25])):
                inputs = torch.tensor([values], dtype=torch.bfloat16)
                reference.zero_grad(set_to_none=True)
                expected_loss = reference(inputs).float().square().mean() * scale
                actual_loss = engine(inputs).float().square().mean() * scale
                torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
                (expected_loss / 2).backward()
                for parameter, master in zip(reference_parameters, masters, strict=True):
                    gradient = parameter.grad.float()
                    if master.grad is None:
                        master.grad = gradient.clone()
                    else:
                        master.grad.add_(gradient)
                engine.backward(actual_loss)
                if microbatch == 0:
                    engine.step()
                    assert not observed_gradients  # accumulation must not update weights

            expected_norm = torch.nn.utils.clip_grad_norm_(masters, threshold, foreach=True)
            if step == 0:
                assert expected_norm < threshold
            else:
                assert expected_norm > threshold
            expected_gradients = [master.grad.detach().clone() for master in masters]
            reference_optimizer.step()
            engine.step()
            assert len(observed_gradients) == len(masters)
            torch.testing.assert_close(zero._global_grad_norm.float(), expected_norm, **tolerance)
            for index, (actual, expected) in enumerate(zip(
                zero.fp32_partitioned_groups_flat, masters, strict=True,
            )):
                torch.testing.assert_close(observed_gradients[index], expected_gradients[index].flatten(), **tolerance)
                torch.testing.assert_close(actual, expected.detach().flatten(), **tolerance)
                actual_state, expected_state = zero.optimizer.state[actual], reference_optimizer.state[expected]
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(actual_state[name].flatten(), expected_state[name].flatten(), **tolerance)
    finally:
        hook.remove()
