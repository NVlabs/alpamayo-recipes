# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported joint clipping boundaries and explicit unsupported-mode behavior."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from alpamayo2edge_sft import deepspeed_clipping as module
from alpamayo2edge_sft.global_norm import validate_norm_algorithm


class Zero:
    def __init__(self, offload=False):
        self.dtype=torch.bfloat16
        self.gradient_accumulation_dtype=torch.float32
        self.loss_scale=1.
        self.clip_grad=1.
        self.offload_optimizer=offload
        self.swap_optimizer=False
        self.model_parallel_group=None
        self.fp32_partitioned_groups_flat=[torch.nn.Parameter(torch.zeros(5))]
        self.fp32_partitioned_groups_flat[0].grad=torch.tensor([2.,-3.,4.,.25,1e-4])
        self.norm_calls=0
        self.clip_calls=[]
    def _get_norm_groups(self):
        self.norm_calls+=1
        return [torch.tensor(7.,dtype=torch.float64)]
    def unscale_and_clip_grads(self,index,norm):
        self.clip_calls.append((index,norm))
        combined=self.loss_scale
        if self.clip_grad>0:
            clip=((norm/self.loss_scale)+1e-6)/self.clip_grad
            clip=torch.clamp(clip,min=1.)
            combined=clip*self.loss_scale
        self.fp32_partitioned_groups_flat[index].grad.mul_(1./combined)


def enable_cpu_fixture(monkeypatch,norm):
    monkeypatch.setattr(module,"zero_norm_unsupported_reason",lambda model,zero:None)
    monkeypatch.setattr(module,"ZeroFp32GlobalNorm",lambda model,zero:lambda:norm)


@pytest.mark.parametrize("threshold",[.25,1.,3.5,100.])
def test_matched_norm_reaches_actual_clipping_and_matches_torch(monkeypatch,threshold):
    zero=Zero()
    zero.clip_grad=threshold
    original=zero.fp32_partitioned_groups_flat[0].grad.clone()
    norm=torch.linalg.vector_norm(original)
    enable_cpu_fixture(monkeypatch,norm)
    assert module._install_fp32_clip_input(zero,object(),"fsdp_fp32")
    groups=zero._get_norm_groups()
    actual_norm=torch.linalg.vector_norm(torch.stack(groups))
    assert torch.equal(actual_norm,norm) and zero.norm_calls==0
    zero.unscale_and_clip_grads(0,actual_norm)
    expected=torch.nn.Parameter(torch.zeros_like(original));expected.grad=original.clone()
    torch.nn.utils.clip_grad_norm_([expected],threshold,foreach=True)
    assert torch.equal(zero.fp32_partitioned_groups_flat[0].grad,expected.grad)
    assert bool(zero.clip_calls)==(threshold==1.)
    if threshold==1.:
        assert zero.clip_calls[0][1].dtype==torch.float32


def test_auto_offload_and_cpu_fallback_preserve_engine_behavior(caplog):
    for offload in (True,False):
        zero=Zero(offload)
        expected=Zero(offload)
        model=torch.nn.Linear(2,2)
        assert module._install_fp32_clip_input(zero,model,"auto")
        assert getattr(zero,module._POLICY_MARKER)["resolved"]=="engine"
        assert not hasattr(zero,module._NORM_MARKER)
        norm=torch.tensor(7.1234567,dtype=torch.float64)
        zero.unscale_and_clip_grads(0,norm)
        expected.unscale_and_clip_grads(0,norm)
        assert torch.equal(zero.fp32_partitioned_groups_flat[0].grad,expected.fp32_partitioned_groups_flat[0].grad)
        before=len(caplog.records)
        assert not module._install_fp32_clip_input(zero,model,"auto")
        assert len(caplog.records)==before
    assert "offload" in caplog.text
    assert "Joint gradient norm uses the engine algorithm" in caplog.text


@pytest.mark.parametrize("offload",[False,True])
def test_strict_unsupported_rejects_before_mutation(offload):
    zero=Zero(offload)
    original_clip,original_norm=zero.unscale_and_clip_grads,zero._get_norm_groups
    with pytest.raises(ValueError,match="unsupported"):
        module._install_fp32_clip_input(zero,torch.nn.Linear(2,2),"fsdp_fp32")
    assert zero.unscale_and_clip_grads==original_clip and zero._get_norm_groups==original_norm
    assert not hasattr(zero,module._MARKER)


def test_disabled_clipping_preserves_original_norm_and_application(monkeypatch):
    zero=Zero();zero.clip_grad=0.
    before=zero.fp32_partitioned_groups_flat[0].grad.clone()
    enable_cpu_fixture(monkeypatch,torch.tensor(123.))
    module._install_fp32_clip_input(zero,object(),"fsdp_fp32")
    groups=zero._get_norm_groups()
    assert zero.norm_calls==1 and groups[0].dtype==torch.float64
    norm=torch.linalg.vector_norm(torch.stack(groups))
    zero.unscale_and_clip_grads(0,norm)
    assert zero.clip_calls[0][1] is norm
    assert torch.equal(before,zero.fp32_partitioned_groups_flat[0].grad)


@pytest.mark.parametrize("value",[float("nan"),float("inf")])
def test_nonfinite_matched_norm_preserves_engine_sentinel_path(monkeypatch,value):
    zero,expected=Zero(),Zero()
    enable_cpu_fixture(monkeypatch,torch.tensor(value))
    module._install_fp32_clip_input(zero,object(),"fsdp_fp32")
    groups=zero._get_norm_groups()
    assert zero.norm_calls==1
    norm=torch.linalg.vector_norm(torch.stack(groups))
    zero.unscale_and_clip_grads(0,norm)
    expected.unscale_and_clip_grads(0,norm)
    assert torch.equal(zero.fp32_partitioned_groups_flat[0].grad,expected.fp32_partitioned_groups_flat[0].grad)


def test_policy_and_installed_norm_replacement_rejected(monkeypatch):
    zero=Zero();enable_cpu_fixture(monkeypatch,torch.tensor(3.))
    module._install_fp32_clip_input(zero,object(),"fsdp_fp32")
    with pytest.raises(RuntimeError,match="policy changed"):
        module._install_fp32_clip_input(zero,object(),"engine")
    zero._get_norm_groups=lambda:[]
    with pytest.raises(RuntimeError,match="norm adapter was replaced"):
        module._install_fp32_clip_input(zero,object(),"fsdp_fp32")


def test_unknown_policy_rejected():
    with pytest.raises(ValueError,match="gradient_norm_algorithm"):
        validate_norm_algorithm("approximate")


def trainer_class(events,reason=None):
    path=Path(module.__file__).with_name("trainer.py")
    cls=next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.ClassDef) and node.name=="Alpamayo2EdgeTrainer")
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="_clip_grad_norm")
    candidate=ast.ClassDef(name="Candidate",bases=[ast.Name(id="Base",ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    class Base:
        def _clip_grad_norm(self,model):
            events.append("engine")
            return "engine_norm"
    from alpamayo2edge_sft.global_norm import fp32_global_norm,clip_fp32_with_norm_
    namespace={"Base":Base,"validate_norm_algorithm":validate_norm_algorithm,
        "fsdp_norm_unsupported_reason":lambda values:reason,
        "fp32_global_norm":fp32_global_norm,"clip_fp32_with_norm_":clip_fp32_with_norm_,
        "logger":SimpleNamespace(warning=lambda *args:events.append("warning"))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate],type_ignores=[])),str(path),"exec"),namespace)
    namespace["Candidate"]._record_gradient_norm_policy = lambda self, policy: setattr(self.args, "gradient_norm_policy", policy)
    return namespace["Candidate"]


def shell(events,policy="auto",reason=None,fsdp=True,deepspeed=False):
    candidate=trainer_class(events,reason)()
    candidate.args=SimpleNamespace(gradient_norm_algorithm=policy,max_grad_norm=.25)
    candidate.is_fsdp_enabled=fsdp;candidate.is_deepspeed_enabled=deepspeed
    candidate.accelerator=SimpleNamespace(state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)),
        unscale_gradients=lambda:events.append("unscale"))
    model=torch.nn.Linear(3,2);model.training_stage="joint"
    for p in model.parameters():p.grad=torch.randn_like(p)
    return candidate,model


def test_actual_trainer_fsdp2_override_matches_explicit_foreach_path():
    events=[];trainer,model=shell(events)
    expected=[p.grad.clone() for p in model.parameters()]
    norm=torch.nn.utils.get_total_norm(expected,2.,False,True)
    coefficient=torch.clamp(.25/(norm+1e-6),max=1.)
    actual=trainer._clip_grad_norm(model)
    assert torch.equal(actual,norm) and events==["unscale"]
    assert all(torch.equal(p.grad,g*coefficient) for p,g in zip(model.parameters(),expected))


@pytest.mark.parametrize("policy",["auto","fsdp_fp32"])
def test_actual_trainer_unsupported_modes(policy):
    events=[];trainer,model=shell(events,policy,reason="mixed mesh")
    before=[p.grad.clone() for p in model.parameters()]
    if policy=="fsdp_fp32":
        with pytest.raises(ValueError,match="mixed mesh"):trainer._clip_grad_norm(model)
        assert events==[]
    else:
        assert trainer._clip_grad_norm(model)=="engine_norm"
        assert trainer._clip_grad_norm(model)=="engine_norm"
        assert events==["warning","engine","engine"]
    assert all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),before))


@pytest.mark.parametrize("setting",["engine","deepspeed","legacy"])
def test_actual_trainer_other_paths_delegate(setting):
    events=[];trainer,model=shell(events)
    if setting=="engine":trainer.args.gradient_norm_algorithm="engine"
    elif setting=="deepspeed":trainer.is_deepspeed_enabled=True
    else:model.training_stage="expert"
    assert trainer._clip_grad_norm(model)=="engine_norm" and events==["engine"]


def test_runtime_resolution_is_persisted_and_recorded_once(tmp_path):
    import json
    from alpamayo2edge_sft.trainer import Alpamayo2EdgeTrainer
    holder=SimpleNamespace(args=SimpleNamespace(output_dir=str(tmp_path)),
                           is_world_process_zero=lambda:True)
    policy=dict(requested="auto",resolved="engine",fallback_reason="optimizer offload")
    Alpamayo2EdgeTrainer._record_gradient_norm_policy(holder,policy)
    destination=tmp_path/"runtime_policy.json"
    assert json.loads(destination.read_text())=={"gradient_norm":policy}
    assert holder.args.gradient_norm_policy==policy
    stamp=destination.stat().st_mtime_ns
    Alpamayo2EdgeTrainer._record_gradient_norm_policy(holder,policy)
    assert destination.stat().st_mtime_ns==stamp
