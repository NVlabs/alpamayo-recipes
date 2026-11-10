# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from alpamayo2edge_sft import global_norm
from alpamayo2edge_sft.models import qk_gradient_layout


class Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,owner):
        ctx.save_for_backward(x,weight)
        ctx.owner=owner
        return x*weight
    @staticmethod
    def backward(ctx,gradient):
        x,weight=ctx.saved_tensors
        ctx.owner.seen.append(gradient)
        return gradient*weight,(gradient*x).sum(tuple(range(x.ndim-1))),None


class Norm(torch.nn.Module):
    def __init__(self,dim=4):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.ones(dim))
        self.seen=[]
    def forward(self,x):
        return Multiply.apply(x,self.weight,self)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training_stage="joint"
        self.q_norm=Norm()
        self.k_norm=Norm()
    def forward(self,x):
        return self.q_norm(x)+self.k_norm(x)


@pytest.fixture(autouse=True)
def attention_type(monkeypatch):
    monkeypatch.setattr(qk_gradient_layout,"_attention_types",lambda:(Attention,))


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_qk_layout_changes_only_backward_storage_and_preserves_state(dtype):
    model=Attention().to(dtype=dtype)
    identities={name:id(value) for name,value in model.named_parameters()}
    saved={name:value.clone() for name,value in model.state_dict().items()}
    x=torch.randn(2,5,3,4,dtype=dtype,requires_grad=True)
    upstream=torch.randn(2,3,5,4,dtype=dtype).transpose(1,2)
    assert not upstream.is_contiguous()
    outputs=[]
    handle=model.q_norm.register_forward_hook(lambda m,a,o:outputs.append(o))
    outside=model.q_norm(x)
    assert qk_gradient_layout.configure_joint_qk_gradient_layout(model)
    inside=model.q_norm(x)
    assert outputs[-1] is inside
    assert torch.equal(inside,outside)
    got=torch.autograd.grad(inside,(x,model.q_norm.weight),upstream)
    assert model.q_norm.seen[-1].is_contiguous()
    expected=torch.autograd.grad(outside,(x,model.q_norm.weight),upstream.contiguous())
    assert all(torch.equal(a,b) for a,b in zip(got,expected))
    model.eval()
    untouched=model.q_norm(x)
    torch.autograd.grad(untouched,(x,model.q_norm.weight),upstream)
    assert not model.q_norm.seen[-1].is_contiguous()
    assert {name:id(value) for name,value in model.named_parameters()}==identities
    assert all(torch.equal(value,saved[name]) for name,value in model.state_dict().items())
    handle.remove()


def test_idempotence_and_instances():
    first,second=Attention(),Attention()
    assert qk_gradient_layout.configure_joint_qk_gradient_layout(first)
    assert not qk_gradient_layout.configure_joint_qk_gradient_layout(first)
    assert not hasattr(second,"_joint_sft_qk_layout")
    assert len(first.q_norm._forward_hooks)==1


def test_no_grad_and_nonattention_unchanged():
    model=torch.nn.Module()
    model.training_stage="joint"
    model.attention=Attention()
    model.unrelated=Norm()
    qk_gradient_layout.configure_joint_qk_gradient_layout(model)
    with torch.no_grad():
        value=model.attention.q_norm(torch.ones(2,3,4))
    assert not value.requires_grad
    assert not model.unrelated._forward_hooks


@pytest.mark.parametrize("use_reentrant",[True,False])
def test_checkpoint_recomputation_receives_contiguous_qk_gradient(use_reentrant):
    from torch.utils.checkpoint import checkpoint
    model=Attention()
    qk_gradient_layout.configure_joint_qk_gradient_layout(model)
    value=torch.randn(2,5,3,4,requires_grad=True)
    upstream=torch.randn(2,3,5,4).transpose(1,2)
    actual=checkpoint(model.q_norm,value,use_reentrant=use_reentrant)
    actual.backward(upstream)
    assert model.q_norm.seen and all(g.is_contiguous() for g in model.q_norm.seen)
    reference=Norm()
    source=value.detach().clone().requires_grad_()
    expected=reference(source)
    expected.backward(upstream.contiguous())
    assert torch.equal(actual,expected)
    assert torch.equal(value.grad,source.grad)
    assert torch.equal(model.q_norm.weight.grad,reference.weight.grad)


@pytest.mark.parametrize("stage",["vlm","expert"])
def test_legacy_training_stages_inert(stage):
    model=Attention()
    qk_gradient_layout.configure_joint_qk_gradient_layout(model)
    model.training_stage=stage
    value=torch.randn(2,5,3,4,requires_grad=True)
    upstream=torch.randn(2,3,5,4).transpose(1,2)
    model.q_norm(value).backward(upstream)
    assert not model.q_norm.seen[-1].is_contiguous()


@pytest.mark.parametrize("threshold",[0.,.25,1.,3.5])
def test_explicit_coefficient_matches_torch_for_thresholds(threshold):
    parameters=[torch.nn.Parameter(torch.zeros(17)),torch.nn.Parameter(torch.zeros(3,5))]
    grads=[torch.randn_like(p) for p in parameters]
    for p,g in zip(parameters,grads):p.grad=g.clone()
    norm=global_norm.fp32_global_norm(grads)
    if threshold>0:
        expected_norm=torch.nn.utils.clip_grad_norm_(parameters,threshold,foreach=True)
        assert torch.equal(norm,expected_norm)
    else:
        expected_norm=norm
    global_norm.clip_fp32_with_norm_(grads,threshold,norm)
    assert all(torch.equal(g,p.grad) for g,p in zip(grads,parameters))


def _worker(rank,world,store,output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+store,rank=rank,world_size=world)
    try:
        model=torch.nn.Module()
        shapes=[(6,3),(1,4),(3,),(3,2)]
        originals=[]
        for index,shape in enumerate(shapes):
            p=torch.nn.Parameter(torch.empty(0))
            p.ds_shape=shape;p.ds_numel=math.prod(shape)
            model.register_parameter(f"arbitrary_{index}",p)
            originals.append((torch.arange(p.ds_numel,dtype=torch.float32)+.12345*(index+1)).reshape(shape))
        parameters=list(model.parameters())
        # Deliberately reverse order and split optimizer groups differently from
        # registration order. Norm must still follow the model's traversal.
        groups=[[parameters[2],parameters[0]],[parameters[3],parameters[1]]]
        pieces={}
        for p,full in zip(parameters,originals):
            width=(p.ds_numel+world-1)//world
            start=rank*width;valid=max(0,min(width,p.ds_numel-start))
            part=torch.full((width,),float('nan'))
            part[:valid]=full.reshape(-1)[start:start+valid]
            pieces[id(p)]=part
        zero=SimpleNamespace(offload_optimizer=False,swap_optimizer=False,model_parallel_group=None,
            dp_process_group=dist.group.WORLD,fp16_groups=groups,
            averaged_gradients=[[pieces[id(p)] for p in group] for group in groups])
        helper=global_norm.ZeroFp32GlobalNorm(model,zero,device_type="cpu")
        before={key:value.view(torch.uint8).clone() for key,value in pieces.items()}
        result=helper()
        reference=[]
        for full in originals:
            # The independent PyTorch sharding implementation is the reference;
            # do not construct expected slices with the helper under test.
            reference.append(distribute_tensor(full.clone(), helper.mesh, [Shard(0)]))
        expected=torch.nn.utils.get_total_norm(reference,2.,False,True).full_tensor()
        assert torch.equal(result,expected)
        fsdp_gradients=[value.clone() for value in reference]
        coefficient=torch.clamp(.25/(result+1e-6),max=1.)
        global_norm.clip_fp32_with_norm_(fsdp_gradients,.25,result)
        assert all(torch.equal(value.to_local(),before.to_local()*coefficient)
                   for value,before in zip(fsdp_gradients,reference))
        assert all(torch.equal(value.view(torch.uint8),before[key]) for key,value in pieces.items())
        assert list(helper.parameters)==parameters
        parameters[0].ds_shape=(3,6)
        with pytest.raises(RuntimeError,match="metadata changed"):
            helper()
        parameters[0].ds_shape=shapes[0]
        Path(output+f'.{rank}').write_text(json.dumps({'norm':float(result),'exact':True}))
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('world',[1,2])
def test_arbitrary_model_order_shapes_groups_and_padding(tmp_path,world):
    mp.start_processes(_worker,args=(world,str(tmp_path/'store'),str(tmp_path/'out')),nprocs=world,
        join=True,start_method='fork')
    values=[json.loads((tmp_path/f'out.{rank}').read_text()) for rank in range(world)]
    assert all(v['norm']==values[0]['norm'] for v in values)



def test_fp32_foreach_norm_matches_torch_without_fixed_kernel_rounding():
    generator=torch.Generator().manual_seed(0)
    gradients=[torch.randn(n,generator=generator) for n in (17,31,63,129,257)]
    actual=global_norm.fp32_global_norm(gradients)
    expected=torch.nn.utils.get_total_norm(gradients,2.,False,True)
    assert actual.dtype == torch.float32
    assert torch.equal(actual,expected)


def test_layout_option_can_disable_and_reenable_hooks():
    model=Attention()
    qk_gradient_layout.configure_joint_qk_gradient_layout(model, enabled=False)
    assert not model.q_norm._forward_hooks
    qk_gradient_layout.configure_joint_qk_gradient_layout(model)
    assert model.q_norm._forward_hooks
    qk_gradient_layout.configure_joint_qk_gradient_layout(model, enabled=False)
    assert not model.q_norm._forward_hooks and not model.k_norm._forward_hooks


def test_real_supported_qwen_attention_classes_are_hooked(monkeypatch, caplog):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
    monkeypatch.undo()
    assert qk_gradient_layout._attention_types() == (Qwen3Attention, Qwen3VLTextAttention)
    model=torch.nn.Module()
    common=dict(hidden_size=32,intermediate_size=64,num_attention_heads=4,
                num_key_value_heads=2,num_hidden_layers=1,head_dim=8)
    model.first=Qwen3Attention(Qwen3Config(**common),layer_idx=0)
    model.second=Qwen3VLTextAttention(Qwen3VLTextConfig(**common),layer_idx=0)
    with caplog.at_level("INFO"):
        qk_gradient_layout.configure_joint_qk_gradient_layout(model)
    assert len(model._joint_sft_qk_layout.handles)==4
    assert "normalization_modules=4" in caplog.text
