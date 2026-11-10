# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real HF expert mask construction, visibility and temporary config scope."""

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import Cosmos3EdgeTextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeTextModel
from torch.nn.attention import sdpa_kernel, SDPBackend

from alpamayo2edge_sft.models.expert_masking import noncausal_expert_mask


def model_and_data(dtype=torch.float32, implementation="sdpa"):
    torch.manual_seed(817)
    config = Cosmos3EdgeTextConfig(vocab_size=128, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        attention_dropout=0., rope_parameters={"rope_type":"default", "rope_theta":10000.,
        "mrope_section":[2,3,3]})
    config._attn_implementation = implementation
    model = Cosmos3EdgeTextModel(config).to(dtype)
    prefix = [(torch.randn(2,2,4,16,dtype=dtype), torch.randn(2,2,4,16,dtype=dtype)) for _ in range(2)]
    inputs = torch.randn(2,3,32,dtype=dtype)
    positions = torch.arange(4,7).view(1,1,3).expand(3,2,3)
    return model, prefix, inputs, positions


def run(model, prefix, inputs, positions, validity, candidate=True):
    cache = DynamicCache([(k.clone(),v.clone()) for k,v in prefix])
    if candidate:
        with noncausal_expert_mask(model.config):
            return model(inputs_embeds=inputs,position_ids=positions,past_key_values=cache,
                         attention_mask=validity,use_cache=True,is_causal=False).last_hidden_state
    bias = inputs.new_zeros((2,1,3,7)).masked_fill(~validity[:,None,None,:],torch.finfo(inputs.dtype).min)
    return model(inputs_embeds=inputs,position_ids=positions,past_key_values=cache,
                 attention_mask=bias,use_cache=True,is_causal=False).last_hidden_state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("implementation", ["sdpa","eager"])
@pytest.mark.parametrize("padded", [False,True])
def test_real_expert_visibility_matches_existing_full_bias(dtype,implementation,padded):
    model,prefix,inputs,positions = model_and_data(dtype,implementation)
    validity=torch.ones(2,7,dtype=torch.bool)
    if padded:
        validity[0,0]=False  # left padding
        validity[1,2:4]=False  # another example's earlier future-start boundary
    before=copy.deepcopy(model.config.to_dict())
    masks=[]
    handle=model.layers[0].self_attn.register_forward_pre_hook(
        lambda module,args,kwargs:masks.append(kwargs["attention_mask"]),with_kwargs=True)
    # Compare visibility under the same math backend. Fused mask/no-mask kernel
    # rounding is the separately measured GPU parity issue, not this unit test.
    with sdpa_kernel(SDPBackend.MATH):
        actual=run(model,prefix,inputs,positions,validity)
    handle.remove()
    with sdpa_kernel(SDPBackend.MATH):
        expected=run(model,prefix,inputs,positions,validity,candidate=False)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert model.config.to_dict()==before
    if not padded:
        assert masks==[None]
    else:
        mask=masks[0]
        seen=mask if mask.dtype==torch.bool else mask.eq(0)
        assert torch.equal(seen.expand(2,1,3,7),validity[:,None,None,:].expand(2,1,3,7))


def test_later_action_tokens_are_visible_and_masked_cache_values_are_not():
    model,prefix,inputs,positions=model_and_data()
    validity=torch.tensor([[0,1,1,1,1,1,1],[1,1,0,0,1,1,1]],dtype=torch.bool)
    original=run(model,prefix,inputs,positions,validity)
    perturbed=inputs.clone(); perturbed[:,-1]+=10*torch.randn_like(perturbed[:,-1])
    assert not torch.allclose(run(model,prefix,perturbed,positions,validity)[:,0],original[:,0])
    hidden_prefix=[(k.clone(),v.clone()) for k,v in prefix]
    for keys,values in hidden_prefix:
        keys[0,:,0,:]=1000;values[0,:,0,:]=-1000
        keys[1,:,2:4,:]=-1000;values[1,:,2:4,:]=1000
    torch.testing.assert_close(run(model,hidden_prefix,inputs,positions,validity),original,rtol=0,atol=0)


def test_context_restores_missing_existing_inherited_flags_and_nested_exceptions():
    class Inherited:
        is_causal=True
    for config in (SimpleNamespace(),SimpleNamespace(is_causal=True),SimpleNamespace(is_causal=False),Inherited()):
        before=vars(config).copy()
        with pytest.raises(RuntimeError,match="sentinel"):
            with noncausal_expert_mask(config):
                assert config.is_causal is False
                with noncausal_expert_mask(config):
                    assert config.is_causal is False
                raise RuntimeError("sentinel")
        assert vars(config)==before
        with noncausal_expert_mask(config,enabled=False):
            assert vars(config)==before
