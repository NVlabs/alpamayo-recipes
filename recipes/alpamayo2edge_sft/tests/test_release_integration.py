# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU integration with the installed inference package, without external assets.

The tokenizer vocabulary and inputs are synthetic; all model layers, trajectory
tokenizers, action-space fitting, and checkpoint I/O use the real implementations.
These tests do not validate a full-size released checkpoint or GPU execution.
"""

import json
import math
import shutil

import pytest
import torch


@pytest.fixture(autouse=True)
def _offline_real_release(pytestconfig, monkeypatch):
    if pytestconfig.getoption("--edge-release-stub"):
        pytest.skip("Requires the installed inference package; release API doubles are excluded")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(713)
            yield
    finally:
        torch.set_num_threads(previous_threads)


def _save_tokenizer(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokens = [f"token_{index}" for index in range(128)]
    tokens[:14] = [
        "<pad>", "<s>", "</s>", "<unk>", "<history_pad>", "<future_pad>",
        "<future_end>", "<history_start>", "<history_end>", "<future_start>",
        "<image>", "<vision_start>", "<vision_end>", "<video>",
    ]
    tokens[100:120] = [f"<i{index}>" for index in range(20)]
    backend = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)},
                                 unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<s>",
        eos_token="</s>", unk_token="<unk>", padding_side="left",
    )
    tokenizer.save_pretrained(directory)


def _tiny_release_model(tokenizer_directory, stage):
    from alpamayo2_edge.config import Alpamayo2EdgeConfig
    from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge
    from alpamayo2edge_sft.models import TrainableAlpamayo2Edge
    from transformers import Cosmos3EdgeConfig, Cosmos3EdgeForConditionalGeneration

    _save_tokenizer(tokenizer_directory)
    text = {
        "model_type": "cosmos3_edge_text", "vocab_size": 128,
        "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        "attention_dropout": 0.0,
        "rope_parameters": {
            "rope_type": "default", "rope_theta": 10000.0,
            "mrope_section": [2, 3, 3],
        },
    }
    action_space = {
        "_target_": "alpamayo2_edge.action_space.UnicycleAccelCurvatureCourseActionSpace",
        "n_waypoints": 4, "accel_mean": 0.123456, "accel_std": 1.234567,
        "curvature_mean": 0.002345, "curvature_std": 0.123456,
        "theta_lambda": 2e-6, "theta_ridge": 3e-8,
        "v_lambda": 4e-6, "v_ridge": 5e-4,
        "a_lambda": 6e-4, "a_ridge": 7e-4,
        "kappa_lambda": 8e-4, "kappa_ridge": 9e-4,
        "slip_gradient": 0.0037,
    }
    config = Alpamayo2EdgeConfig(
        dtype="float32", vlm_name_or_path=str(tokenizer_directory),
        vlm_config=Cosmos3EdgeConfig(
            text_config=text, image_token_id=10, video_token_id=13,
            vision_start_token_id=11, vision_end_token_id=12,
            projector_hidden_size=48,
            vision_config={
                "num_hidden_layers": 1, "hidden_size": 32, "intermediate_size": 48,
                "num_attention_heads": 2, "patch_size": 2,
                "num_patches": 4, "spatial_merge_size": 2,
            },
        ),
        expert_config={
            "llm_config": {**text, "hidden_size": 32},
            "action_space_cfg": action_space,
            "diffusion_cfg": {"_target_": "alpamayo2_edge.diffusion.flow_matching.FlowMatching"},
            "action_in_proj_cfg": {
                "_target_": "alpamayo2_edge.models.action_in_proj.PerWaypointActionInProjV2",
                "hidden_size": 32, "num_enc_layers": 1,
                "num_fourier_feats": 8, "max_freq": 17.0,
            },
            "action_out_proj_cfg": {"_target_": "torch.nn.Linear"},
        },
        hist_traj_tokenizer_cfg={
            "_target_": "alpamayo2_edge.models.delta_tokenizer.DeltaTrajectoryTokenizer",
            "pad_origin_at_beginning": False, "num_bins": 10,
        },
        future_traj_tokenizer_cfg={
            "_target_": (
                "alpamayo2_edge.action_space.discrete_action_space.DiscreteTrajectoryTokenizer"
            ),
            "_recursive_": False,
            "action_space_cfg": action_space,
            "dims_min": [-2.0, -1.0], "dims_max": [2.0, 1.0], "num_bins": 10,
        },
        history_vocab_size=10, future_vocab_size=10,
        tokens_per_history_traj=57, tokens_per_future_traj=8,
        traj_ids={
            "history_id0": 100, "future_id0": 110, "history_pad": 4, "future_pad": 5,
            "history_start": 7, "history_end": 8, "future_start": 9, "future_end": 6,
        },
    )
    config._attn_implementation = "sdpa"
    model = TrainableAlpamayo2Edge(config, training_stage=stage).to("cpu")
    assert isinstance(model, Alpamayo2Edge)
    assert isinstance(model.vlm, Cosmos3EdgeForConditionalGeneration)
    assert type(model.expert).__module__ == "alpamayo2_edge.models.expert"
    return model


def _batch():
    # Four image tokens correspond to sixteen 2x2 RGB patches after spatial merging.
    ids = torch.tensor([[0, 1, 11, 10, 10, 10, 10, 12, 14, 7]
                        + [4] * 57 + [8, 15, 9] + [5] * 8 + [6, 2]])
    labels_mask = torch.zeros_like(ids, dtype=torch.bool)
    labels_mask[:, 68:] = True  # Assistant text, then the future trajectory and EOS.
    history = torch.zeros(1, 1, 20, 3)
    history[0, 0, :, 0] = torch.arange(-19, 1) * 0.1
    future = torch.zeros(1, 1, 4, 3)
    future[0, 0, :, 0] = torch.arange(1, 5) * 0.1
    return {
        "tokenized_data": {
            "input_ids": ids, "attention_mask": ids.ne(0).long(),
            # Exercise the recipe's modality-ID compatibility boundary as well.
            "pixel_values": torch.randn(16, 12),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        },
        "labels_mask": labels_mask,
        "traj_data": {
            "ego_history_xyz": history,
            "ego_history_rot": torch.eye(3).expand(1, 1, 20, 3, 3).clone(),
            "ego_future_xyz": future,
            "ego_future_rot": torch.eye(3).expand(1, 1, 4, 3, 3).clone(),
        },
    }


def _assert_active_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def _fixed_output(model, batch):
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(719)
        output = model(**batch)
    predictions = output.supervised_logits if model.training_stage == "vlm" else output.logits
    return output.loss, predictions


@pytest.mark.parametrize("stage", ["vlm", "expert", "joint"])
def test_real_release_training_and_strict_checkpoint_roundtrip(tmp_path, stage):
    from alpamayo2edge_sft.models import TrainableAlpamayo2Edge

    tokenizer_directory = tmp_path / "original-tokenizer"
    model = _tiny_release_model(tokenizer_directory, stage)
    batch = _batch()
    frozen = None if stage == "joint" else (model.expert if stage == "vlm" else model.vlm)
    active = model if stage == "joint" else (model.vlm if stage == "vlm" else model.expert)
    frozen_before = {
        name: parameter.detach().clone() for name, parameter in (frozen.named_parameters() if frozen is not None else [])
    }
    changed_parameter = (
        model.vlm.lm_head.weight if stage == "vlm" else model.expert.action_out_proj.weight
    )
    changed_before = changed_parameter.detach().clone()
    model.train()
    assert active.training
    if frozen is not None:
        assert not frozen.training
    assert all(parameter.requires_grad for parameter in active.parameters())
    if frozen is not None:
        assert not any(parameter.requires_grad for parameter in frozen.parameters())
    output = model(**batch)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss) and output.loss.item() > 0
    if stage == "vlm":
        assert output.future_traj_ce.item() > 0 and output.others_ce.item() > 0
    else:
        assert output.logits.shape == (1, 4, 2)
    output.loss.backward()
    if frozen is not None:
        assert all(parameter.grad is None for parameter in frozen.parameters())
    if stage in {"vlm", "joint"}:
        _assert_active_gradients(model.vlm.model.visual)
        _assert_active_gradients(model.vlm.model.projector)
        _assert_active_gradients(model.vlm.model.language_model)
        _assert_active_gradients(model.vlm.lm_head)
    if stage in {"expert", "joint"}:
        for module in (
            model.expert.expert, model.expert.action_in_proj, model.expert.action_out_proj
        ):
            _assert_active_gradients(module)
    torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad),
                    lr=1e-3).step()
    assert not torch.equal(changed_parameter, changed_before)
    for name, parameter in (frozen.named_parameters() if frozen is not None else []):
        torch.testing.assert_close(parameter, frozen_before[name], rtol=0, atol=0)

    expected_loss, expected_predictions = _fixed_output(model, batch)
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(checkpoint)
    saved = json.loads((checkpoint / "config.json").read_text())
    assert saved["architectures"] == ["Alpamayo2Edge"]
    assert saved["vlm_class"] == "Cosmos3EdgeForConditionalGeneration"
    assert saved["vlm_config"]["model_type"] == "cosmos3_edge"
    assert saved["vlm_config"]["vision_config"]["model_type"] == "cosmos3_edge_vision"
    assert saved["expert_config"]["llm_config"]["model_type"] == "cosmos3_edge_text"
    if stage == "joint":
        assert saved["qk_gradient_layout"] is False
        assert not hasattr(model, "_joint_sft_qk_layout")
    assert saved["training_stage"] == stage
    assert saved["cotrain_expert_vlm"] is (stage == "joint")
    assert saved["cotrain_detach_expert_kv"] is True
    assert saved["loss_weights"] == model.config.loss_weights
    assert saved["lm_token_loss_weights"] == model.config.lm_token_loss_weights
    # Loading must use the newly saved tokenizer assets, not the original fixture path.
    shutil.rmtree(tokenizer_directory)
    restored = TrainableAlpamayo2Edge.from_pretrained(
        checkpoint, local_files_only=True, require_complete_weights=True,
        dtype=torch.float32, attn_implementation="sdpa",
    )
    assert restored.training_stage == stage
    assert restored.config.loss_weights == model.config.loss_weights
    assert restored.config.lm_token_loss_weights == model.config.lm_token_loss_weights
    assert restored.tokenizer.get_vocab() == model.tokenizer.get_vocab()
    assert dict(restored.named_parameters()).keys() == dict(model.named_parameters()).keys()
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        assert dict(restored.named_parameters())[name].requires_grad == parameter.requires_grad
    projection = restored.expert.action_in_proj
    for encoder in [*projection.sinus, projection.timestep_fourier_encoder]:
        expected = torch.logspace(0, math.log10(17.0), encoder.out_dim // 2)[None]
        torch.testing.assert_close(encoder.freqs, expected, rtol=0, atol=0)
        assert encoder.freqs.dtype == torch.float32
    for space in (restored.expert.action_space, restored.future_traj_tokenizer.action_space):
        assert type(space).__name__ == "UnicycleAccelCurvatureCourseActionSpace"
        for name in (
            "theta_lambda", "theta_ridge", "v_lambda", "v_ridge", "a_lambda", "a_ridge",
            "kappa_lambda", "kappa_ridge", "slip_gradient",
        ):
            assert getattr(space, name) == model.config.expert_config.action_space_cfg[name]
        for name, expected in {
            "accel_mean": 0.123456, "accel_std": 1.234567,
            "curvature_mean": 0.002345, "curvature_std": 0.123456,
        }.items():
            buffer = getattr(space, name)
            assert buffer.device.type == "cpu", f"{name} was not materialized on CPU"
            torch.testing.assert_close(buffer, torch.tensor(expected), rtol=0, atol=0)
    actual_loss, actual_predictions = _fixed_output(restored, batch)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual_predictions, expected_predictions, rtol=1e-6, atol=1e-7)


def test_cosmos3_edge_rejects_qk_normalization_workaround(tmp_path):
    model = _tiny_release_model(tmp_path / "tokenizer", "joint")
    assert not hasattr(model.expert.expert.layers[0].self_attn, "q_norm")
    assert not hasattr(model.vlm.model.language_model.layers[0].self_attn, "k_norm")
    model.config.qk_gradient_layout = True
    with pytest.raises(ValueError, match="no Q/K normalization"):
        model.set_training_stage("joint")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_joint_vision_precision_scope_and_public_export(tmp_path, dtype):
    from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge
    from alpamayo2edge_sft.models.vision_precision import joint_sft_vision_precision

    model = _tiny_release_model(tmp_path / "tokenizer", "joint").to(dtype=dtype)
    batch = _batch()
    before = {name: value.clone() for name, value in model.named_buffers()}
    state_keys = set(model.state_dict())
    # Install once before the observation hook, so it sees the effective inputs.
    with joint_sft_vision_precision(model.vlm):
        pass
    # C3's SigLIP2 vision encoder has learned positions and no vision RoPE.
    visual = model.vlm.model.visual
    assert not hasattr(visual, "blocks")
    assert not hasattr(model.vlm, "_joint_sft_vision_precision")
    with torch.no_grad():
        prepared = model._prepare_vlm_inputs(batch["tokenized_data"])
        native = model.vlm(**prepared, use_cache=False).logits
        with joint_sft_vision_precision(model.vlm):
            scoped = model.vlm(**prepared, use_cache=False).logits
    torch.testing.assert_close(native, scoped, rtol=0, atol=0)
    losses = []
    for training in (True, False):
        model.train(training)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(719)
            losses.append(model(**batch).loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert set(model.state_dict()) == state_keys
    for name, original in before.items():
        actual = dict(model.named_buffers())[name]
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    checkpoint = tmp_path / "public-export"
    model.save_pretrained(checkpoint)
    exported, info = Alpamayo2Edge.from_pretrained(
        checkpoint, local_files_only=True, dtype=dtype,
        attn_implementation="sdpa", output_loading_info=True,
    )
    assert type(exported) is Alpamayo2Edge
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert set(exported.state_dict()) == state_keys
    assert not hasattr(exported.vlm, "_joint_sft_vision_precision")
    assert json.loads((checkpoint / "config.json").read_text())["architectures"] == ["Alpamayo2Edge"]


@pytest.mark.parametrize("stage", ["joint","expert"])
def test_expert_mask_scope_preserves_gradients_and_public_inference(tmp_path, stage):
    import copy
    from transformers.cache_utils import DynamicCache

    model=_tiny_release_model(tmp_path/'tokenizer',stage)
    batch=_batch()
    model.train()
    before=copy.deepcopy(model.config.to_dict())
    expert_predictions=[]
    handle=model.expert.action_out_proj.register_forward_hook(lambda module,args,output:expert_predictions.append(output))
    output=model(**batch)
    handle.remove()
    # Backpropagate expert output alone; detached conditioning must block VLM gradients.
    expert_predictions[0].float().square().mean().backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.expert.parameters())
    assert model.config.to_dict()==before
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(**batch).loss)
    assert model.config.to_dict()==before
    target=tmp_path/'checkpoint';model.save_pretrained(target)
    from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge
    restored=Alpamayo2Edge.from_pretrained(target,local_files_only=True)
    assert type(restored) is Alpamayo2Edge
    assert getattr(restored.expert.expert.config,"is_causal",True)==getattr(model.expert.expert.config,"is_causal",True)
    # Exercise the inherited inference denoiser after training and after native
    # reload. It continues to accept the public explicit4D noncausal mask.
    prefix=[(torch.randn(1,2,2,16),torch.randn(1,2,2,16)) for _ in range(2)]
    positions=torch.arange(2,6).view(1,1,4).expand(3,1,4)
    outputs=[]
    with torch.no_grad():
        for inference_model in (model,restored):
            cache=DynamicCache([(k.clone(),v.clone()) for k,v in prefix])
            outputs.append(inference_model._sample_actions(cache,2,positions,
                torch.zeros(1,1,4,6),1,torch.device('cpu'),
                {'inference_step':2,'generator':torch.Generator().manual_seed(129)}))
            assert cache.get_seq_length()==2
    torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)


def test_real_release_trajectory_free_joint_checkpoint_roundtrip(tmp_path):
    """The real action projections/expert execute with connected zero gradients."""
    from alpamayo2edge_sft.models import TrainableAlpamayo2Edge

    model = _tiny_release_model(tmp_path / 'tokenizer', 'joint')
    batch = _batch()
    ids = batch['tokenized_data']['input_ids']
    for token in (4, 5, 9):
        ids[ids == token] = 14
    batch['traj_data'] = [None]
    output = model(**batch)
    assert torch.isfinite(output.loss) and output.vlm_ce.item() > 0
    assert output.expert_diffusion.item() == 0
    output.loss.backward()
    for name, parameter in model.expert.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) == 0, name
    _assert_active_gradients(model.vlm.model.visual)
    _assert_active_gradients(model.vlm.model.language_model)
    expected_loss, expected_predictions = _fixed_output(model, batch)
    checkpoint = tmp_path / 'text-checkpoint'
    model.save_pretrained(checkpoint)
    restored = TrainableAlpamayo2Edge.from_pretrained(
        checkpoint, local_files_only=True, require_complete_weights=True,
        dtype=torch.float32, attn_implementation='sdpa',
    )
    loss, predictions = _fixed_output(restored, batch)
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(predictions, expected_predictions, rtol=0, atol=0)
