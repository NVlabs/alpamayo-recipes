# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU integration tests against the installed public Edge inference package."""

from types import SimpleNamespace

import pytest
import torch

from transformers import GenerationConfig, Cosmos3EdgeConfig

from alpamayo2edge_sft.models.trainable_alpamayo2_edge import TrainableAlpamayo2Edge


@pytest.fixture
def model(monkeypatch):
    """Use actual vision, language, expert and diffusion modules with small widths."""
    pytest.importorskip("alpamayo2_edge.models.expert_utils")
    from alpamayo2_edge.config import Alpamayo2EdgeConfig

    text = {
        "model_type": "cosmos3_edge_text",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [2, 3, 3],
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    config = Alpamayo2EdgeConfig(
        dtype="float32",
        vlm_name_or_path="test-tokenizer",
        vlm_config=Cosmos3EdgeConfig(
            text_config=text,
            image_token_id=10,
            video_token_id=13,
            vision_start_token_id=11,
            vision_end_token_id=12,
            projector_hidden_size=48,
            vision_config={
                "num_hidden_layers": 1,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_attention_heads": 2,
                "num_patches": 4,
                "patch_size": 2,
                "spatial_merge_size": 2,
            },
        ),
        expert_config={
            "llm_config": {**text, "hidden_size": 32},
            "action_space_cfg": {
                "_target_": "alpamayo2_edge.action_space.UnicycleAccelCurvatureCourseActionSpace",
                "n_waypoints": 4,
                "dt": 0.1,
            },
            "diffusion_cfg": {"_target_": "alpamayo2_edge.diffusion.flow_matching.FlowMatching"},
            "action_in_proj_cfg": {
                "_target_": "alpamayo2_edge.models.action_in_proj.PerWaypointActionInProjV2",
                "hidden_size": 32,
                "num_enc_layers": 1,
            },
            "action_out_proj_cfg": {"_target_": "torch.nn.Linear"},
        },
        traj_ids={
            "history_id0": 100,
            "future_id0": 110,
            "future_start": 9,
            "history_pad": 98,
            "future_pad": 99,
        },
        hist_traj_tokenizer_cfg={
            "_target_": "alpamayo2_edge.models.delta_tokenizer.DeltaTrajectoryTokenizer",
            "pad_origin_at_beginning": False,
            "num_bins": 10,
        },
        history_vocab_size=10,
        future_vocab_size=10,
    )
    monkeypatch.setattr(
        "alpamayo2_edge.models.alpamayo2_edge.build_alpamayo2_edge_tokenizer",
        lambda *args: SimpleNamespace(pad_token_id=0),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return TrainableAlpamayo2Edge(config).eval()


def _data(multimodal=False):
    ids = torch.tensor(
        [[0, 0, 1, 11, 10, 10, 10, 10, 12, 9], [1, 3, 4, 11, 10, 10, 10, 10, 12, 9]]
        if multimodal
        else [[0, 0, 1, 3, 4, 5, 6, 7, 8, 9], [1, 3, 4, 5, 6, 7, 8, 3, 6, 9]]
    )
    tokenized = {"input_ids": ids, "attention_mask": ids.ne(0).long()}
    if multimodal:
        tokenized.update(
            mm_token_type_ids=ids.eq(10).long(),
            image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
            pixel_values=torch.randn(32, 12, generator=torch.Generator().manual_seed(19)),
        )
    history = torch.zeros(2, 1, 20, 3)
    history[..., 0] = torch.arange(20) * 0.1
    return {
        "tokenized_data": tokenized,
        "ego_history_xyz": history,
        "ego_history_rot": torch.eye(3).expand(2, 1, 20, 3, 3),
        # A sampler must never use future ground truth, even when it is present.
        "ego_future_xyz": torch.full((2, 1, 4, 3), float("nan")),
    }


@pytest.mark.parametrize("multimodal", [False, True])
@torch.inference_mode()
def test_prefill_matches_stock_hf_cache_with_left_padding_and_stale_rope(
    model, monkeypatch, multimodal
):
    data = _data(multimodal)
    tokenized = data["tokenized_data"]
    expected = model.vlm.generate(
        **tokenized,
        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        ),
    )
    expected_deltas = model.vlm.model.rope_deltas.clone()
    model.vlm.model.rope_deltas = torch.full((7, 1), 13, dtype=torch.long)
    captured = []

    def sample_actions(cache, length, positions, mask, batch, device, diffusion_kwargs):
        assert length == tokenized["input_ids"].shape[1]
        assert cache.get_seq_length() == length
        assert batch == 12
        captured.append((positions, mask))
        visible = tokenized["attention_mask"].repeat_interleave(6, dim=0).bool()
        for actual_layer, expected_layer in zip(cache.layers, expected.past_key_values.layers):
            for name in ("keys", "values"):
                actual_values = getattr(actual_layer, name).transpose(1, 2)
                expected_values = getattr(expected_layer, name).repeat_interleave(6, dim=0)
                torch.testing.assert_close(
                    actual_values[visible], expected_values.transpose(1, 2)[visible],
                    atol=2e-6, rtol=2e-5,
                )
        return torch.zeros(batch, 4, 2, device=device)

    monkeypatch.setattr(model, "_sample_actions", sample_actions)
    monkeypatch.setattr(model.vlm, "generate", lambda **kwargs: pytest.fail("Must not decode CoT"))
    # Old packed fixtures omit the processor field; the adapter must reconstruct it.
    if multimodal:
        del tokenized["mm_token_type_ids"]
    xyz, rot, logprob = model.sample_trajectories_from_data(
        data, num_traj_sets=2, num_traj_samples=3
    )
    assert xyz.shape == (2, 2, 3, 4, 3)
    assert rot.shape == (2, 2, 3, 4, 3, 3)
    assert logprob.shape == (2, 2, 3, 4)
    positions, mask = captured[0]
    offset = tokenized["attention_mask"].sum(-1, keepdim=True) + expected_deltas
    expected_positions = offset.repeat_interleave(6, dim=0) + torch.arange(4)
    torch.testing.assert_close(positions, expected_positions[None].expand(3, -1, -1))
    assert mask.shape == (12, 1, 4, 14)
    assert (mask[:6, :, :, :2] < -1e20).all()
    assert (mask[..., 10:] == 0).all()  # All action tokens attend to each other.


def test_real_public_diffusion_is_seeded_and_preserves_reusable_inputs(model, monkeypatch):
    data = _data(multimodal=True)
    original = {key: value.clone() for key, value in data["tokenized_data"].items()}
    monkeypatch.setattr(model.vlm, "generate", lambda **kwargs: pytest.fail("Must not decode CoT"))
    results = []
    for seed in (29, 29, 31):
        results.append(model.sample_trajectories_from_data(
            data,
            num_traj_sets=2,
            num_traj_samples=2,
            diffusion_kwargs={"generator": torch.Generator().manual_seed(seed)},
        )[0])
    assert results[0].shape == (2, 2, 2, 4, 3)
    assert all(torch.isfinite(result).all() for result in results)
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    assert not torch.equal(results[0], results[2])
    for key, value in original.items():
        torch.testing.assert_close(data["tokenized_data"][key], value)


@pytest.mark.parametrize(
    "change", ["missing_anchor", "duplicate_anchor", "future", "future_pad", "masked_anchor"]
)
def test_invalid_prompt_rejected_before_public_prefill(model, monkeypatch, change):
    data = _data()
    tokenized = data["tokenized_data"]
    if change == "missing_anchor":
        tokenized["input_ids"][0, -1] = 8
    elif change == "duplicate_anchor":
        tokenized["input_ids"][0, 4] = 9
    elif change == "future":
        tokenized["input_ids"][0, 4] = 110
    elif change == "future_pad":
        tokenized["input_ids"][0, 4] = 99
    else:
        tokenized["attention_mask"][0, -1] = 0
    monkeypatch.setattr(model.vlm.model, "forward", lambda **kwargs: pytest.fail("Invalid input"))
    with pytest.raises(ValueError):
        model.sample_trajectories_from_data(data)


@pytest.mark.parametrize("multimodal", [False, True])
def test_native_cot_generation_reaches_expert_with_generated_prefix(model, monkeypatch, multimodal):
    """Exercise real HF generation, prefix caching, and diffusion on a tiny CPU model."""
    data = _data(multimodal)
    tokenized = data["tokenized_data"]
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if name in tokenized:
            tokenized[name] = tokenized[name][:, :-1]
    history_count = model.config.tokens_per_history_traj
    tokenized["input_ids"] = torch.nn.functional.pad(
        tokenized["input_ids"], (0, history_count), value=model.config.traj_ids["history_pad"]
    )
    tokenized["attention_mask"] = torch.nn.functional.pad(
        tokenized["attention_mask"], (0, history_count), value=1
    )
    if "mm_token_type_ids" in tokenized:
        tokenized["mm_token_type_ids"] = torch.nn.functional.pad(
            tokenized["mm_token_type_ids"], (0, history_count), value=0
        )
    original = {key: value.clone() for key, value in tokenized.items()}
    prompt_length = tokenized["input_ids"].shape[1]
    generate = model.vlm.generate
    generated = []

    class ReasoningThenAnchor:
        def __call__(self, ids, scores):
            next_token = (42, 9, 8)[min(ids.shape[1] - prompt_length, 2)]
            scores.fill_(float("-inf"))
            scores[:, next_token] = 0
            return scores

    def generate_with_known_continuation(**kwargs):
        kwargs["logits_processor"].append(ReasoningThenAnchor())
        output = generate(**kwargs)
        generated.append(output.sequences[:, prompt_length:].clone())
        return output

    monkeypatch.setattr(model.vlm, "generate", generate_with_known_continuation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        xyz, rot, logprob = model.sample_trajectories_from_data(
            data, generate_cot=True, num_traj_sets=2, num_traj_samples=2,
            max_generation_length=4,
        )
    assert len(generated) == 1
    assert generated[0][:, :2].tolist() == [[42, 9]] * 8
    assert xyz.shape == (2, 2, 2, 4, 3)
    assert rot.shape == (2, 2, 2, 4, 3, 3)
    assert logprob.shape == (2, 2, 2, 4)
    assert torch.isfinite(xyz).all() and torch.isfinite(rot).all()
    for key, value in original.items():
        torch.testing.assert_close(tokenized[key], value)


def test_cot_generation_rejects_anchored_prompt_before_decoding(model, monkeypatch):
    monkeypatch.setattr(model.vlm, "generate", lambda **kwargs: pytest.fail("Invalid input"))
    with pytest.raises(ValueError, match="fresh assistant prompt"):
        model.sample_trajectories_from_data(_data(), generate_cot=True)


@torch.inference_mode()
def test_8192_padded_prefill_matches_native_cache_on_visible_tokens(model, monkeypatch):
    """Run the old full-padding shape through the real public VLM and sampler."""
    data = _data()
    data["tokenized_data"] = {key: value[:1, 2:] for key, value in data["tokenized_data"].items()}
    data["ego_history_xyz"] = data["ego_history_xyz"][:1]
    data["ego_history_rot"] = data["ego_history_rot"][:1]
    tokenized = data["tokenized_data"]
    visible_length = tokenized["input_ids"].shape[1]
    expected = model.vlm.generate(
        **tokenized,
        generation_config=GenerationConfig(
            do_sample=False, max_new_tokens=1, return_dict_in_generate=True,
            use_cache=True, pad_token_id=0, eos_token_id=None,
        ),
    ).past_key_values
    data["tokenized_data"] = {
        key: torch.nn.functional.pad(value, (8192 - visible_length, 0), value=0)
        for key, value in tokenized.items()
    }
    def sample_actions(cache, length, positions, mask, batch, device, diffusion_kwargs):
        assert length == 8192
        assert (mask[..., :8192 - visible_length] < -1e20).all()
        for actual_layer, expected_layer in zip(cache.layers, expected.layers, strict=True):
            for name in ("keys", "values"):
                torch.testing.assert_close(
                    getattr(actual_layer, name)[..., -visible_length:, :],
                    getattr(expected_layer, name), atol=2e-6, rtol=2e-5,
                )
        return torch.zeros(batch, 4, 2)
    monkeypatch.setattr(model, "_sample_actions", sample_actions)
    xyz, _, logprob = model.sample_trajectories_from_data(data)
    assert torch.isfinite(xyz).all()
    assert (logprob == 0).all()  # API placeholder, never a candidate likelihood.
