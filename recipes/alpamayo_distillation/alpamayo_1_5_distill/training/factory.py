# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model factories and online teacher-target construction."""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from alpamayo1_5.models.token_utils import StopAfterEOS, to_special_token
from transformers import LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5_distill.data.dataset import recursive_collate
from alpamayo1_5_distill.models.expert_context import find_future_start_offsets
from alpamayo1_5_distill.models.modeling_distill import DistillAlpamayo1_5, ExpertLogitsProcessor
from alpamayo1_5_distill.training import processor as helper

logger = logging.getLogger(__name__)


class InvalidTeacherGenerationError(RuntimeError):
    """Raised when an online teacher rollout is unsafe to use as supervision."""


class InvalidStudentGenerationError(RuntimeError):
    """Raised when an on-policy student rollout cannot be safely scored."""


_FORBIDDEN_TEACHER_SUFFIX_TOKENS = (
    "answer_start",
    "answer_end",
    "question_start",
    "question_end",
)

# Native Qwen3-VL visual delimiters/placeholders belong only in the prompt.
_QWEN_VISION_CONTROL_TOKEN_IDS = (151652, 151653, 151654, 151655)


def _navigation_text(sample: dict[str, Any]) -> str | None:
    """Return the route condition selected by the dataset for this sample."""
    nav_text = sample.get("nav_text")
    enabled = bool(sample.get("nav_enabled", nav_text is not None))
    return str(nav_text) if enabled and nav_text is not None else None


def _create_sample_message(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the shared teacher/student prompt, including its route condition."""
    return helper.create_message(
        frames=sample["image_frames"].flatten(0, 1),
        camera_indices=sample["camera_indices"],
        nav_text=_navigation_text(sample),
    )


def _dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {name}") from error
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name} is not a torch dtype")
    return dtype


def build_teacher_model(
    name_or_path: str = "nvidia/Alpamayo-1.5-10B",
    vlm_name_or_path: str | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
) -> DistillAlpamayo1_5:
    """Load the frozen AM1.5 teacher used to produce online targets."""
    config = DistillAlpamayo1_5.config_with_vlm_override(
        name_or_path,
        vlm_name_or_path,
    )
    model = DistillAlpamayo1_5.from_pretrained(
        name_or_path,
        config=config,
        dtype=_dtype(dtype),
        attn_implementation=attn_implementation,
    )
    model.requires_grad_(False)
    return model


def build_student_model(
    teacher_name_or_path: str = "nvidia/Alpamayo-1.5-10B",
    teacher_vlm_name_or_path: str | None = None,
    student_vlm_name_or_path: str = "nvidia/Cosmos-Reason2-2B",
    dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    train_timestep_sampler: str = "beta",
    freeze_vision: bool = True,
    gradient_checkpointing: bool = False,
) -> DistillAlpamayo1_5:
    """Create the Cosmos-Reason2-2B VLM with the compact action expert."""
    model = DistillAlpamayo1_5.from_distillation_student(
        teacher_name_or_path=teacher_name_or_path,
        teacher_vlm_name_or_path=teacher_vlm_name_or_path,
        train_timestep_sampler=train_timestep_sampler,
        vlm_name_or_path=student_vlm_name_or_path,
        model_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if gradient_checkpointing:
        # Qwen disables use_cache under checkpointing, while the action expert
        # explicitly needs the differentiable per-layer VLM cache.
        raise ValueError("gradient checkpointing is incompatible with KV-cache action distillation")
    if freeze_vision:
        vision = getattr(model.vlm, "visual", None)
        if vision is None:
            vision = getattr(getattr(model.vlm, "model", None), "visual", None)
        if vision is None:
            raise AttributeError("Cannot locate the Qwen3-VL vision encoder")
        vision.requires_grad_(False)
    return model


def _single_record_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] != 1:
        raise ValueError(f"online teacher expects one sample at a time, got {tensor.shape}")
    return tensor[0].detach().contiguous()


_FLAT_MULTIMODAL_KEYS = {
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
}


def _single_record_tokenized_tensor(
    key: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Remove only real batch dimensions from processor outputs."""
    if key in _FLAT_MULTIMODAL_KEYS:
        return tensor.detach().contiguous()
    return _single_record_tensor(tensor)


def _log_invalid_teacher_generation(
    model: DistillAlpamayo1_5,
    sample: dict[str, Any],
    generated_sequence: torch.Tensor,
    *,
    prompt_length: int,
    max_new_tokens: int,
    reason: str = "invalid expert boundary",
    forbidden_tokens: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Record enough context to identify and inspect a failed teacher rollout."""
    suffix = generated_sequence[0, prompt_length:].detach().cpu()
    try:
        decoded_suffix = model.tokenizer.decode(
            suffix.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        logger.exception(
            "Failed to decode invalid online teacher generation for clip_id=%r, t0_us=%r",
            sample.get("clip_id"),
            sample.get("t0_us"),
        )
        decoded_suffix = "<decode failed; inspect generated_token_ids>"

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.error(
        "Invalid online teacher generation: rank=%d, clip_id=%r, t0_us=%r, "
        "reason=%r, forbidden_tokens=%s, "
        "prompt_length=%d, generated_suffix_length=%d, configured_max_new_tokens=%d, "
        "generated_token_ids=%s, decoded_suffix=%r",
        rank,
        sample.get("clip_id"),
        sample.get("t0_us"),
        reason,
        forbidden_tokens or {},
        prompt_length,
        suffix.numel(),
        max_new_tokens,
        suffix.tolist(),
        decoded_suffix,
    )


def _forbidden_teacher_suffix_tokens(
    tokenizer: Any,
    generated_sequence: torch.Tensor,
    *,
    prompt_length: int,
) -> dict[str, dict[str, Any]]:
    """Return forbidden VQA tokens found in the generated suffix."""
    suffix = generated_sequence[:, prompt_length:]
    found: dict[str, dict[str, Any]] = {}
    for token_name in _FORBIDDEN_TEACHER_SUFFIX_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(to_special_token(token_name))
        if token_id is None:
            raise ValueError(f"tokenizer has no ID for {to_special_token(token_name)}")
        positions = suffix.eq(int(token_id)).nonzero(as_tuple=False)
        if positions.numel():
            found[token_name] = {
                "token_id": int(token_id),
                "suffix_positions": positions[:, 1].detach().cpu().tolist(),
            }
    return found


def build_student_inference_batch(
    model: DistillAlpamayo1_5,
    samples: list[dict[str, Any]],
    *,
    processor: Any | None = None,
) -> dict[str, Any]:
    """Tokenize raw PhysicalAI-AV samples for student trajectory sampling."""
    if not samples:
        raise ValueError("student inference batch must contain at least one sample")
    if processor is None:
        processor = helper.get_processor(
            model.tokenizer,
            getattr(model.config, "vlm_name_or_path", None),
        )
    records = []
    for sample in samples:
        messages = _create_sample_message(sample)
        tokenized = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        records.append(
            {
                "tokenized_data": {
                    key: _single_record_tokenized_tensor(key, value)
                    for key, value in tokenized.items()
                },
                "ego_history_xyz": _single_record_tensor(sample["ego_history_xyz"]),
                "ego_history_rot": _single_record_tensor(sample["ego_history_rot"]),
                "ego_future_xyz": _single_record_tensor(sample["ego_future_xyz"]),
                "ego_future_rot": _single_record_tensor(sample["ego_future_rot"]),
                "clip_id": str(sample["clip_id"]),
                "t0_us": int(sample["t0_us"]),
            }
        )
    batch = recursive_collate(records)
    for key in _FLAT_MULTIMODAL_KEYS:
        values = [
            record["tokenized_data"][key] for record in records if key in record["tokenized_data"]
        ]
        if values:
            batch["tokenized_data"][key] = torch.cat(values, dim=0)
    return helper.to_device(batch, device=next(model.parameters()).device)


def _forward_teacher_distillation_targets(
    model: DistillAlpamayo1_5,
    teacher_batch: dict[str, Any],
    history: dict[str, torch.Tensor],
    *,
    compute_teacher_vector: bool,
) -> dict[str, torch.Tensor]:
    """Run the teacher action expert only when its vector target is requested."""
    if compute_teacher_vector:
        return model.forward_distill(teacher_batch)

    vlm_inputs = dict(teacher_batch["tokenized_data"])
    vlm_input_ids = vlm_inputs.pop("input_ids")
    vlm_input_ids = model.fuse_traj_tokens(vlm_input_ids, history)
    vlm_outputs = model.vlm(
        input_ids=vlm_input_ids,
        **vlm_inputs,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return {
        "student_logits": vlm_outputs.logits,
        "student_hidden": vlm_outputs.hidden_states[-1],
    }


def _build_teacher_logit_targets(
    logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Build dense targets for ``top_k=-1`` or sparse top-k targets otherwise."""

    if top_k == -1:
        return {"teacher_logits": logits}
    if top_k <= 0:
        raise ValueError("top_k must be -1 for dense KL or a positive integer")

    top_k = min(top_k, logits.shape[-1])
    topk_logits, topk_indices = logits.topk(top_k, dim=-1)
    full_log_norm = torch.logsumexp(logits / temperature, dim=-1)
    topk_mass = torch.exp(torch.logsumexp(topk_logits / temperature, dim=-1) - full_log_norm)
    return {
        "teacher_topk_indices": topk_indices,
        "teacher_topk_logits": topk_logits,
        "teacher_residual_mass": 1.0 - topk_mass,
    }


def _build_teacher_targets(
    model: DistillAlpamayo1_5,
    sample: dict[str, Any],
    *,
    top_k: int = 64,
    temperature: float = 0.5,
    max_new_tokens: int = 256,
    processor: Any | None = None,
    compute_teacher_vector: bool = True,
) -> dict[str, Any]:
    """Generate aligned online language, feature, and action targets."""
    if top_k != -1 and top_k <= 0:
        raise ValueError("top_k must be -1 for dense KL or a positive integer")
    device = next(model.parameters()).device
    messages = _create_sample_message(sample)
    if processor is None:
        processor = helper.get_processor(
            model.tokenizer,
            getattr(model.config, "vlm_name_or_path", None),
        )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = helper.to_device(prompt, device=device)
    history = {
        "ego_history_xyz": sample["ego_history_xyz"].to(device),
        "ego_history_rot": sample["ego_history_rot"].to(device),
    }
    prompt_ids = model.fuse_traj_tokens(prompt["input_ids"], history)

    generation_config = copy.deepcopy(model.vlm.generation_config)

    def _as_id_list(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(token_id) for token_id in value]

    chat_eos_ids = _as_id_list(generation_config.eos_token_id) or _as_id_list(
        getattr(model.vlm.config, "eos_token_id", None)
    )
    forbidden_vqa_ids = [
        int(model.tokenizer.convert_tokens_to_ids(to_special_token(token_name)))
        for token_name in _FORBIDDEN_TEACHER_SUFFIX_TOKENS
    ]
    # Vision controls, VQA delimiters, and chat EOS tokens must never appear in
    # generated reasoning.
    # The post-generation validation below remains the authoritative safety check.
    ban_ids = list(
        dict.fromkeys(
            [
                *_QWEN_VISION_CONTROL_TOKEN_IDS,
                *forbidden_vqa_ids,
                *chat_eos_ids,
            ]
        )
    )
    bad_words_ids = [[token_id] for token_id in ban_ids]
    generation_config.top_p = 1.0
    generation_config.top_k = None
    generation_config.temperature = 1.0
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.num_return_sequences = 1
    generation_config.eos_token_id = None
    generation_config.forced_eos_token_id = None
    generation_config.max_new_tokens = max_new_tokens + 1
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    eos_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    generation_inputs = {key: value for key, value in prompt.items() if key != "input_ids"}
    # Transformers can merge checkpoint defaults back into GenerationConfig. Repeat
    # safety-critical overrides as direct kwargs so StopAfterEOS exclusively owns
    # termination and greedy decoding remains deterministic.
    generation_inputs.update(
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=max_new_tokens + 1,
        eos_token_id=None,
        forced_eos_token_id=None,
        output_logits=False,
    )
    generated = model.vlm.generate(
        input_ids=prompt_ids,
        **generation_inputs,
        generation_config=generation_config,
        stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_id)]),
        logits_processor=LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=model.config.traj_token_start_idx,
                    traj_vocab_size=model.config.traj_vocab_size,
                )
            ]
        ),
        bad_words_ids=bad_words_ids,
    )
    generated_sequence = generated.sequences
    forbidden_tokens = _forbidden_teacher_suffix_tokens(
        model.tokenizer,
        generated_sequence,
        prompt_length=prompt_ids.shape[1],
    )
    if forbidden_tokens:
        _log_invalid_teacher_generation(
            model,
            sample,
            generated_sequence,
            prompt_length=prompt_ids.shape[1],
            max_new_tokens=max_new_tokens,
            reason="forbidden VQA token in generated suffix",
            forbidden_tokens=forbidden_tokens,
        )
        raise InvalidTeacherGenerationError(
            "online teacher rollout contains forbidden VQA token(s): " + ", ".join(forbidden_tokens)
        )
    try:
        future_start_offsets = find_future_start_offsets(
            generated_sequence,
            eos_id,
            search_start=prompt_ids.shape[1],
            cache_seq_len=generated_sequence.shape[1],
            require_trailing_token=True,
            allow_missing=False,
            context="online teacher generation",
        )
    except ValueError as error:
        _log_invalid_teacher_generation(
            model,
            sample,
            generated_sequence,
            prompt_length=prompt_ids.shape[1],
            max_new_tokens=max_new_tokens,
            reason="missing or invalid <traj_future_start> boundary",
        )
        raise InvalidTeacherGenerationError(
            "online teacher rollout has no valid <traj_future_start> boundary"
        ) from error
    # Keep the cache-flush token so training and rollout share one sequence
    # contract. Language losses and the action expert mask it below.
    sequence = generated_sequence
    tokenized_data = {
        key: value for key, value in prompt.items() if key not in {"input_ids", "attention_mask"}
    }
    tokenized_data["input_ids"] = sequence
    tokenized_data["attention_mask"] = torch.ones_like(sequence)

    labels = torch.full_like(sequence, -100)
    future_start_offset = int(future_start_offsets[0].item())
    labels[:, prompt_ids.shape[1] : future_start_offset] = sequence[
        :, prompt_ids.shape[1] : future_start_offset
    ]
    target_action = model.action_space.traj_to_action(
        sample["ego_history_xyz"][:, -1].to(device),
        sample["ego_history_rot"][:, -1].to(device),
        sample["ego_future_xyz"][:, -1].to(device),
        sample["ego_future_rot"][:, -1].to(device),
    ).to(model.expert.dtype)
    flow_noise = torch.randn_like(target_action)
    _, flow_timestep, _ = model.diffusion.sample_training_pair(
        target_action,
        noise=flow_noise,
    )
    teacher_batch = {
        "tokenized_data": tokenized_data,
        "labels": labels,
        "ego_history_xyz": sample["ego_history_xyz"].to(device),
        "ego_history_rot": sample["ego_history_rot"].to(device),
        "target_action": target_action,
        "flow_noise": flow_noise,
        "flow_timestep": flow_timestep,
        "return_hidden_states": True,
        "_teacher_prompt_length": torch.full(
            (sequence.shape[0],),
            prompt_ids.shape[1],
            dtype=torch.long,
            device=sequence.device,
        ),
    }
    outputs = _forward_teacher_distillation_targets(
        model,
        teacher_batch,
        history,
        compute_teacher_vector=compute_teacher_vector,
    )

    teacher_logits = outputs.pop("student_logits")
    logit_targets = _build_teacher_logit_targets(
        teacher_logits,
        top_k=top_k,
        temperature=temperature,
    )
    del teacher_logits
    loss_mask = torch.zeros_like(sequence, dtype=torch.bool)
    loss_mask[:, prompt_ids.shape[1] - 1 : future_start_offset - 1] = True
    prompt_mask = torch.zeros_like(sequence, dtype=torch.bool)
    prompt_attention_mask = prompt.get("attention_mask")
    if isinstance(prompt_attention_mask, torch.Tensor):
        prompt_mask[:, : prompt_ids.shape[1]] = prompt_attention_mask.bool()
    else:
        prompt_mask[:, : prompt_ids.shape[1]] = True
    hidden_mask = labels.ne(-100)

    record_tokenized = {
        key: _single_record_tokenized_tensor(key, value) for key, value in tokenized_data.items()
    }
    record = {
        # Transient metadata consumed by build_online_teacher_batch. Inferring this
        # from labels is incorrect because post-reasoning tokens are also ignored.
        "_teacher_prompt_length": int(prompt_ids.shape[1]),
        "tokenized_data": record_tokenized,
        "labels": _single_record_tensor(labels),
        "loss_mask": _single_record_tensor(loss_mask),
        "prompt_mask": _single_record_tensor(prompt_mask),
        "hidden_mask": _single_record_tensor(hidden_mask),
        "ego_history_xyz": _single_record_tensor(sample["ego_history_xyz"]),
        "ego_history_rot": _single_record_tensor(sample["ego_history_rot"]),
        "ego_future_xyz": _single_record_tensor(sample["ego_future_xyz"]),
        "ego_future_rot": _single_record_tensor(sample["ego_future_rot"]),
        "target_action": _single_record_tensor(target_action),
        "flow_noise": _single_record_tensor(flow_noise),
        "flow_timestep": _single_record_tensor(flow_timestep),
        "teacher_temperature": temperature,
        "teacher_hidden": _single_record_tensor(outputs["student_hidden"]).to(torch.bfloat16),
        "clip_id": str(sample["clip_id"]),
        "t0_us": int(sample["t0_us"]),
    }
    for key, value in logit_targets.items():
        target = _single_record_tensor(value)
        if key in {"teacher_logits", "teacher_topk_logits"}:
            target = target.to(torch.bfloat16)
        elif key == "teacher_residual_mass":
            target = target.to(torch.float32)
        record[key] = target
    if compute_teacher_vector:
        record["teacher_vector"] = _single_record_tensor(outputs["student_vector"]).to(
            torch.bfloat16
        )
    return record


def _build_model_prompt(
    model: DistillAlpamayo1_5,
    sample: dict[str, Any],
    processor: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    """Build and trajectory-fuse one model-specific multimodal prompt."""
    device = next(model.parameters()).device
    messages = _create_sample_message(sample)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = helper.to_device(prompt, device=device)
    history = {
        "ego_history_xyz": sample["ego_history_xyz"].to(device),
        "ego_history_rot": sample["ego_history_rot"].to(device),
    }
    prompt_ids = model.fuse_traj_tokens(prompt["input_ids"], history)
    return prompt, prompt_ids, history


@torch.no_grad()
def _generate_student_sequence(
    model: DistillAlpamayo1_5,
    prompt: dict[str, torch.Tensor],
    prompt_ids: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> torch.Tensor:
    """Sample a student reasoning prefix using the inference boundary contract."""
    generation_config = copy.deepcopy(model.vlm.generation_config)

    def _as_id_list(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        return [int(token_id) for token_id in value]

    chat_eos_ids = _as_id_list(generation_config.eos_token_id) or _as_id_list(
        getattr(model.vlm.config, "eos_token_id", None)
    )
    forbidden_vqa_ids = [
        int(model.tokenizer.convert_tokens_to_ids(to_special_token(token_name)))
        for token_name in _FORBIDDEN_TEACHER_SUFFIX_TOKENS
    ]
    ban_ids = list(
        dict.fromkeys(
            [
                *_QWEN_VISION_CONTROL_TOKEN_IDS,
                *forbidden_vqa_ids,
                *chat_eos_ids,
            ]
        )
    )
    generation_config.top_p = top_p
    generation_config.top_k = None
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_beams = 1
    generation_config.num_return_sequences = 1
    generation_config.eos_token_id = None
    generation_config.forced_eos_token_id = None
    generation_config.max_new_tokens = max_new_tokens + 1
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    future_start_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    generation_inputs = {key: value for key, value in prompt.items() if key != "input_ids"}
    generation_inputs.update(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_beams=1,
        num_return_sequences=1,
        max_new_tokens=max_new_tokens + 1,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=None,
        forced_eos_token_id=None,
        output_logits=False,
        return_dict_in_generate=True,
    )
    was_training = model.vlm.training
    model.vlm.eval()
    try:
        generated = model.vlm.generate(
            input_ids=prompt_ids,
            **generation_inputs,
            generation_config=generation_config,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=future_start_id)]),
            logits_processor=LogitsProcessorList(
                [
                    ExpertLogitsProcessor(
                        traj_token_offset=model.config.traj_token_start_idx,
                        traj_vocab_size=model.config.traj_vocab_size,
                    )
                ]
            ),
            bad_words_ids=[[token_id] for token_id in ban_ids],
        )
    finally:
        if was_training:
            model.vlm.train()
    return generated.sequences


def _build_on_policy_targets(
    teacher: DistillAlpamayo1_5,
    student: DistillAlpamayo1_5,
    sample: dict[str, Any],
    *,
    teacher_processor: Any,
    student_processor: Any,
    temperature: float,
    top_p: float,
    divergence_temperature: float,
    max_new_tokens: int,
    compute_teacher_vector: bool,
) -> dict[str, Any]:
    """Sample from the student and score that exact sequence with the teacher."""
    student_prompt, student_prompt_ids, student_history = _build_model_prompt(
        student,
        sample,
        student_processor,
    )
    teacher_prompt, teacher_prompt_ids, teacher_history = _build_model_prompt(
        teacher,
        sample,
        teacher_processor,
    )
    if not torch.equal(
        student_prompt_ids,
        teacher_prompt_ids.to(device=student_prompt_ids.device),
    ):
        raise ValueError("teacher and student processed prompt token IDs differ")

    sequence = _generate_student_sequence(
        student,
        student_prompt,
        student_prompt_ids,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    prompt_length = student_prompt_ids.shape[1]
    forbidden_tokens = _forbidden_teacher_suffix_tokens(
        student.tokenizer,
        sequence,
        prompt_length=prompt_length,
    )
    if forbidden_tokens:
        _log_invalid_teacher_generation(
            student,
            sample,
            sequence,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            reason="forbidden VQA token in on-policy student suffix",
            forbidden_tokens=forbidden_tokens,
        )
        raise InvalidStudentGenerationError(
            "on-policy student rollout contains forbidden VQA token(s)"
        )
    future_start_id = student.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    try:
        future_start_offsets = find_future_start_offsets(
            sequence,
            future_start_id,
            search_start=prompt_length,
            cache_seq_len=sequence.shape[1],
            require_trailing_token=True,
            allow_missing=False,
            context="on-policy student rollout",
        )
    except ValueError as error:
        _log_invalid_teacher_generation(
            student,
            sample,
            sequence,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            reason="missing or invalid <traj_future_start> boundary",
        )
        raise InvalidStudentGenerationError(
            "on-policy student rollout has no valid <traj_future_start> boundary"
        ) from error

    tokenized_data = {
        key: value
        for key, value in student_prompt.items()
        if key not in {"input_ids", "attention_mask"}
    }
    tokenized_data["input_ids"] = sequence
    tokenized_data["attention_mask"] = torch.ones_like(sequence)
    teacher_tokenized_data = {
        key: value
        for key, value in teacher_prompt.items()
        if key not in {"input_ids", "attention_mask"}
    }
    teacher_tokenized_data["input_ids"] = sequence.to(device=next(teacher.parameters()).device)
    teacher_tokenized_data["attention_mask"] = torch.ones_like(teacher_tokenized_data["input_ids"])

    future_start_offset = int(future_start_offsets[0].item())
    target_action = student.action_space.traj_to_action(
        sample["ego_history_xyz"][:, -1].to(student_prompt_ids.device),
        sample["ego_history_rot"][:, -1].to(student_prompt_ids.device),
        sample["ego_future_xyz"][:, -1].to(student_prompt_ids.device),
        sample["ego_future_rot"][:, -1].to(student_prompt_ids.device),
    ).to(student.expert.dtype)
    flow_noise = torch.randn_like(target_action)
    _, flow_timestep, _ = student.diffusion.sample_training_pair(
        target_action,
        noise=flow_noise,
    )
    teacher_batch = {
        "tokenized_data": teacher_tokenized_data,
        "ego_history_xyz": teacher_history["ego_history_xyz"],
        "ego_history_rot": teacher_history["ego_history_rot"],
        "target_action": target_action.to(next(teacher.parameters()).device),
        "flow_noise": flow_noise.to(next(teacher.parameters()).device),
        "flow_timestep": flow_timestep.to(next(teacher.parameters()).device),
        "return_hidden_states": True,
        "_teacher_prompt_length": torch.full(
            (sequence.shape[0],),
            prompt_length,
            dtype=torch.long,
            device=teacher_tokenized_data["input_ids"].device,
        ),
    }
    outputs = _forward_teacher_distillation_targets(
        teacher,
        teacher_batch,
        teacher_history,
        compute_teacher_vector=compute_teacher_vector,
    )

    labels = torch.full_like(sequence, -100)
    on_policy_kl_mask = torch.zeros_like(sequence, dtype=torch.bool)
    on_policy_kl_mask[:, prompt_length:future_start_offset] = True
    prompt_mask = torch.zeros_like(sequence, dtype=torch.bool)
    prompt_attention_mask = student_prompt.get("attention_mask")
    if isinstance(prompt_attention_mask, torch.Tensor):
        prompt_mask[:, :prompt_length] = prompt_attention_mask.bool()
    else:
        prompt_mask[:, :prompt_length] = True
    hidden_mask = on_policy_kl_mask.clone()

    record = {
        "_teacher_prompt_length": int(prompt_length),
        "tokenized_data": {
            key: _single_record_tokenized_tensor(key, value)
            for key, value in tokenized_data.items()
        },
        "labels": _single_record_tensor(labels),
        "loss_mask": _single_record_tensor(torch.zeros_like(sequence, dtype=torch.bool)),
        "on_policy_kl_mask": _single_record_tensor(on_policy_kl_mask),
        "prompt_mask": _single_record_tensor(prompt_mask),
        "hidden_mask": _single_record_tensor(hidden_mask),
        "ego_history_xyz": _single_record_tensor(student_history["ego_history_xyz"]),
        "ego_history_rot": _single_record_tensor(student_history["ego_history_rot"]),
        "ego_future_xyz": _single_record_tensor(
            sample["ego_future_xyz"].to(student_prompt_ids.device)
        ),
        "ego_future_rot": _single_record_tensor(
            sample["ego_future_rot"].to(student_prompt_ids.device)
        ),
        "target_action": _single_record_tensor(target_action),
        "flow_noise": _single_record_tensor(flow_noise),
        "flow_timestep": _single_record_tensor(flow_timestep),
        "teacher_temperature": divergence_temperature,
        "teacher_logits": _single_record_tensor(outputs["student_logits"]).to(torch.bfloat16),
        "teacher_hidden": _single_record_tensor(outputs["student_hidden"]).to(torch.bfloat16),
        "clip_id": str(sample["clip_id"]),
        "t0_us": int(sample["t0_us"]),
    }
    if compute_teacher_vector:
        record["teacher_vector"] = _single_record_tensor(outputs["student_vector"]).to(
            torch.bfloat16
        )
    return record


@torch.no_grad()
def build_on_policy_batch(
    teacher: DistillAlpamayo1_5,
    samples: list[dict[str, Any]],
    *,
    student_model: DistillAlpamayo1_5,
    teacher_processor: Any | None = None,
    student_processor: Any | None = None,
    sample_temperature: float = 0.6,
    sample_top_p: float = 0.98,
    divergence_temperature: float = 1.0,
    max_new_tokens: int = 256,
    compute_teacher_vector: bool = True,
) -> dict[str, Any]:
    """Return a student-ready batch built from stochastic student rollouts."""
    teacher_processor = teacher_processor or helper.get_processor(
        teacher.tokenizer,
        getattr(teacher.config, "vlm_name_or_path", None),
    )
    student_processor = student_processor or helper.get_processor(
        student_model.tokenizer,
        getattr(student_model.config, "vlm_name_or_path", None),
    )
    records = [
        _build_on_policy_targets(
            teacher,
            student_model,
            sample,
            teacher_processor=teacher_processor,
            student_processor=student_processor,
            temperature=sample_temperature,
            top_p=sample_top_p,
            divergence_temperature=divergence_temperature,
            max_new_tokens=max_new_tokens,
            compute_teacher_vector=compute_teacher_vector,
        )
        for sample in samples
    ]
    batch = recursive_collate(records)
    batch["on_policy"] = True
    for key in _FLAT_MULTIMODAL_KEYS:
        values = [
            record["tokenized_data"][key] for record in records if key in record["tokenized_data"]
        ]
        if values:
            batch["tokenized_data"][key] = torch.cat(values, dim=0)
    return helper.to_device(batch, device=next(student_model.parameters()).device)


@torch.no_grad()
def build_online_teacher_batch(
    model: DistillAlpamayo1_5,
    samples: list[dict[str, Any]],
    *,
    student_model: DistillAlpamayo1_5 | None = None,
    teacher_processor: Any | None = None,
    student_processor: Any | None = None,
    top_k: int = 64,
    temperature: float = 0.5,
    max_new_tokens: int = 256,
    compute_teacher_vector: bool = True,
) -> dict[str, Any]:
    """Run the frozen teacher now and return a student-ready device batch."""
    records = []
    for sample in samples:
        record = _build_teacher_targets(
            model,
            sample,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            processor=teacher_processor,
            compute_teacher_vector=compute_teacher_vector,
        )
        if student_model is not None:
            messages = _create_sample_message(sample)
            active_student_processor = student_processor or helper.get_processor(
                student_model.tokenizer,
                getattr(student_model.config, "vlm_name_or_path", None),
            )
            student_prompt = active_student_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = next(student_model.parameters()).device
            student_prompt = helper.to_device(student_prompt, device=device)
            student_prompt_ids = student_model.fuse_traj_tokens(
                student_prompt["input_ids"],
                {
                    "ego_history_xyz": sample["ego_history_xyz"].to(device),
                    "ego_history_rot": sample["ego_history_rot"].to(device),
                },
            )
            prompt_length = int(record["_teacher_prompt_length"])
            teacher_prefix = record["tokenized_data"]["input_ids"][:prompt_length]
            if not torch.equal(student_prompt_ids[0], teacher_prefix.to(device)):
                raise ValueError("teacher and student processed prompt token IDs differ")
            for key, value in student_prompt.items():
                if key not in {"input_ids", "attention_mask"}:
                    record["tokenized_data"][key] = _single_record_tokenized_tensor(key, value)
        records.append(record)
    batch = recursive_collate(records)
    for key in _FLAT_MULTIMODAL_KEYS:
        values = [
            record["tokenized_data"][key] for record in records if key in record["tokenized_data"]
        ]
        if values:
            batch["tokenized_data"][key] = torch.cat(values, dim=0)
    device = next(model.parameters()).device
    return helper.to_device(batch, device=device)
