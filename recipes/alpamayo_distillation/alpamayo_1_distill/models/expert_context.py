"""Utilities for aligning the diffusion expert with a generated VLM prefix."""

from __future__ import annotations

import torch


def extend_prompt_attention_mask(
    prompt_attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    prompt_seq_len: int,
    total_seq_len: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Extend a prompt mask exactly as autoregressive generation does.

    Hugging Face generation appends a visible attention-mask slot after every decode
    step, including slots that later hold forced padding for an already-finished row.
    Re-forwarding generated sequences must preserve that behavior so Qwen's
    ``rope_deltas`` remain anchored to the original prompt.
    """
    if batch_size <= 0 or prompt_seq_len <= 0 or total_seq_len < prompt_seq_len:
        raise ValueError(
            "invalid generation attention-mask dimensions: "
            f"batch_size={batch_size}, prompt_seq_len={prompt_seq_len}, "
            f"total_seq_len={total_seq_len}"
        )

    if prompt_attention_mask is None:
        prompt_valid = torch.ones(
            (batch_size, prompt_seq_len), dtype=torch.long, device=device
        )
    else:
        if prompt_attention_mask.shape != (batch_size, prompt_seq_len):
            raise ValueError(
                "prompt_attention_mask must have shape "
                f"[{batch_size}, {prompt_seq_len}], got {prompt_attention_mask.shape}"
            )
        prompt_valid = prompt_attention_mask.to(device=device)

    generated_valid = torch.ones(
        (batch_size, total_seq_len - prompt_seq_len),
        dtype=prompt_valid.dtype,
        device=device,
    )
    return torch.cat((prompt_valid, generated_valid), dim=1)


def find_prompt_end_offsets(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    context: str = "expert prompt",
) -> torch.LongTensor:
    """Return the position immediately after each prompt's last valid token."""
    if input_ids.ndim != 2:
        raise ValueError(f"{context}: expected input_ids with shape [B, L], got {input_ids.shape}")

    batch_size, seq_len = input_ids.shape
    if seq_len == 0:
        raise ValueError(f"{context}: prompt cannot be empty")
    if attention_mask is None:
        return torch.full(
            (batch_size,), seq_len, dtype=torch.long, device=input_ids.device
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"{context}: attention_mask must have shape {input_ids.shape}, "
            f"got {attention_mask.shape}"
        )

    valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    has_token = valid.any(dim=1)
    if not bool(has_token.all()):
        empty = (~has_token).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        raise ValueError(f"{context}: no valid prompt tokens for batch indices {empty}")

    positions = torch.arange(1, seq_len + 1, device=input_ids.device)
    return positions.unsqueeze(0).masked_fill(~valid, 0).amax(dim=1)


def find_future_start_offsets(
    sequences: torch.Tensor,
    future_start_id: int,
    *,
    search_start: int = 0,
    cache_seq_len: int | None = None,
    require_trailing_token: bool = False,
    context: str = "expert context",
) -> torch.LongTensor:
    """Return the position immediately after the first generated future-start marker.

    The marker is mandatory: without it there is no semantically valid boundary between
    reasoning tokens and diffusion-expert tokens.  When ``cache_seq_len`` is supplied, the
    marker must also already be represented in the KV cache.
    """
    if sequences.ndim != 2:
        raise ValueError(f"{context}: expected sequences with shape [B, L], got {sequences.shape}")
    if not 0 <= search_start < sequences.shape[1]:
        raise ValueError(
            f"{context}: search_start={search_start} is outside sequence length "
            f"{sequences.shape[1]}"
        )

    marker_mask = sequences[:, search_start:].eq(future_start_id)
    has_marker = marker_mask.any(dim=1)
    if not bool(has_marker.all()):
        missing = (~has_marker).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        raise ValueError(
            f"{context}: missing <traj_future_start> in generated suffix for batch indices "
            f"{missing}; increase the reasoning generation limit or reject these samples"
        )

    marker_positions = marker_mask.to(torch.int64).argmax(dim=1) + search_start
    offsets = marker_positions + 1

    if require_trailing_token:
        marker_at_end = offsets >= sequences.shape[1]
        if bool(marker_at_end.any()):
            indices = marker_at_end.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            raise ValueError(
                f"{context}: <traj_future_start> was emitted in the final available slot "
                f"for batch indices {indices}; reserve one trailing cache-flush token"
            )

    if cache_seq_len is not None:
        if cache_seq_len < 0:
            raise ValueError(f"{context}: cache_seq_len must be non-negative, got {cache_seq_len}")
        uncached = offsets > cache_seq_len
        if bool(uncached.any()):
            indices = uncached.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            bad_offsets = offsets[uncached].detach().cpu().tolist()
            raise ValueError(
                f"{context}: <traj_future_start> is not yet present in the KV cache for "
                f"batch indices {indices} (offsets={bad_offsets}, cache_seq_len={cache_seq_len}); "
                "reserve one generation step after the marker"
            )

    return offsets


def append_suffix_after_offsets(
    sequences: torch.Tensor,
    offsets: torch.LongTensor,
    suffix_tokens: torch.Tensor,
    pad_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append a per-row suffix immediately after each row's offset.

    The returned tensors are padded to ``max(offsets) + suffix_len``. This is
    useful when rows emit a boundary token at different positions and a
    teacher-forced target should begin right after each boundary.
    """
    if sequences.ndim != 2:
        raise ValueError(f"expected sequences with shape [B, L], got {sequences.shape}")
    if suffix_tokens.ndim != 2:
        raise ValueError(f"expected suffix_tokens with shape [B, S], got {suffix_tokens.shape}")
    if offsets.ndim != 1:
        raise ValueError(f"expected offsets with shape [B], got {offsets.shape}")

    batch_size, seq_len = sequences.shape
    if suffix_tokens.shape[0] != batch_size or offsets.shape[0] != batch_size:
        raise ValueError(
            "batch size mismatch: "
            f"sequences={batch_size}, offsets={offsets.shape[0]}, "
            f"suffix_tokens={suffix_tokens.shape[0]}"
        )
    if suffix_tokens.shape[1] <= 0:
        raise ValueError("suffix_tokens must contain at least one token")
    if bool(((offsets < 0) | (offsets > seq_len)).any()):
        raise ValueError(
            f"offsets must be within [0, {seq_len}], got "
            f"{offsets.detach().cpu().tolist()}"
        )
    if attention_mask is not None and attention_mask.shape != sequences.shape:
        raise ValueError(
            f"attention_mask must have shape {sequences.shape}, got {attention_mask.shape}"
        )

    suffix_tokens = suffix_tokens.to(device=sequences.device, dtype=sequences.dtype)
    offsets = offsets.to(device=sequences.device)

    max_offset = int(offsets.max().item())
    suffix_len = suffix_tokens.shape[1]
    total_len = max_offset + suffix_len

    extended = sequences.new_full((batch_size, total_len), pad_token_id)
    mask_dtype = attention_mask.dtype if attention_mask is not None else torch.long
    extended_attention_mask = torch.zeros(
        (batch_size, total_len),
        dtype=mask_dtype,
        device=sequences.device,
    )
    suffix_mask = torch.zeros(
        (batch_size, total_len),
        dtype=torch.bool,
        device=sequences.device,
    )

    for row in range(batch_size):
        offset = int(offsets[row].item())
        extended[row, :offset] = sequences[row, :offset]
        if attention_mask is None:
            extended_attention_mask[row, :offset] = 1
        else:
            extended_attention_mask[row, :offset] = attention_mask[row, :offset]

        suffix_start = offset
        suffix_end = suffix_start + suffix_len
        extended[row, suffix_start:suffix_end] = suffix_tokens[row]
        extended_attention_mask[row, suffix_start:suffix_end] = 1
        suffix_mask[row, suffix_start:suffix_end] = True

    return extended, extended_attention_mask, suffix_mask


def build_expert_attention_mask(
    offsets: torch.LongTensor,
    *,
    cache_seq_len: int,
    query_seq_len: int,
    attention_implementation: str | None,
    cache_valid_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build a backend-compatible non-causal mask for expert queries.

    FlashAttention consumes a 2-D boolean key-padding mask (``True`` means visible).
    Eager/SDPA consume a 4-D additive mask.  In both representations, cached keys at or
    after each sample's future-start offset are hidden and all newly supplied expert keys
    remain visible.
    """
    if offsets.ndim != 1:
        raise ValueError(f"expected offsets with shape [B], got {offsets.shape}")
    if cache_seq_len < 0 or query_seq_len <= 0:
        raise ValueError(
            f"invalid expert mask lengths: cache_seq_len={cache_seq_len}, "
            f"query_seq_len={query_seq_len}"
        )
    if bool(((offsets < 0) | (offsets > cache_seq_len)).any()):
        raise ValueError(
            f"expert offsets must be within [0, {cache_seq_len}], got "
            f"{offsets.detach().cpu().tolist()}"
        )

    batch_size = offsets.shape[0]
    cache_positions = torch.arange(cache_seq_len, device=offsets.device)
    visible_cache = cache_positions.unsqueeze(0) < offsets.unsqueeze(1)

    if cache_valid_mask is not None:
        if cache_valid_mask.shape != (batch_size, cache_seq_len):
            raise ValueError(
                "cache_valid_mask must have shape "
                f"[{batch_size}, {cache_seq_len}], got {cache_valid_mask.shape}"
            )
        visible_cache &= cache_valid_mask.to(device=offsets.device, dtype=torch.bool)

    visible_queries = torch.ones(
        (batch_size, query_seq_len), dtype=torch.bool, device=offsets.device
    )
    key_padding_mask = torch.cat((visible_cache, visible_queries), dim=1)

    if (attention_implementation or "").startswith("flash_attention"):
        return key_padding_mask

    if not dtype.is_floating_point:
        raise ValueError(f"additive expert attention mask requires floating dtype, got {dtype}")
    additive_mask = torch.zeros(
        (batch_size, 1, query_seq_len, cache_seq_len + query_seq_len),
        dtype=dtype,
        device=offsets.device,
    )
    return additive_mask.masked_fill(
        ~key_padding_mask[:, None, None, :], torch.finfo(dtype).min
    )


def build_expert_position_ids(
    offsets: torch.LongTensor,
    rope_deltas: torch.Tensor,
    *,
    query_seq_len: int,
    device: torch.device | str | None = None,
) -> torch.LongTensor:
    """Build the shared Qwen MRoPE positions used by training and inference."""
    if offsets.ndim != 1:
        raise ValueError(f"expected offsets with shape [B], got {offsets.shape}")
    if rope_deltas.ndim == 1:
        rope_deltas = rope_deltas.unsqueeze(1)
    if rope_deltas.shape != (offsets.shape[0], 1):
        raise ValueError(
            f"expected rope_deltas with shape [{offsets.shape[0]}, 1], "
            f"got {rope_deltas.shape}"
        )
    if query_seq_len <= 0:
        raise ValueError(f"query_seq_len must be positive, got {query_seq_len}")

    target_device = offsets.device if device is None else torch.device(device)
    base_positions = torch.arange(query_seq_len, device=target_device)
    base_positions = base_positions.view(1, 1, -1).expand(3, offsets.shape[0], -1).clone()
    delta = rope_deltas.to(target_device) + offsets.to(target_device).unsqueeze(1)
    return base_positions + delta
