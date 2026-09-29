# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assistant-label masking for Alpamayo 2 Super SFT."""

import numpy as np
import torch
from transformers import AutoTokenizer


def get_assistant_mask(
    tokenizer: AutoTokenizer,
    tokens: torch.Tensor | list[int],
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>",
    role: str = "assistant",
) -> torch.Tensor:
    """Generate a mask covering assistant content and its end token.

    Args:
        tokenizer: Tokenizer used to resolve role and boundary token IDs.
        tokens: One tokenized conversation.
        bos_token: Conversation-turn start token.
        eos_token: Conversation-turn end token.
        role: Role whose response should be supervised.

    Returns:
        A boolean mask with the same shape as ``tokens``.
    """
    start_offset = 3
    end_offset = 1
    np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)

    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    role_id = tokenizer.convert_tokens_to_ids(role)
    start_indices = np.where(np_tokens == bos_token_id)[0]
    end_indices = np.where(np_tokens == eos_token_id)[0]

    masks = np.zeros_like(np_tokens, dtype=bool)
    if len(start_indices) != len(end_indices):
        raise ValueError(
            f"Number of bos ({len(start_indices)}) does not match eos ({len(end_indices)})"
        )
    for start, end in zip(start_indices, end_indices):
        if start + 1 < len(np_tokens) and np_tokens[start + 1] == role_id:
            masks[start + start_offset : end + end_offset] = True

    return torch.from_numpy(masks)
