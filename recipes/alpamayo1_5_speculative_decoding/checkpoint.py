# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load the same VLM from Stage-1 SFT or full Alpamayo checkpoints."""

# SFT stores the same trajectory tokenizer settings under the R1 package name.
_PUBLIC_MODULES = {
    "alpamayo_r1.models.delta_tokenizer.": "alpamayo1_5.models.delta_tokenizer.",
    "alpamayo_r1.models.action_in_proj.": "alpamayo1_5.models.action_in_proj.",
    "alpamayo_r1.action_space.": "alpamayo1_5.action_space.",
    "alpamayo_r1.diffusion.": "alpamayo1_5.diffusion.",
}

# These belong to the trajectory expert, not the target VLM being distilled.
_EXPERT_PREFIXES = (
    "expert.",
    "action_space.",
    "diffusion.",
    "action_in_proj.",
    "action_out_proj.",
    "expert_hist_traj_tokenizer.",
    "hist_traj_embed.",
    "hidden_proj.",
)


def public_config(value):
    """Remap package metadata without changing dimensions, tokens or weights."""
    if isinstance(value, list):
        return [public_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: public_config(item) for key, item in value.items()}
    target = result.get("_target_")
    if isinstance(target, str):
        for old, new in _PUBLIC_MODULES.items():
            if target.startswith(old):
                result["_target_"] = new + target[len(old) :]
                break
    return result


def check_loading(audit):
    """Never accept missing/mismatched VLM weights or arbitrary extra modules."""
    extra = audit.get("unexpected_keys", [])
    unknown = [key for key in extra if not key.startswith(_EXPERT_PREFIXES)]
    if unknown or any(audit.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint does not contain a complete compatible target VLM: {audit}")
    return sorted(extra)
