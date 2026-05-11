# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Chat template for the Alpamayo R1 (v1) model family.

Mirrors the original ``alpamayo1_sft`` ``conversation.py``: supports
user components ``prompt``, ``image``, ``traj_history`` and assistant
components ``cot``, ``meta_action``, ``traj_future``. Unknown
components are silently skipped, matching the original ``match`` block.
"""

from typing import Any

import torch

from . import components


class R1ChatTemplate:
    """Compose conversation messages for Alpamayo R1."""

    def build_conversation(
        self,
        data: dict[str, Any],
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        generation_mode: bool,
        include_camera_ids: bool = False,
        camera_ids: torch.Tensor | None = None,
        include_frame_nums: bool = False,
    ) -> list[dict[str, Any]]:
        """Compose the conversation messages for the VLA model.

        Args:
            data (dict): The data dictionary containing the information to construct the prompt.

        Returns:
            messages (list[dict[str, str]]): The list of message dictionaries for the VLA model.
        """
        system_messages: dict[str, Any] = {
            "role": "system",
            "content": components.construct_system_prompt(),
        }
        user_messages: dict[str, Any] = {"role": "user", "content": []}
        assistant_messages: dict[str, Any] = {"role": "assistant", "content": []}
        last_component = components_order[-1]
        for component in components_order:
            ask_for_component = generation_mode and component == last_component
            self._add_component(
                component=component,
                data=data,
                ask_for_component=ask_for_component,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                num_tokens_per_history_traj=num_tokens_per_history_traj,
                num_tokens_per_future_traj=num_tokens_per_future_traj,
                components_order=components_order,
                components_prompt=components_prompt,
                generation_mode=generation_mode,
                include_camera_ids=include_camera_ids,
                camera_ids=camera_ids,
                include_frame_nums=include_frame_nums,
            )

        return [system_messages, user_messages, assistant_messages]

    def _add_component(
        self,
        component: str,
        data: dict[str, Any],
        ask_for_component: bool,
        user_messages: dict[str, Any],
        assistant_messages: dict[str, Any],
        num_tokens_per_history_traj: int,
        num_tokens_per_future_traj: int,
        components_order: list[str],
        components_prompt: list[str],
        generation_mode: bool,
        include_camera_ids: bool,
        camera_ids: torch.Tensor | None,
        include_frame_nums: bool,
    ) -> None:
        match component:
            # these are user components
            case "prompt":
                user_messages["content"].extend(
                    components.construct_user_prompt(
                        components_order=components_order,
                        components_prompt=components_prompt,
                        generation_mode=generation_mode,
                    )
                )
            case "image":
                user_messages["content"].extend(
                    components.construct_image(
                        data=data,
                        include_camera_ids=include_camera_ids,
                        camera_ids=camera_ids,
                        include_frame_nums=include_frame_nums,
                    )
                )
            case "traj_history":
                user_messages["content"].extend(
                    components.construct_traj_history(
                        num_tokens_per_history_traj=num_tokens_per_history_traj
                    )
                )
            # these are assistant components
            case "cot":
                assistant_messages["content"].extend(
                    components.construct_cot(data=data, ask_for_component=ask_for_component)
                )
            case "meta_action":
                assistant_messages["content"].extend(
                    components.construct_meta_action(
                        data=data, ask_for_component=ask_for_component
                    )
                )
            case "traj_future":
                assistant_messages["content"].extend(
                    components.construct_traj_future(
                        num_tokens_per_future_traj=num_tokens_per_future_traj,
                        ask_for_component=ask_for_component,
                    )
                )
