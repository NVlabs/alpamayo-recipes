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

"""Chat template for the Alpamayo R1.5 model family.

Extends :class:`R1ChatTemplate` with three additional components used
for navigation-conditioned and VQA training: ``route``, ``question``,
``answer``.
"""

from typing import Any

import torch

from . import components
from .r1 import R1ChatTemplate


class R1_5ChatTemplate(R1ChatTemplate):
    """Compose conversation messages for Alpamayo R1.5."""

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
            case "route":
                user_messages["content"].extend(components.construct_route(data=data))
            case "question":
                user_messages["content"].extend(
                    components.construct_question(
                        data=data, ask_for_component=ask_for_component
                    )
                )
            case "answer":
                assistant_messages["content"].extend(
                    components.construct_answer(data=data, ask_for_component=ask_for_component)
                )
            case _:
                # delegate to v1 dispatch (prompt, image, traj_history, cot, meta_action, traj_future);
                # unknown components are silently skipped there
                super()._add_component(
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
