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

"""Per-component builders for VLA chat templates.

Each ``construct_*`` returns a list of message-content dicts for one
conversation component. These are pure helpers shared across model
versions; per-version ordering and dispatch live in ``r1.py`` /
``r1_5.py``.
"""

from typing import Any

import torch

import alpamayo.common.constants as constants
from alpamayo_r1.models.base_model import SPECIAL_TOKENS


def get_component_str(
    start_str: str,
    end_str: str,
    content_str: str | None = None,
    padding_str: str | None = None,
    ask_for_component: bool = False,
) -> str:
    # always add the start string
    component_str = [start_str]

    # if ask this component, we only add start string
    if not ask_for_component:
        assert (content_str is None) != (padding_str is None), (
            "Exactly one of content_str or padding_str must be provided"
        )
        if content_str is not None:
            # add content string directly
            component_str.append(content_str)
        elif padding_str is not None:
            # use padding string as placeholder
            component_str.append(padding_str)
        component_str.append(end_str)
    return "".join(component_str)


def construct_system_prompt() -> list[dict[str, str]]:
    """Construct the system message for the VLA model.

    Returns:
        system_prompt (list): The list of system message prompts for the VLA model.
    """
    system_prompt = "You are a driving assistant that generates safe and accurate actions."
    return [{"type": "text", "text": system_prompt}]


def construct_user_prompt(
    components_order: list[str], components_prompt: list[str], generation_mode: bool
) -> list[dict[str, str]]:
    """Construct the input prompt for the VLA model.

    Args:
        components_order (list[str]): The order of the components.
        components_prompt (list[str]): The prompt of the components.
        generation_mode (bool): Whether to use the generation mode.

    Returns:
        prompt (list): The list of message dictionaries for the VLA model.
    """
    template = {
        "cot": "output the chain-of-thought reasoning of the driving process",
        "meta_action": "output meta actions",
        "traj_future": "output the future trajectory",
    }

    # in generation mode, use components_prompt, in training mode, use components_order
    components = components_prompt if generation_mode else components_order

    prompt_components = []
    for component in components:
        if component in template.keys():
            prompt_components.append(template[component])

    prompt = ", then ".join(prompt_components) + "."
    return [{"type": "text", "text": prompt}]


def construct_image(
    data: dict[str, Any],
    include_camera_ids: bool,
    camera_ids: torch.Tensor,
    include_frame_nums: bool,
) -> list[dict[str, str]]:
    """Construct the image description prompt for the VLA model.

    Args:
        data (dict): The images with shape (num_chunk, num_frames, 3, H_new, W_new).
        include_camera_ids (bool): Whether to include camera IDs as text before images.
        camera_ids (torch.Tensor): The sorted camera IDs of the images.
        include_frame_nums (bool): Whether to include frame numbers as text before images.

    Returns:
        image_prompt (list): The list of image description prompts for the VLA model.
    """
    # assert camera_ids is in ascending order
    assert torch.all(camera_ids == torch.sort(camera_ids)[0])

    images = data["image_frames"]
    messages = []
    for i, view_images in enumerate(images):
        cam_id = camera_ids[i].item()
        # Add camera name as text before image sequence
        if include_camera_ids:
            messages.append(
                {"type": "text", "text": f"{constants.CAMERA_INDICES_TO_DISPLAY_NAMES[cam_id]}: "}
            )
        for frame_idx, frame_im in enumerate(view_images):
            # Add frame number as text before each image sequence
            if include_frame_nums:
                messages.append({"type": "text", "text": f"frame {frame_idx} "})
            messages.append({"type": "image", "image": frame_im})

    return messages


def construct_traj_history(num_tokens_per_history_traj: int) -> list[dict[str, str]]:
    """Construct the trajectory history prompt for the VLA model.

    Args:
        num_tokens_per_history_traj (int): The number of tokens per history trajectory.

    Returns:
        traj_history_component (list): The list of trajectory history prompts for the VLA model.
    """
    return [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["traj_history_start"],
                end_str=SPECIAL_TOKENS["traj_history_end"],
                padding_str=SPECIAL_TOKENS["traj_history"] * num_tokens_per_history_traj,
            ),
        }
    ]


def construct_traj_future(
    num_tokens_per_future_traj: int, ask_for_component: bool = False
) -> list[dict[str, str]]:
    """Construct the trajectory future prompt for the VLA model.

    Args:
        num_tokens_per_future_traj (int): The number of tokens per future trajectory.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        traj_future_component (list): The list of trajectory future prompts for the VLA model.
    """
    return [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["traj_future_start"],
                end_str=SPECIAL_TOKENS["traj_future_end"],
                padding_str=SPECIAL_TOKENS["traj_future"] * num_tokens_per_future_traj,
                ask_for_component=ask_for_component,
            ),
        }
    ]


def construct_cot(data: dict[str, Any], ask_for_component: bool = False) -> list[dict[str, str]]:
    """Construct the chain-of-thought prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        cot_component (list): The list of chain-of-thought prompts for the VLA model.
    """
    # if not asking for cot, we must have the cot in data
    cot = None
    if not ask_for_component:
        assert "cot" in data, "cot not found in data but `cot` in `components_order`"
        cot = data["cot"]

    return [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["cot_start"],
                end_str=SPECIAL_TOKENS["cot_end"],
                content_str=cot,
                ask_for_component=ask_for_component,
            ),
        }
    ]


def construct_meta_action(
    data: dict[str, Any], ask_for_component: bool = False
) -> list[dict[str, str]]:
    """Construct the meta action prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        meta_action_component (list): The list of meta action prompts for the VLA model.
    """
    # if not asking for meta_action, we must have the meta_action in data
    meta_action = None
    if not ask_for_component:
        assert "meta_action_strings" in data, (
            "meta_action not found in data but `meta_action` in `components_order`"
        )
        meta_action = data["meta_action_strings"]

    return [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["meta_action_start"],
                end_str=SPECIAL_TOKENS["meta_action_end"],
                content_str=meta_action,
                ask_for_component=ask_for_component,
            ),
        }
    ]


def construct_route(data: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the navigation route component for the VLA model.

    Wraps a navigation instruction (e.g. ``"Turn left onto Main St in 40m"``)
    in ``<|route_start|>...<|route_end|>`` tokens. When ``nav_text`` is absent
    from data, returns an empty list so the component is silently skipped.

    Args:
        data: Data dictionary; expects an optional ``"nav_text"`` key.

    Returns:
        List of message content dicts (empty if no nav text).
    """
    nav_text = data.get("nav_text")
    if nav_text is None:
        return []
    return [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["route_start"],
                end_str=SPECIAL_TOKENS["route_end"],
                content_str=nav_text,
            ),
        }
    ]


def construct_question(
    data: dict[str, Any], ask_for_component: bool = False
) -> list[dict[str, str]]:
    """Construct the question component for VQA training.

    Args:
        data: Data dictionary; expects a ``"question"`` key when not asking.
        ask_for_component: If True, only emit the start token (generation mode).

    Returns:
        List of message content dicts.
    """
    question = None
    if not ask_for_component:
        assert "question" in data, "question not found in data but `question` in `components_order`"
        question = data["question"]

    return [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["question_start"],
                end_str=SPECIAL_TOKENS["question_end"],
                content_str=question,
                ask_for_component=ask_for_component,
            ),
        }
    ]


def construct_answer(data: dict[str, Any], ask_for_component: bool = False) -> list[dict[str, str]]:
    """Construct the answer component for VQA training.

    Args:
        data: Data dictionary; expects an ``"answer"`` key when not asking.
        ask_for_component: If True, only emit the start token (generation mode).

    Returns:
        List of message content dicts.
    """
    answer = None
    if not ask_for_component:
        assert "answer" in data, "answer not found in data but `answer` in `components_order`"
        answer = data["answer"]

    return [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["answer_start"],
                end_str=SPECIAL_TOKENS["answer_end"],
                content_str=answer,
                ask_for_component=ask_for_component,
            ),
        }
    ]
