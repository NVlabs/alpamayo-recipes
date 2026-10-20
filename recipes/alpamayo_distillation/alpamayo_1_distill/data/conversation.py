"""Compose the conversation template for VLM models."""

from typing import Any

import torch

from alpamayo1_distill.vla_constant import SPECIAL_TOKENS
from alpamayo1_distill.data.constants import CAMERA_INDICES_TO_DISPLAY_NAMES


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

    Args:
        config (DataPreprocessConfig): The configuration for the data processing.

    Returns:
        system_prompt (list): The list of system message prompts for the VLA model.
    """
    system_prompt = "You are a driving assistant that generates safe and accurate actions."
    return [{"type": "text", "text": system_prompt}]


def construct_user_prompt(
    components_order: list[str], components_prompt: list[str], generation_mode: bool
) -> list[dict[str, str]]:
    """Construct the input prompt for the VLA  model.

    Args:
        components_order (list[str]): The order of the components.
        components_prompt (list[str]): The prompt of the components.
        generation_mode (bool): Whether to use the generation mode.

    Returns:
        prompt (list): The list of message dictionaries for the VLA model.
    """
    # templates
    template = {
        "cot": "output the chain-of-thought reasoning of the driving process",
        "meta_action": "output meta actions",
        "traj_future": "output the future trajectory",
    }

    # in generation mode, use components_prompt, in training mode, use components_order
    components = components_prompt if generation_mode else components_order

    prompt_components = []
    # remember to preserve the order
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
    for chunk, images_chunk in enumerate(images):
        prev_cam_id = None
        frame_idx = 0
        for i, image in enumerate(images_chunk):
            #print(image.shape)
            cam_id = camera_ids[i].item()
            # Reset frame index when camera ID changes
            if prev_cam_id is not None and cam_id != prev_cam_id:
                frame_idx = 0
            # Add camera name as text before image sequence
            if include_camera_ids and frame_idx == 0:
                    messages.append(
                    {"type": "text", "text": f"{CAMERA_INDICES_TO_DISPLAY_NAMES[cam_id]}: "}
                )
            # Add frame number as text before each image sequence
            if include_frame_nums:
                messages.append({"type": "text", "text": f"frame {frame_idx} "})
            messages.append({"type": "image", "image": image})
            prev_cam_id = cam_id
            frame_idx += 1
    return messages


def construct_traj_history(num_tokens_per_history_traj: int) -> list[dict[str, str]]:
    """Construct the trajectory history prompt for the VLA model.

    Args:
        num_tokens_per_history_traj (int): The number of tokens per history trajectory.

    Returns:
        traj_history_component (list): The list of trajectory history prompts for the VLA model.
    """
    traj_history_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["traj_history_start"],
                end_str=SPECIAL_TOKENS["traj_history_end"],
                padding_str=SPECIAL_TOKENS["traj_history"] * num_tokens_per_history_traj,
            ),
        }
    ]
    return traj_history_component


def construct_route_xy(data: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the route XY prompt for the VLA model."""
    # remove batch dimension and only keep XY
    route_xy = data["route_xy"][0, ..., 0:2]
    # filter out nan values
    valid_mask = ~torch.isnan(route_xy).any(dim=-1)
    route_xy = route_xy[valid_mask].numpy().tolist()
    route_xy_str = [f"[{', '.join([f'{v:.2f}' for v in row])}]" for row in route_xy]
    route_xy_str = ", ".join(route_xy_str)

    route_xy_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str="The route waypoints are: ",
                end_str=". ",
                content_str=route_xy_str,
            ),
        }
    ]
    return route_xy_component


def construct_question(data: dict[str, Any]) -> list[dict[str, str]]:
    """Construct the question prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.

    Returns:
        question_component (list): The list of question prompts for the VLA model.
    """
    question_component = [
        {
            "type": "text",
            "text": get_component_str(
                start_str=SPECIAL_TOKENS["question_start"],
                end_str=SPECIAL_TOKENS["question_end"],
                content_str=data["question"],
            ),
        }
    ]
    return question_component


def construct_traj_future(
    num_tokens_per_future_traj: int, ask_for_component: bool = False
) -> list[dict[str, str]]:
    """Construct the trajectory future prompt for the VLA model.

    Args:
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        traj_future_component (list): The list of trajectory future prompts for the VLA model.
    """
    traj_future_component = [
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
    return traj_future_component


def construct_answer(data: dict[str, Any], ask_for_component: bool = False) -> list[dict[str, str]]:
    """Construct the answer prompt for the VLA model.

    Args:
        data (dict): The data dictionary containing the information to construct the prompt.
        ask_for_component (bool): Whether to ask the model to generate this component.

    Returns:
        answer_component (list): The list of answer prompts for the VLA model.
    """
    # if not asking for answer, we must have the answer in data
    answer = None
    if not ask_for_component:
        assert "answer" in data, "answer not found in data but `answer` in `components_order`"
        answer = data["answer"]

    answer_component = [
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
    return answer_component


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

    cot_component = [
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
    return cot_component


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

    meta_action_component = [
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
    return meta_action_component


def build_conversation(
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
        "content": construct_system_prompt(),
    }
    #print("in build conversation:")
    #print("components_order:", components_order)
    #print("components_prompt:", components_prompt)
    user_messages: dict[str, Any] = {"role": "user", "content": []}
    assistant_messages: dict[str, Any] = {"role": "assistant", "content": []}
    last_component = components_order[-1]
    for component in components_order:
        ask_for_component = generation_mode and component == last_component
        match component:
            # these are user components
            case "prompt":
                user_messages["content"].extend(
                    construct_user_prompt(
                        components_order=components_order,
                        components_prompt=components_prompt,
                        generation_mode=generation_mode,
                    )
                )
            case "image":
                user_messages["content"].extend(
                    construct_image(
                        data=data,
                        include_camera_ids=include_camera_ids,
                        camera_ids=camera_ids,
                        include_frame_nums=include_frame_nums,
                    )
                )
            case "route_xy":
                user_messages["content"].extend(construct_route_xy(data=data))
            case "traj_history":
                user_messages["content"].extend(
                    construct_traj_history(num_tokens_per_history_traj=num_tokens_per_history_traj)
                )
            case "question":
                user_messages["content"].extend(construct_question(data=data))
            # these are assistant components
            case "answer":
                assistant_messages["content"].extend(
                    construct_answer(data=data, ask_for_component=ask_for_component)
                )
            case "cot":
                assistant_messages["content"].extend(
                    construct_cot(data=data, ask_for_component=ask_for_component)
                )
            case "meta_action":
                assistant_messages["content"].extend(
                    construct_meta_action(data=data, ask_for_component=ask_for_component)
                )
            case "traj_future":
                assistant_messages["content"].extend(
                    construct_traj_future(
                        num_tokens_per_future_traj=num_tokens_per_future_traj,
                        ask_for_component=ask_for_component,
                    )
                )

    messages = [system_messages, user_messages, assistant_messages]
    return messages
