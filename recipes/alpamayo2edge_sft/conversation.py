# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training conversations using the Edge inference package's image conventions."""

from typing import Any

import torch
from alpamayo2_edge.chat_template import (
    CAMERA_DISPLAY_NAMES,
    DEFAULT_SYSTEM_PROMPT,
    relative_time_labels,
)

_TASKS = {
    "cot": "output the chain-of-thought reasoning of the driving process",
    "cot_auto_labeling": "output a comprehensive analysis of the driving scene in JSON format",
    "meta_action": "output meta actions",
    "traj_future": "output the future trajectory",
}


def _text(value: str) -> dict[str, str]:
    if not isinstance(value, str):
        raise TypeError("Conversation text components must contain strings")
    return {"type": "text", "text": value}


def build_conversation(
    data: dict[str, Any],
    num_tokens_per_history_traj: int,
    num_tokens_per_future_traj: int,
    components_order: list[str],
    components_prompt: list[str] | None,
    generation_mode: bool,
    include_camera_ids: bool = False,
    token_layout: str = "camera_ts",
    frame_label: str = "relative_time",
    encode_as_video: bool = False,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Add supervised assistant targets to checkpoint-native Edge input formatting.

    The inference package composes a navigation/CoT generation prompt only. SFT
    also needs trajectory-only prompts and complete assistant responses. Keep
    that task assembly here while sharing its camera labels and time formatting.
    ``generation_mode`` is handled by the processor: trajectory-only prompts
    end at future-start; CoT generation omits assistant targets before this
    builder runs and starts a fresh assistant turn in the native chat template.
    """
    del generation_mode
    if token_layout != "camera_ts" or encode_as_video:
        raise ValueError("Edge SFT conversations require camera_ts layout and individual images")
    if frame_label not in {"relative_time", "frame_num", "none"}:
        raise ValueError(f"Unsupported Edge frame label: {frame_label!r}")
    if "prompt" in components_order:
        boundary = components_order.index("prompt") + 1
    elif "question" in components_order:
        boundary = components_order.index("question") + 1
    else:
        raise ValueError("Conversation components require prompt or question")

    def image_content() -> list[dict[str, Any]]:
        images = data["image_frames"]
        cameras = data["camera_indices"]
        if len(images) != len(cameras):
            raise ValueError("Each image must have one camera ID")
        labels = None
        if frame_label == "relative_time":
            timestamps = data["absolute_timestamps"]
            if timestamps.dtype != torch.int64:
                raise TypeError("absolute_timestamps must use int64 microseconds")
            if timestamps.shape not in ((len(images),), (len(images), 1)):
                raise ValueError("Each image must have one absolute timestamp")
            labels = relative_time_labels(timestamps)
        content = []
        previous_camera = None
        frame_index = 0
        for index, (image, camera) in enumerate(zip(images, cameras.tolist(), strict=True)):
            if camera != previous_camera:
                frame_index = 0
                if include_camera_ids:
                    content.append(_text(CAMERA_DISPLAY_NAMES[camera] + ": "))
            if labels is not None:
                content.append(_text(labels[index]))
            elif frame_label == "frame_num":
                content.append(_text(f"frame {frame_index} "))
            content.append({"type": "image", "image": image, "camera_id": int(camera)})
            previous_camera = camera
            frame_index += 1
        return content

    def component_content(component: str) -> list[dict[str, Any]]:
        if component == "image":
            return image_content()
        if component in {"traj_history", "traj_future"}:
            count = (
                num_tokens_per_history_traj
                if component == "traj_history"
                else num_tokens_per_future_traj
            )
            return [
                _text(
                    f"<|{component}_start|>"
                    + f"<|{component}|>" * count
                    + f"<|{component}_end|>"
                )
            ]
        if component == "prompt":
            if not components_prompt or any(item not in _TASKS for item in components_prompt):
                raise ValueError("components_prompt must contain supported nonempty task names")
            task = ", then ".join(_TASKS[c] for c in components_prompt)
            return [_text("Your task: " + task + ".")]
        if component == "nav_instruction":
            nav = data["nav_text"]
            if not isinstance(nav, (list, tuple)) or not nav or not isinstance(nav[0], str):
                raise ValueError("nav_text must contain a navigation string")
            if not nav[0]:
                raise ValueError("nav_text must not be empty")
            return [_text("Navigation input is: " + nav[0])]
        if component == "cot_history":
            history = data[component]
            if isinstance(history, str):
                return [_text(history)]
            lines = ["CoT history:"]
            for entry in history:
                lines.extend([f"<t={entry['delta_us'] / 1_000_000:.1f}s>", entry["cot"]])
            return [_text("\n".join(lines) if history else "CoT history:\n(none)")]
        if component in {"question", "answer", "cot", "cot_auto_labeling", "meta_action"}:
            key = "meta_action_strings" if component == "meta_action" else component
            return [_text(data[key])]
        raise ValueError(f"Unsupported Edge SFT conversation component: {component!r}")

    def message(role: str, components: list[str]) -> dict[str, Any]:
        content = []
        for component in components:
            parts = component_content(component)
            if parts:
                if content:
                    content.append(_text("\n"))
                content.extend(parts)
        return {"role": role, "content": content}

    effective_system = data.get("system_prompt")
    if effective_system is None:
        effective_system = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": [_text(str(effective_system))]},
        message("user", components_order[:boundary]),
    ]
    assistant = message("assistant", components_order[boundary:])
    if assistant["content"]:
        messages.append(assistant)
    return messages
