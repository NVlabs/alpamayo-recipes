# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Driving questions used for teacher-labeled PhysicalAI-AV VQA."""

from __future__ import annotations


# These prompts cover the LingoQA competencies of description, identification,
# counting, localization, anticipation, attention, action, and justification.
# They are phrased generally so every prompt remains meaningful for an arbitrary
# PhysicalAI-AV clip. The first two also match the public Alpamayo 1.5 VQA
# inference notebook examples.
PAI_VQA_QUESTIONS = (
    "Describe the scene.",
    "What are the key traffic elements visible in this scene and how should they "
    "influence driving behavior?",
    "What is the road layout and what type of driving environment is shown?",
    "Which road users and vehicles are visible, and where are they located?",
    "Are there any pedestrians or cyclists that require attention?",
    "Are there any traffic lights or signs, and what do they indicate?",
    "Are there any junctions, crossings, parked vehicles, or construction zones visible?",
    "What is the ego vehicle currently doing?",
    "What should the ego vehicle do next?",
    "Why is that the appropriate driving action in this scene?",
    "What are the most important hazards or risks in this scene?",
    "What should the driver pay attention to right now?",
    "How are nearby road users likely to affect the ego vehicle's path?",
    "What changes in the scene should the driver anticipate?",
    "Is the current speed and lane position appropriate for the scene?",
    "How do visibility, weather, and road conditions affect safe driving here?",
    "What is the safest driving behavior for this situation, and why?",
    "What is the reasoning process of driving?",
)
