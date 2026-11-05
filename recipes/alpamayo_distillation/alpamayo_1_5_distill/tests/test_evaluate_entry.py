# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_evaluate_entry_uses_shared_validation() -> None:
    evaluate_path = Path(__file__).parents[1] / "evaluate.py"
    module = ast.parse(evaluate_path.read_text(encoding="utf-8"))
    calls = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_validation" in calls
    assert "load_config" in calls
