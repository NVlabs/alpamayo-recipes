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

import torch
from alpamayo1_5_distill.data.dataset import recursive_collate


def test_recursive_collate_pads_nested_variable_tensors():
    batch = [
        {"tokens": torch.tensor([1, 2]), "nested": {"features": torch.ones(2, 3)}},
        {"tokens": torch.tensor([3]), "nested": {"features": torch.full((1, 2), 2.0)}},
    ]

    result = recursive_collate(batch)

    torch.testing.assert_close(result["tokens"], torch.tensor([[1, 2], [3, 0]]))
    torch.testing.assert_close(result["tokens_mask"], torch.tensor([[True, True], [True, False]]))
    assert result["nested"]["features"].shape == (2, 2, 3)
    assert result["nested"]["features_mask"].sum().item() == 8
    assert result["nested"]["features"][1, 0, 0].item() == 2.0
    assert result["nested"]["features"][1, 1, 2].item() == 0.0
