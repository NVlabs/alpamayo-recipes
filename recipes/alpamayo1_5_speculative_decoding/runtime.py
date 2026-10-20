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

"""One CUDA device per torchrun worker."""

import os
import random
from contextlib import AbstractContextManager

import numpy as np
import torch
import torch.distributed as dist


class Runtime(AbstractContextManager):
    def __init__(self, output, seed):
        self.output = output
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not torch.cuda.is_available() or self.local_rank >= torch.cuda.device_count():
            raise RuntimeError("Each torchrun worker must have a visible CUDA device")
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        random.seed(seed + self.rank)
        np.random.seed((seed + self.rank) % 2**32)
        torch.manual_seed(seed + self.rank)

    def __enter__(self):
        if self.world_size > 1:
            dist.init_process_group("nccl")
        self.output.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self):
        if dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])

    def sum(self, tensor):
        if dist.is_initialized():
            dist.all_reduce(tensor)
        return tensor
