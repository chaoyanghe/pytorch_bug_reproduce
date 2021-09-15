# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright 2018, NVIDIA CORPORATION.
# Copyright 2020, Amazon.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Activations."""

import math
import torch

__all__ = ["gelu", "swish"]


@torch.jit.script
def f_gelu(x):
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)


@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))


@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)


@torch.jit.script
def h_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return h_gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "tanh": torch.nn.functional.tanh,
    "swish": swish,
}
