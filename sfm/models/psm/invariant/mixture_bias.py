# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSMBias(nn.Module):
    """
    Class for the invariant encoder bias in the PSM model.
    """

    def __init__(self, args, psm_config):
        """
        Initialize the PSMBias class.
        """
        super(PSMBias, self).__init__()

        rpe_heads = psm_config.num_attention_heads * psm_config.num_encoder_layers

        self.gbf = GaussianLayer(
            psm_config.num_3d_bias_kernel, psm_config.num_attention_heads
        )
        self.gbf_proj = NonLinear(psm_config.num_3d_bias_kernel, rpe_heads)

        if psm_config.num_3d_bias_kernel != psm_config.embedding_dim:
            self.edge_proj = nn.Linear(
                psm_config.num_3d_bias_kernel, psm_config.embedding_dim
            )
        else:
            self.edge_proj = None

        self.mask_bias = nn.Embedding(
            1, psm_config.num_attention_heads, padding_idx=None
        )

    def forward(
        self, batch_data: Dict, masked_token_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the PSMBias class.
        """

        pass


@torch.jit.script
def gaussian(x, mean, std):
    pi = torch.pi
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x
