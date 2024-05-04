# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.models.psm.psm_config import PSMConfig


class PSMBias(nn.Module):
    """
    Class for the invariant encoder bias in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig, key_prefix: str = ""):
        """
        Initialize the PSMBias class.
        """
        super(PSMBias, self).__init__()

        rpe_heads = psm_config.num_attention_heads

        self.gbf = GaussianLayer(psm_config.num_3d_bias_kernel, psm_config.num_edges)
        self.gbf_proj = NonLinear(psm_config.num_3d_bias_kernel, rpe_heads)
        self.pos_embedding_proj = nn.Linear(
            psm_config.num_3d_bias_kernel, psm_config.encoder_embed_dim
        )

        self.psm_config = psm_config
        self.key_prefix = key_prefix

    #     # # make sure attn_bias has gradient
    #     self.pos_embedding_proj.weight.register_hook(self.print_grad)

    # def print_grad(self, grad):
    #     print(torch.max(grad))
    #     return grad

    def forward(
        self,
        batch_data: Dict,
        masked_token_type: torch.Tensor,
        padding_mask: torch.Tensor,
        pbc_expand_batched: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the PSMBias class.
        Args:
            batch_data: Input data for the forward pass.
            masked_token_type: The masked token type [B, L].
            padding_mask: The padding mask [B, L].
            pbc_expand_batched: PBC expanded information
        """

        pos = batch_data[f"{self.key_prefix}pos"]
        n_graph, n_node = pos.size()[:2]

        if pbc_expand_batched is not None:
            expand_pos = torch.cat(
                [pos, pbc_expand_batched[f"{self.key_prefix}expand_pos"]], dim=1
            )
            n_expand_node = expand_pos.size()[1]
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            dist = delta_pos.norm(dim=-1).view(-1, n_node, n_expand_node)
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)

        masked_token_type_i = (
            masked_token_type.unsqueeze(-1).repeat(1, 1, n_node).unsqueeze(-1)
        )
        masked_token_type_j = (
            masked_token_type.unsqueeze(1).repeat(1, n_node, 1).unsqueeze(-1)
        )
        pair_token_type = torch.cat([masked_token_type_i, masked_token_type_j], dim=-1)

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_pair_token_type = torch.gather(
                pair_token_type,
                dim=2,
                index=outcell_index.unsqueeze(1).unsqueeze(-1).repeat(1, n_node, 1, 2),
            )
            pair_token_type = torch.cat(
                [pair_token_type, expand_pair_token_type], dim=2
            )
            pbc_expand_batched["expand_node_type_edge"] = pair_token_type
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
        else:
            local_attention_weight = None

        edge_feature = self.gbf(dist, pair_token_type.long())
        graph_attn_bias = self.gbf_proj(edge_feature)
        pos_embedding_feature = self.pos_embedding_proj(edge_feature)

        if pbc_expand_batched is not None:
            expand_mask = pbc_expand_batched["expand_mask"]
            full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
            graph_attn_bias = graph_attn_bias.masked_fill(
                full_mask.unsqueeze(1).unsqueeze(-1), float("-inf")
            )
            pos_embedding_feature = pos_embedding_feature.masked_fill(
                full_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )
        else:
            graph_attn_bias = graph_attn_bias.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(-1), float("-inf")
            )
            pos_embedding_feature = pos_embedding_feature.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
            )

        graph_attn_bias = graph_attn_bias.masked_fill(
            padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0
        )

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2)

        if local_attention_weight is not None:
            pos_embedding_feature = (
                pos_embedding_feature * local_attention_weight.unsqueeze(-1)
            )
        pos_embedding_feature = pos_embedding_feature.sum(dim=-2)

        return graph_attn_bias, pos_embedding_feature


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
