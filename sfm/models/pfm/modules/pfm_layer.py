# -*- coding: utf-8 -*-
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.FairseqDropout import FairseqDropout
from torch import Tensor

# from fairseq import utils


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class ResidueFeature(nn.Module):
    """
    Compute residule features, three parts are included
    1. learnable embedding of residue type
    2. Prior of residue features
    3. learnable embedding of angles
    """

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_in_degree,
        num_out_degree,
        hidden_dim,
        n_layers,
        no_2d=False,
    ):
        super(ResidueFeature, self).__init__()
        # self.num_heads = num_heads
        # self.num_atoms = num_atoms
        # self.no_2d = no_2d

        # # 1 for graph token
        # self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        # self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        # self.out_degree_encoder = nn.Embedding(
        #     num_out_degree, hidden_dim, padding_idx=0
        # )

        # self.graph_token = nn.Embedding(1, hidden_dim)

        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, mask_2d=None):
        pass
        # x, in_degree, out_degree = (
        #     batched_data["x"],
        #     batched_data["in_degree"],
        #     batched_data["out_degree"],
        # )
        # n_graph, n_node = x.size()[:2]

        # # node feauture + graph token
        # node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        # if not self.no_2d:
        #     degree_feature = self.in_degree_encoder(
        #         in_degree
        #     ) + self.out_degree_encoder(out_degree)
        #     if mask_2d is not None:
        #         degree_feature = degree_feature * mask_2d[:, None, None]
        #     node_feature = node_feature + degree_feature

        # graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        # graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        # return graph_node_feature


class Edge3DEmbedding(nn.Module):
    def __init__(self, num_edges, embed_dim, num_kernel):
        super(Graph3DBias, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        self.gbf = GaussianLayer(self.num_kernel, num_edges)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

    def forward(self, pos, node_type_edge, padding_mask, node_mask):
        n_graph, n_node, _ = pos.shape

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        if node_mask is not None:
            node_mask_i = node_mask.unsqueeze(-2).repeat(1, 1, n_node, 1)
            node_mask_j = node_mask.unsqueeze(1).repeat(1, n_node, 1, 1)
            new_node_mask = torch.cat([node_mask_i, node_mask_j], dim=-1).bool()
            node_type_edge = node_type_edge.masked_fill(new_node_mask, 0).to(
                node_type_edge
            )

        edge_feature = self.gbf(
            dist,
            torch.zeros_like(dist).long()
            if node_type_edge is None
            else node_type_edge.long(),
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        if node_mask is not None:
            merge_edge_features = merge_edge_features.masked_fill_(
                padding_mask.unsqueeze(-1) + node_mask.bool(), 0.0
            )
        else:
            merge_edge_features = merge_edge_features.masked_fill_(
                padding_mask.unsqueeze(-1), 0.0
            )

        return edge_feature, merge_edge_features, delta_pos


class Graph2DBias(nn.Module):
    """
    Compute attention bias for each head.
    Consider to include MSA here
    """

    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
        no_2d=False,
    ):
        super(Graph3DBias, self).__init__()

    def forward(self, batched_data, mask_2d=None):
        pass


class Graph3DBias(nn.Module):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(self, num_heads, num_kernel):
        super(Graph3DBias, self).__init__()
        self.num_heads = num_heads
        self.num_kernel = num_kernel

        self.gbf_proj = NonLinear(self.num_kernel, self.num_heads)

    def forward(self, padding_mask, edge_feature):
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        return graph_attn_bias


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
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
