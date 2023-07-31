# -*- coding: utf-8 -*-
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from .graphormer_layers import (
    GaussianLayer,
    Graph3DBias,
    GraphAttnBias,
    GraphNodeFeature,
    NonLinear,
    init_params,
)


class GraphNodeFeatureDiff(GraphNodeFeature):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self,
        args,
        num_heads,
        num_atoms,
        num_in_degree,
        num_out_degree,
        hidden_dim,
        n_layers,
        no_2d=False,
        t_timesteps=1010,
    ):
        super(GraphNodeFeatureDiff, self).__init__(
            args,
            num_heads,
            num_atoms,
            num_in_degree,
            num_out_degree,
            hidden_dim,
            n_layers,
            no_2d,
        )
        self.time_embedding = nn.Embedding(t_timesteps, hidden_dim)

    def forward(self, batched_data, t=0, mask_2d=None):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]

        if not self.no_2d:
            degree_feature = self.in_degree_encoder(
                in_degree
            ) + self.out_degree_encoder(out_degree)
            if mask_2d is not None:
                degree_feature = degree_feature * mask_2d[:, None, None]
            node_feature = node_feature + degree_feature

        # # remove node_mask: no need to add mask to 1d and 2d in docking
        if not self.args.ft:
            # mask_embedding = self.atom_mask_embedding.weight.sum(dim=0)
            time_embedding = (
                torch.zeros_like(node_feature).to(node_feature)
                + self.time_embedding(t)[:, None, :]
            )
            # node_mask = batched_data["node_mask"]
            # node_feature[node_mask.bool().squeeze(-1)] = mask_embedding
            # time_embedding = time_embedding.masked_fill(~node_mask.bool(), 0.0)
            node_feature = node_feature + time_embedding

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBiasDiff(GraphAttnBias):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        args,
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
        super(GraphAttnBiasDiff, self).__init__(
            args,
            num_heads,
            num_atoms,
            num_edges,
            num_spatial,
            num_edge_dis,
            hidden_dim,
            edge_type,
            multi_hop_max_dist,
            n_layers,
            no_2d,
        )

    def forward(self, batched_data, mask_2d=None):
        graph_attn_bias = super(GraphAttnBiasDiff, self).forward(
            batched_data, mask_2d, no_mask=True
        )
        return graph_attn_bias


class Graph3DBiasDiff(Graph3DBias):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(
        self,
        args,
        num_heads,
        num_edges,
        n_layers,
        embed_dim,
        num_kernel,
        no_share_rpe=False,
    ):
        super(Graph3DBiasDiff, self).__init__(
            args, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe
        )

    def forward(self, batched_data):
        graph_attn_bias, merge_edge_features, delta_pos = super(
            Graph3DBiasDiff, self
        ).forward(batched_data)
        return graph_attn_bias, merge_edge_features, delta_pos
