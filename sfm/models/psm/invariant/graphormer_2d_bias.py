# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from sfm.logging import logger

try:
    import graphormer_preprocess_cuda

    FOUND_CUDA_GRAPHORMER_PREPROCESS = True
except:
    FOUND_CUDA_GRAPHORMER_PREPROCESS = False
    logger.info(
        "graphormer_preprocess_cuda is not found. Using CPU molecule preprocessing for 2D structures."
    )

from sfm.models.psm.psm_config import PSMConfig


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = psm_config.num_attention_heads
        if not psm_config.share_attention_bias:
            self.num_heads *= psm_config.num_encoder_layers + 1
        self.multi_hop_max_dist = psm_config.multi_hop_max_dist
        self.edge_hidden_dim = (
            psm_config.encoder_embed_dim // psm_config.num_attention_heads
        )

        if self.psm_config.use_graphormer_path_edge_feature:
            self.edge_encoder = nn.Embedding(
                psm_config.num_edges + 1, self.edge_hidden_dim, padding_idx=0
            )
            self.edge_dis_encoder = nn.Embedding(
                psm_config.num_edge_dis * self.edge_hidden_dim * self.num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(
            psm_config.num_spatial + 10, self.num_heads, padding_idx=0
        )
        self.graph_token_virtual_distance = nn.Embedding(1, self.num_heads)

        self.psm_config = psm_config

    def forward(self, batched_data):
        attn_bias, node_attr, is_molecule = (
            batched_data["attn_bias"],
            batched_data["node_attr"],
            batched_data["is_molecule"],
        )
        n_graph, n_node = node_attr.size()[:2]

        num_molecules = torch.sum(is_molecule.long())
        if num_molecules > 0:
            if self.psm_config.preprocess_2d_bond_features_with_cuda:
                if not FOUND_CUDA_GRAPHORMER_PREPROCESS:
                    raise ValueError(
                        "graphormer_preprocess_cuda is not installed by preprocess_2d_bond_features_with_cuda is enabled."
                    )
                adj, attn_edge_type = (
                    batched_data["adj"],
                    batched_data["attn_edge_type"],
                )
                spatial_pos = torch.zeros(
                    [n_graph, n_node, n_node], device=attn_bias.device, dtype=torch.long
                )
                floyd_pred = torch.zeros(
                    [n_graph, n_node, n_node], device=attn_bias.device, dtype=torch.long
                )
                graphormer_preprocess_cuda.floyd_warshall_batch(
                    adj.long(),
                    batched_data["num_atoms"],
                    n_node,
                    510,
                    spatial_pos,
                    floyd_pred,
                )
                max_dist = (torch.max(spatial_pos) - 1).cpu().item()
                multi_hop_max_dist = (
                    max_dist
                    if max_dist <= self.multi_hop_max_dist
                    else self.multi_hop_max_dist
                )
                edge_input = torch.zeros(
                    [
                        n_graph,
                        n_node,
                        n_node,
                        multi_hop_max_dist,
                        attn_edge_type.size()[-1],
                    ],
                    device=spatial_pos.device,
                    dtype=torch.long,
                )
                graphormer_preprocess_cuda.gen_edge_input_batch(
                    510,
                    n_node,
                    attn_edge_type.size()[-1],
                    multi_hop_max_dist,
                    batched_data["num_atoms"],
                    floyd_pred,
                    spatial_pos,
                    attn_edge_type,
                    edge_input,
                )
            else:
                spatial_pos, edge_input = (
                    batched_data["spatial_pos"],
                    batched_data["edge_input"],
                )

        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        if num_molecules > 0:
            # spatial pos
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = (
                graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
            )

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if num_molecules > 0 and self.psm_config.use_graphormer_path_edge_feature:
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, n_node * n_node * n_graph, self.edge_hidden_dim
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.edge_hidden_dim, self.num_heads
                )[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)
            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)

            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input

        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
        if self.psm_config.share_attention_bias:
            graph_attn_bias = (
                graph_attn_bias.reshape(
                    n_graph,
                    self.psm_config.num_attention_heads,
                    n_node + 1,
                    n_node + 1,
                )
                .contiguous()
            )
        else:
            graph_attn_bias = (
                graph_attn_bias.reshape(
                    n_graph,
                    self.psm_config.num_encoder_layers + 1,
                    self.psm_config.num_attention_heads,
                    n_node + 1,
                    n_node + 1,
                )
                .permute(1, 0, 2, 3, 4)
                .contiguous()
            )

        return graph_attn_bias
