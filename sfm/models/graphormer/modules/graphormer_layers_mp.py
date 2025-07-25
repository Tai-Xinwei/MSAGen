# -*- coding: utf-8 -*-
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from megatron.core import parallel_state, tensor_parallel
from sfm.logging import logger
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.sfmmodule import SFMModule


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeatureMP(SFMModule):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self,
        mp_config,
        args,
        num_heads,
        num_atoms,
        num_in_degree,
        num_out_degree,
        hidden_dim,
        n_layers,
        no_2d=False,
    ):
        super(GraphNodeFeatureMP, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.no_2d = no_2d
        self.args = args

        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.atom_mask_embedding = nn.Embedding(9, hidden_dim, padding_idx=None)

        # padded_num_atoms = (
        #     num_atoms // mp_config.tensor_model_parallel_size + 1
        # ) * mp_config.tensor_model_parallel_size

        # self.atom_encoder = tensor_parallel.VocabParallelEmbedding(
        #     padded_num_atoms,
        #     hidden_dim,
        #     config=mp_config,
        #     init_method=mp_config.init_method,
        # )

        # No need to split, TP will cause unnecessary communication
        # padded_num_in_degree = (
        #     (num_in_degree - 1) // mp_config.tensor_model_parallel_size + 1
        # ) * mp_config.tensor_model_parallel_size

        # self.in_degree_encoder = tensor_parallel.VocabParallelEmbedding(
        #     padded_num_in_degree,
        #     hidden_dim,
        #     config=mp_config,
        #     init_method=mp_config.init_method,
        # )

        # padded_num_out_degree = (
        #     (num_out_degree - 1) // mp_config.tensor_model_parallel_size + 1
        # ) * mp_config.tensor_model_parallel_size

        # self.out_degree_encoder = tensor_parallel.VocabParallelEmbedding(
        #     padded_num_out_degree,
        #     hidden_dim,
        #     config=mp_config,
        #     init_method=mp_config.init_method,
        # )

        # self.graph_token = tensor_parallel.VocabParallelEmbedding(
        #     4, hidden_dim, config=mp_config, init_method=mp_config.init_method
        # )
        # self.atom_mask_embedding = tensor_parallel.VocabParallelEmbedding(
        #     12, hidden_dim, config=mp_config, init_method=mp_config.init_method
        # )

    def forward(self, inputs_tuple: tuple):
        x, in_degree, out_degree, node_mask, mask_2d = inputs_tuple

        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(dim=-2)  # [n_graph, n_node, n_hidden]
        # node_feature = self.atom_encoder(x)[:, :, 0, :]

        if not self.no_2d:
            degree_feature = self.in_degree_encoder(
                in_degree
            ) + self.out_degree_encoder(out_degree)
            if mask_2d is not None:
                degree_feature = degree_feature * mask_2d[:, None, None]
            node_feature = node_feature + degree_feature

        # if self.add_3d: ## should be modified to self.add_3d
        # @ Roger added: mask atom
        if (
            not self.no_2d
            and not self.args.ft
            and not self.args.infer
            and node_mask is not None
        ):
            mask_embedding = self.atom_mask_embedding.weight.sum(dim=0)
            node_feature[
                node_mask.bool().squeeze(-1).contiguous()
            ] = mask_embedding.contiguous()

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        # print("GraphNodeFeaturePipe: ", graph_node_feature.grad)

        return graph_node_feature


class GraphAttnBiasMP(SFMModule):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        mp_config,
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
        super(GraphAttnBiasMP, self).__init__()
        self.num_heads_per_tprank = num_heads  # // mp_config.tensor_model_parallel_size
        self.multi_hop_max_dist = multi_hop_max_dist
        self.no_2d = no_2d
        self.args = args
        self.param_type = mp_config.params_dtype
        self.edge_encoder = nn.Embedding(num_edges + 1, hidden_dim, padding_idx=0)

        # size of num_heads = num_heads * (nlayer + 1)
        self.edge_type = edge_type
        self.edge_hidden_dim = hidden_dim

        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * hidden_dim * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(
            num_spatial + 10, num_heads, padding_idx=0
        )

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # padded_num_edges = (
        #     num_edges // mp_config.tensor_model_parallel_size + 1
        # ) * mp_config.tensor_model_parallel_size

        # self.edge_encoder = tensor_parallel.VocabParallelEmbedding(
        #     padded_num_edges,
        #     hidden_dim,
        #     config=mp_config,
        #     init_method=mp_config.init_method,
        # )

        # if self.edge_type == "multi_hop":
        #     self.edge_dis_encoder = tensor_parallel.VocabParallelEmbedding(
        #         num_edge_dis * hidden_dim * num_heads,
        #         1,
        #         config=mp_config,
        #         init_method=mp_config.init_method,
        #     )

        # self.spatial_pos_encoder = tensor_parallel.RowParallelLinear(
        #     num_heads,
        #     num_spatial + 10,
        #     config=mp_config,
        #     init_method=mp_config.output_layer_init_method,
        #     bias=False,
        #     input_is_parallel=True,
        # )

        # self.graph_token_virtual_distance = tensor_parallel.RowParallelLinear(
        #     num_heads,
        #     1,
        #     config=mp_config,
        #     init_method=mp_config.output_layer_init_method,
        #     bias=False,
        #     input_is_parallel=True,
        # )

    def forward(self, input_tuple: tuple):
        (
            attn_bias,
            spatial_pos,
            x,
            edge_input,
            attn_edge_type,
            node_mask,
            mask_2d,
        ) = input_tuple

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads_per_tprank, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if not self.no_2d:
            if not self.args.ft and not self.args.infer and node_mask is not None:
                # @ Roger added: mask atom
                mask_spatial_pos_value = self.spatial_pos_encoder.weight.shape[0] - 1
                spatial_pos = spatial_pos.masked_fill_(
                    node_mask.squeeze(-1).unsqueeze(1).bool(), mask_spatial_pos_value
                )

            # spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(
            #     0, 3, 1, 2
            # )

            # transfer spatial_pos to one-hot
            spatial_pos_onehot = F.one_hot(
                spatial_pos, num_classes=self.spatial_pos_encoder.weight.shape[0]
            ).to(self.param_type)

            spatial_pos_bias = torch.einsum(
                "abcd,de->abce", spatial_pos_onehot, self.spatial_pos_encoder.weight
            ).permute(0, 3, 1, 2)

            if mask_2d is not None:
                spatial_pos_bias = spatial_pos_bias * mask_2d[:, None, None, None]
            graph_attn_bias[:, :, 1:, 1:] = (
                graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
            )

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(
            1, self.num_heads_per_tprank, 1
        )
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        if not self.no_2d:
            if self.edge_type == "multi_hop":
                spatial_pos_ = spatial_pos.clone()
                spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
                # set 1 to 1, x > 1 to x - 1
                spatial_pos_ = torch.where(
                    spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_
                )
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
                        -1, self.edge_hidden_dim, self.num_heads_per_tprank
                    )[:max_dist, :, :],
                )

                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads_per_tprank
                ).permute(1, 2, 3, 0, 4)

                edge_input = (
                    edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
                ).permute(0, 3, 1, 2)
            else:
                # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
                edge_input = (
                    self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
                )

            if mask_2d is not None:
                edge_input = edge_input * mask_2d[:, None, None, None]

            # @ Roger added: mask atom
            if not self.args.ft and not self.args.infer and node_mask is not None:
                edge_input = edge_input.masked_fill_(
                    node_mask.squeeze(-1)[:, None, None, :].bool(), 0.0
                )

            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class Graph3DBiasMP(SFMModule):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    def __init__(
        self,
        mp_config,
        args,
        num_heads,
        num_edges,
        n_layers,
        embed_dim,
        num_kernel,
        no_share_rpe=False,
    ):
        super(Graph3DBiasMP, self).__init__()
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share_rpe = no_share_rpe
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.args = args

        rpe_heads = (
            self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
        )
        rpe_heads_per_tprank = rpe_heads // mp_config.tensor_model_parallel_size

        self.gbf = GaussianLayerMP(self.num_kernel, num_edges)
        self.gbf_proj = NonLinearMP(mp_config, self.num_kernel, rpe_heads_per_tprank)

        self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)

    def forward(self, input_tuple: tuple):
        pos, x, node_type_edge, node_mask = input_tuple

        padding_mask = x.eq(0).all(dim=-1)
        n_graph, n_node, _ = pos.shape

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-5)

        if not self.args.ft and node_mask is not None:
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
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        if not self.args.ft and node_mask is not None:
            merge_edge_features = merge_edge_features.masked_fill_(
                padding_mask.unsqueeze(-1) + node_mask.bool(), 0.0
            )
        else:
            merge_edge_features = merge_edge_features.masked_fill_(
                padding_mask.unsqueeze(-1), 0.0
            )

        return graph_attn_bias, merge_edge_features, delta_pos


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayerMP(SFMModule):
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


class NonLinearMP(SFMModule):
    def __init__(
        self,
        config,
        input,
        output_size,
        hidden=None,
        enable_expert_tensor_parallelism: bool = False,
        moe: bool = False,
    ):
        super(NonLinearMP, self).__init__()

        if hidden is None:
            hidden = input

        self.gate_proj = tensor_parallel.ColumnParallelLinear(
            input,
            hidden,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

        self.down_proj = self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            hidden,
            output_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            moe=moe,
            enable_expert_tensor_parallelism=enable_expert_tensor_parallelism,
        )

    def forward(self, x):
        x = self.layer1(x)[0]
        x = F.gelu(x)
        x = self.layer2(x)[0]
        return x
