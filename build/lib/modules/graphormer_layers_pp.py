import math
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .get_activation_fn import get_activation_fn
from .graphormer_layers import (
    Graph3DBias,
    GraphAttnBias,
    GraphNodeFeature,
    NodeTaskHead,
)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeaturePipe(GraphNodeFeature):
    """
    Compute node features for each node in the graph.
    """

    # def __init__(self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers, no_2d=False, add_3d=False, args=None):
    #     super().__init__(num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers)
    #     self.num_heads = num_heads
    #     self.num_atoms = num_atoms
    #     self.no_2d = no_2d
    #     self.add_3d = add_3d

    #     # 1 for graph token
    #     self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
    #     self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
    #     self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

    #     self.graph_token = nn.Embedding(1, hidden_dim)

    #     self.atom_mask_embedding = nn.Embedding(9, hidden_dim, padding_idx=None)

    #     # self.register_buffer('node_2d_factor', torch.ones(1).float())
    #     #
    #     # self.node_2d_dropout = FairseqDropout(
    #     #     0.5, module_name=self.__class__.__name__
    #     # )

    #     self.apply(lambda module: init_params(module, n_layers=n_layers))
    #     self.args = args

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
        if not self.no_2d and not self.args.ft and not self.args.infer:
            mask_embedding = self.atom_mask_embedding.weight.sum(dim=0)
            node_feature[
                node_mask.bool().squeeze(-1).contiguous()
            ] = mask_embedding.contiguous()

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        # print("GraphNodeFeaturePipe: ", graph_node_feature.grad)

        return graph_node_feature


class GraphAttnBiasPipe(GraphAttnBias):
    """
    Compute attention bias for each head.
    """

    # def __init__(self, num_heads, num_atoms, num_edges, num_spatial, num_edge_dis, hidden_dim, edge_type, multi_hop_max_dist, n_layers, no_2d=False, add_3d=False, args=None):
    #     super().__init__(num_heads, num_atoms, num_edges, num_spatial, num_edge_dis, hidden_dim, edge_type, multi_hop_max_dist, n_layers)
    #     self.num_heads = num_heads
    #     self.multi_hop_max_dist = multi_hop_max_dist
    #     self.no_2d = no_2d
    #     self.add_3d = add_3d

    #     self.edge_encoder = nn.Embedding(num_edges + 1, hidden_dim, padding_idx=0)

    #     # full_path_information
    #     # self.node_encoder = nn.Embedding(num_atoms + 1, num_heads, padding_idx=0)
    #     self.edge_type = edge_type
    #     if self.edge_type == 'multi_hop':
    #         self.edge_dis_encoder = nn.Embedding(
    #             num_edge_dis * hidden_dim * num_heads, 1)
    #         # full_path_information
    #         # self.node_dis_encoder = nn.Embedding(
    #         #     (num_edge_dis+1 ) * num_heads * num_heads, 1
    #         # )
    #     self.spatial_pos_encoder = nn.Embedding(num_spatial + 10, num_heads, padding_idx=0)

    #     # self.register_buffer('edge_2d_factor', torch.ones(1).float())
    #     # self.edge_2d_dropout = FairseqDropout(
    #     #     0.5, module_name=self.__class__.__name__
    #     # )

    #     # spd_count
    #     # self.spatial_pos_count_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

    #     self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    #     self.apply(lambda module: init_params(module, n_layers=n_layers))
    #     self.args = args

    def forward(self, input_tuple):
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
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if not self.no_2d:
            if not self.args.ft and not self.args.infer:
                # @ Roger added: mask atom
                mask_spatial_pos_value = self.spatial_pos_encoder.weight.shape[0] - 1
                spatial_pos = spatial_pos.masked_fill_(
                    node_mask.squeeze(-1).unsqueeze(1).bool(), mask_spatial_pos_value
                )

            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            if mask_2d is not None:
                spatial_pos_bias = spatial_pos_bias * mask_2d[:, None, None, None]
            graph_attn_bias[:, :, 1:, 1:] = (
                graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
            )

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
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
                        -1, self.edge_hidden_dim, self.num_heads
                    )[:max_dist, :, :],
                )
                edge_input = edge_input_flat.reshape(
                    max_dist, n_graph, n_node, n_node, self.num_heads
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
            if not self.args.ft and not self.args.infer:
                edge_input = edge_input.masked_fill_(
                    node_mask.squeeze(-1)[:, None, None, :].bool(), 0.0
                )

            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class Graph3DBiasPipe(Graph3DBias):
    """
    Compute 3D attention bias according to the position information for each head.
    """

    # def __init__(self, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe=False, args=None):
    #     super().__init__(num_heads, num_edges, n_layers, embed_dim, num_kernel)
    #     self.num_heads = num_heads
    #     self.num_edges = num_edges
    #     self.n_layers = n_layers
    #     self.no_share_rpe = no_share_rpe
    #     self.num_kernel = num_kernel
    #     self.embed_dim = embed_dim

    #     rpe_heads = self.num_heads * self.n_layers if self.no_share_rpe else self.num_heads
    #     self.gbf = GaussianLayer(self.num_kernel, num_edges)
    #     self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)

    #     if self.num_kernel != self.embed_dim:
    #         self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
    #     else:
    #         self.edge_proj = None

    #     self.mask_bias = nn.Embedding(1, num_heads, padding_idx=None)
    #     self.args = args
    #     # self.register_buffer('pos_3d_factor', torch.tensor(0.5).float())
    #     # self.pos_3d_dropout = FairseqDropout(
    #     #     0.5, module_name=self.__class__.__name__
    #     # )

    def forward(self, input_tuple: tuple):
        pos, x, node_type_edge, node_mask = input_tuple

        padding_mask = x.eq(0).all(dim=-1)
        n_graph, n_node, _ = pos.shape

        # @ Roger added:
        # pos = pos.masked_fill(node_mask.bool(), 0.0)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-5)

        if not self.args.ft and not self.args.infer:
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

        # @ Roger added: mask atom

        # graph_attn_bias[node_mask.squeeze(-1)[:, None, :].bool().repeat(1, n_node, 1)] = self.mask_bias.weight.squeeze(0)

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        if not self.args.ft and not self.args.infer:
            merge_edge_features = merge_edge_features.masked_fill_(
                padding_mask.unsqueeze(-1) + node_mask.bool(), 0.0
            )
        else:
            merge_edge_features = merge_edge_features.masked_fill_(
                padding_mask.unsqueeze(-1), 0.0
            )
        # pos_3d_factor = self.pos_3d_dropout(self.pos_3d_factor)

        return graph_attn_bias, merge_edge_features, delta_pos


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


class NodeTaskHeadPipe(NodeTaskHead):
    # def __init__(
    #     self,
    #     embed_dim: int,
    #     num_heads: int,
    # ):
    #     super().__init__(embed_dim, num_heads)
    #     self.embed_dim = embed_dim
    #     self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
    #     self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
    #     self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
    #     self.num_heads = num_heads
    #     self.scaling = (embed_dim // num_heads) ** -0.5
    #     self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
    #     self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
    #     self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

    #     # self.dropout_module = FairseqDropout(
    #         # 0.1, module_name=self.__class__.__name__
    #     # )
    #     self.dropout_module = nn.Dropout(0.1)

    def forward(
        self,
        input_tuple: tuple,
    ) -> Tensor:
        query, attn_bias, delta_pos = input_tuple
        query = query.contiguous().transpose(0, 1)
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        # attn_probs_float = utils.softmax(attn.view(-1, n_node, n_node) + attn_bias.contiguous().view(-1, n_node, n_node), dim=-1, onnx_trace=False)
        attn_probs_float = nn.functional.softmax(
            attn.view(-1, n_node, n_node)
            + attn_bias.contiguous().view(-1, n_node, n_node),
            dim=-1,
        )

        # print("396:", q, k, v)
        # print("397", attn, attn_probs_float)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = self.dropout_module(attn_probs).view(
            bsz, self.num_heads, n_node, n_node
        )
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]

        # print("404", attn_probs, rot_attn_probs)
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        # print(rot_attn_probs.shape, v.unsqueeze(2).shape)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        # print(rot_attn_probs.shape, v.shape)
        # x = torch.einsum("ijkmn,ijnq->ijkmq", rot_attn_probs, v)
        # print("407", rot_attn_probs, x, v)
        # exit()
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj1(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj1(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()

        # print("413", x, f1, f2, f3)
        return cur_force


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout=0.0
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        # self.ln = LayerNorm(inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        # x = self.ln(x)
        x = self.out_proj(x)
        return x
