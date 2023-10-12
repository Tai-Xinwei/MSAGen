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

from sfm.logging import logger
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.get_activation_fn import get_activation_fn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
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
    ):
        super(GraphNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.no_2d = no_2d
        self.args = args
        # 1 for graph token
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            num_out_degree, hidden_dim, padding_idx=0
        )

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.atom_mask_embedding = nn.Embedding(9, hidden_dim, padding_idx=None)

    def forward(self, batched_data, mask_2d=None, no_mask=False):
        x, in_degree, out_degree = (
            batched_data["x"],
            batched_data["in_degree"],
            batched_data["out_degree"],
        )
        n_graph, n_node = x.size()[:2]

        # node feauture + graph token
        node_feature = (
            self.atom_encoder(x).sum(dim=-2).contiguous()
        )  # [n_graph, n_node, n_hidden]

        if not self.no_2d:
            degree_feature = self.in_degree_encoder(
                in_degree
            ) + self.out_degree_encoder(out_degree)
            if mask_2d is not None:
                degree_feature = degree_feature * mask_2d[:, None, None]
            node_feature = node_feature + degree_feature

        # @ Roger added: mask atom
        if not self.args.ft and not no_mask:
            mask_embedding = self.atom_mask_embedding.weight.sum(dim=0)
            node_mask = batched_data["node_mask"]
            node_feature[node_mask.bool().squeeze(-1)] = mask_embedding

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature


class GraphAttnBias(nn.Module):
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
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.no_2d = no_2d
        self.args = args
        self.edge_encoder = nn.Embedding(num_edges + 1, hidden_dim, padding_idx=0)

        self.edge_type = edge_type
        self.edge_hidden_dim = hidden_dim
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * hidden_dim * num_heads, 1
            )
            # full_path_information
            # self.node_dis_encoder = nn.Embedding(
            #     (num_edge_dis+1 ) * num_heads * num_heads, 1
            # )
        self.spatial_pos_encoder = nn.Embedding(
            num_spatial + 10, num_heads, padding_idx=0
        )

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    def forward(self, batched_data, mask_2d=None, no_mask=False):
        attn_bias, spatial_pos, x = (
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["x"],
        )
        edge_input, attn_edge_type = (
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )
        if not self.args.ft:
            node_mask = batched_data["node_mask"]

        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        if not self.no_2d:
            # @ Roger added: mask atom
            if not self.args.ft and not no_mask:
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
            if not self.args.ft and not no_mask:
                edge_input = edge_input.masked_fill_(
                    node_mask.squeeze(-1)[:, None, None, :].bool(), 0.0
                )

            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias


class Graph3DBias(nn.Module):
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
        super(Graph3DBias, self).__init__()
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
        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, rpe_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

        self.mask_bias = nn.Embedding(1, num_heads, padding_idx=None)

        # self.register_buffer('pos_3d_factor', torch.tensor(0.5).float())
        # self.pos_3d_dropout = FairseqDropout(
        #     0.5, module_name=self.__class__.__name__
        # )

    def forward(self, batched_data, pos):
        x, node_type_edge = (
            batched_data["x"],
            batched_data["node_type_edge"],
        )  # pos shape: [n_graphs, n_nodes, 3]

        if not self.args.ft:
            node_mask = batched_data["node_mask"]
        # pos.requires_grad_(True)

        padding_mask = x.eq(0).all(dim=-1)
        n_graph, n_node, _ = pos.shape

        # @ Roger added:
        # pos = pos.masked_fill(node_mask.bool(), 0.0)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-5)

        if not self.args.ft:
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

        if not self.args.ft:
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


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.k_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.v_proj: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1: Callable[[Tensor], Tensor] = nn.Linear(
            embed_dim, 1, bias=False
        )
        # self.force_proj2: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)
        # self.force_proj3: Callable[[Tensor], Tensor] = nn.Linear(embed_dim, 1)

        self.dropout_module = FairseqDropout(0.1, module_name=self.__class__.__name__)

    def forward(
        self,
        query: Tensor,
        attn_bias: Tensor,
        delta_pos: Tensor,
    ) -> Tensor:
        query = query.contiguous().transpose(0, 1)
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        attn_probs_float = nn.functional.softmax(
            attn.view(-1, n_node, n_node)
            + attn_bias.contiguous().view(-1, n_node, n_node),
            dim=-1,
        )
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = self.dropout_module(attn_probs).view(
            bsz, self.num_heads, n_node, n_node
        )
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj1(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj1(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj1(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
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


class EquivariantVectorOutput(nn.Module):
    def __init__(self, hidden_channels=768, activation="silu", d_tilde=1):
        super(EquivariantVectorOutput, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1, activation=activation),
            ]
        )

        self.reset_parameters(d_tilde)

    def reset_parameters(self, d_tilde):
        for layer in self.output_network:
            layer.reset_parameters(d_tilde)

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze(-1)


class EquivariantLayerNorm(nn.Module):
    r"""Rotationally-equivariant Vector Layer Normalization
    Expects inputs with shape (N, n, d), where N is batch size, n is vector dimension, d is width/number of vectors.
    """
    __constants__ = ["normalized_shape", "elementwise_linear"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_linear: bool

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_linear: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EquivariantLayerNorm, self).__init__()

        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.elementwise_linear = elementwise_linear
        if self.elementwise_linear:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter(
                "weight", None
            )  # Without bias term to preserve equivariance!

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_linear:
            nn.init.ones_(self.weight)

    def mean_center(self, input):
        return input - input.mean(-1, keepdim=True)

    def covariance(self, input):
        return 1 / self.normalized_shape[0] * input @ input.transpose(-1, -2)

    def symsqrtinv(self, matrix):
        """Compute the inverse square root of a positive definite matrix.
        Based on https://github.com/pytorch/pytorch/issues/25481
        """
        _, s, v = matrix.svd()
        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))
        return (v * 1 / torch.sqrt(s + self.eps).unsqueeze(-2)) @ v.transpose(-2, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float64)  # Need double precision for accurate inversion.
        input = self.mean_center(input)
        # We use different diagonal elements in case input matrix is approximately zero,
        # in which case all singular values are equal which is problematic for backprop.
        # See e.g. https://pytorch.org/docs/stable/generated/torch.svd.html
        reg_matrix = (
            torch.diag(torch.tensor([1.0, 2.0, 3.0]))
            .unsqueeze(0)
            .to(input.device)
            .type(input.dtype)
        )
        covar = self.covariance(input) + self.eps * reg_matrix
        covar_sqrtinv = self.symsqrtinv(covar)
        return (covar_sqrtinv @ input).to(self.weight.dtype) * self.weight.reshape(
            1, 1, self.normalized_shape[0]
        )

    def extra_repr(self) -> str:
        return "{normalized_shape}, " "elementwise_linear={elementwise_linear}".format(
            **self.__dict__
        )


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=64, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class Distance(nn.Module):
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_num_neighbors=32,
        return_vecs=True,
        loop=True,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(
            pos,
            r=self.cutoff_upper,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0)).to(edge_vec)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # TODO: return only `edge_index` and `edge_weight` once
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels=256,
        num_rbf=64,
        distance_influence="both",
        num_heads=8,
        activation=nn.SiLU,
        attn_activation="silu",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        d_tilde=1.0,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        act_class_mapping = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.reset_parameters(d_tilde)

    def reset_parameters(self, d_tilde):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, vec: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        x = x.reshape(-1, self.hidden_channels).to(q)
        vec = vec.reshape(-1, 3, self.hidden_channels)

        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec
        return dx, dvec

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        # update scalar features
        x = x * attn.unsqueeze(2)
        # update vector features
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(
            2
        ).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class_mapping = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self, d_tilde):
        nn.init.xavier_uniform_(self.vec1_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        nn.init.xavier_uniform_(
            self.update_net[0].weight, gain=1.0 / math.sqrt(d_tilde)
        )
        if self.out_channels == 1:
            nn.init.xavier_uniform_(self.vec2_proj.weight, gain=1.0 / d_tilde)
            nn.init.xavier_uniform_(self.update_net[2].weight, gain=1.0 / d_tilde)
        else:
            nn.init.xavier_uniform_(
                self.vec2_proj.weight, gain=1.0 / math.sqrt(d_tilde)
            )
            nn.init.xavier_uniform_(
                self.update_net[2].weight, gain=1.0 / math.sqrt(d_tilde)
            )

        self.update_net[0].bias.data.fill_(0)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(-2) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
