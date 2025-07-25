# -*- coding: utf-8 -*-
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timestep_encoder import TimeStepEncoder


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class ResidueFeatureV0(nn.Module):
    """
    Compute residule features, three parts are included
    1. learnable embedding of residue type
    2. Prior of residue features
    3. learnable embedding of angles
    """

    def __init__(
        self,
        num_residues,
        hidden_dim,
        max_len=1024,
        prop_feat=True,
        angle_feat=True,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(ResidueFeatureV0, self).__init__()

        self.num_residues = num_residues
        self.hidden_dim = hidden_dim
        self.prop_feat = prop_feat
        self.angle_feat = angle_feat

        self.token_embed = nn.Embedding(num_residues, hidden_dim)
        self.atom_mask_embedding = nn.Embedding(9, hidden_dim, padding_idx=None)

    def forward(self, batched_data, time=None, mask_aa=None, mask_pos=None):
        with torch.no_grad():
            if "x_new" in batched_data.keys():
                aa_seq = batched_data["x_new"]
            else:
                aa_seq = batched_data["x"]

        x = self.token_embed(aa_seq)

        return x


class ResidueFeatureV1(nn.Module):
    """
    Compute residule features, three parts are included
    1. learnable embedding of residue type
    2. Prior of residue features
    3. learnable embedding of angles
    """

    def __init__(
        self,
        num_residues,
        hidden_dim,
        max_len=1024,
        prop_feat=True,
        angle_feat=True,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(ResidueFeatureV1, self).__init__()

        self.num_residues = num_residues
        self.hidden_dim = hidden_dim
        self.prop_feat = prop_feat
        self.angle_feat = angle_feat

        self.token_embed = nn.Embedding(num_residues, hidden_dim // 2)
        self.bpe_embed = nn.Embedding(16384, hidden_dim // 2)
        self.atom_mask_embedding = nn.Embedding(9, hidden_dim // 2, padding_idx=None)

    def forward(self, batched_data, time=None, mask_aa=None, mask_pos=None):
        if "x_new" in batched_data.keys():
            x_aa = self.token_embed(batched_data["x_new"])
        else:
            x_aa = self.token_embed(batched_data["x"])

        mask_embedding = self.atom_mask_embedding.weight.sum(dim=0)
        x_bpe = self.bpe_embed(batched_data["bpe"])
        x_bpe[mask_aa.bool().squeeze(-1)] = mask_embedding

        x = torch.cat([x_aa, x_bpe], dim=-1)

        return x


class ResidueFeature(nn.Module):
    """
    Compute residule features, three parts are included
    1. learnable embedding of residue type
    2. Prior of residue features
    3. learnable embedding of angles
    """

    def __init__(
        self,
        num_residues,
        hidden_dim,
        max_len=1024,
        prop_feat=True,
        angle_feat=True,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(ResidueFeature, self).__init__()

        self.num_residues = num_residues
        self.hidden_dim = hidden_dim
        self.prop_feat = prop_feat
        self.angle_feat = angle_feat

        self.token_embed = nn.Embedding(num_residues, hidden_dim)
        self.atom_mask_embedding = nn.Embedding(9, hidden_dim, padding_idx=None)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            hidden_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

        if self.prop_feat:
            # [ Chemical polarity ]
            ### 0 - Polar
            ### 1 - Nonpolar
            ### 2 - Brønsted base
            ### 3 - Brønsted acid
            ### 4 - Brønsted acid and base
            ### 5 - Basic polar
            ### 6 - Unknown
            # [ Net charge at pH 7.4 ]
            ### 0 - Neutral
            ### 1 - Positive
            ### 2 - Negative
            ### 3 - Unknown
            # [ Hydropathy index ]
            ### real value
            # [ Molecular mass ]
            ### real value
            self.prop_nets = nn.ModuleDict(
                {
                    "chem_polar": nn.Embedding(7, hidden_dim),
                    "net_charge": nn.Embedding(4, hidden_dim),
                    "hydropathy": nn.Linear(1, hidden_dim, bias=False),
                    "mol_mass": nn.Linear(1, hidden_dim, bias=False),
                }
            )

        if self.angle_feat:
            self.angle_embed = nn.Linear(3, hidden_dim, bias=False)

    def forward(self, batched_data, time=None, mask_aa=None, mask_pos=None):
        x = self.token_embed(batched_data["x"])

        mask_embedding = self.atom_mask_embedding.weight.sum(dim=0)

        if self.prop_feat:
            for prop_name, prop_net in self.prop_nets.items():
                prop_data = batched_data[prop_name]
                if prop_data.dtype == torch.float:
                    prop_data = prop_data.to(x.dtype)
                x = x + prop_net(prop_data)

        if self.angle_feat:
            angle_data = batched_data["ang"].to(x.dtype) / 180.0
            anlge_mask = angle_data == float("inf")
            x = x + self.angle_embed(angle_data.masked_fill(anlge_mask, 0.0))

        x[mask_aa.bool().squeeze(-1)] = mask_embedding

        if time is not None:
            time_embedding = (
                torch.zeros_like(x).to(x) + self.time_embedding(time)[:, None, :]
            )
            t0 = torch.zeros_like(time).to(time)
            t0_emb = torch.zeros_like(x).to(x) + self.time_embedding(t0)[:, None, :]
            time_embedding = torch.where(mask_pos.bool(), time_embedding, t0_emb)

            x = x + time_embedding

        return x


class Edge3DEmbedding(nn.Module):
    def __init__(self, num_edges, embed_dim, num_kernel):
        super(Edge3DEmbedding, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)

    def forward(self, pos, node_type_edge, padding_mask, mask_aa):
        n_graph, n_node, _ = pos.shape

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        if mask_aa is not None:
            node_mask_i = mask_aa.unsqueeze(-2).repeat(1, 1, n_node, 1)
            node_mask_j = mask_aa.unsqueeze(1).repeat(1, n_node, 1, 1)
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


# class RBF(nn.Module):
#     def __init__(self, K):
#         super(RBF, self).__init__()
#         self.K = K
#         self.centres = nn.Parameter(torch.Tensor(K, 1))
#         self.log_sigmas = nn.Parameter(torch.Tensor(K))
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.normal_(self.centres, 0, 1)
#         nn.init.constant_(self.log_sigmas, 0)

#     #  - Input: (N, in_features) where N is an arbitrary batch size
#     #  - Output: (N, out_features) where N is an arbitrary batch size
#     def forward(self, input):
#         size = (input.size(0), self.K, 1)
#         x = input.unsqueeze(1).expand(size)
#         c = self.centres.unsqueeze(0).expand(size)
#         distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
#         return torch.exp(-1*distances.pow(2))


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
        # self.layer1.weight.register_hook(self.print_grad)

    # make sure attn_bias has gradient
    def print_grad(self, grad):
        print(torch.max(grad))
        return grad

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x
