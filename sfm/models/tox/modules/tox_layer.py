# -*- coding: utf-8 -*-
import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timestep_encoder import DiffNoise, TimeStepEncoder


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
        pfm_config,
        num_residues,
        hidden_dim,
        max_len=1024,
        prop_feat=False,
        angle_feat=False,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(ResidueFeature, self).__init__()

        self.num_residues = num_residues
        self.hidden_dim = hidden_dim
        self.prop_feat = prop_feat
        self.angle_feat = angle_feat
        self.label_smooth = 0.05

        self.token_embed = nn.Embedding(num_residues, hidden_dim)
        # self.token_embed = nn.Linear(num_residues, hidden_dim, bias=False)
        self.atom_mask_embedding = nn.Embedding(9, hidden_dim, padding_idx=None)

        # self.time_embedding = TimeStepEncoder(
        #     t_timesteps,
        #     hidden_dim,
        #     timestep_emb_type=time_embedding_type,
        #     mlp=time_embedding_mlp,
        # )

    def forward(self, batched_data, time_aa=None, mask_aa=None, mask_pos=None):
        x = self.token_embed(batched_data["x"])

        mask_embedding = self.atom_mask_embedding.weight.sum(dim=0)

        x[mask_aa.bool().squeeze(-1)] = mask_embedding

        return x


class Edge3DEmbedding(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Edge3DEmbedding, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            self.num_kernel,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

    def forward(
        self, pos, angle, node_type_edge, padding_mask, mask_aa, mask_pos, time_pos
    ):
        n_graph, n_node, _ = pos.shape

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-5)

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

        if time_pos is not None and mask_pos is not None:
            t = self.time_embedding(time_pos).unsqueeze(1).unsqueeze(1)
            t_dist = t.repeat(1, n_node, 1, 1) + t.repeat(1, 1, n_node, 1)
            time_embedding = torch.zeros_like(edge_feature).to(edge_feature) + t_dist

            t0 = torch.zeros_like(time_pos).to(time_pos)
            t0_dist = self.time_embedding(t0).unsqueeze(1).unsqueeze(1)
            t_dist = t0_dist.repeat(1, n_node, 1, 1) + t0_dist.repeat(1, 1, n_node, 1)
            t0_emb = torch.zeros_like(edge_feature).to(edge_feature) + t0_dist

            mask_pos_i = mask_pos.unsqueeze(-2).repeat(1, 1, n_node, 1)
            mask_pos_j = mask_pos.unsqueeze(1).repeat(1, n_node, 1, 1)
            mask_dist = (mask_pos_i | mask_pos_j).bool()

            time_embedding = torch.where(mask_dist.bool(), time_embedding, t0_emb)

            # edge_feature.masked_fill_(mask_dist, 0.0)
            edge_feature = edge_feature + time_embedding

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        merge_edge_features.masked_fill_(padding_mask.unsqueeze(-1), 0.0)

        return edge_feature, merge_edge_features, delta_pos


class Node3DEmbedding(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Node3DEmbedding, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        inter_dim = embed_dim // 2

        self.pos_emb = NonLinear(3, inter_dim)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            embed_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

        self.angle_embed = NonLinear(3, inter_dim)

    def forward(
        self, pos, angle, node_type_edge, padding_mask, mask_aa, mask_pos, time_pos
    ):
        n_graph, n_node, _ = pos.shape

        node3dfeature = self.pos_emb(pos.type_as(self.pos_emb.layer1.weight))

        angle_mask = angle == float("inf")
        angle = angle.to(node3dfeature.dtype)
        angle_feature = self.angle_embed(angle.masked_fill(angle_mask, 0.0))

        node3dfeature = torch.cat([node3dfeature, angle_feature], dim=-1)

        if time_pos is not None and mask_pos is not None:
            time_embedding = self.time_embedding(time_pos).unsqueeze(1)
            t0 = torch.zeros_like(time_pos).to(time_pos)
            t0_emb = self.time_embedding(t0).unsqueeze(1)

            time_embedding = torch.where(mask_pos.bool(), time_embedding, t0_emb)

            # edge_feature.masked_fill_(mask_dist, 0.0)
            node3dfeature = node3dfeature + time_embedding

        node3dfeature = node3dfeature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        return node3dfeature, None, None


class Node3DEmbeddingv2(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Node3DEmbeddingv2, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        inter_dim = embed_dim // 2

        self.gbf = GaussianLayer(num_kernel, inter_dim)
        self.feature_proj = NonLinear(num_kernel, inter_dim)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            embed_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

        self.angle_embed = NonLinear(3, inter_dim)

    def forward(
        self, pos, angle, node_type_edge, padding_mask, mask_aa, mask_pos, time_pos
    ):
        n_graph, n_node, _ = pos.shape

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1e-5)

        pos_feature = self.gbf(dist)

        pos_feature = pos_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_pos_features = pos_feature.sum(dim=-2)
        node3dfeature = self.feature_proj(sum_pos_features)

        angle_mask = angle == float("inf")
        angle = angle.to(node3dfeature.dtype)
        angle_feature = self.angle_embed(angle.masked_fill(angle_mask, 0.0))

        node3dfeature = torch.cat([node3dfeature, angle_feature], dim=-1)

        if time_pos is not None and mask_pos is not None:
            time_embedding = self.time_embedding(time_pos).unsqueeze(1)
            t0 = torch.zeros_like(time_pos).to(time_pos)
            t0_emb = self.time_embedding(t0).unsqueeze(1)

            time_embedding = torch.where(mask_pos.bool(), time_embedding, t0_emb)

            # edge_feature.masked_fill_(mask_dist, 0.0)
            node3dfeature = node3dfeature + time_embedding

        return node3dfeature, None, None


class Mix3DEmbedding(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Mix3DEmbedding, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        # inter_dim = embed_dim // 2

        self.mix_emb = NonLinear(6, embed_dim)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            embed_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

    def forward(
        self, pos, angle, node_type_edge, padding_mask, mask_aa, mask_pos, time_pos
    ):
        angle = angle.to(pos.dtype)

        node6d = torch.cat([pos, angle], dim=-1)

        node3dfeature = self.mix_emb(node6d.type_as(self.mix_emb.layer1.weight))

        if time_pos is not None and mask_pos is not None:
            time_embedding = self.time_embedding(time_pos).unsqueeze(1)
            t0 = torch.zeros_like(time_pos).to(time_pos)
            t0_emb = self.time_embedding(t0).unsqueeze(1)

            time_embedding = torch.where(mask_pos.bool(), time_embedding, t0_emb)

            node3dfeature = node3dfeature + time_embedding

        node3dfeature = node3dfeature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        return node3dfeature, None, None


class Mix3DEmbeddingV2(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Mix3DEmbeddingV2, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.time_embedding_dim = embed_dim

        self.angle_emb = AngleEmb(1, self.time_embedding_dim)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            self.time_embedding_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

        self.feature_proj = nn.Linear(3 * self.time_embedding_dim, embed_dim)
        self.cls_embedding = nn.Embedding(1, embed_dim, padding_idx=None)
        self.eos_embedding = nn.Embedding(1, embed_dim, padding_idx=None)

    def forward(
        self,
        pos,
        angle,
        padding_mask,
        mask_aa,
        mask_pos,
        mask_angle,
        angle_mask,
        time_pos,
        time_angle,
        aa_seq,
    ):
        bs, nnode, _ = angle.shape
        angle = angle.to(self.angle_emb.layer1.weight.dtype)
        angle = angle[:, :, :3].reshape(bs, -1, 1)
        angle_mask = angle_mask[:, :, :3]

        cls_mask = (aa_seq == 0).unsqueeze(-1)
        eos_mask = (aa_seq == 2).unsqueeze(-1)
        angle_feat = self.angle_emb(angle)

        node6dfeature = angle_feat

        if time_pos is not None and mask_angle is not None:
            if time_pos.dim() == 2:
                time_pos = time_pos.unsqueeze(-1).repeat(1, 1, 3)
            elif time_pos.dim() == 1:
                time_pos = time_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, nnode, 3)

            t0 = torch.zeros_like(
                time_pos, dtype=time_pos.dtype, device=time_pos.device
            )

            time_pos = time_pos.masked_fill(~angle_mask, 1.0).view(-1)
            t0 = t0.masked_fill(~angle_mask, 1.0).view(-1)

            time_embedding_pos = self.time_embedding(time_pos).view(
                bs, -1, self.time_embedding_dim
            )
            t0_emb = self.time_embedding(t0).view(bs, -1, self.time_embedding_dim)
            t_mask = mask_angle.bool() & (~cls_mask) & (~eos_mask)
            time_embedding_pos = torch.where(
                t_mask.repeat(1, 3, 1), time_embedding_pos, t0_emb
            )

            node6dfeature = node6dfeature + time_embedding_pos

        node6dfeature = self.feature_proj(node6dfeature.view(bs, nnode, -1))

        node6dfeature = torch.where(cls_mask, self.cls_embedding.weight, node6dfeature)
        node6dfeature = torch.where(eos_mask, self.eos_embedding.weight, node6dfeature)
        node6dfeature = node6dfeature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        return node6dfeature, None, None


class Mix3DEmbeddingV3(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Mix3DEmbeddingV3, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim

        # inter_dim = embed_dim // 2
        self.trans_pos = TransNet(3, embed_dim // 2)
        self.trans_feat = TransNet(embed_dim // 2, embed_dim // 2)
        self.pos_emb = NonLinear(3, embed_dim // 2)
        self.angle_emb = NonLinear(3, embed_dim // 2)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            embed_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

    def forward(
        self,
        pos,
        angle,
        node_type_edge,
        padding_mask,
        mask_aa,
        mask_pos,
        mask_angle,
        time_pos,
        time_angle,
    ):
        angle = angle.to(self.trans_pos.layer1.weight.dtype)
        pos = pos.to(self.trans_pos.layer1.weight.dtype)

        T_mat = self.trans_pos(pos)
        pos = pos + torch.bmm(pos, T_mat)
        pos_feat = self.pos_emb(pos)
        # T_feat = self.trans_feat(pos_feat)
        # pos_feat = pos_feat + torch.bmm(pos_feat, T_feat)

        # sin_ang = torch.sin(angle)
        # cos_ang = torch.cos(angle)
        # angle = torch.cat([sin_ang, cos_ang], dim=-1)
        angle_feat = self.angle_emb(angle)

        node6dfeature = torch.cat([pos_feat, angle_feat], dim=-1)
        # node6dfeature = angle_feat

        if time_pos is not None and mask_pos is not None:
            time_embedding_pos = self.time_embedding(time_pos).unsqueeze(1)
            t0 = torch.zeros_like(time_pos).to(time_pos)
            t0_emb = self.time_embedding(t0).unsqueeze(1)
            time_embedding_pos = torch.where(
                mask_pos.bool(), time_embedding_pos, t0_emb
            )
            node6dfeature = node6dfeature + time_embedding_pos

        # if time_angle is not None and mask_angle is not None:
        #     time_embedding_ang = self.time_embedding(time_angle).unsqueeze(1)
        #     t0 = torch.zeros_like(time_pos).to(time_pos)
        #     t0_emb = self.time_embedding(t0).unsqueeze(1)
        #     time_embedding_ang = torch.where(
        #         mask_angle.bool(), time_embedding_ang, t0_emb
        #     )
        #     angle_feat = angle_feat + time_embedding_ang

        node6dfeature = node6dfeature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        # T_mat = self.trans_feat(node6dfeature)
        # node6dfeature = node6dfeature + torch.bmm(node6dfeature, T_mat)

        return node6dfeature, None, None


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

    def forward(self, x, edge_types=None):
        if edge_types is not None:
            mul = self.mul(edge_types).sum(dim=-2)
            bias = self.bias(edge_types).sum(dim=-2)
            x = mul * x.unsqueeze(-1) + bias
        else:
            x = x.unsqueeze(-1)
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, output_size, bias=False)
        # self.layer1.weight.register_hook(self.print_grad)

    # # make sure attn_bias has gradient
    # def print_grad(self, grad):
    #     print(torch.max(grad))
    #     return grad

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class AngleEmb(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(AngleEmb, self).__init__()

        if hidden is None:
            hidden = output_size // 4
        self.layer1 = nn.Linear(input, hidden, bias=False)
        self.layer2 = nn.Linear(hidden, output_size, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class TransNet(nn.Module):
    def __init__(self, input, hidden):
        super(TransNet, self).__init__()

        if hidden is None:
            hidden = input
        self.hidden = hidden
        self.input = input

        self.layer1 = nn.Linear(input, hidden // 2, bias=False)
        self.layer2 = nn.Linear(hidden // 2, hidden, bias=False)

        self.layer3 = nn.Linear(hidden, hidden // 2, bias=False)
        self.layer4 = nn.Linear(hidden // 2, input * input, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)  # [B, N, hidden]

        x = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, hidden]
        x = x.view(-1, self.hidden)  # [B, hidden]

        x = self.layer3(x)
        x = F.gelu(x)
        x = self.layer4(x)

        x = x.view(-1, self.input, self.input)

        return x


class TransNet_V2(nn.Module):
    def __init__(self, input, hidden):
        super(TransNet, self).__init__()

        if hidden is None:
            hidden = input
        self.hidden = hidden
        self.input = input

        self.layer1 = nn.Linear(input, hidden // 2, bias=False)
        self.layer2 = nn.Linear(hidden // 2, hidden, bias=False)

        self.layer3 = nn.Linear(hidden, hidden // 2, bias=False)
        self.layer4 = nn.Linear(hidden // 2, input * input, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)  # [B, N, hidden]

        x = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, hidden]
        x = x.view(-1, self.hidden)  # [B, hidden]

        x = self.layer3(x)
        x = F.gelu(x)
        x = self.layer4(x)

        x = x.view(-1, self.input, self.input)

        return x
