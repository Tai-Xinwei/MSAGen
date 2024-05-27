# -*- coding: utf-8 -*-
import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sfm.modules.mem_eff_attn import MemEffSelfAttn

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
        self.unkangle_embedding = nn.Embedding(
            1, self.time_embedding_dim, padding_idx=None
        )

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
        angle = angle.masked_fill(~angle_mask, 0.0)
        angle = angle.to(self.angle_emb.layer1.weight.dtype)
        angle = angle[:, :, :3].reshape(bs, -1, 1)
        angle_mask = angle_mask[:, :, :3]

        cls_mask = (aa_seq == 0).unsqueeze(-1)
        eos_mask = (aa_seq == 2).unsqueeze(-1)
        angle_feat = self.angle_emb(angle)
        angle_feat = torch.where(
            angle_mask.reshape(bs, -1, 1), angle_feat, self.unkangle_embedding.weight
        )

        node6dfeature = angle_feat

        if time_pos is not None and mask_angle is not None:
            if time_pos.dim() == 2:
                time_pos = time_pos.unsqueeze(-1).repeat(1, 1, 3)
            elif time_pos.dim() == 1:
                time_pos = time_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, nnode, 3)

            t0 = torch.zeros_like(
                time_pos, dtype=time_pos.dtype, device=time_pos.device
            )

            time_pos = time_pos.masked_fill(~angle_mask, 0.0).view(-1)
            t0 = t0.masked_fill(~angle_mask, 0.0).view(-1)

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


class Mix3DEmbeddingV4(nn.Module):
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
        super(Mix3DEmbeddingV4, self).__init__()
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

        self.feature_proj = nn.Linear(6 * self.time_embedding_dim, embed_dim)
        self.cls_embedding = nn.Embedding(1, embed_dim, padding_idx=None)
        self.eos_embedding = nn.Embedding(1, embed_dim, padding_idx=None)
        self.unkangle_embedding = nn.Embedding(
            1, self.time_embedding_dim, padding_idx=None
        )

    def forward(
        self,
        pos,
        angle,
        padding_mask,
        mask_aa,
        mask_pos,
        mask_angle,
        angle_mask,
        bond_angle_mask,
        time_pos,
        time_angle,
        aa_seq,
    ):
        bs, nnode, _ = angle.shape
        angle_mask = angle_mask[:, :, :3]
        angle_mask = torch.cat([angle_mask, bond_angle_mask], dim=-1)

        angle = angle.masked_fill(~angle_mask, 0.0)
        angle = angle.to(self.angle_emb.layer1.weight.dtype)
        angle = angle.reshape(bs, -1, 1)

        cls_mask = (aa_seq == 0).unsqueeze(-1)
        eos_mask = (aa_seq == 2).unsqueeze(-1)
        angle_feat = self.angle_emb(angle)
        angle_feat = torch.where(
            angle_mask.reshape(bs, -1, 1), angle_feat, self.unkangle_embedding.weight
        )

        node6dfeature = angle_feat

        if time_pos is not None and mask_angle is not None:
            if time_pos.dim() == 2:
                time_pos = time_pos.unsqueeze(-1).repeat(1, 1, 6)
            elif time_pos.dim() == 1:
                time_pos = time_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, nnode, 3)

            t0 = torch.zeros_like(
                time_pos, dtype=time_pos.dtype, device=time_pos.device
            )

            time_pos = time_pos.masked_fill(~angle_mask, 0.0).view(-1)
            t0 = t0.masked_fill(~angle_mask, 0.0).view(-1)

            time_embedding_pos = self.time_embedding(time_pos).view(
                bs, -1, self.time_embedding_dim
            )
            t0_emb = self.time_embedding(t0).view(bs, -1, self.time_embedding_dim)
            t_mask = mask_angle.bool() & (~cls_mask) & (~eos_mask)
            time_embedding_pos = torch.where(
                t_mask.repeat(1, 6, 1), time_embedding_pos, t0_emb
            )

            node6dfeature = node6dfeature + time_embedding_pos

        node6dfeature = self.feature_proj(node6dfeature.view(bs, nnode, -1))

        node6dfeature = torch.where(cls_mask, self.cls_embedding.weight, node6dfeature)
        node6dfeature = torch.where(eos_mask, self.eos_embedding.weight, node6dfeature)
        node6dfeature = node6dfeature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        return node6dfeature, None, None


class Mix3DEmbeddingV5(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel=64,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Mix3DEmbeddingV5, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.time_embedding_dim = embed_dim // 2

        self.angle_emb = AngleEmb(1, self.time_embedding_dim)
        self.pos_emb = NonLinear(1, self.time_embedding_dim)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            self.time_embedding_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

        self.ang_feature_proj = nn.Linear(
            6 * self.time_embedding_dim, self.time_embedding_dim
        )
        self.pos_feature_proj = nn.Linear(
            self.time_embedding_dim, self.time_embedding_dim
        )

        self.cls_embedding = nn.Embedding(1, embed_dim, padding_idx=None)
        self.eos_embedding = nn.Embedding(1, embed_dim, padding_idx=None)

        self.unkangle_embedding = nn.Embedding(
            1, self.time_embedding_dim, padding_idx=None
        )
        self.unkpos_embedding = nn.Embedding(
            1, self.time_embedding_dim, padding_idx=None
        )

    def cal_dist(self, pos, aa_seq):
        Bs, L = pos.shape[:2]
        dist_sum = torch.zeros(Bs, L, device=pos.device, dtype=pos.dtype)
        for i in range(Bs):
            mask_start_idx = (aa_seq[i] == 0).nonzero(as_tuple=True)[0]
            mask_end_idx = (aa_seq[i] == 2).nonzero(as_tuple=True)[0]

            for j in range(len(mask_end_idx)):
                s_idx = mask_start_idx[j] + 1
                e_idx = mask_end_idx[j]
                delta_pos = pos[i, s_idx:e_idx].unsqueeze(0) - pos[
                    i, s_idx:e_idx
                ].unsqueeze(1)
                dist_sum[i, s_idx:e_idx] = delta_pos.norm(dim=-1).mean(dim=-1)

            if len(mask_start_idx) > len(mask_end_idx):
                s_idx = mask_start_idx[-1] + 1
                delta_pos = pos[i, s_idx:].unsqueeze(0) - pos[i, s_idx:].unsqueeze(1)
                dist_sum[i, s_idx:] = delta_pos.norm(dim=-1).mean(dim=-1)

        return dist_sum.unsqueeze(-1)

    def forward(
        self,
        pos,
        angle,
        padding_mask,
        mask_aa,
        mask_pos,
        mask_angle,
        angle_mask,
        bond_angle_mask,
        time_pos,
        time_angle,
        aa_seq,
    ):
        bs, nnode, _ = angle.shape
        angle_mask = angle_mask[:, :, :3]
        angle_mask = torch.cat([angle_mask, bond_angle_mask], dim=-1)

        angle = angle.masked_fill(~angle_mask, 0.0)
        angle = angle.to(self.angle_emb.layer1.weight.dtype)
        angle = angle.reshape(bs, -1, 1)

        cls_mask = (aa_seq == 0).unsqueeze(-1)
        eos_mask = (aa_seq == 2).unsqueeze(-1)
        angle_feat = self.angle_emb(angle)
        angle_feat = torch.where(
            angle_mask.reshape(bs, -1, 1), angle_feat, self.unkangle_embedding.weight
        )

        pos = pos.to(angle.dtype)
        dist_sum = self.cal_dist(pos[:, :, 1, :], aa_seq)
        pos_feat = self.pos_emb(dist_sum)
        pos_mask = ~(cls_mask | eos_mask)
        pos_feat = torch.where(pos_mask, pos_feat, self.unkpos_embedding.weight)

        if time_pos is not None and mask_angle is not None:
            if time_pos.dim() == 2:
                time_pos = time_pos.unsqueeze(-1)
                time_angle = time_pos.repeat(1, 1, 6)
            elif time_pos.dim() == 1:
                time_pos = time_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, nnode, 3)

            t0_ang = torch.zeros_like(
                time_angle, dtype=time_angle.dtype, device=time_angle.device
            )
            t0_pos = torch.zeros_like(
                time_pos, dtype=time_pos.dtype, device=time_pos.device
            )

            time_angle = time_angle.masked_fill(~angle_mask, 0.0).view(-1)
            t0_ang = t0_ang.masked_fill(~angle_mask, 0.0).view(-1)
            t0_pos = t0_pos.masked_fill(~pos_mask, 0.0).view(-1)

            time_embedding_pos = self.time_embedding(time_pos).view(
                bs, -1, self.time_embedding_dim
            )
            time_embedding_ang = self.time_embedding(time_angle).view(
                bs, -1, self.time_embedding_dim
            )
            t0_ang_emb = self.time_embedding(t0_ang).view(
                bs, -1, self.time_embedding_dim
            )
            t0_pos_emb = self.time_embedding(t0_pos).view(
                bs, -1, self.time_embedding_dim
            )

            t_mask = mask_angle.bool() & (~cls_mask) & (~eos_mask)
            time_embedding_ang = torch.where(
                t_mask.repeat(1, 6, 1), time_embedding_ang, t0_ang_emb
            )
            time_embedding_pos = torch.where(t_mask, time_embedding_pos, t0_pos_emb)

            angle_feat = angle_feat + time_embedding_ang
            pos_feat = pos_feat + time_embedding_pos

        angle_feat = self.ang_feature_proj(angle_feat.view(bs, nnode, -1))
        pos_feat = self.pos_feature_proj(pos_feat.view(bs, nnode, -1))

        node9dfeature = torch.cat([angle_feat, pos_feat], dim=-1)

        node9dfeature = torch.where(cls_mask, self.cls_embedding.weight, node9dfeature)
        node9dfeature = torch.where(eos_mask, self.eos_embedding.weight, node9dfeature)
        node9dfeature = node9dfeature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        return node9dfeature, None, None


class Mix3DEmbeddingV6(nn.Module):
    def __init__(
        self,
        pfm_config,
        num_edges,
        embed_dim,
        num_kernel=32,
        t_timesteps=1010,
        time_embedding_type="positional",
        time_embedding_mlp=True,
    ):
        super(Mix3DEmbeddingV6, self).__init__()
        self.num_edges = num_edges
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim
        self.time_embedding_dim = embed_dim // 2

        # self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.pos_emb = nn.Linear(3, self.num_kernel, bias=False)
        self.pos_feature_emb = nn.Linear(
            self.num_kernel, self.time_embedding_dim, bias=False
        )

        self.angle_emb = AngleEmb(1, self.time_embedding_dim)

        self.time_embedding = TimeStepEncoder(
            t_timesteps,
            self.time_embedding_dim,
            timestep_emb_type=time_embedding_type,
            mlp=time_embedding_mlp,
        )

        self.ang_feature_proj = nn.Linear(
            6 * self.time_embedding_dim, self.time_embedding_dim, bias=False
        )
        self.pos_feature_proj = nn.Linear(
            self.time_embedding_dim, self.time_embedding_dim, bias=False
        )

        self.cls_embedding = nn.Embedding(1, embed_dim, padding_idx=None)
        self.eos_embedding = nn.Embedding(1, embed_dim, padding_idx=None)

        self.unkangle_embedding = nn.Embedding(
            1, self.time_embedding_dim, padding_idx=None
        )
        self.unkpos_embedding = nn.Embedding(
            1, self.time_embedding_dim, padding_idx=None
        )

    def cal_dist(self, pos, aa_seq):
        Bs, L = pos.shape[:2]
        sum_dist_features = torch.zeros(
            Bs, L, self.num_kernel, device=pos.device, dtype=pos.dtype
        )
        for i in range(Bs):
            mask_start_idx = (aa_seq[i] == 0).nonzero(as_tuple=True)[0]
            mask_end_idx = (aa_seq[i] == 2).nonzero(as_tuple=True)[0]

            for j in range(len(mask_end_idx)):
                s_idx = mask_start_idx[j] + 1
                e_idx = mask_end_idx[j]
                delta_pos = pos[i, s_idx:e_idx].unsqueeze(0) - pos[
                    i, s_idx:e_idx
                ].unsqueeze(1)
                pos_p = self.pos_emb(pos[i, s_idx:e_idx])
                sum_dist_features[i, s_idx:e_idx, :] = torch.matmul(
                    1.0 / (delta_pos.norm(dim=-1) + 1.0), pos_p
                )

            if len(mask_start_idx) > len(mask_end_idx):
                s_idx = mask_start_idx[-1] + 1
                delta_pos = pos[i, s_idx:].unsqueeze(0) - pos[i, s_idx:].unsqueeze(1)
                pos_p = self.pos_emb(pos[i, s_idx:])
                sum_dist_features[i, s_idx:, :] = torch.matmul(
                    1.0 / (delta_pos.norm(dim=-1) + 1.0), pos_p
                )

        merge_dist_features = self.pos_feature_emb(sum_dist_features)
        return merge_dist_features

    def forward(
        self,
        pos,
        angle,
        padding_mask,
        mask_aa,
        mask_pos,
        mask_angle,
        angle_mask,
        bond_angle_mask,
        time_pos,
        time_angle,
        aa_seq,
    ):
        bs, nnode, _ = angle.shape
        angle_mask = angle_mask[:, :, :3]
        angle_mask = torch.cat([angle_mask, bond_angle_mask], dim=-1)

        angle = angle.masked_fill(~angle_mask, 0.0)
        angle = angle.to(self.angle_emb.layer1.weight.dtype)
        angle = angle.reshape(bs, -1, 1)

        cls_mask = (aa_seq == 0).unsqueeze(-1)
        eos_mask = (aa_seq == 2).unsqueeze(-1)
        angle_feat = self.angle_emb(angle)
        angle_feat = torch.where(
            angle_mask.reshape(bs, -1, 1), angle_feat, self.unkangle_embedding.weight
        )

        pos = pos.to(angle.dtype)
        pos_feat = self.cal_dist(pos[:, :, 1, :], aa_seq)
        pos_mask = ~(cls_mask | eos_mask)
        pos_feat = torch.where(pos_mask, pos_feat, self.unkpos_embedding.weight)

        if time_pos is not None and mask_angle is not None:
            if time_pos.dim() == 2:
                time_pos = time_pos.unsqueeze(-1)
                time_angle = time_pos.repeat(1, 1, 6)
            elif time_pos.dim() == 1:
                time_pos = time_pos.unsqueeze(-1).unsqueeze(-1).repeat(1, nnode, 3)

            t0_ang = torch.zeros_like(
                time_angle, dtype=time_angle.dtype, device=time_angle.device
            )
            t0_pos = torch.zeros_like(
                time_pos, dtype=time_pos.dtype, device=time_pos.device
            )

            time_angle = time_angle.masked_fill(~angle_mask, 0.0).view(-1)
            t0_ang = t0_ang.masked_fill(~angle_mask, 0.0).view(-1)
            t0_pos = t0_pos.masked_fill(~pos_mask, 0.0).view(-1)

            time_embedding_pos = self.time_embedding(time_pos).view(
                bs, -1, self.time_embedding_dim
            )
            time_embedding_ang = self.time_embedding(time_angle).view(
                bs, -1, self.time_embedding_dim
            )
            t0_ang_emb = self.time_embedding(t0_ang).view(
                bs, -1, self.time_embedding_dim
            )
            t0_pos_emb = self.time_embedding(t0_pos).view(
                bs, -1, self.time_embedding_dim
            )

            t_mask = mask_angle.bool() & (~cls_mask) & (~eos_mask)
            time_embedding_ang = torch.where(
                t_mask.repeat(1, 6, 1), time_embedding_ang, t0_ang_emb
            )
            time_embedding_pos = torch.where(t_mask, time_embedding_pos, t0_pos_emb)

            angle_feat = angle_feat + time_embedding_ang
            pos_feat = pos_feat + time_embedding_pos

        angle_feat = self.ang_feature_proj(angle_feat.view(bs, nnode, -1))
        pos_feat = self.pos_feature_proj(pos_feat.view(bs, nnode, -1))

        node9dfeature = torch.cat([angle_feat, pos_feat], dim=-1)

        node9dfeature = torch.where(cls_mask, self.cls_embedding.weight, node9dfeature)
        node9dfeature = torch.where(eos_mask, self.eos_embedding.weight, node9dfeature)
        node9dfeature = node9dfeature.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        return node9dfeature, None, None


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
            x = x.expand(-1, -1, -1, self.K)
        else:
            x = x.unsqueeze(-1)
            x = x.expand(-1, -1, self.K)
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


class PosEmb(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(PosEmb, self).__init__()

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


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj = nn.Linear(embed_dim, 1, bias=False)

    def forward(
        self,
        query: Tensor,
        delta_pos: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:
        query = query.contiguous()
        bsz, n_node, _ = query.size()
        q = (
            self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        min_dtype = torch.finfo(k.dtype).min
        attn = attn.masked_fill(attn_mask.unsqueeze(1), min_dtype)

        attn_probs_float = nn.functional.softmax(attn.view(-1, n_node, n_node), dim=-1)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = attn_probs.view(bsz, self.num_heads, n_node, n_node)

        delta_pos = delta_pos.masked_fill(attn_mask.unsqueeze(-1), 0.0)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        f1 = self.force_proj(x[:, :, 0, :]).view(bsz, n_node, 1)
        f2 = self.force_proj(x[:, :, 1, :]).view(bsz, n_node, 1)
        f3 = self.force_proj(x[:, :, 2, :]).view(bsz, n_node, 1)
        cur_force = torch.cat([f1, f2, f3], dim=-1).float()
        return cur_force
