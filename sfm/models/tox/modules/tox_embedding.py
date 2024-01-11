# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from sfm.logging import logger
from sfm.modules.FairseqDropout import FairseqDropout

from .tox_layer import (
    Edge3DEmbedding,
    Mix3DEmbedding,
    Mix3DEmbeddingV2,
    Mix3DEmbeddingV3,
    Node3DEmbedding,
    Node3DEmbeddingv2,
    ResidueFeature,
)


class TOXEmbedding(nn.Module):
    def __init__(
        self,
        pfm_config,
    ):
        super().__init__()
        self.pfm_config = pfm_config
        self.residue_feature = ResidueFeature(
            pfm_config,
            num_residues=pfm_config.num_residues,
            hidden_dim=pfm_config.embedding_dim,
            max_len=1024,
            prop_feat=False,
            angle_feat=True,
        )

        self.edge_3d_emb = (
            Edge3DEmbedding(
                pfm_config,
                num_edges=pfm_config.num_edges,
                embed_dim=pfm_config.embedding_dim,
                num_kernel=pfm_config.num_3d_bias_kernel,
            )
            if pfm_config.add_3d
            else None
        )

        # TODO: 2D attention bias needs carefully designed, features such as MSA should be included
        # 2D attention bias is impletemented in pfmencoderlayers to avoid redundent communication
        # self.graph_attn_bias = graph2dBias()

        # 3D attention bias is impletemented in pfmencoderlayers to avoid redundent communication
        # self.graph_3d_bias = Graph3DBias()

    def forward(
        self,
        batched_data,
        padding_mask,
        pos=None,
        mask_aa=None,
        mask_pos=None,
        time_pos=None,
        time_aa=None,
    ):
        x = self.residue_feature(
            batched_data,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
            time_aa=time_aa,
            angle_feat=False,
        )
        edge_feature = None
        delta_pos = None
        if mask_pos is not None and self.pfm_config.add_3d:
            node_type_edge = batched_data["node_type_edge"]
            edge_feature, merged_edge_features, delta_pos = self.edge_3d_emb(
                pos,
                batched_data["ang"],
                node_type_edge,
                padding_mask,
                mask_aa,
                mask_pos,
                time_pos,
            )

            if self.pfm_config.noise_mode == "mae":
                merged_edge_features = merged_edge_features.masked_fill(mask_pos, 0.0)
                delta_pos = delta_pos.masked_fill(mask_pos.unsqueeze(-1), 0.0)

            # add 3d edge feature in node feature
            # x = x + merged_edge_features * 0.01

            # TEST: set 3d edge_feature to zero
            # edge_feature = torch.zeros_like(edge_feature)

        return x, edge_feature, delta_pos


class TOXmixEmbedding(nn.Module):
    def __init__(
        self,
        pfm_config,
    ):
        super().__init__()
        self.pfm_config = pfm_config
        self.residue_feature = ResidueFeature(
            pfm_config,
            num_residues=pfm_config.num_residues,
            hidden_dim=pfm_config.embedding_dim // 2,
            max_len=1024,
            prop_feat=False,
            angle_feat=False,
        )

        self.edge_3d_emb = (
            Mix3DEmbeddingV2(
                pfm_config,
                num_edges=pfm_config.num_edges,
                embed_dim=pfm_config.embedding_dim // 2,
                num_kernel=pfm_config.num_3d_bias_kernel,
            )
            if pfm_config.add_3d
            else None
        )

        # TODO: 2D attention bias needs carefully designed, features such as MSA should be included
        # 2D attention bias is impletemented in pfmencoderlayers to avoid redundent communication
        # self.graph_attn_bias = graph2dBias()

        # 3D attention bias is impletemented in pfmencoderlayers to avoid redundent communication
        # self.graph_3d_bias = Graph3DBias()

    def forward(
        self,
        batched_data,
        padding_mask,
        pos=None,
        angle=None,
        mask_aa=None,
        mask_pos=None,
        mask_angle=None,
        time_pos=None,
        time_aa=None,
        time_angle=None,
    ):
        x = self.residue_feature(
            batched_data,
            mask_aa=mask_aa,
            mask_pos=mask_pos,
            time_aa=time_aa,
        )
        edge_feature = None
        delta_pos = None
        if mask_pos is not None and self.pfm_config.add_3d:
            node_type_edge = batched_data["node_type_edge"]
            node3dfeature, merged_edge_features, delta_pos = self.edge_3d_emb(
                pos,
                angle,
                node_type_edge,
                padding_mask,
                mask_aa,
                mask_pos,
                mask_angle,
                time_pos,
                time_angle,
            )

            # add 3d edge feature in node feature
            x = torch.cat([x, node3dfeature], dim=-1)

        return x, edge_feature, delta_pos
