# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from sfm.logging import logger
from sfm.modules.FairseqDropout import FairseqDropout

from .pfm_layer import Edge3DEmbedding, ResidueFeature


class PFMEmbedding(nn.Module):
    def __init__(
        self,
        pfm_config,
    ):
        super().__init__()
        self.pfm_config = pfm_config
        self.residue_feature = ResidueFeature(
            num_residues=pfm_config.num_residues,
            hidden_dim=pfm_config.embedding_dim,
            max_len=1024,
            prop_feat=True,
            angle_feat=True,
        )

        self.edge_3d_emb = (
            Edge3DEmbedding(
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
        self, batched_data, padding_mask, pos=None, mask_aa=None, mask_pos=None
    ):
        x = self.residue_feature(batched_data, mask_aa=mask_aa)
        edge_feature = None
        delta_pos = None
        if mask_pos is not None and self.pfm_config.add_3d:
            node_type_edge = batched_data["node_type_edge"]
            edge_feature, merged_edge_features, delta_pos = self.edge_3d_emb(
                pos, node_type_edge, padding_mask, mask_pos
            )

            merged_edge_features = merged_edge_features.masked_fill(mask_pos, 0.0)
            delta_pos = delta_pos.masked_fill(mask_pos.unsqueeze(-1), 0.0)

            x = x + merged_edge_features * 0.01
            # edge_feature = torch.zeros_like(edge_feature)

        return x, edge_feature, delta_pos
