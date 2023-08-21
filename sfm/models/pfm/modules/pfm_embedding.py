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
            num_heads=pfm_config.num_attention_heads,
            num_atoms=pfm_config.num_atoms,
            num_in_degree=pfm_config.num_in_degree,
            num_out_degree=pfm_config.num_out_degree,
            hidden_dim=pfm_config.embedding_dim,
            n_layers=pfm_config.num_encoder_layers,
            no_2d=pfm_config.no_2d,
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
        self, batched_data, padding_mask, node_mask, mask_2d=None, mask_3d=None
    ):
        x = self.residue_feature(batched_data, mask_2d=mask_2d)
        if self.pfm_config.add_3d:
            pos = batched_data["pos"]
            node_type_edge = batched_data["node_type_edge"]
            edge_feature, merged_edge_features, delta_pos = self.edge_3d_emb(
                pos, node_type_edge, padding_mask, node_mask
            )
            if mask_3d is not None:
                merged_edge_features, delta_pos = (
                    merged_edge_features * mask_3d[:, None, None],
                    delta_pos * mask_3d[:, None, None, None],
                )

            x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        return x, pos, edge_feature, delta_pos
