# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Optional

import torch
import torch.nn as nn

from sfm.models.psm.psm_config import PSMConfig

from ..modules.pbc import CellExpander
from .geomformer import GeomFormer


class EquivariantEncoder(nn.Module):
    def __init__(self):
        super(EquivariantEncoder, self).__init__()

    def forward(self, x):
        pass


class EquivariantDecoder(nn.Module):
    def __init__(self, psm_config: PSMConfig):
        super(EquivariantDecoder, self).__init__()
        self.psm_config = psm_config

        # use GeoMFormer as equivariant decoder
        self.model = GeomFormer(
            psm_config=psm_config,
            num_pred_attn_layer=psm_config.num_pred_attn_layer,
            embedding_dim=psm_config.embedding_dim,
            num_attention_heads=psm_config.num_attention_heads,
            ffn_embedding_dim=psm_config.decoder_ffn_dim,
            dropout=psm_config.dropout,
            attention_dropout=psm_config.attention_dropout,
            activation_dropout=psm_config.activation_dropout,
            num_3d_bias_kernel=psm_config.num_3d_bias_kernel,
            num_edges=psm_config.num_edges,
            num_atoms=psm_config.num_atoms,
        )

        self.cell_expander = CellExpander(
            psm_config.pbc_expanded_distance_cutoff,
            psm_config.pbc_expanded_token_cutoff,
            psm_config.pbc_expanded_num_cell_per_direction,
            psm_config.pbc_multigraph_cutoff,
        )

    def forward(
        self,
        batched_data,
        x,
        mixed_attn_bias,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        pos = batched_data["pos"]
        return self.model(
            batched_data, x, pos, mixed_attn_bias, padding_mask, pbc_expand_batched
        )
