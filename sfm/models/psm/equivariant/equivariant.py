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
            ffn_embedding_dim=psm_config.ffn_embedding_dim,
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
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        pos = batched_data["pos"]
        if pbc_expand_batched is None:
            if (
                "pbc" in batched_data
                and batched_data["pbc"] is not None
                and torch.any(batched_data["pbc"])
            ):
                pos = batched_data["pos"]
                pbc = batched_data["pbc"]
                atoms = batched_data["masked_token_type"]
                cell = batched_data["cell"]
                pbc_expand_batched = self.cell_expander.expand(
                    pos, pbc, atoms, cell, self.psm_config.pbc_use_local_attention
                )

                n_node = atoms.size()[-1]
                masked_token_type_i = (
                    atoms.unsqueeze(-1).repeat(1, 1, n_node).unsqueeze(-1)
                )
                masked_token_type_j = (
                    atoms.unsqueeze(1).repeat(1, n_node, 1).unsqueeze(-1)
                )
                pair_token_type = torch.cat(
                    [masked_token_type_i, masked_token_type_j], dim=-1
                )
                outcell_index = pbc_expand_batched["outcell_index"]
                expand_pair_token_type = torch.gather(
                    pair_token_type,
                    dim=2,
                    index=outcell_index.unsqueeze(1)
                    .unsqueeze(-1)
                    .repeat(1, n_node, 1, 2),
                )
                pair_token_type = torch.cat(
                    [pair_token_type, expand_pair_token_type], dim=2
                )
                pbc_expand_batched["expand_node_type_edge"] = pair_token_type
            else:
                pbc_expand_batched = None

        return self.model(batched_data, x, pos, padding_mask, pbc_expand_batched)
