# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.invariant.graphormer_2d_bias import GraphAttnBias
from sfm.models.psm.invariant.mixture_bias import PSMBias
from sfm.models.psm.modules.timestep_encoder import TimeStepEncoder
from sfm.models.psm.psm_config import PSMConfig


class PSMMixEmbedding(nn.Module):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig):
        """
        Initialize the PSMMixEmbedding class.
        ## [1, 128]: atom type; [129, 159] amino acid type
        """
        super(PSMMixEmbedding, self).__init__()

        ## [1, 128]: atom type; [129, 154] amino acid type
        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim)

        # embedding for 2D
        self.atom_feature_embed = nn.Embedding(
            psm_config.num_atom_features, psm_config.encoder_embed_dim
        )

        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps,
            psm_config.embedding_dim,
            psm_config.diffusion_time_step_encoder_type,
        )
        self.pos_embedding_bias = PSMBias(psm_config, key_prefix="")
        self.init_pos_embedding_bias = PSMBias(psm_config, key_prefix="init_")

        if psm_config.use_2d_bond_features:
            self.graph_2d_attention_bias = GraphAttnBias(psm_config)

        self.psm_config = psm_config

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        clean_mask: Optional[Tensor] = None,
        aa_mask: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ignore_mlm_from_decoder_feature: bool = False,
    ) -> Tensor:
        """
        Forward pass of the PSMMixEmbedding class.
        Args:
            batched_data: Input data for the forward pass.
        Returns:
            x: The embedding representation.
            padding_mask: The padding mask.
        """
        token_id = batched_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1

        is_molecule = batched_data["is_molecule"]
        batch_size, max_num_nodes = padding_mask.size()[:2]

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        batched_data["masked_token_type"] = mask_token_type
        x = self.embed(mask_token_type)

        if (
            self.psm_config.use_2d_atom_features
            and "node_attr" in batched_data
            and is_molecule.any()
        ):
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~is_molecule.unsqueeze(-1).unsqueeze(-1), 0.0
            )
            x += atom_feature_embedding

        is_protein = batched_data["is_protein"].any(dim=-1).all()

        if time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)
        else:
            time_embed = None

        if is_protein and (not ignore_mlm_from_decoder_feature) and (not self.psm_config.mlm_from_decoder_feature):
            return x, padding_mask, time_embed, None

        if time_embed is not None:
            x += time_embed

        pos_attn_bias, pos_embedding = self.pos_embedding_bias(
            batched_data, padding_mask, pbc_expand_batched
        )
        x += pos_embedding

        batch_size, _, max_num_nodes, max_num_expanded_nodes = pos_attn_bias.size()
        if self.psm_config.share_attention_bias:
            pos_attn_bias = pos_attn_bias.reshape(
                batch_size,
                self.psm_config.num_attention_heads,
                max_num_nodes,
                max_num_expanded_nodes,
            ).contiguous()
        else:
            pos_attn_bias = (
                pos_attn_bias.reshape(
                    batch_size,
                    self.psm_config.num_encoder_layers + 1,
                    self.psm_config.num_attention_heads,
                    max_num_nodes,
                    max_num_expanded_nodes,
                )
                .permute(1, 0, 2, 3, 4)
                .contiguous()
            )

        if "init_pos" in batched_data and (batched_data["init_pos"] != 0.0).any():
            init_pos = batched_data["init_pos"]
            init_pos_attn_bias, init_pos_embedding = self.init_pos_embedding_bias(
                batched_data, padding_mask, pbc_expand_batched
            )
            init_pos_mask = (
                (init_pos != 0.0).any(dim=-1, keepdim=False).any(dim=-1, keepdim=False)
            )
            x[init_pos_mask, :] += init_pos_embedding[init_pos_mask, :]

            init_pos_attn_bias = init_pos_attn_bias.masked_fill(
                ~init_pos_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            if self.psm_config.share_attention_bias:
                init_pos_attn_bias = init_pos_attn_bias.reshape(
                    batch_size,
                    self.psm_config.num_attention_heads,
                    max_num_nodes,
                    max_num_expanded_nodes,
                ).contiguous()
            else:
                init_pos_attn_bias = (
                    init_pos_attn_bias.reshape(
                        batch_size,
                        self.psm_config.num_encoder_layers + 1,
                        self.psm_config.num_attention_heads,
                        max_num_nodes,
                        max_num_expanded_nodes,
                    )
                    .permute(1, 0, 2, 3, 4)
                    .contiguous()
                )
            pos_attn_bias += init_pos_attn_bias

        if self.psm_config.use_2d_bond_features and is_molecule.any():
            graph_2d_attention_bias = self.graph_2d_attention_bias(batched_data)
            graph_2d_attention_bias = graph_2d_attention_bias.masked_fill(
                ~is_molecule.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0
            )

            # TODO:(shiyu) need extra handling if considering catalyst systems
            if self.psm_config.share_attention_bias:
                pos_attn_bias[
                    :, :, :max_num_nodes, :max_num_nodes
                ] += graph_2d_attention_bias[:, :, 1:, 1:]
            else:
                pos_attn_bias[
                    :, :, :, :max_num_nodes, :max_num_nodes
                ] += graph_2d_attention_bias[:, :, :, 1:, 1:]

        return x, padding_mask, time_embed, pos_attn_bias
