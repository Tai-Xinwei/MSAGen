# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..modules.pbc import CellExpander
from ..psm_config import PSMConfig
from .graphormer_sentence_encoder_layer import GraphormerSentenceEncoderLayer


class PSMEncoder(nn.Module):
    """
    Class for the invariant encoder in the PSM model.
    """

    def __init__(self, args, psm_config: PSMConfig):
        """
        Initialize the PSMEncoder class.
        """
        super(PSMEncoder, self).__init__()

        self.layers = nn.ModuleList([])

        for nl in range(psm_config.num_encoder_layers):
            self.layers.extend(
                [
                    self.build_transformer_sentence_encoder_layer(
                        psm_config=psm_config,
                        embedding_dim=psm_config.embedding_dim,
                        ffn_embedding_dim=psm_config.ffn_embedding_dim,
                        num_attention_heads=psm_config.num_attention_heads,
                        dropout=psm_config.dropout,
                        attention_dropout=psm_config.attn_dropout,
                        activation_dropout=psm_config.act_dropout,
                        activation_fn=psm_config.activation_fn,
                        nl=nl,
                        args=args,
                    )
                ]
            )

        self.cell_expander = CellExpander(
            psm_config.pbc_expanded_distance_cutoff,
            psm_config.pbc_expanded_token_cutoff,
            psm_config.pbc_expanded_num_cell_per_direction,
            psm_config.pbc_multigraph_cutoff,
        )

        self.psm_config = psm_config

    def build_transformer_sentence_encoder_layer(
        self,
        psm_config,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        nl,
        args,
    ):
        """
        Build the transformer sentence encoder layer.
        Args:
            embedding_dim: Dimension of the embedding.
            ffn_embedding_dim: Dimension of the feed-forward network embedding.
            num_attention_heads: Number of attention heads.
            dropout: Dropout rate.
            attention_dropout: Attention dropout rate.
            activation_dropout: Activation dropout rate.
            activation_fn: Activation function.
            nl: Number of layers.
            args: Command line arguments.
        Returns:
            GraphormerSentenceEncoderLayer: The transformer sentence encoder layer.
        """
        return GraphormerSentenceEncoderLayer(
            psm_config=psm_config,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            nl=nl,
            args=args,
            pp_mode=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batch_data: Dict,
        masked_token_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the PSMEncoder class.
        Args:
            x (torch.Tensor): Input tensor, [B, L, H].
            padding_mask (torch.Tensor): Padding mask, [B, L].
            batch_data (Dict): Input data for the forward pass.
            masked_token_type (torch.Tensor): The masked token type, [B, L].
        Returns:
            torch.Tensor: Encoded tensor, [B, L, H].
        """
        attn_mask = None
        if (
            "pbc" in batch_data
            and batch_data["pbc"] is not None
            and torch.any(batch_data["pbc"])
        ):
            pos = batch_data["pos"]
            pbc = batch_data["pbc"]
            atoms = batch_data["token_id"]
            cell = batch_data["cell"]
            pbc_expand_batched = self.cell_expander.expand(
                pos, pbc, atoms, cell, self.psm_config.pbc_use_local_attention
            )
        else:
            pbc_expand_batched = None

        for nl, layer in enumerate(self.layers):
            x, _ = layer(
                x,
                batch_data,
                masked_token_type,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                pbc_expand_batched=pbc_expand_batched,
            )

        return x, pbc_expand_batched
