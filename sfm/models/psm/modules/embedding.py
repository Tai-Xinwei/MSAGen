# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

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
        ## 0-31: amino acid embedding; 32-159: atom type embedding
        """
        super(PSMMixEmbedding, self).__init__()

        ## 0-31: amino acid embedding; 32-159: atom type embedding
        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim)
        self.time_step_encoder = TimeStepEncoder(
            psm_config.num_timesteps, psm_config.embedding_dim, "positional"
        )
        self.pos_embedding_bias = PSMBias(psm_config)
        self.init_pos_embedding_bias = PSMBias(psm_config)

    def forward(
        self,
        batch_data: Dict,
        time_step: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
    ) -> Tensor:
        """
        Forward pass of the PSMMixEmbedding class.
        Args:
            batch_data: Input data for the forward pass.
        Returns:
            x: The embedding representation.
            padding_mask: The padding mask.
        """
        token_id = batch_data["token_id"]
        padding_mask = token_id.eq(0)  # B x T x 1

        # TODO: need to implement the masked token type here or in the encoder
        mask_token_type = batch_data["token_id"]

        x = self.embed(token_id)

        if time_step is not None:
            time_embed = self.time_step_encoder(time_step)
            x += time_embed.unsqueeze(1)

        _, pos_embedding = self.pos_embedding_bias(
            batch_data, token_id, padding_mask, pbc_expand_batched
        )
        x += pos_embedding

        return x, padding_mask, mask_token_type
