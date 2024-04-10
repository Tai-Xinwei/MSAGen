# -*- coding: utf-8 -*-
import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sfm.logging import logger


class PSMMixEmbedding(nn.Module):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config):
        """
        Initialize the PSMMixEmbedding class.
        ## 0-31: amino acid embedding; 32-159: atom type embedding
        """
        super(PSMMixEmbedding, self).__init__()

        ## 0-31: amino acid embedding; 32-159: atom type embedding
        self.embed = nn.Embedding(160, psm_config.encoder_embed_dim)

    def forward(self, batch_data: Dict) -> Tensor:
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
        return x, padding_mask, mask_token_type
