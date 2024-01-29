# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from sfm.logging import logger
from sfm.modules.FairseqDropout import FairseqDropout

from .pfm_layer import Edge3DEmbedding, ResidueFeatureV0, ResidueFeatureV1


class PFMEmbedding(nn.Module):
    def __init__(
        self,
        pfm_config,
    ):
        super().__init__()
        self.pfm_config = pfm_config
        self.residue_feature = ResidueFeatureV0(
            num_residues=pfm_config.num_residues,
            hidden_dim=pfm_config.embedding_dim,
            max_len=1024,
            prop_feat=False,
            angle_feat=False,
        )

    def forward(
        self,
        batched_data,
        padding_mask,
        pos=None,
        mask_aa=None,
        mask_pos=None,
        time=None,
    ):
        x = self.residue_feature(
            batched_data, mask_aa=mask_aa, mask_pos=mask_pos, time=time
        )
        edge_feature = None
        delta_pos = None

        return x, edge_feature, delta_pos
