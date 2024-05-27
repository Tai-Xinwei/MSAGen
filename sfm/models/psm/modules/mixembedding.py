# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.invariant.mixture_bias import PSMBias
from sfm.models.psm.modules.timestep_encoder import TimeStepEncoder
from sfm.models.psm.psm_config import PSMConfig


class PSMMix3dEmbedding(nn.Module):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig):
        """
        Initialize the PSMMixEmbedding class.
        ## [1, 128]: atom type; [129, 159] amino acid type
        """
        super(PSMMix3dEmbedding, self).__init__()

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
        # self.pos_embedding_bias = PSMBias(psm_config, key_prefix="")
        # self.init_pos_embedding_bias = PSMBias(psm_config, key_prefix="init_")
        self.pos_emb = nn.Linear(3, psm_config.num_3d_bias_kernel, bias=False)
        self.pos_feature_emb = nn.Linear(
            psm_config.num_3d_bias_kernel, psm_config.embedding_dim, bias=False
        )

        self.psm_config = psm_config

    def _pos_emb(self, batched_data: Dict, padding_mask: torch.Tensor):
        pos = batched_data["pos"]
        pos = pos.to(self.pos_emb.weight.dtype)
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = 1.0 / (delta_pos.norm(dim=-1) + 1.0)
        dist = dist.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        pos_emb = self.pos_emb(pos)
        dist = torch.nn.functional.softmax(dist, dim=1)
        pos_feature_emb = torch.matmul(
            dist, pos_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        )
        pos_feature_emb = self.pos_feature_emb(pos_feature_emb)
        pos_feature_emb = pos_feature_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return pos_feature_emb

    def forward(
        self,
        batched_data: Dict,
        time_step: Optional[Tensor] = None,
        clean_mask: Optional[Tensor] = None,
        aa_mask: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
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

        if aa_mask is not None:
            mask_token_type = token_id.masked_fill(
                aa_mask, 157
            )  # 157 is the mask token
        else:
            mask_token_type = token_id

        batched_data["masked_token_type"] = mask_token_type
        x = self.embed(mask_token_type)

        if self.psm_config.use_2d_atom_features and "node_attr" in batched_data:
            atom_feature_embedding = self.atom_feature_embed(
                batched_data["node_attr"][:, :, 1:]
            ).sum(
                dim=-2
            )  # B x T x #ATOM_FEATURE x D -> # B x T x D
            x += atom_feature_embedding

        if time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)
            x += time_embed.unsqueeze(1)

        # _, pos_embedding = self.pos_embedding_bias(
        #     batched_data, mask_token_type, padding_mask, pbc_expand_batched
        # )
        pos_embedding = self._pos_emb(batched_data, padding_mask)
        x += pos_embedding

        # if "init_pos" in batched_data:
        #     init_pos = batched_data["init_pos"]
        #     if pbc_expand_batched is not None:
        #         pos = batched_data["pos"]
        #         outcell_index = (
        #             pbc_expand_batched["outcell_index"].unsqueeze(-1).repeat(1, 1, 3)
        #         )
        #         expand_pos_no_offset = torch.gather(pos, dim=1, index=outcell_index)
        #         offset = batched_data["expand_pos"] - expand_pos_no_offset
        #         init_expand_pos_no_offset = torch.gather(
        #             init_pos, dim=1, index=outcell_index
        #         )
        #         init_expand_pos = torch.cat(
        #             [init_pos, init_expand_pos_no_offset + offset]
        #         )
        #         init_expand_pos = init_expand_pos.masked_fill(
        #             torch.cat(
        #                 [padding_mask, pbc_expand_batched["expand_mask"]], dim=1
        #             ).unsqueeze(-1),
        #             0.0,
        #         )
        #         batched_data["init_expand_pos"] = init_expand_pos
        #     _, init_pos_embedding = self.init_pos_embedding_bias(
        #         batched_data, mask_token_type, padding_mask, pbc_expand_batched
        #     )
        #     init_pos_mask = (
        #         (init_pos != 0.0).any(dim=-1, keepdim=False).any(dim=-1, keepdim=False)
        #     )
        #     x[init_pos_mask, :] += init_pos_embedding[init_pos_mask, :]

        return x, padding_mask, mask_token_type
