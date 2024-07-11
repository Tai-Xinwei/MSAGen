# -*- coding: utf-8 -*-
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.modules.timestep_encoder import TimeStepEncoder
from sfm.models.psm.psm_config import PSMConfig


class PSMMix3DEquivEmbedding(nn.Module):
    """
    Class for the embedding layer in the PSM model.
    """

    def __init__(self, psm_config: PSMConfig):
        """
        Initialize the PSMMixEmbedding class.
        ## [1, 128]: atom type; [129, 159] amino acid type
        """
        super(PSMMix3DEquivEmbedding, self).__init__()

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
        self.pos_emb = nn.Linear(1, psm_config.num_3d_bias_kernel, bias=False)
        self.pos_feature_emb = nn.Linear(
            psm_config.num_3d_bias_kernel, psm_config.embedding_dim, bias=False
        )

        self.scaling = (psm_config.num_3d_bias_kernel) ** -0.5

        self.psm_config = psm_config

    def _pos_emb(
        self,
        pos: Optional[torch.Tensor],
        expand_pos: torch.Tensor,
        padding_mask: torch.Tensor,
        pbc_expand_batched: Optional[Dict[str, Tensor]] = None,
    ):
        pos = pos.to(self.pos_emb.weight.dtype)
        if pbc_expand_batched is not None:
            expand_pos = expand_pos.to(self.pos_emb.weight.dtype)
            expand_pos = torch.cat([pos, expand_pos], dim=1)
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            expand_mask = padding_mask
            expand_pos = pos
            local_attention_weight = None

        dist = 1.0 / (delta_pos.norm(dim=-1) + 1.0)
        min_dtype = torch.finfo(dist.dtype).min
        dist = dist.masked_fill(expand_mask.unsqueeze(1), min_dtype)
        dist = dist.masked_fill(padding_mask.unsqueeze(-1), min_dtype)
        if local_attention_weight is not None:
            local_attention_weight = local_attention_weight.to(dtype=pos.dtype)
            dist = dist.masked_fill(local_attention_weight <= 1e-5, min_dtype)

        pos_emb = (
            self.pos_emb(expand_pos.norm(dim=-1).unsqueeze(-1))
            .masked_fill(expand_mask.unsqueeze(-1), 0.0)
            .float()
        )
        dist = torch.nn.functional.softmax(dist.float() * self.scaling, dim=-1)
        if local_attention_weight is not None:
            dist = dist * local_attention_weight
        pos_feature_emb = torch.matmul(dist, pos_emb).to(self.pos_emb.weight.dtype)
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
        is_molecule = batched_data["is_molecule"]

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
            atom_feature_embedding = atom_feature_embedding.masked_fill(
                ~is_molecule.unsqueeze(-1).unsqueeze(-1), 0.0
            )
            x += atom_feature_embedding

        if time_step is not None:
            time_embed = self.time_step_encoder(time_step, clean_mask)
            x += time_embed  # .unsqueeze(1)

        pos_embedding = self._pos_emb(
            batched_data["pos"],
            pbc_expand_batched["expand_pos"]
            if pbc_expand_batched is not None
            else None,
            padding_mask,
            pbc_expand_batched=pbc_expand_batched,
        )
        x += pos_embedding

        if "init_pos" in batched_data and (batched_data["init_pos"] != 0.0).any():
            init_pos = batched_data["init_pos"]
            init_pos_embedding = self._pos_emb(
                init_pos,
                batched_data["init_expand_pos"]
                if pbc_expand_batched is not None
                else None,
                padding_mask,
                pbc_expand_batched=pbc_expand_batched,
            )
            init_pos_mask = (
                (init_pos != 0.0).any(dim=-1, keepdim=False).any(dim=-1, keepdim=False)
            )
            x[init_pos_mask, :] += init_pos_embedding[init_pos_mask, :]

        return x, padding_mask, time_embed, None
