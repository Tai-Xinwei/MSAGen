# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.modules.multihead_attention import (
    MemEffAttnWithProteinRotaryEmbedding,
)
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.layer_norm import AdaNorm
from sfm.modules.mem_eff_attn import MemEffAttn


class VectorVanillaTransformer(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        super().__init__()
        self.psm_config = psm_config
        embed_dim = psm_config.encoder_embed_dim

        self.layers = nn.ModuleList([])
        self.vec_project = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 3, bias=False),
        )

        for _ in range(psm_config.num_pred_attn_layer):
            self.layers.append(VectorTransformerBlock(psm_config))

        self.adaLN_modulation_vec = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )
        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )

    def modulate(self, x, scale, shift):
        return x * scale + shift

    def forward(
        self,
        batched_data: Dict,
        x,
        pos_emb,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        x = x.transpose(0, 1)
        encoder_x = x

        bsz, n_node, _ = x.size()

        vec = self.vec_project(pos_emb).view(bsz, n_node, 3, -1)
        # vec = pos_emb

        for layer in self.layers:
            x, vec = layer(batched_data, x, vec, padding_mask, pbc_expand_batched)

        residue_x = x
        residue_vec = vec

        scale_vec, shift_vec = self.adaLN_modulation_vec(vec).chunk(2, dim=-1)
        scale_x, shift_x = self.adaLN_modulation_x(x).chunk(2, dim=-1)

        x = self.modulate(x, scale_vec.norm(dim=-2), shift_vec.norm(dim=-2))
        vec = self.modulate(vec, scale_x.unsqueeze(-2), shift_x.unsqueeze(-2))

        x = x + residue_x
        vec = vec + residue_vec

        if self.psm_config.decoder_feat4energy:
            return x, vec
        else:
            return encoder_x, vec


class VectorTransformerBlock(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        # )
        super().__init__()
        self.psm_config = psm_config
        embed_dim = psm_config.encoder_embed_dim
        self.ln_x = nn.LayerNorm(embed_dim)
        self.ln_vec = nn.LayerNorm(embed_dim)

        self.adaLN_modulation_vec = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )
        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )
        self.adaLN_modulation_x_pre_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )

        if psm_config.only_use_rotary_embedding_for_protein:
            attn_cls = MemEffAttnWithProteinRotaryEmbedding
        else:
            attn_cls = MemEffAttn

        self.attn = attn_cls(
            psm_config.embedding_dim,
            psm_config.num_attention_heads,
            dropout=psm_config.dropout,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=True,
            use_smooth_softmax=psm_config.use_smooth_softmax,
            smooth_factor=psm_config.smooth_factor,
        )

        self.x_mlp_norm = nn.LayerNorm(psm_config.embedding_dim)
        self.vec_mlp_norm = nn.LayerNorm(psm_config.embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(
                psm_config.embedding_dim, psm_config.ffn_embedding_dim, bias=False
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.ffn_embedding_dim, psm_config.embedding_dim, bias=False
            ),
        )

    def modulate(self, x, scale, shift=None):
        if shift is not None:
            return x * scale + shift
        else:
            return x * scale

    def forward(
        self,
        batched_data: Dict,
        x,
        vec,
        padding_mask,
        delta_pos: Tensor = None,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        residue_x = x
        residue_vec = vec

        x = self.ln_x(x)
        vec = self.ln_vec(vec)

        scale_vec, shift_vec = self.adaLN_modulation_vec(vec).chunk(2, dim=-1)
        scale_x, shift_x = self.adaLN_modulation_x(x).chunk(2, dim=-1)

        x = self.modulate(x, scale_vec.sum(dim=-2), shift_vec.sum(dim=-2))
        vec = self.modulate(vec, scale_x.unsqueeze(-2), shift_x.unsqueeze(-2))

        if self.psm_config.only_use_rotary_embedding_for_protein:
            x = self.attn(
                x.transpose(0, 1),
                key_padding_mask=padding_mask,
                is_protein=batched_data["is_protein"],
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
            )[0].transpose(0, 1)
        else:
            x = self.attn(
                x.transpose(0, 1),
                key_padding_mask=padding_mask,
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
            )[0].transpose(0, 1)

        x = x + residue_x
        vec = vec + residue_vec

        residue_x = x
        residue_vec = vec

        x = self.x_mlp_norm(x)
        vec = self.vec_mlp_norm(vec)

        scale_x_mlp, shift_x_mlp = self.adaLN_modulation_x_pre_mlp(x).chunk(2, dim=-1)
        x = self.mlp(x)
        vec = self.modulate(vec, scale_x_mlp.unsqueeze(-2), shift_x_mlp.unsqueeze(-2))

        x = x + residue_x
        vec = vec + residue_vec

        return x, vec
