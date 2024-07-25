# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.mem_eff_attn import MemEffAttn


class VecMemEffAttnWithProteinRotaryEmbedding(MemEffAttn):
    def forward(
        self,
        q,
        k,
        v,
        key_padding_mask,
        attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ):
        if pbc_expand_batched is not None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-2], k.size()[-1])
            )

            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            expand_mask = None

        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]
        pos_feature_num = q.shape[2]

        if self.rot_emb is not None and is_protein is not None and is_protein.any():
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, tgt_len, self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            is_protein_expanded = (
                is_protein.unsqueeze(1)
                .repeat(1, pos_feature_num * self.num_heads, 1)
                .view(bsz * pos_feature_num * self.num_heads, tgt_len, 1)
            )
            rot_q, rot_k = self.rot_emb(
                q,
                k,
                v,
                position_ids=position_ids,
                nhead=pos_feature_num * self.num_heads,
            )
            q = torch.where(is_protein_expanded, rot_q, q)
            k = torch.where(is_protein_expanded, rot_k, k)
            q = (
                q.reshape(bsz, pos_feature_num, self.num_heads, tgt_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
        else:
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )

        if key_padding_mask is not None:
            if pbc_expand_batched is not None:
                expand_mask = pbc_expand_batched["expand_mask"]
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=-1)
            if key_padding_mask.bool().any():
                attn_mask = torch.zeros(
                    (bsz, self.num_heads, tgt_len, src_len),
                    device=q.device,
                    dtype=q.dtype,
                )
                if attn_bias is not None:
                    attn_mask += attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_mask = None
                if attn_bias is not None:
                    attn_mask = attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=q.dtype)
            attn = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout,
                attn_mask=attn_mask,
                is_causal=False,
            )

        attn = (
            attn.transpose(1, 2)
            .reshape(bsz, tgt_len, self.num_heads, 3, self.head_dim)
            .transpose(2, 3)
            .reshape(bsz, tgt_len, 3, self.embed_dim)
        )

        attn = self.out_proj(attn)

        return attn


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

        pos = batched_data["pos"].to(x.dtype)

        if pbc_expand_batched is not None:
            expand_pos = pbc_expand_batched["expand_pos"]
            expand_pos = torch.cat([pos, expand_pos], dim=1)
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )
            extend_n_node = expand_pos.shape[1]
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            expand_mask = padding_mask
            expand_pos = pos
            extend_n_node = n_node

        dist = delta_pos.norm(dim=-1).view(-1, n_node, extend_n_node)
        dist = dist.masked_fill(padding_mask.unsqueeze(-1), 1e6)
        dist = dist.masked_fill(expand_mask.unsqueeze(1), 1e6)
        dist = 1.0 / (dist + 1.0)
        dist = dist.unsqueeze(1).expand(-1, self.psm_config.num_attention_heads, -1, -1)

        # vec = self.vec_project(pos_emb).view(bsz, n_node, 3, -1)
        vec = self.vec_project(x).view(bsz, n_node, 3, -1)

        for layer in self.layers:
            x, vec = layer(batched_data, x, vec, dist, padding_mask, pbc_expand_batched)

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

        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(psm_config.embedding_dim, psm_config.embedding_dim, bias=False),
        )
        self.adaLN_modulation_vec_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )

        self.attn = VecMemEffAttnWithProteinRotaryEmbedding(
            psm_config.embedding_dim,
            psm_config.num_attention_heads,
            dropout=psm_config.dropout,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=False,
            use_smooth_softmax=psm_config.use_smooth_softmax,
            smooth_factor=psm_config.smooth_factor,
        )

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
        dist,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        residue_x = x
        residue_vec = vec

        x = self.ln_x(x)
        vec = self.ln_vec(vec)

        scale_x = self.adaLN_modulation_x(x)

        vec = self.modulate(vec, scale_x.unsqueeze(-2))

        vec = self.attn(
            vec,
            vec,
            vec,
            attn_bias=dist,
            key_padding_mask=padding_mask,
            is_protein=batched_data["is_protein"],
            position_ids=batched_data["position_ids"],
            pbc_expand_batched=pbc_expand_batched,
        )

        x = x + residue_x
        vec = vec + residue_vec

        residue_x = x
        residue_vec = vec

        vec = self.vec_mlp_norm(vec)

        vec = self.mlp(vec)
        scale_vec_mlp, shift_vec_mlp = self.adaLN_modulation_vec_mlp(vec).chunk(
            2, dim=-1
        )
        x = self.modulate(x, scale_vec_mlp.sum(dim=-2), shift_vec_mlp.sum(dim=-2))

        x = x + residue_x
        vec = vec + residue_vec

        return x, vec
