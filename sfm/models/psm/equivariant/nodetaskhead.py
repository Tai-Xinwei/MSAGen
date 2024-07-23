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


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        # )
        super().__init__()
        self.psm_config = psm_config
        embed_dim = psm_config.encoder_embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        if psm_config.decoder_feat4energy:
            self.v_proj_energy = nn.Linear(embed_dim, embed_dim, bias=False)

        self.num_heads = psm_config.encoder_attention_heads
        self.scaling = (embed_dim // psm_config.encoder_attention_heads) ** -0.5
        self.embed_dim = embed_dim

        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        if psm_config.decoder_feat4energy:
            self.o_proj_energy = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        batched_data: Dict,
        x,
        pos_emb,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)
        bsz, n_node, _ = x.size()
        pos = batched_data["pos"]

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
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1.0)

        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)
        if self.psm_config.decoder_feat4energy:
            v_e = self.v_proj_energy(x)

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
        else:
            outcell_index = None

        if outcell_index is not None:
            outcell_index = outcell_index.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)

            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
            if self.psm_config.decoder_feat4energy:
                expand_v_e = torch.gather(v_e, dim=1, index=outcell_index)
                v_e = torch.cat([v_e, expand_v_e], dim=1)

        q = q.view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        k = k.view(bsz, extend_n_node, self.num_heads, -1).transpose(1, 2)
        v = v.view(bsz, extend_n_node, self.num_heads, -1).transpose(1, 2)
        if self.psm_config.decoder_feat4energy:
            v_e = v_e.view(bsz, extend_n_node, self.num_heads, -1).transpose(1, 2)

        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        min_dtype = torch.finfo(k.dtype).min
        attn = attn.masked_fill(expand_mask.unsqueeze(1).unsqueeze(2), min_dtype)
        attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1), min_dtype)

        attn_probs_float = nn.functional.softmax(attn.float(), dim=-1)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = attn_probs.view(bsz, self.num_heads, n_node, extend_n_node)

        delta_pos = delta_pos.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]

        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        decoder_vec_output = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head, 3, n, d]
        decoder_vec_output = (
            decoder_vec_output.permute(0, 3, 2, 1, 4)
            .contiguous()
            .view(bsz, n_node, 3, -1)
        )
        decoder_vec_output = self.o_proj(decoder_vec_output)

        if self.psm_config.decoder_feat4energy:
            decoder_x_output = attn_probs @ v_e
            decoder_x_output = decoder_x_output.permute(0, 2, 1, 3).reshape(
                bsz, n_node, -1
            )
            decoder_x_output = self.o_proj_energy(decoder_x_output)
        else:
            decoder_x_output = x

        return decoder_x_output, decoder_vec_output


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
        bsz, n_node, _ = x.size()

        # pos = batched_data["pos"].to(x.dtype)
        # # if pbc_expand_batched is not None:
        # #     expand_pos = pbc_expand_batched["expand_pos"]
        # #     expand_pos = torch.cat([pos, expand_pos], dim=1)
        # #     delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
        # #     expand_mask = torch.cat(
        # #         [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
        # #     )
        # #     extend_n_node = expand_pos.shape[1]
        # # else:
        # delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
        # expand_mask = padding_mask
        # extend_n_node = n_node

        # dist = delta_pos.norm(dim=-1).view(-1, n_node, extend_n_node)
        # dist = dist.masked_fill(padding_mask.unsqueeze(-1), 1e6)
        # dist = dist.masked_fill(expand_mask.unsqueeze(1), 1e6)
        # delta_pos = 1.0 / (dist + 1.0)

        # pos_emb = pos_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        # delta_pos = delta_pos.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        # delta_pos = delta_pos.masked_fill(expand_mask.unsqueeze(1).unsqueeze(-1), 0.0)
        # vec = self.vec_project(pos_emb).view(bsz, n_node, 3, -1)
        # vec = vec * delta_pos.sum(dim=-2).unsqueeze(-1).type_as(vec)  # [bsz, n, n_expand, 3, H]

        vec = pos_emb
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

        return x, vec


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

    def modulate(self, x, scale, shift):
        return x * scale + shift

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

        x = self.modulate(x, scale_vec.norm(dim=-2), shift_vec.norm(dim=-2))
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


class VectorOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(VectorOutput, self).__init__()
        self.adanorm = AdaNorm(hidden_channels)
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(self, x, v):
        v = self.adanorm(v)
        v = self.output_network(v)
        return v.squeeze(-1)


class VectorProjOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(VectorProjOutput, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3, bias=False),
        )

    def forward(self, x, v):
        x = self.output_network(x)
        return x
