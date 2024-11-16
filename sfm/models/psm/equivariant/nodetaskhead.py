# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict, Optional

import torch
import torch.nn as nn
from sympy import ff
from torch import Tensor

from sfm.models.psm.invariant.dit_encoder import DiTBlock
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
        attn_bias,
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
        attn = attn + attn_bias
        min_dtype = torch.finfo(k.dtype).min
        attn = attn.masked_fill(expand_mask.unsqueeze(1).unsqueeze(2), min_dtype)
        attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1), min_dtype)

        attn_probs_float = nn.functional.softmax(attn.float(), dim=-1)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = attn_probs.view(bsz, self.num_heads, n_node, extend_n_node)

        # [bsz, n, n_extend, 3]
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


class VectorOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(VectorOutput, self).__init__()
        # self.adanorm = AdaNorm(hidden_channels)
        # self.output_network = nn.Linear(hidden_channels, 1, bias=False)
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(self, x, v):
        # v = self.adanorm(v)
        v = self.output_network(v)
        return v.squeeze(-1)


class VectorProjOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(VectorProjOutput, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 3, bias=False),
        )

    def forward(self, x, v):
        x = self.output_network(x)
        return x


class ForceVecOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(ForceVecOutput, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 3 * hidden_channels, bias=False),
        )
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(self, x, v):
        x = self.mlp(x)
        x = x.view(x.size(0), x.size(1), 3, -1)
        x = self.proj(x).squeeze(-1)
        return x


class VectorGatedOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(VectorGatedOutput, self).__init__()
        self.up_proj = nn.Sequential(
            nn.Linear(hidden_channels, 3 * hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(3 * hidden_channels),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
        )
        self.vec_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(self, x, v):
        gate = self.gate_proj(x)
        vec = self.up_proj(x)
        vec = vec.view(x.size(0), x.size(1), 3, -1)
        x = self.vec_proj(vec * gate.unsqueeze(-2)).squeeze(-1)
        return x


class ConditionVectorGatedOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(ConditionVectorGatedOutput, self).__init__()
        self.up_proj = nn.Linear(hidden_channels, 3 * hidden_channels, bias=False)
        self.gate_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.shift_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.vec_proj = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(self, x, c):
        gate = self.gate_proj(c)
        shift = self.shift_proj(c)
        vec = self.up_proj(x)
        vec = vec.view(x.size(0), x.size(1), 3, -1)
        x = self.vec_proj(vec * (1 + gate.unsqueeze(-2)) + shift.unsqueeze(-2)).squeeze(
            -1
        )
        return x


class ScalarGatedOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(ScalarGatedOutput, self).__init__()
        self.up_proj = nn.Sequential(
            nn.Linear(hidden_channels, 3 * hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(3 * hidden_channels),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
        )
        self.vec_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=True),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1, bias=True),
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        vec = self.up_proj(x)
        x = vec.view(x.size(0), x.size(1), 3, -1).mean(dim=-2)
        x = self.vec_proj(x * gate)
        return x


class ForceGatedOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(ForceGatedOutput, self).__init__()
        self.up_proj = nn.Sequential(
            nn.Linear(hidden_channels, 3 * hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(3 * hidden_channels),
        )
        self.gate_proj = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 3, bias=False),
        )
        self.vec_proj = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 1, bias=False),
        )

    def forward(self, x, v):
        gate = self.gate_proj(x)
        vec = self.up_proj(x)
        vec = vec.view(x.size(0), x.size(1), 3, -1)
        x = self.vec_proj(vec).squeeze(-1) * gate
        return x


class DiffusionModule(nn.Module):
    def __init__(
        self,
        args,
        psm_config: PSMConfig,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        self.pos_emb = nn.Linear(3, psm_config.embedding_dim, bias=False)

        self.pair_feat_bias = nn.Sequential(
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.encoder_pair_embed_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.num_attention_heads,
                bias=False,
            ),
        )

        for nl in range(psm_config.num_pred_attn_layer):
            self.layers.extend([DiTBlock(args, psm_config)])

    def forward(
        self,
        batched_data: Dict,
        x,
        time_emb,
        attn_bias,
        padding_mask,
        mixed_attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
        pair_feat: Optional[Tensor] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)

        pos_embedding = self.pos_emb(
            batched_data["pos"].to(self.pos_emb.weight.dtype)
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if pair_feat is not None:
            pair_feat_bias = self.pair_feat_bias(pair_feat).permute(0, 3, 1, 2)
        else:
            pair_feat_bias = None

        if mixed_attn_bias is not None:
            mixed_attn_bias = mixed_attn_bias + pair_feat_bias
        else:
            mixed_attn_bias = pair_feat_bias

        for _, layer in enumerate(self.layers):
            pos_embedding = layer(
                pos_embedding + time_emb,
                x,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=ifbackprop,
            )

        return pos_embedding


class DiffusionModule2(nn.Module):
    def __init__(
        self,
        args,
        psm_config: PSMConfig,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        self.pos_emb = nn.Linear(3, psm_config.embedding_dim, bias=False)

        self.pair_feat_bias = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.num_attention_heads,
                bias=False,
            ),
        )

        self.pair2node = nn.Sequential(
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.encoder_pair_embed_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.encoder_pair_embed_dim, psm_config.embedding_dim, bias=False
            ),
        )

        for nl in range(psm_config.num_pred_attn_layer):
            self.layers.extend([DiTBlock(args, psm_config)])

    def forward(
        self,
        batched_data: Dict,
        x,
        time_emb,
        attn_bias,
        padding_mask,
        mixed_attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
        pair_feat: Optional[Tensor] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)

        pos_embedding = self.pos_emb(
            batched_data["pos"].to(self.pos_emb.weight.dtype)
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if pair_feat is not None:
            pair_feat_bias = self.pair_feat_bias(pair_feat).permute(0, 3, 1, 2)

            pair_feat = pair_feat.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(1), 0.0
            )
            pair_feat = pair_feat.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(2), 0.0
            )

            feat2nodeindex = torch.topk(pair_feat_bias.mean(dim=1), 3, dim=1)[1]
            feat2node = torch.gather(
                pair_feat,
                1,
                feat2nodeindex.unsqueeze(-1).expand(-1, -1, -1, pair_feat.size(-1)),
            ).mean(dim=1)
            feat2node = self.pair2node(feat2node)
        else:
            pair_feat_bias = None
            feat2node = None

        if mixed_attn_bias is not None:
            mixed_attn_bias = mixed_attn_bias + pair_feat_bias
        else:
            mixed_attn_bias = pair_feat_bias

        if feat2node is not None:
            x = x + feat2node

        for _, layer in enumerate(self.layers):
            pos_embedding = layer(
                pos_embedding + time_emb,
                x,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=ifbackprop,
            )

        return pos_embedding


class DiffusionModule3(nn.Module):
    def __init__(
        self,
        args,
        psm_config: PSMConfig,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        self.pos_emb = nn.Linear(3, psm_config.embedding_dim, bias=False)

        self.pair_feat_bias = nn.Sequential(
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.encoder_pair_embed_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.num_attention_heads,
                bias=False,
            ),
        )

        self.pair2node = nn.Sequential(
            nn.Linear(
                psm_config.encoder_pair_embed_dim,
                psm_config.encoder_pair_embed_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.encoder_pair_embed_dim, psm_config.embedding_dim, bias=False
            ),
        )

        for nl in range(psm_config.num_pred_attn_layer):
            self.layers.extend(
                [
                    DiTBlock(
                        args,
                        psm_config,
                        embedding_dim=psm_config.decoder_hidden_dim,
                        ffn_embedding_dim=psm_config.decoder_ffn_dim,
                    )
                ]
            )

    def forward(
        self,
        batched_data: Dict,
        x,
        time_emb,
        attn_bias,
        padding_mask,
        mixed_attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
        pair_feat: Optional[Tensor] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)

        pos_embedding = self.pos_emb(
            batched_data["pos"].to(self.pos_emb.weight.dtype)
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if pair_feat is not None:
            pair_feat_bias = self.pair_feat_bias(pair_feat).permute(0, 3, 1, 2)

            pair_feat = pair_feat.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(1), 0.0
            )
            pair_feat = pair_feat.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(2), 0.0
            )

            # feat2nodeindex = torch.topk(pair_feat_bias.mean(dim=1), 20, dim=1)[1]
            # feat2node = torch.gather(
            #     pair_feat,
            #     1,
            #     feat2nodeindex.unsqueeze(-1).expand(-1, -1, -1, pair_feat.size(-1)),
            # ).mean(dim=1)
            # feat2node = self.pair2node(feat2node)
            feat2node = None
        else:
            pair_feat_bias = None
            feat2node = None

        if mixed_attn_bias is not None:
            mixed_attn_bias = mixed_attn_bias + pair_feat_bias
        else:
            mixed_attn_bias = pair_feat_bias

        if feat2node is not None:
            x = x + feat2node

        for _, layer in enumerate(self.layers):
            pos_embedding = layer(
                pos_embedding + time_emb,
                x,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=ifbackprop,
            )

        return pos_embedding
