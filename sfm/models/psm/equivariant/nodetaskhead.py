# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.invariant.dit_encoder import DiTBlock
from sfm.models.psm.modules.embedding import PSMMixEmbedding
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


class AAVectorProjOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(AAVectorProjOutput, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, 111, bias=False),
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
        padding_mask,
        mixed_attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
        pair_feat: Optional[Tensor] = None,
        dist_map: Optional[Tensor] = None,
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
        padding_mask,
        mixed_attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
        pair_feat: Optional[Tensor] = None,
        dist_map: Optional[Tensor] = None,
        clean_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)

        if pbc_expand_batched is not None:
            # use pbc and multi-graph
            # expand_pos = torch.cat([batched_data["pos"], pbc_expand_batched["expand_pos"]], dim=1)
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )
        else:
            # expand_pos = batched_data["pos"]
            expand_mask = padding_mask

        pos_embedding = self.pos_emb(
            batched_data["pos"].to(self.pos_emb.weight.dtype)
        ).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if pair_feat is not None:
            if pbc_expand_batched is not None:
                expand_mask = torch.cat(
                    [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
                )
            else:
                expand_mask = padding_mask

            pair_feat_bias = self.pair_feat_bias(pair_feat).permute(0, 3, 1, 2)

            pair_feat = pair_feat.masked_fill(
                expand_mask.unsqueeze(-1).unsqueeze(1), 0.0
            )
            pair_feat = pair_feat.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(2), 0.0
            )

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

        return pos_embedding, None


class AADiffusionModule(nn.Module):
    def __init__(
        self,
        args,
        psm_config: PSMConfig,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        self.atom_pos_emb = nn.Linear(3, psm_config.embedding_dim, bias=False)
        self.residue_pos_emb = nn.Linear(111, psm_config.embedding_dim, bias=False)

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

        for _ in range(psm_config.num_pred_attn_layer):
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
        padding_mask,
        mixed_attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
        pair_feat: Optional[Tensor] = None,
        dist_map: Optional[Tensor] = None,
        clean_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)

        if pbc_expand_batched is not None:
            # use pbc and multi-graph
            # expand_pos = torch.cat([batched_data["pos"], pbc_expand_batched["expand_pos"]], dim=1)
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )
        else:
            # expand_pos = batched_data["pos"]
            expand_mask = padding_mask

        # mix pos embedding
        if batched_data["is_protein"].any():
            B, L, _, _ = batched_data["pos"].shape
            pos_embedding_res = self.residue_pos_emb(
                batched_data["pos"].view(B, L, -1).to(self.residue_pos_emb.weight.dtype)
            ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
            pos_embedding_atom = self.atom_pos_emb(
                batched_data["pos"][:, :, 1, :].to(self.residue_pos_emb.weight.dtype)
            ).masked_fill(padding_mask.unsqueeze(-1), 0.0)
            pos_embedding = torch.where(
                batched_data["is_protein"].unsqueeze(-1),
                pos_embedding_res,
                pos_embedding_atom,
            )
        else:
            pos_embedding = self.atom_pos_emb(
                batched_data["pos"][:, :, 0, :].to(self.residue_pos_emb.weight.dtype)
            ).masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if pair_feat is not None:
            if pbc_expand_batched is not None:
                expand_mask = torch.cat(
                    [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
                )
            else:
                expand_mask = padding_mask

            pair_feat_bias = self.pair_feat_bias(pair_feat).permute(0, 3, 1, 2)

            pair_feat = pair_feat.masked_fill(
                expand_mask.unsqueeze(-1).unsqueeze(1), 0.0
            )
            pair_feat = pair_feat.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(2), 0.0
            )

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


class InvariantDiffusionModule(nn.Module):
    def __init__(
        self,
        args,
        psm_config: PSMConfig,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        self.embedding = PSMMixEmbedding(psm_config)

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

        for nl in range(psm_config.num_structure_encoder_layer):
            self.layers.extend(
                [
                    DiTBlock(
                        args,
                        psm_config,
                        embedding_dim=psm_config.structure_hidden_dim,
                        ffn_embedding_dim=psm_config.structure_ffn_dim,
                    )
                ]
            )

    def forward(
        self,
        batched_data: Dict,
        x,
        time_emb,
        padding_mask,
        mixed_attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
        pair_feat: Optional[Tensor] = None,
        dist_map: Optional[Tensor] = None,
        clean_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)

        pos_embedding, _, _, pos_attn_bias = self.embedding(
            batched_data,
            time_step=None,
            aa_mask=None,
            clean_mask=clean_mask,
            pbc_expand_batched=pbc_expand_batched,
            ignore_mlm_from_decoder_feature=True,
        )

        if pair_feat is not None:
            if pbc_expand_batched is not None:
                expand_mask = torch.cat(
                    [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
                )
            else:
                expand_mask = padding_mask

            pair_feat_bias = self.pair_feat_bias(pair_feat).permute(0, 3, 1, 2)

            pair_feat = pair_feat.masked_fill(
                expand_mask.unsqueeze(-1).unsqueeze(1), 0.0
            )
            pair_feat = pair_feat.masked_fill(
                padding_mask.unsqueeze(-1).unsqueeze(2), 0.0
            )

            feat2node = None
        else:
            pair_feat_bias = None
            feat2node = None

        if mixed_attn_bias is not None:
            mixed_attn_bias = mixed_attn_bias + pair_feat_bias + pos_attn_bias
        else:
            mixed_attn_bias = pair_feat_bias + pos_attn_bias

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

        return pos_embedding, mixed_attn_bias


class GaussianDiffusion_SEQDIFF:
    """
    T = number of timesteps to set up diffuser with

    schedule = type of noise schedule to use linear, cosine, gaussian

    noise = type of ditribution to sample from; DEFAULT - normal_gaussian

    """

    def __init__(
        self,
        T=1000,
        schedule="sqrt",
        sample_distribution="normal",
        sample_distribution_gmm_means=[-1.0, 1.0],
        sample_distribution_gmm_variances=[1.0, 1.0],
        F=1,
    ):
        # Use float64 for accuracy.
        betas = np.array(get_named_beta_schedule(schedule, T), dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        self.F = F

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)

        # sample_distribution_params
        self.sample_distribution = sample_distribution
        self.sample_distribution_gmm_means = [
            float(mean) for mean in sample_distribution_gmm_means
        ]
        self.sample_distribution_gmm_variances = [
            float(variance) for variance in sample_distribution_gmm_variances
        ]

        if self.sample_distribution == "normal":
            self.noise_function = torch.randn_like
        else:
            self.noise_function = self.randnmixture_like

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, mask=None, DEVICE=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """

        # noise_function is determined in init depending on type of noise specified
        noise = self.noise_function(x_start) * (self.F**2)
        if DEVICE is not None:
            noise = noise.to(DEVICE)

        assert noise.shape == x_start.shape
        x_sample = (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        if mask is not None:
            x_sample[mask] = x_start[mask]

        return x_sample

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape

        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)

        posterior_log_variance_clipped = _extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def randnmixture_like(self, tensor_like, number_normal=3, weights_normal=None):
        if (
            self.sample_distribution_gmm_means
            and self.sample_distribution_gmm_variances
        ):
            assert len(self.sample_distribution_gmm_means) == len(
                self.sample_distribution_gmm_variances
            )

        if not weights_normal:
            mix = torch.distributions.Categorical(
                torch.ones(len(self.sample_distribution_gmm_means))
            )  # number_normal
        else:
            assert len(weights_normal) == number_normal
            mix = torch.distributions.Categorical(weights_normal)
        # comp = torch.distributions.Normal(torch.randn(number_normal), torch.rand(number_normal))
        comp = torch.distributions.Normal(
            torch.tensor(self.sample_distribution_gmm_means),
            torch.tensor(self.sample_distribution_gmm_variances),
        )
        # comp = torch.distributions.Normal([-3, 3], [1, 1])
        # comp = torch.distributions.Normal([-3, 0, 3], [1, 1, 1])
        # comp = torch.distributions.Normal([-3, 0, 3], [1, 1, 1])
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
        return torch.tensor(
            [gmm.sample() for _ in range(np.prod(tensor_like.shape))]
        ).reshape(tensor_like.shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def _extract(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
