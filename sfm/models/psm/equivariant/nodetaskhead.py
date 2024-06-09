# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sfm.models.psm.psm_config import PSMConfig


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        # )
        super().__init__()
        embed_dim = psm_config.encoder_embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_heads = psm_config.encoder_attention_heads
        self.scaling = (embed_dim // psm_config.encoder_attention_heads) ** -0.5

    def forward(
        self,
        batched_data: Dict,
        x,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ) -> Tensor:
        x = x.transpose(0, 1)
        bsz, n_node, _ = x.size()
        pos = batched_data["pos"]
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        dist = dist.masked_fill(padding_mask.unsqueeze(-1), 10000.0)
        dist = dist.masked_fill(padding_mask.unsqueeze(1), 10000.0)
        delta_pos /= dist.unsqueeze(-1) + 1e-4

        q = (
            self.q_proj(x).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(x).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        min_dtype = torch.finfo(k.dtype).min
        attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), min_dtype)
        attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(-1), min_dtype)

        attn_probs_float = nn.functional.softmax(attn.float(), dim=-1)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = attn_probs.view(bsz, self.num_heads, n_node, n_node)

        delta_pos = delta_pos.masked_fill(padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0)
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        decoder_vec_output = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        decoder_vec_output = (
            decoder_vec_output.permute(0, 3, 2, 1, 4)
            .contiguous()
            .view(bsz, n_node, 3, -1)
        )

        decoder_x_output = x

        return decoder_x_output, decoder_vec_output


class Non_equi_head(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        # )
        super().__init__()
        embed_dim = psm_config.encoder_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim * 3),
        )

    def forward(
        self,
        batched_data: Dict,
        x,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ) -> Tensor:
        # query = query.contiguous()
        x = x.transpose(0, 1)
        bsz, n_node, _ = x.size()

        decoder_x_output = x
        decoder_vec_output = self.mlp(x).view(bsz, n_node, 3, -1)

        return decoder_x_output, decoder_vec_output


class VectorOutput(nn.Module):
    def __init__(self, hidden_channels=768):
        super(VectorOutput, self).__init__()
        self.output_network = nn.Linear(hidden_channels, 1, bias=False)

    def forward(self, x, v):
        v = self.output_network(v)
        return v.squeeze(-1)
