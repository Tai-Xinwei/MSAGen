# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


class NodeTaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        # self.force_proj = nn.Linear(embed_dim, 1, bias=False)

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
        pos = batched_data["pos"]
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1).view(-1, n_node, n_node)
        delta_pos /= dist.unsqueeze(-1) + 1e-5

        q = (
            self.q_proj(x).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
            * self.scaling
        )
        k = self.k_proj(x).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

        attn = q @ k.transpose(-1, -2)  # [bsz, head, n, n]
        min_dtype = torch.finfo(k.dtype).min
        attn = attn.masked_fill(~padding_mask.unsqueeze(1).unsqueeze(-1), min_dtype)

        attn_probs_float = nn.functional.softmax(attn.view(-1, n_node, n_node), dim=-1)
        attn_probs = attn_probs_float.type_as(attn)
        attn_probs = attn_probs.view(bsz, self.num_heads, n_node, n_node)

        delta_pos = delta_pos.masked_fill(
            ~padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0
        )
        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(
            attn_probs
        )  # [bsz, head, n, n, 3]
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)  # [bsz, head , 3, n, d]
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)
        # f1 = self.force_proj(x[:, :, 0, :]).view(bsz, n_node, 1, -1)
        # f2 = self.force_proj(x[:, :, 1, :]).view(bsz, n_node, 1, -1)
        # f3 = self.force_proj(x[:, :, 2, :]).view(bsz, n_node, 1, -1)
        # cur_force = torch.cat([f1, f2, f3], dim=-2)
        return x
