# -*- coding: utf-8 -*-
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.droppath import DropPath
from sfm.modules.FairseqDropout import FairseqDropout

# from fairseq import utils
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.mem_eff_attn import MemEffAttn, MemEffSelfAttn


class PSMPlainEncoderLayer(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()

        self.psm_config = psm_config

        # Initialize blocks
        self.activation_fn = get_activation_fn(psm_config.activation_fn)
        self.self_attn = self.build_self_attention(
            psm_config.embedding_dim,
            psm_config.num_attention_heads,
            dropout=psm_config.dropout,
            add_rope=True,
        )

        self.fc1 = self.build_fc1(
            psm_config.embedding_dim,
            psm_config.ffn_embedding_dim,
        )
        self.fc2 = self.build_fc2(
            psm_config.ffn_embedding_dim,
            psm_config.embedding_dim,
        )

        # sandwitch layernorm
        self.top_layer_norm = nn.LayerNorm(psm_config.embedding_dim)
        self.mid_layer_norm = nn.LayerNorm(psm_config.embedding_dim)

        self.args = args

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.top_layer_norm.reset_parameters()
        self.mid_layer_norm.reset_parameters()

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        d_tilde=1,
        add_rope=False,
    ):
        return MemEffSelfAttn(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            d_tilde=d_tilde,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=add_rope,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        masked_token_type: torch.Tensor,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        residual = x
        x = self.top_layer_norm(x)
        x, _ = self.self_attn(
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
            attn_mask=None,
        )
        x = residual + x

        residual = x
        x = self.mid_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, None
