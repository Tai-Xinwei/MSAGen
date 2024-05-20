# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from sfm.models.psm.invariant.mixture_bias import PSMBias
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.droppath import DropPath
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.mem_eff_attn import MemEffAttn
from sfm.modules.multihead_attention import MultiheadAttention


class GraphormerSentenceEncoderLayer(nn.Module):
    """
    Implements a Graphormer Encoder Layer
    """

    def __init__(
        self,
        psm_config: PSMConfig,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        droppath_prob: float = 0.0,
        nl: int = 0,
        self_attn_mask: Optional[torch.Tensor] = None,
        args=None,
        pp_mode: bool = True,  # used in pipeline mode or not
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        if droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = FairseqDropout(
                dropout, module_name=self.__class__.__name__
            )

        self.pre_attn_norm = nn.LayerNorm(self.embedding_dim)
        self.pre_mlp_norm = nn.LayerNorm(self.embedding_dim)

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            d_tilde=args.d_tilde,
            use_memory_efficient_attention=psm_config.use_memory_efficient_attention,
        )

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
        )

        self.attn_bias = self.build_attn_bias(psm_config)

        self.nl = nl
        self.args = args
        self.self_attn_mask = self_attn_mask

        if pp_mode:  # create dummy parameter only when used in pipeline mode
            self.dummy = nn.Linear(1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.pre_attn_norm.reset_parameters()
        self.pre_mlp_norm.reset_parameters()

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        d_tilde=1,
        use_memory_efficient_attention=False,
    ):
        if use_memory_efficient_attention:
            return MemEffAttn(
                embed_dim,
                num_attention_heads,
                dropout=dropout,
                self_attention=self_attention,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                d_tilde=d_tilde,
                layer_norm=False,
                k_bias=False,
                q_bias=False,
                v_bias=False,
                o_bias=False,
            )
        else:
            return MultiheadAttention(
                embed_dim,
                num_attention_heads,
                dropout=dropout,
                self_attention=self_attention,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                d_tilde=d_tilde,
                layer_norm=False,
                k_bias=False,
                q_bias=False,
                v_bias=False,
                o_bias=False,
            )

    def build_attn_bias(self, psm_config):
        return PSMBias(psm_config)

    def forward(
        self,
        x: torch.Tensor,
        batched_data: Dict,
        masked_token_type: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        graph_2d_attention_bias: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        Args:
            x: Input tensor [T x B x C].
            batched_data: Input data for the forward pass.
            masked_token_type: The masked token type [B, L].
            self_attn_mask: The self-attention mask [B, L, L].
            self_attn_padding_mask: The self-attention padding mask [B, L].
            pbc_expand_batched: The pbc expand batched data.
            graph_2d_attention_bias: 2D attention bias, if use bond features in molecules
        """
        # # TODO: graphormer stype attn bias
        self_attn_bias, _ = self.attn_bias(
            batched_data,
            masked_token_type,
            self_attn_padding_mask,
            pbc_expand_batched=pbc_expand_batched,
        )

        # expand graph_2d_attention_bias according to PBC, just to match the shape of 3D attention bias
        # for periodic systems graph_2d_attention_bias should be all zero
        # here we use pbc and token id together to differentiate molecules from periodic systems and proteins
        if graph_2d_attention_bias is not None:
            expanded_graph_2d_attention_bias = torch.zeros_like(self_attn_bias)
            token_id = batched_data["token_id"]
            pbc = batched_data["pbc"]
            molecule_mask = torch.all(token_id <= 128, dim=-1) & ~torch.any(pbc, dim=-1)
            n_node = graph_2d_attention_bias.size()[-1] - 1
            expanded_graph_2d_attention_bias[
                :, :, :n_node, :n_node
            ] = graph_2d_attention_bias[
                :, :, 1:, 1:
            ]  # leave out virtual node in 2D attention bias
            expanded_graph_2d_attention_bias.masked_fill(
                ~molecule_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), 0.0
            )
            self_attn_bias += expanded_graph_2d_attention_bias

        # x: T x B x C
        residual = x
        x = self.pre_attn_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            pbc_expand_batched=pbc_expand_batched,
        )

        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.pre_mlp_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x, attn
