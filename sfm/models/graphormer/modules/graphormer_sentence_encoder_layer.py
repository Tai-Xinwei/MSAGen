# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.modules.droppath import DropPath
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import Fp32LayerNorm, LayerNorm
from sfm.modules.multihead_attention import MultiheadAttention
from sfm.modules.quant_noise import quant_noise


class GraphormerSentenceEncoderLayer(nn.Module):
    """
    Implements a Graphormer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
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
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        nl: int = 0,
        self_attn_mask: Optional[torch.Tensor] = None,
        args=None,
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

        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

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
        )

        # @ Roger added:
        self.sandwich_ln = sandwich_ln

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # layer norm associated with the self attention layer, sandwich
        # self.self_attn_sandwich_layer_norm = LayerNorm(self.embedding_dim, export=export) if self.sandwich_ln else None

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # self.final_sandwich_layer_norm = LayerNorm(self.embedding_dim, export=export) if self.sandwich_ln else None
        self.final_layer_norm_2 = LayerNorm(ffn_embedding_dim, export=export)

        self.nl = nl
        self.args = args
        self.self_attn_mask = self_attn_mask

        self.dummy = nn.Linear(1, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.self_attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()
        self.final_layer_norm_2.reset_parameters()

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        d_tilde=1,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            d_tilde=d_tilde,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_bias: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        # x: T x B x C
        residual = x
        x = self.self_attn_layer_norm(x)

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
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)

        x = self.final_layer_norm_2(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x, attn


class GraphormerSentenceEncoderLayer_PP(GraphormerSentenceEncoderLayer):
    """
    Implements a Graphormer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        graphormer_config,
        args,
        nl=0,
    ) -> None:
        super().__init__(
            graphormer_config.embedding_dim,
            graphormer_config.ffn_embedding_dim,
            graphormer_config.num_attention_heads,
            graphormer_config.dropout,
            graphormer_config.attn_dropout,
            graphormer_config.act_dropout,
            activation_fn=graphormer_config.activation_fn,
            sandwich_ln=graphormer_config.sandwich_ln,
            args=args,
            nl=nl,
        )
        self.graphormer_config = graphormer_config

    @classmethod
    def config(cls):
        return GraphormerConfig(
            hidden_size=cls.embedding_dim,
            intermediate_size=cls.ffn_embedding_dim,
            num_attention_heads=cls.num_attention_heads,
            hidden_act="relu",
        )

    def forward(self, input_tuple: tuple):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        if self.graphormer_config.use_pbc:
            (
                x,
                self_attn_padding_mask,
                self_attn_bias,
                input_ids,
                llm_mask,
                outcell_index,
                expand_mask,
            ) = input_tuple
            pbc_expand_batched = {
                "outcell_index": outcell_index,
                "expand_mask": expand_mask,
            }
        else:
            x, self_attn_padding_mask, self_attn_bias, input_ids, llm_mask = input_tuple
            pbc_expand_batched = None

        assert type(x) == torch.Tensor
        assert type(self_attn_bias) == torch.Tensor
        assert type(self_attn_padding_mask) == torch.Tensor
        # assert type(delta_pos) == torch.Tensor

        self_attn_mask = None
        attn_bias_temp = self_attn_bias[:, self.nl, :, :, :]
        # x: T x B x C

        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=attn_bias_temp,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            pbc_expand_batched=pbc_expand_batched,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.final_layer_norm_2(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if self.nl == self.graphormer_config.encoder_layers - 1:
            return_tuple = (
                x.contiguous(),
                self_attn_padding_mask.contiguous(),
                llm_mask.contiguous(),
                input_ids.contiguous(),
            )
        else:
            return_tuple = (
                x.contiguous(),
                self_attn_padding_mask.contiguous(),
                self_attn_bias.contiguous(),
                input_ids.contiguous(),
                llm_mask.contiguous(),
            )
            if self.graphormer_config.use_pbc:
                return_tuple += (
                    pbc_expand_batched["outcell_index"],
                    pbc_expand_batched["expand_mask"],
                )

        return return_tuple
