# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn

from megatron.core import parallel_state, tensor_parallel
from megatron.model.transformer import ParallelAttention
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.modules.droppath import DropPath
from sfm.modules.FairseqDropout import FairseqDropout
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.parallelattentionbias import TPMultiheadAttention
from sfm.modules.quant_noise import quant_noise

try:
    from apex.normalization import FusedLayerNorm as LayerNormTP
except:
    raise ImportError("Please install apex from install/install_megatron.sh")


class GraphormerSentenceEncoderLayerMP(nn.Module):
    """
    Implements a Graphormer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        graphormer_config,
        mp_config,
        args,
        nl=0,
    ) -> None:
        super().__init__()
        self.nl = nl
        self.args = args
        self.graphormer_config = graphormer_config
        self.mp_config = mp_config

        # Initialize parameters
        self.embedding_dim = graphormer_config.embedding_dim
        self.ffn_embedding_dim = graphormer_config.ffn_embedding_dim
        self.num_attention_heads = graphormer_config.num_attention_heads
        self.attention_dropout = graphormer_config.attention_dropout
        self.activation_dropout = graphormer_config.activation_dropout
        self.activation_fn = graphormer_config.activation_fn
        self.sandwich_ln = graphormer_config.sandwich_ln

        self.dropout_module = FairseqDropout(
            graphormer_config.dropout, module_name=self.__class__.__name__
        )

        self.activation_dropout_module = FairseqDropout(
            graphormer_config.activation_dropout, module_name=self.__class__.__name__
        )

        self.activation_fn = get_activation_fn(graphormer_config.activation_fn)
        self.self_attn = self.build_self_attention_TP(
            self.embedding_dim,
            self.num_attention_heads,
            dropout=self.attention_dropout,
            d_tilde=args.d_tilde,
        )

        self.self_attn_layer_norm = LayerNormTP(self.embedding_dim)

        self.fc1 = self.build_fc1_TP(
            self.embedding_dim,
            self.ffn_embedding_dim,
        )

        self.fc2 = self.build_fc2_TP(
            self.ffn_embedding_dim,
            self.embedding_dim,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNormTP(self.embedding_dim)

        # self.final_sandwich_layer_norm = LayerNorm(self.embedding_dim, export=export) if self.sandwich_ln else None
        self.final_layer_norm_2 = LayerNormTP(
            self.ffn_embedding_dim // mp_config.tensor_model_parallel_size
        )

    def reset_parameters(self):
        # self.fc1.reset_parameters()
        # self.fc2.reset_parameters()
        self.self_attn_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()
        self.final_layer_norm_2.reset_parameters()

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc1_TP(self, input_dim, output_dim):
        return tensor_parallel.ColumnParallelLinear(
            input_dim,
            output_dim,
            config=self.mp_config,
            init_method=self.mp_config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
        )

    def build_fc2_TP(self, input_dim, output_dim):
        return tensor_parallel.RowParallelLinear(
            input_dim,
            output_dim,
            config=self.mp_config,
            init_method=self.mp_config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
        )

    def build_self_attention_TP(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        d_tilde=1,
    ):
        return TPMultiheadAttention(
            embed_dim,
            num_attention_heads,
            self.graphormer_config,
            self.mp_config,
            query_bias=True,
            key_bias=False,
            value_bias=True,
            output_bias=True,
            dropout=dropout,
            num_key_value_heads=self.mp_config.num_key_value_heads,
            d_tilde=d_tilde,
        )

    def forward(self, input_tuple: tuple):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """

        x, self_attn_padding_mask, self_attn_bias, input_ids, llm_mask = input_tuple
        # value_tensor, self_attn_padding_mask, shape_tensor = input_tuple
        # x, self_attn_bias, delta_pos = self.tensors_decode(value_tensor, shape_tensor)
        # x = x.to(torch.float16)
        # print("layer", self.nl)
        assert type(x) == torch.Tensor
        assert type(self_attn_bias) == torch.Tensor
        assert type(self_attn_padding_mask) == torch.Tensor
        # assert type(delta_pos) == torch.Tensor

        seq_len, _, _ = x.shape
        attn_bias_temp = self_attn_bias[:, self.nl, :, :, :]
        assert (
            self.num_attention_heads % self.mp_config.tensor_model_parallel_size == 0
        ), f"num_attention_heads {self.num_attention_heads} must be divisible by tensor_model_parallel_size {self.mp_config.tensor_model_parallel_size}"
        # num_head_pre_tprank = (
        #     self.num_attention_heads // self.mp_config.tensor_model_parallel_size
        # )
        self_attn_padding_mask_temp = (
            self_attn_padding_mask.unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, seq_len)
        )
        # x: T x B x C
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            hidden_states=x,
            key=x,
            value=x,
            attn_bias=attn_bias_temp,
            attention_mask=self_attn_padding_mask_temp,
            # attention_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x = self.final_layer_norm_2(x)
        x, _ = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if self.nl == self.graphormer_config.encoder_layers - 1:
            return (
                x.contiguous(),
                self_attn_padding_mask.contiguous(),
                input_ids.contiguous(),
                llm_mask.contiguous(),
            )
        else:
            return (
                x.contiguous(),
                self_attn_padding_mask.contiguous(),
                self_attn_bias.contiguous(),
                input_ids.contiguous(),
                llm_mask.contiguous(),
            )
