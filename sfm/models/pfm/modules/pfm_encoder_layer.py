# -*- coding: utf-8 -*-
from typing import Callable, Optional

import torch
import torch.nn as nn

from sfm.modules.droppath import DropPath
from sfm.modules.FairseqDropout import FairseqDropout

# from fairseq import utils
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.layer_norm import LayerNorm
from sfm.modules.mem_eff_attn import MemEffAttn
from sfm.modules.multihead_attention import MultiheadAttention
from sfm.modules.multihead_attention_flash import FlashAttn
from sfm.modules.quant_noise import quant_noise

from .pfm_layer import Graph2DBias, Graph3DBias


class PFMEncoderLayer(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
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
        pfm_config=None,
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
        self.pfm_config = pfm_config

        self.nl = nl
        self.args = args
        self.self_attn_mask = self_attn_mask

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
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            d_tilde=args.d_tilde,
            add_rope=True,
        )

        self.sandwich_ln = sandwich_ln

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

        # sandwitch layernorm
        self.top_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.mid_layer_norm = LayerNorm(self.embedding_dim, export=export)
        # self.final_layer_norm = LayerNorm(ffn_embedding_dim, export=export)

        # self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        # self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # self.top_layer_norm = nn.LayerNorm(self.embedding_dim)
        # self.mid_layer_norm = nn.LayerNorm(self.embedding_dim)

        # TODO: 2D attention bias needs carefully designed, features such as MSA should be included
        # self.graph_attn_bias = graph2dBias()

        # # TODO: reuse the 3D attention bias from Graphormer, modification may needed
        # self.graph_3d_bias = (
        #     Graph3DBias(
        #         num_heads=pfm_config.num_attention_heads,
        #         num_kernel=pfm_config.num_3d_bias_kernel,
        #     )
        #     if pfm_config.add_3d
        #     else None
        # )

        # dummy param for lora, do not remove
        self.dummy = nn.Linear(1, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.top_layer_norm.reset_parameters()
        self.mid_layer_norm.reset_parameters()

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        q_noise,
        qn_block_size,
        d_tilde=1,
        add_rope=False,
    ):
        # TODO: needs to be replaced by flash-att
        if self.args.flash_attn:
            return FlashAttn(
                embed_dim,
                num_attention_heads,
                dropout=dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                d_tilde=d_tilde,
                add_rope=add_rope,
                layer_norm=False,
            )
        else:
            return MemEffAttn(
                # return MultiheadAttention(
                embed_dim,
                num_attention_heads,
                dropout=dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                d_tilde=d_tilde,
                add_rope=add_rope,
                layer_norm=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_feature: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        mask_pos: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        self_3d_attn_bias = None
        # if self.pfm_config.add_3d:
        #     # [bs, nHead, nnode, nnode]
        #     self_3d_attn_bias = self.graph_3d_bias(self_attn_padding_mask, edge_feature)
        #     # mae task need to mask the 3d attn bias
        #     if mask_pos is not None and self.pfm_config.noise_mode == "mae":
        #         self_3d_attn_bias = self_3d_attn_bias.masked_fill_(
        #             (
        #                 (self_3d_attn_bias != float("-inf")) * mask_pos[:, None, :, :]
        #             ).bool(),
        #             0.0,
        #         )

        residual = x
        x = self.top_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_3d_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            # position_ids=position_ids,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.mid_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x, self_3d_attn_bias
