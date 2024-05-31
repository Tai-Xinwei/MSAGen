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
from sfm.modules.quant_noise import quant_noise


class ToxInternalEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer.
    """

    def __init__(
        self,
        droppath_prob: float = 0.0,
        args=None,
    ) -> None:
        super().__init__()
        self.args = self.check_args(args)

        if args.droppath_prob > 0.0:
            self.dropout_module = DropPath(droppath_prob)
        else:
            self.dropout_module = FairseqDropout(
                args.dropout, module_name=self.__class__.__name__
            )

        self.activation_dropout_module = FairseqDropout(
            args.activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = get_activation_fn(args.activation_fn)
        self.self_attn = MemEffAttn(
            # return MultiheadAttention(
            embed_dim=args.embedding_dim,
            num_heads=args.num_attention_heads,
            dropout=args.attntion_dropout,
            q_noise=args.q_noise,
            qn_block_size=args.qn_block_size,
            d_tilde=args.d_tilde,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=args.add_rope,
        )

        self.fc1 = self.build_fc1(
            args.embedding_dim,
            args.ffn_embedding_dim,
            q_noise=args.q_noise,
            qn_block_size=args.qn_block_size,
        )
        self.fc2 = self.build_fc2(
            args.ffn_embedding_dim,
            args.embedding_dim,
            q_noise=args.q_noise,
            qn_block_size=args.qn_block_size,
        )

        # sandwitch layernorm
        self.top_layer_norm = LayerNorm(args.embedding_dim, export=False)
        self.mid_layer_norm = LayerNorm(args.embedding_dim, export=False)
        self.final_layer_norm = LayerNorm(args.ffn_embedding_dim, export=False)

        self.args = args
        self.reset_parameters()

    def check_args(self, args):
        required_lst = [
            "droppath_prob",
            "dropout",
            "activation_dropout",
            "attntion_dropout",
            "activation_fn",
            "embedding_dim",
            "num_attention_heads",
            "q_noise",
            "qn_block_size",
            "d_tilde",
            "add_rope",
            "ffn_embedding_dim",
        ]
        for k in required_lst:
            assert hasattr(args, k), f"args must have {k}"
        return args

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.top_layer_norm.reset_parameters()
        self.mid_layer_norm.reset_parameters()
        self.final_layer_norm.reset_parameters()

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x = self.top_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=None,  # we do not use attn_bias for now.
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        residual = x
        x = self.mid_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.final_layer_norm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        return x
