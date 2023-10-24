# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .FairseqDropout import FairseqDropout
from .layer_norm import Fp32LayerNorm, LayerNorm
from .quant_noise import quant_noise
from .rotary_embedding import RotaryEmbedding


class MemEffAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        d_tilde=1,
        add_rope=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (
            (self.head_dim / d_tilde) ** 0.5
        ) / self.head_dim  # when d_tilt == 1, match with original transformer scale

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        # @ shengjie added: no key bias for stability
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=False), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.layer_norm = LayerNorm(embed_dim)

        self.reset_parameters(d_tilde)

        self.onnx_trace = False

        self.rot_emb = None
        if add_rope:
            self.rot_emb = RotaryEmbedding(dim=self.head_dim)

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self, d_tilde=1):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.v_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
            nn.init.xavier_uniform_(
                self.q_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
            )
        else:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(d_tilde))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(d_tilde))

        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        self.layer_norm.reset_parameters()

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            pass

        tgt_len, bsz, embed_dim = query.size()

        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if k is not None:
            k = (
                k.contiguous()
                .view(src_len, bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(src_len, bsz, self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        # add rope
        if self.rot_emb:
            q, k = self.rot_emb(q, k)

        if key_padding_mask is not None and attn_bias is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

            attn_mask = (
                key_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .repeat(1, self.num_heads, tgt_len, 1)
                .bool()
            )

            attn_bias = attn_bias.masked_fill_(attn_mask, float("-inf"))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        with torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_mem_efficient=True, enable_flash=False
        ):
            attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=self.dropout
            )

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # attn = self.layer_norm(attn)
        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    # def upgrade_state_dict_named(self, state_dict, name):
    #     prefix = name + "." if name != "" else ""
    #     items_to_add = {}
    #     keys_to_remove = []
    #     for k in state_dict.keys():
    #         if k.endswith(prefix + "in_proj_weight"):
    #             # in_proj_weight used to be q + k + v with same dimensions
    #             dim = int(state_dict[k].shape[0] / 3)
    #             items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
    #             items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
    #             items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

    #             keys_to_remove.append(k)

    #             k_bias = prefix + "in_proj_bias"
    #             if k_bias in state_dict.keys():
    #                 dim = int(state_dict[k].shape[0] / 3)
    #                 items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
    #                 items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
    #                     dim : 2 * dim
    #                 ]
    #                 items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

    #                 keys_to_remove.append(prefix + "in_proj_bias")

    #     for k in keys_to_remove:
    #         del state_dict[k]

    #     for key, value in items_to_add.items():
    #         state_dict[key] = value
