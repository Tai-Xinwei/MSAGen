# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sfm.logging import logger
from sfm.modules.quant_noise import quant_noise
from sfm.modules.rotary_embedding import SFMRotaryEmbedding


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
        k_bias=False,
        q_bias=True,
        v_bias=True,
        o_bias=True,
        q_noise=0.0,
        qn_block_size=8,
        d_tilde=1,
        add_rope=False,
        layer_norm=False,
        add_quant_noise=False,
        use_smooth_softmax=False,
        smooth_factor=0.0,
        use_no_pre_cutoff_softmax=False,
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
        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
        self.scaling = (
            (self.head_dim / d_tilde) ** 0.5
        ) / self.head_dim  # when d_tilt == 1, match with original transformer scale

        if add_quant_noise:
            self.k_proj = quant_noise(
                nn.Linear(self.kdim, embed_dim, bias=k_bias), q_noise, qn_block_size
            )
            self.v_proj = quant_noise(
                nn.Linear(self.vdim, embed_dim, bias=v_bias), q_noise, qn_block_size
            )
            self.q_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=q_bias), q_noise, qn_block_size
            )

            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=o_bias), q_noise, qn_block_size
            )
        else:
            self.k_proj = nn.Linear(self.kdim, embed_dim, bias=k_bias)
            self.v_proj = nn.Linear(self.vdim, embed_dim, bias=v_bias)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=q_bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=o_bias)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.reset_parameters(d_tilde)

        self.onnx_trace = False

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)
        self.use_no_pre_cutoff_softmax = use_no_pre_cutoff_softmax

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
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def forward(
        self,
        query,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        math_kernel: bool = False,
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

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
        else:
            outcell_index = None
            expand_mask = None
            local_attention_weight = None

        if outcell_index is not None:
            outcell_index = (
                outcell_index.transpose(1, 0).unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            expand_k = torch.gather(k, dim=0, index=outcell_index)
            expand_v = torch.gather(v, dim=0, index=outcell_index)

            k = torch.cat([k, expand_k], dim=0)
            v = torch.cat([v, expand_v], dim=0)

            src_len = k.size()[0]

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if k is not None:
            k = (
                k.contiguous()
                .view(src_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(src_len, bsz * self.num_heads, self.head_dim)
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
            q, k = self.rot_emb(q, k, v, position_ids, self.num_heads)

        if key_padding_mask is not None:
            if outcell_index is not None:
                assert expand_mask is not None
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=1)
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

            if key_padding_mask.bool().any():
                attn_mask = torch.zeros(
                    (bsz, self.num_heads, tgt_len, src_len),
                    device=q.device,
                    dtype=q.dtype,
                )
                if attn_bias is not None:
                    attn_mask += attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )
                attn_mask = attn_mask.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_mask = None
                if attn_bias is not None:
                    attn_mask = attn_bias.contiguous().view(
                        bsz, self.num_heads, tgt_len, src_len
                    )

        if local_attention_weight is not None:
            local_attention_weight = local_attention_weight.to(dtype=q.dtype)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.use_smooth_softmax:
                attn_weights = (
                    attn_weights + self.smooth_factor
                ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
            elif self.use_no_pre_cutoff_softmax:
                pass
            else:
                attn_weights = attn_weights.masked_fill(
                    local_attention_weight.unsqueeze(1) <= 1e-5, float("-inf")
                )

            if attn_mask is not None:
                attn_weights += attn_mask

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            attn_probs = nn.functional.softmax(attn_weights, dim=-1)

            if local_attention_weight is not None:
                attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
                attn_probs = attn_probs * local_attention_weight.unsqueeze(1)
                attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

            attn = torch.bmm(attn_probs, v)
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        else:
            # if attn_bias is not None:
            # raise NotImplementedError("mem efficient attn not support attn_bias")

            # FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed.
            # Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
            # with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, self.num_heads, src_len, self.head_dim)
            v = v.view(bsz, self.num_heads, src_len, self.head_dim)

            if math_kernel:
                context = sdpa_kernel([SDPBackend.MATH])
            else:
                context = sdpa_kernel(
                    [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                )

            with context:
                attn = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout,
                    attn_mask=attn_mask,
                    is_causal=False,
                )

            attn = (
                attn.transpose(1, 2)
                .contiguous()
                .view(bsz, tgt_len, embed_dim)
                .transpose(0, 1)
            )

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)

        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class MemEffSelfAttn(nn.Module):
    """Multi-headed self attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=False,
        k_bias=False,
        q_bias=True,
        v_bias=True,
        o_bias=True,
        q_noise=0.0,
        qn_block_size=8,
        d_tilde=1,
        add_rope=False,
        layer_norm=False,
        add_quant_noise=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (
            (self.head_dim / d_tilde) ** 0.5
        ) / self.head_dim  # when d_tilt == 1, match with original transformer scale

        if add_quant_noise:
            self.qkv_proj = quant_noise(nn.Linear(embed_dim, 3 * embed_dim, bias=bias))
            self.out_proj = quant_noise(
                nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
            )
        else:
            self.qkv_proj = nn.Linear(
                embed_dim, 3 * embed_dim, bias=bias
            )  # TODO: add an option here
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.layer_norm = nn.LayerNorm(embed_dim) if layer_norm else None
        self.reset_parameters(d_tilde)
        self.onnx_trace = False

        self.rot_emb = (
            SFMRotaryEmbedding(
                self.head_dim,
            )
            if add_rope
            else None
        )

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self, d_tilde=1):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        nn.init.xavier_uniform_(
            self.qkv_proj.weight, gain=1.0 / (math.sqrt(2 * d_tilde))
        )
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.layer_norm is not None:
            self.layer_norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        before_softmax: bool = False,
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
        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        qkv = self.qkv_proj(x)
        # Separate Q, K, V from linear output
        qkv = qkv.view(tgt_len, bsz, 3, self.num_heads * self.head_dim).transpose(0, 1)
        q, k, v = qkv.chunk(3, dim=-2)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        # TODO: add rope
        if self.rot_emb is not None:
            q, k = self.rot_emb(q, k, v, position_ids, self.num_heads)

        # do not support attn_bias, just key_padding_mask
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                (bsz, self.num_heads, tgt_len, tgt_len), device=q.device, dtype=q.dtype
            )
            attn_mask.masked_fill_(
                key_padding_mask.view(bsz, 1, 1, tgt_len).to(torch.bool), float("-inf")
            )

        # FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed.
        # Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attn = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout,
                attn_mask=attn_mask,
                is_causal=False,
            )  # [B, H, L, D]

        attn = (
            attn.transpose(1, 2)  # [B, L, H, D]
            .contiguous()
            .view(bsz, tgt_len, embed_dim)  # [B, L, H*D]
        )

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)
        attn = self.out_proj(attn).transpose(0, 1)
        attn_weights: Optional[Tensor] = None
        return attn, attn_weights
