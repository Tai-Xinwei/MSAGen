# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .FairseqDropout import FairseqDropout
from .layer_norm import Fp32LayerNorm, LayerNorm
from .quant_noise import quant_noise
from .rotary_embedding import SFM2DRotaryEmbedding, SFMRotaryEmbedding


class MultiheadAttention(nn.Module):
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
        self_attention=True,
        q_noise=0.0,
        qn_block_size=8,
        d_tilde=1,
        k_bias=False,
        q_bias=True,
        v_bias=True,
        o_bias=True,
        add_rope=False,
        layer_norm=False,
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
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
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

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=q_bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=v_bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=o_bias), q_noise, qn_block_size
        )

        if layer_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.reset_parameters(d_tilde)

        self.onnx_trace = False

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
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
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
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
            need_weights = True

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

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
        else:
            outcell_index = None
            expand_mask = None
            local_attention_weight = None

        if outcell_index is not None:
            if position_ids is not None:
                torch.gather(position_ids, dim=1, index=outcell_index)

            outcell_index = (
                outcell_index.transpose(1, 0).unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            expand_k = torch.gather(k, dim=0, index=outcell_index)
            expand_v = torch.gather(v, dim=0, index=outcell_index)

            k = torch.cat([k, expand_k], dim=0)  # [L_expand, B,]
            v = torch.cat([v, expand_v], dim=0)
            if position_ids is not None:
                position_ids = (
                    torch.arange(k.shape[0], device=k.device, dtype=k.dtype)
                    .unsqueeze(0)
                    .repeat(v.shape[1], 1)
                )
            # position_ids = torch.cat([position_ids, expand_position_ids], dim=1)

            src_len = k.size()[0]

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
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

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if local_attention_weight is not None:
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
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(
            attn_weights.float(), dim=-1
        ).type_as(attn_weights)

        if local_attention_weight is not None:
            attn_weights_float = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights_float = attn_weights_float * local_attention_weight.unsqueeze(
                1
            )
            attn_weights_float = attn_weights_float.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_weights = attn_weights_float.type_as(attn_weights)

        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)

        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class MSA_MultiheadAttention(nn.Module):
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
        self_attention=True,
        q_noise=0.0,
        qn_block_size=8,
        d_tilde=1,
        k_bias=False,
        q_bias=True,
        v_bias=True,
        o_bias=True,
        add_rope=False,
        layer_norm=False,
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
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
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

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=q_bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=v_bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=o_bias), q_noise, qn_block_size
        )

        if layer_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.reset_parameters(d_tilde)

        self.onnx_trace = False

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
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
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
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
            need_weights = True

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
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
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
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(
            attn_weights.float(), dim=-1
        ).type_as(attn_weights)

        attn_weights = attn_weights_float.type_as(attn_weights)

        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)

        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


class RowSelfAttention(nn.Module):
    """Compute self-attention over rows of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        k_bias=False,
        q_bias=False,
        v_bias=False,
        o_bias=False,
        add_rope=False,
        max_tokens_per_msa: int = 2**16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "rhnij"
        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFM2DRotaryEmbedding(dim=self.head_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=v_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=q_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=o_bias)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def row_causal_mask(self, attn_weights):
        r, h, n, i, j = attn_weights.shape
        nw = torch.zeros_like(attn_weights)
        cumsum = torch.zeros_like(attn_weights[0])
        for k in range(r):
            cumsum += attn_weights[k]
            nw[k] = cumsum / (k + 1)
        return nw

    def _batched_forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[
                    :, start : start + max_rows
                ]
                if self_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(
                x[start : start + max_rows], attn_probs
            )
            outputs.append(output)

        output = torch.cat(outputs, 0)
        return output, attn_probs

    def compute_attention_weights(
        self,
        x,
        scaling: float,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        position_ids=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        k = self.k_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(
                4
            ).to(q)

        if self.rot_emb:
            q, k = self.rot_emb(
                q.view(
                    num_rows, num_cols, batch_size * self.num_heads, self.head_dim
                ).permute(2, 0, 1, 3),
                k.view(
                    num_rows, num_cols, batch_size * self.num_heads, self.head_dim
                ).permute(2, 0, 1, 3),
                position_ids,
                self.num_heads,
            )
            q = q.view(
                batch_size, self.num_heads, num_rows, num_cols, self.head_dim
            ).permute(2, 3, 0, 1, 4)
            k = k.view(
                batch_size, self.num_heads, num_rows, num_cols, self.head_dim
            ).permute(2, 3, 0, 1, 4)
        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        # add row causal mask
        attn_weights = self.row_causal_mask(attn_weights)  # rhnij
        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]

        if self_attn_padding_mask is not None:
            # attn_weights = attn_weights.masked_fill(
            #     self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
            #     -10000,
            # )
            key_padding_mask = self_attn_padding_mask.permute(1, 0, 2)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            attn_weights = attn_weights.masked_fill(key_padding_mask, -10000)

        return attn_weights

    def compute_attention_update(
        self,
        x,
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        position_ids=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (
            num_rows * num_cols > self.max_tokens_per_msa
        ) and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(
                x, scaling, self_attn_mask, self_attn_padding_mask, position_ids
            )
            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs


class ColumnSelfAttention(nn.Module):
    """Compute self-attention over columns of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        k_bias=False,
        q_bias=False,
        v_bias=False,
        o_bias=False,
        add_rope=False,
        dropout=0.0,
        max_tokens_per_msa: int = 2**16,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFM2DRotaryEmbedding(dim=self.head_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=v_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=q_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=o_bias)
        self.dropout_module = nn.Dropout(dropout)

    def build_column_causal_mask(self, num_cols, device):
        # shape: [cols, cols]
        return torch.tril(torch.ones((num_cols, num_cols), device=device)).bool()

    def _batched_forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                x[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[
                    :, :, start : start + max_cols
                ]
                if self_attn_padding_mask is not None
                else None,
            )
            outputs.append(output)
            attns.append(attn)
        output = torch.cat(outputs, 1)
        attns = torch.cat(attns, 1)
        return output, attns

    def compute_attention_update(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        position_ids=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with padding
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=x.device,
                dtype=x.dtype,
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            k = self.k_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            v = self.v_proj(x).view(
                num_rows, num_cols, batch_size, self.num_heads, self.head_dim
            )
            q *= self.scaling
            if self.rot_emb:
                q, k = self.rot_emb(
                    q.view(
                        num_rows, num_cols, batch_size * self.num_heads, self.head_dim
                    ).permute(2, 0, 1, 3),
                    k.view(
                        num_rows, num_cols, batch_size * self.num_heads, self.head_dim
                    ).permute(2, 0, 1, 3),
                    position_ids,
                    self.num_heads,
                )
                q = q.view(
                    batch_size, self.num_heads, num_rows, num_cols, self.head_dim
                ).permute(2, 3, 0, 1, 4)
                k = k.view(
                    batch_size, self.num_heads, num_rows, num_cols, self.head_dim
                ).permute(2, 3, 0, 1, 4)
            attn_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)

            # add column causal mask
            col_causal_mask = self.build_column_causal_mask(num_rows, x.device)
            attn_weights = attn_weights.masked_fill(
                ~col_causal_mask[None, None, None, :, :], -10000
            )

            if self_attn_mask is not None:
                raise NotImplementedError
            if self_attn_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.contiguous().view(
                num_rows, num_cols, batch_size, embed_dim
            )
            output = self.out_proj(context)
        return output, attn_probs

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        position_ids=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (
            num_rows * num_cols
        ) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                x,
                self_attn_mask,
                self_attn_padding_mask,
            )
        else:
            return self.compute_attention_update(
                x, self_attn_mask, self_attn_padding_mask, position_ids
            )


class CrossAttention2D(nn.Module):
    """Compute cross-attention over rows of a 2D input."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        k_bias=False,
        q_bias=False,
        v_bias=False,
        o_bias=False,
        max_tokens_per_msa: int = 2**16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "rhnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=k_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=v_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=q_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=o_bias)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[
                    :, start : start + max_rows
                ]
                if self_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(
                x[start : start + max_rows], attn_probs
            )
            outputs.append(output)
        print("batched_forward")
        output = torch.cat(outputs, 0)
        return output, attn_probs

    def compute_attention_weights(
        self,
        x,
        c,
        scaling: float,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        k = self.k_proj(c).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(
                4
            ).to(q)

        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]
        # print(self_attn_padding_mask.sum())
        # if self_attn_padding_mask is not None:
        #     attn_weights = attn_weights.masked_fill(
        #         self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2).to(torch.bool),
        #         float("-inf"),
        #     )
        #     print(attn_weights)
        return attn_weights

    def compute_attention_update(
        self,
        x,
        c,
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(c).view(
            num_rows, num_cols, batch_size, self.num_heads, self.head_dim
        )
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        c,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (
            num_rows * num_cols > self.max_tokens_per_msa
        ) and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(
                x, c, scaling, self_attn_mask, self_attn_padding_mask
            )
            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, c, attn_probs)
            return output, attn_probs


class CrossAttention(nn.Module):
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
        self_attention=True,
        q_noise=0.0,
        qn_block_size=8,
        d_tilde=1,
        k_bias=False,
        q_bias=True,
        v_bias=True,
        o_bias=True,
        add_rope=False,
        layer_norm=False,
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
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
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

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=k_bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=q_bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=v_bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=o_bias), q_noise, qn_block_size
        )

        if layer_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.reset_parameters(d_tilde)

        self.onnx_trace = False

        self.rot_emb = None
        if add_rope:
            self.rot_emb = SFMRotaryEmbedding(dim=self.head_dim)

        self.use_smooth_softmax = use_smooth_softmax
        self.smooth_factor = smooth_factor
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
        condition,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
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
            need_weights = True

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
        k = self.k_proj(condition)
        v = self.v_proj(condition)

        q *= self.scaling

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
        else:
            outcell_index = None
            expand_mask = None
            local_attention_weight = None

        if outcell_index is not None:
            if position_ids is not None:
                torch.gather(position_ids, dim=1, index=outcell_index)

            outcell_index = (
                outcell_index.transpose(1, 0).unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            expand_k = torch.gather(k, dim=0, index=outcell_index)
            expand_v = torch.gather(v, dim=0, index=outcell_index)

            k = torch.cat([k, expand_k], dim=0)  # [L_expand, B,]
            v = torch.cat([v, expand_v], dim=0)
            if position_ids is not None:
                position_ids = (
                    torch.arange(k.shape[0], device=k.device, dtype=k.dtype)
                    .unsqueeze(0)
                    .repeat(v.shape[1], 1)
                )
            # position_ids = torch.cat([position_ids, expand_position_ids], dim=1)

            src_len = k.size()[0]

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
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

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.contiguous().view(
                bsz * self.num_heads, tgt_len, src_len
            )

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if local_attention_weight is not None:
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
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = nn.functional.softmax(
            attn_weights.float(), dim=-1
        ).type_as(attn_weights)

        if local_attention_weight is not None:
            attn_weights_float = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights_float = attn_weights_float * local_attention_weight.unsqueeze(
                1
            )
            attn_weights_float = attn_weights_float.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_weights = attn_weights_float.type_as(attn_weights)

        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.layer_norm is not None:
            attn = self.layer_norm(attn)

        attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
