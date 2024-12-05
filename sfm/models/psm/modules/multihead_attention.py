# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
# from torch.nn.attention import SDPBackend, sdpa_kernel

from sfm.modules.mem_eff_attn import MemEffAttn, MemEffSelfAttn
from sfm.modules.multihead_attention import MultiheadAttention


class MultiheadAttentionWithProteinRotaryEmbedding(MultiheadAttention):
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
        is_protein: Optional[torch.Tensor] = None,
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
        if self.rot_emb and is_protein.any():
            is_protein = (
                is_protein.unsqueeze(1)
                .repeat(1, self.num_heads, 1)
                .view(bsz * self.num_heads, tgt_len, 1)
            )
            q_rope, k_rope = self.rot_emb(q, k, v, position_ids, self.num_heads)
            q = torch.where(is_protein, q_rope, q)
            k = torch.where(is_protein, k_rope, k)

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

        attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)

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


class MemEffAttnWithProteinRotaryEmbedding(MemEffAttn):
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
        is_protein: Optional[torch.Tensor] = None,
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
        if self.rot_emb and is_protein.any() and src_len == tgt_len:
            is_protein = (
                is_protein.unsqueeze(1)
                .repeat(1, self.num_heads, 1)
                .view(bsz * self.num_heads, tgt_len, 1)
            )
            q_rope, k_rope = self.rot_emb(q, k, v, position_ids, self.num_heads)
            q = torch.where(is_protein, q_rope, q)
            k = torch.where(is_protein, k_rope, k)

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

        if local_attention_weight is not None or attn_bias is not None:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            if attn_mask is not None:
                attn_weights += attn_mask

            if local_attention_weight is not None:
                local_attention_weight = local_attention_weight.to(dtype=q.dtype)
                if self.use_smooth_softmax:
                    attn_weights = (
                        attn_weights + self.smooth_factor
                    ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
                else:
                    attn_weights = attn_weights.masked_fill(
                        local_attention_weight.unsqueeze(1) <= 1e-5, float("-inf")
                    )

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            attn_probs = nn.functional.softmax(attn_weights, dim=-1)

            if local_attention_weight is not None:
                attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
                attn_probs = attn_probs * local_attention_weight.unsqueeze(1)
                attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

            attn = torch.bmm(attn_probs, v)
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        # if attn_bias is not None:
        # raise NotImplementedError("mem efficient attn not support attn_bias")

        # FutureWarning: torch.backends.cuda.sdp_kernel() is deprecated. In the future, this context manager will be removed.
        # Please see, torch.nn.attention.sdpa_kernel() for the new context manager, with updated signature.
        # with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        else:
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


class MemEffSelfAttnWithProteinRotaryEmbedding(MemEffSelfAttn):
    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        is_protein: Optional[torch.Tensor] = None,
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
            position_ids (Tensor, optional): position IDs.
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

        # add rope
        if self.rot_emb and is_protein.any():
            is_protein = (
                is_protein.unsqueeze(1)
                .repeat(1, self.num_heads, 1)
                .view(bsz * self.num_heads, tgt_len, 1)
            )
            q_rope, k_rope = self.rot_emb(
                q, k, v, position_ids=position_ids, nhead=self.num_heads
            )
            q = torch.where(is_protein, q_rope, q)
            k = torch.where(is_protein, k_rope, k)

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
