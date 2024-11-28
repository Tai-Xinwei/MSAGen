# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from sfm.models.psm.modules.multihead_attention import (
    MemEffAttnWithProteinRotaryEmbedding,
)
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.mem_eff_attn import MemEffAttn


class CrossVecMemEffAttnWithProteinRotaryEmbedding(MemEffAttn):
    def forward(
        self,
        x,
        vec,
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

        bsz, tgt_len, embed_dim = x.size()
        vec_dim = vec.size(-1)
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(vec)

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            expand_mask = pbc_expand_batched["expand_mask"]
            local_attention_weight = pbc_expand_batched["local_attention_weight"]
        else:
            outcell_index = None
            expand_mask = None
            local_attention_weight = None

        if outcell_index is not None:
            outcell_index = outcell_index.unsqueeze(-1).expand(-1, -1, embed_dim)
            expand_k = torch.gather(k, dim=1, index=outcell_index)
            outcell_index = outcell_index.unsqueeze(-2).expand(-1, -1, vec_dim, -1)
            expand_v = torch.gather(v, dim=1, index=outcell_index)

            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)

            src_len = k.size()[1]

        if self.rot_emb is not None and is_protein is not None and is_protein.any():
            q = (
                q.reshape(bsz, tgt_len, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
                .reshape(bsz * self.num_heads, tgt_len, self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
                .reshape(bsz * self.num_heads, src_len, self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, vec_dim, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * vec_dim * self.num_heads, src_len, self.head_dim)
            )
            is_protein_expanded = (
                is_protein.unsqueeze(1)
                .repeat(1, vec_dim * self.num_heads, 1)
                .view(bsz * vec_dim * self.num_heads, tgt_len, 1)
            )
            rot_q, rot_k = self.rot_emb(
                q,
                k,
                v,
                position_ids=position_ids,
                nhead=vec_dim * self.num_heads,
            )
            q = torch.where(is_protein_expanded, rot_q, q)
            k = torch.where(is_protein_expanded, rot_k, k)
            q = (
                q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
                .permute(0, 2, 1, 3)
                .reshape(bsz, self.num_heads, tgt_len, self.head_dim)
            )
            k = (
                k.reshape(bsz, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 1, 3)
                .reshape(bsz, self.num_heads, src_len, self.head_dim)
            )
            v = (
                v.reshape(bsz, vec_dim, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, vec_dim * self.head_dim)
            )
        else:
            q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3
            )
            k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3
            )
            v = (
                v.reshape(bsz, src_len, vec_dim, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, vec_dim * self.head_dim)
            )

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

        if local_attention_weight is not None:
            local_attention_weight = local_attention_weight.to(dtype=q.dtype)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.use_smooth_softmax:
                attn_weights = (
                    attn_weights + self.smooth_factor
                ) * local_attention_weight.unsqueeze(1) - self.smooth_factor
            else:
                attn_weights = attn_weights.masked_fill(
                    local_attention_weight.unsqueeze(1) <= 1e-5, float("-inf")
                )

            if attn_mask is not None:
                attn_weights += attn_mask

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


class VecMemEffAttnWithProteinRotaryEmbedding(MemEffAttn):
    def forward(
        self,
        q,
        k,
        v,
        key_padding_mask,
        attn_bias: Optional[Tensor] = None,
        pbc_expand_batched: Optional[Dict] = None,
        is_protein: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        math_kernel: bool = False,
    ):
        if pbc_expand_batched is not None:
            outcell_index = (
                pbc_expand_batched["outcell_index"]
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, k.size()[-2], k.size()[-1])
            )

            expand_k = torch.gather(k, dim=1, index=outcell_index)
            expand_v = torch.gather(v, dim=1, index=outcell_index)
            k = torch.cat([k, expand_k], dim=1)
            v = torch.cat([v, expand_v], dim=1)
            expand_mask = pbc_expand_batched["expand_mask"]
        else:
            expand_mask = None

        bsz, tgt_len, src_len = q.shape[0], q.shape[1], k.shape[1]
        pos_feature_num = q.shape[2]

        if self.rot_emb is not None and is_protein is not None and is_protein.any():
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, tgt_len, self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz * pos_feature_num * self.num_heads, src_len, self.head_dim)
            )
            is_protein_expanded = (
                is_protein.unsqueeze(1)
                .repeat(1, pos_feature_num * self.num_heads, 1)
                .view(bsz * pos_feature_num * self.num_heads, tgt_len, 1)
            )
            rot_q, rot_k = self.rot_emb(
                q,
                k,
                v,
                position_ids=position_ids,
                nhead=pos_feature_num * self.num_heads,
            )
            q = torch.where(is_protein_expanded, rot_q, q)
            k = torch.where(is_protein_expanded, rot_k, k)
            q = (
                q.reshape(bsz, pos_feature_num, self.num_heads, tgt_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, pos_feature_num, self.num_heads, src_len, self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
        else:
            q = (
                q.reshape(bsz, tgt_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, tgt_len, pos_feature_num * self.head_dim)
            )
            k = (
                k.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )
            v = (
                v.reshape(bsz, src_len, pos_feature_num, self.num_heads, self.head_dim)
                .permute(0, 3, 1, 2, 4)
                .reshape(bsz, self.num_heads, src_len, pos_feature_num * self.head_dim)
            )

        if key_padding_mask is not None:
            if pbc_expand_batched is not None:
                expand_mask = pbc_expand_batched["expand_mask"]
                key_padding_mask = torch.cat([key_padding_mask, expand_mask], dim=-1)
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

        if math_kernel:
            context = sdpa_kernel([SDPBackend.MATH])
        else:
            context = sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            )

        with context:
            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=q.dtype)
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
            .reshape(bsz, tgt_len, self.num_heads, 3, self.head_dim)
            .transpose(2, 3)
            .reshape(bsz, tgt_len, 3, self.embed_dim)
        )

        attn = self.out_proj(attn)

        return attn


class VectorVanillaTransformer(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        super().__init__()
        self.psm_config = psm_config
        embed_dim = psm_config.encoder_embed_dim

        self.layers = nn.ModuleList([])
        self.vec_project = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 3, bias=False),
        )

        for _ in range(psm_config.num_pred_attn_layer):
            self.layers.append(VectorTransformerBlock(psm_config))

        self.adaLN_modulation_vec = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )
        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )

        self.ln_x = nn.LayerNorm(embed_dim)
        self.ln_vec = nn.LayerNorm(embed_dim)

    def modulate(self, x, scale, shift):
        return x * scale + shift

    def forward(
        self,
        batched_data: Dict,
        x,
        graph_attn_bias,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
    ):
        x = x.transpose(0, 1)
        encoder_x = x
        bsz, n_node, embed_dim = x.size()

        pos = batched_data["pos"].to(x.dtype)

        if pbc_expand_batched is not None:
            expand_pos = pbc_expand_batched["expand_pos"]
            expand_pos = torch.cat([pos, expand_pos], dim=1)
            delta_pos = pos.unsqueeze(2) - expand_pos.unsqueeze(1)
            expand_mask = torch.cat(
                [padding_mask, pbc_expand_batched["expand_mask"]], dim=-1
            )
            extend_n_node = expand_pos.shape[1]
            outcell_index = pbc_expand_batched["outcell_index"]
            outcell_index = outcell_index.unsqueeze(-1).expand(-1, -1, embed_dim)

            expand_x = torch.gather(x, dim=1, index=outcell_index)
            expand_x = torch.cat([x, expand_x], dim=1)
        else:
            delta_pos = pos.unsqueeze(2) - pos.unsqueeze(1)
            expand_mask = padding_mask
            expand_pos = pos
            extend_n_node = n_node
            expand_x = x

        dist = delta_pos.norm(dim=-1).view(-1, n_node, extend_n_node)
        dist = dist.masked_fill(padding_mask.unsqueeze(-1), 1e6)
        dist = dist.masked_fill(expand_mask.unsqueeze(1), 1e6)
        delta_pos = delta_pos / (dist.unsqueeze(-1) + 1.0)  # B, L, L_extend, 3

        vec = torch.einsum("blmk,bmh->blkh", delta_pos, expand_x)

        for layer in self.layers:
            x, vec = layer(
                batched_data,
                x,
                vec,
                graph_attn_bias,
                padding_mask,
                pbc_expand_batched,
                ifbackprop=True,
            )

        x = self.ln_x(x)
        vec = self.ln_vec(vec)

        if self.psm_config.decoder_feat4energy:
            return x, vec
        else:
            return encoder_x, vec


class VectorTransformerBlock(nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
    ):
        # )
        super().__init__()
        self.psm_config = psm_config
        embed_dim = psm_config.encoder_embed_dim
        self.ln_x = nn.LayerNorm(embed_dim)
        self.ln_vec = nn.LayerNorm(embed_dim)

        # self.adaLN_modulation_x = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(psm_config.embedding_dim, psm_config.embedding_dim, bias=False),
        # )
        self.adaLN_modulation_vec = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )
        # self.adaLN_modulation_x_mlp = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(
        #         psm_config.embedding_dim, psm_config.embedding_dim, bias=False
        #     ),
        # )
        self.adaLN_modulation_vec_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 2 * psm_config.embedding_dim, bias=False
            ),
        )

        self.attn_vec = VecMemEffAttnWithProteinRotaryEmbedding(
            psm_config.embedding_dim,
            psm_config.num_attention_heads,
            dropout=psm_config.dropout,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=False,
            use_smooth_softmax=psm_config.use_smooth_softmax,
            smooth_factor=psm_config.smooth_factor,
        )

        # self.attn_x = MemEffAttnWithProteinRotaryEmbedding(
        #     psm_config.embedding_dim,
        #     psm_config.num_attention_heads,
        #     dropout=psm_config.dropout,
        #     k_bias=False,
        #     q_bias=False,
        #     v_bias=False,
        #     o_bias=False,
        #     add_rope=False,
        #     use_smooth_softmax=psm_config.use_smooth_softmax,
        #     smooth_factor=psm_config.smooth_factor,
        # )

        self.vec_mlp_norm = nn.LayerNorm(psm_config.embedding_dim)
        self.x_mlp_norm = nn.LayerNorm(psm_config.embedding_dim)

        self.mlp_vec = nn.Sequential(
            nn.Linear(
                psm_config.embedding_dim, psm_config.ffn_embedding_dim, bias=False
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.ffn_embedding_dim, psm_config.embedding_dim, bias=False
            ),
        )
        # self.mlp_x = nn.Sequential(
        #     nn.Linear(
        #         psm_config.embedding_dim, psm_config.ffn_embedding_dim, bias=False
        #     ),
        #     nn.SiLU(),
        #     nn.Linear(
        #         psm_config.ffn_embedding_dim, psm_config.embedding_dim, bias=False
        #     ),
        # )

    def modulate(self, x, scale, shift=None):
        if shift is not None:
            return x * scale + shift
        else:
            return x * scale

    def forward(
        self,
        batched_data: Dict,
        x,
        vec,
        graph_attn_bias,
        padding_mask,
        pbc_expand_batched: Optional[Dict] = None,
        ifbackprop: bool = False,
    ):
        math_kernel = ifbackprop and pbc_expand_batched is not None

        residue_x = x
        residue_vec = vec

        x = self.ln_x(x)
        vec = self.ln_vec(vec)

        vec = self.attn_vec(
            vec,
            vec,
            vec,
            attn_bias=graph_attn_bias,
            key_padding_mask=padding_mask,
            is_protein=batched_data["is_protein"],
            position_ids=batched_data["position_ids"],
            pbc_expand_batched=pbc_expand_batched,
            math_kernel=math_kernel,
        )

        scale_vec, shift_vec = self.adaLN_modulation_vec(vec).chunk(2, dim=-1)

        x = self.modulate(x, scale_vec.norm(dim=-2), shift_vec.norm(dim=-2))

        x = x + residue_x
        vec = vec + residue_vec

        residue_x = x
        residue_vec = vec

        x = self.x_mlp_norm(x)
        vec = self.vec_mlp_norm(vec)

        vec = self.mlp_vec(vec)

        scale_vec_mlp, shift_vec_mlp = self.adaLN_modulation_vec_mlp(vec).chunk(
            2, dim=-1
        )
        x = self.modulate(x, scale_vec_mlp.norm(dim=-2), shift_vec_mlp.norm(dim=-2))

        x = x + residue_x
        vec = vec + residue_vec

        return x, vec
