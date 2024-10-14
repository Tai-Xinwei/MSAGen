# -*- coding: utf-8 -*-
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from sklearn.metrics import pair_confusion_matrix

from sfm.models.psm.modules.multihead_attention import (
    MemEffAttnWithProteinRotaryEmbedding,
)
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.mem_eff_attn import MemEffAttn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTPairBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            psm_config.embedding_dim, elementwise_affine=False, eps=1e-6
        )
        self.psm_config = psm_config

        if psm_config.only_use_rotary_embedding_for_protein:
            attn_cls = MemEffAttnWithProteinRotaryEmbedding
        else:
            attn_cls = MemEffAttn

        self.attn = attn_cls(
            psm_config.embedding_dim,
            psm_config.num_attention_heads,
            dropout=psm_config.dropout,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=True,
            use_smooth_softmax=psm_config.use_smooth_softmax,
            smooth_factor=psm_config.smooth_factor,
        )

        self.norm2 = nn.LayerNorm(
            psm_config.embedding_dim, elementwise_affine=False, eps=1e-6
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                psm_config.embedding_dim, psm_config.ffn_embedding_dim, bias=False
            ),
            nn.SiLU(),
            nn.Linear(
                psm_config.ffn_embedding_dim, psm_config.embedding_dim, bias=False
            ),
        )

        self.adaLN_modulation = nn.Linear(
            psm_config.embedding_dim, 2 * psm_config.encoder_pair_embed_dim, bias=False
        )
        self.pair_feat_bias = nn.Sequential(
            nn.SiLU(),
            nn.Linear(psm_config.encoder_pair_embed_dim, 1, bias=False),
        )

    def forward(
        self,
        x,
        c,
        pair_feat,
        padding_mask,
        batched_data,
        pbc_expand_batched=None,
        mixed_attn_bias=None,
        ifbackprop=False,
    ):
        math_kernel = ifbackprop  # and pbc_expand_batched is not None

        p1, p2 = self.adaLN_modulation(c).chunk(2, dim=-1)
        pair_feat_i = torch.einsum("lbh,kbh->lkbh", p1, p2).permute(2, 0, 1, 3)
        if pair_feat is not None:
            pair_feat = pair_feat + pair_feat_i
        else:
            pair_feat = pair_feat_i

        pair_feat_bias = self.pair_feat_bias(pair_feat)
        if mixed_attn_bias is not None:
            mixed_attn_bias = mixed_attn_bias + pair_feat_bias
        else:
            mixed_attn_bias = pair_feat_bias

        if self.psm_config.only_use_rotary_embedding_for_protein:
            x = x + self.attn(
                self.norm1(x).transpose(0, 1),
                key_padding_mask=padding_mask,
                is_protein=batched_data["is_protein"],
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                attn_bias=mixed_attn_bias,
                math_kernel=math_kernel,
            )[0].transpose(0, 1)
        else:
            x = x + self.attn(
                self.norm1(x).transpose(0, 1),
                key_padding_mask=padding_mask,
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                attn_bias=mixed_attn_bias,
                math_kernel=math_kernel,
            )[0].transpose(0, 1)

        x = x + self.mlp(self.norm2(x))
        return x, pair_feat


class PSMPDiTPairEncoder(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()

        self.layers = nn.ModuleList([])

        for nl in range(psm_config.num_encoder_layers):
            self.layers.extend([DiTPairBlock(args, psm_config)])

        # dummy param for lora, do not remove
        self.dummy = nn.Linear(1, 1, bias=False)

    # @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        pbc_expand_batched: Optional[Dict] = None,
        mixed_attn_bias: Optional[torch.Tensor] = None,
        ifbackprop: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        pair_feat = None
        for layer_index, layer in enumerate(self.layers):
            x, pair_feat = layer(
                x,
                c,
                pair_feat,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
                mixed_attn_bias=mixed_attn_bias,
                ifbackprop=ifbackprop,
            )
        return x
