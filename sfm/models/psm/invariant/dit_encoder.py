# -*- coding: utf-8 -*-
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from sfm.models.psm.modules.multihead_attention import (
    MemEffAttnWithProteinRotaryEmbedding,
)
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.mem_eff_attn import MemEffAttn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                psm_config.embedding_dim, 6 * psm_config.embedding_dim, bias=False
            ),
        )

    def forward(
        self,
        x,
        c,
        padding_mask,
        batched_data,
        pbc_expand_batched=None,
        ifbackprop=False,
    ):
        math_kernel = ifbackprop and pbc_expand_batched is not None

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=2)
        if self.psm_config.only_use_rotary_embedding_for_protein:
            x = x + gate_msa * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa).transpose(0, 1),
                key_padding_mask=padding_mask,
                is_protein=batched_data["is_protein"],
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                math_kernel=math_kernel,
            )[0].transpose(0, 1)
        else:
            x = x + gate_msa * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa).transpose(0, 1),
                key_padding_mask=padding_mask,
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                math_kernel=math_kernel,
            )[0].transpose(0, 1)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PSMDiTEncoder(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()

        self.layers = nn.ModuleList([])

        for nl in range(psm_config.num_encoder_layers):
            self.layers.extend([DiTBlock(args, psm_config)])

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
        ifbackprop: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        for layer_index, layer in enumerate(self.layers):
            x = layer(
                x,
                c,
                padding_mask,
                batched_data,
                pbc_expand_batched=pbc_expand_batched,
                ifbackprop=ifbackprop,
            )
        return x
