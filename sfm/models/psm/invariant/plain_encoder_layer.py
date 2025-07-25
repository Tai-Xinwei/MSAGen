# -*- coding: utf-8 -*-
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfm.models.psm.modules.multihead_attention import (
    MemEffAttnWithProteinRotaryEmbedding,
    MultiheadAttentionWithProteinRotaryEmbedding,
)
from sfm.models.psm.psm_config import PSMConfig
from sfm.modules.droppath import DropPath
from sfm.modules.FairseqDropout import FairseqDropout

# from fairseq import utils
from sfm.modules.get_activation_fn import get_activation_fn
from sfm.modules.mem_eff_attn import MemEffAttn
from sfm.modules.multihead_attention import MultiheadAttention


class PSMPlainEncoderLayer(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(self, args, psm_config: PSMConfig):
        super().__init__()

        self.psm_config = psm_config

        # Initialize blocks
        self.activation_fn = get_activation_fn(psm_config.activation_fn)
        self.self_attn = self.build_self_attention(
            psm_config.embedding_dim,
            psm_config.num_attention_heads,
            dropout=psm_config.dropout,
            add_rope=True,
        )

        self.fc1 = self.build_fc1(
            psm_config.embedding_dim,
            psm_config.ffn_embedding_dim,
        )
        self.fc2 = self.build_fc2(
            psm_config.ffn_embedding_dim,
            psm_config.embedding_dim,
        )

        # sandwitch layernorm
        self.top_layer_norm = nn.LayerNorm(psm_config.embedding_dim)
        self.mid_layer_norm = nn.LayerNorm(psm_config.embedding_dim)

        self.args = args

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.top_layer_norm.reset_parameters()
        self.mid_layer_norm.reset_parameters()

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        d_tilde=1,
        add_rope=False,
    ):
        if self.psm_config.only_use_rotary_embedding_for_protein:
            attn_cls = MemEffAttnWithProteinRotaryEmbedding
        else:
            attn_cls = MemEffAttn
        return attn_cls(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            d_tilde=d_tilde,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=add_rope,
            use_smooth_softmax=self.psm_config.use_smooth_softmax,
            smooth_factor=self.psm_config.smooth_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        mixed_attn_bias: Optional[torch.Tensor] = None,
        ifbackprop: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        math_kernel = ifbackprop  # and pbc_expand_batched is not None

        residual = x
        x = self.top_layer_norm(x)
        if self.psm_config.only_use_rotary_embedding_for_protein:
            x, _ = self.self_attn(
                x,
                key_padding_mask=padding_mask,
                need_weights=False,
                attn_mask=None,
                is_protein=batched_data["is_protein"],
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                attn_bias=mixed_attn_bias,
                math_kernel=math_kernel,
            )
        else:
            x, _ = self.self_attn(
                x,
                key_padding_mask=padding_mask,
                need_weights=False,
                attn_mask=None,
                # position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                attn_bias=mixed_attn_bias,
                math_kernel=math_kernel,
            )
        x = residual + x

        residual = x
        x = self.mid_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, None


class PSMPairPlainEncoderLayer(nn.Module):
    """
    Implements a Transformer-M Encoder Layer.
    """

    def __init__(
        self,
        args,
        psm_config: PSMConfig,
        embedding_dim: int = None,
        ffn_embedding_dim: int = None,
        encoder_pair_embed_dim: int = None,
        num_attention_heads: int = None,
    ):
        super().__init__()

        if embedding_dim is None:
            embedding_dim = psm_config.embedding_dim

        if ffn_embedding_dim is None:
            ffn_embedding_dim = psm_config.ffn_embedding_dim

        if num_attention_heads is None:
            num_attention_heads = psm_config.num_attention_heads

        if encoder_pair_embed_dim is None:
            encoder_pair_embed_dim = psm_config.encoder_pair_embed_dim

        self.psm_config = psm_config

        # Initialize blocks
        self.activation_fn = get_activation_fn(psm_config.activation_fn)
        self.self_attn = self.build_self_attention(
            embedding_dim,
            num_attention_heads,
            dropout=psm_config.dropout,
            add_rope=True,
        )

        self.fc1 = self.build_fc1(
            embedding_dim,
            ffn_embedding_dim,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            embedding_dim,
        )

        # sandwitch layernorm
        self.top_layer_norm = nn.LayerNorm(embedding_dim)
        self.mid_layer_norm = nn.LayerNorm(embedding_dim)

        self.args = args

        self.pair_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(
                embedding_dim,
                encoder_pair_embed_dim * 2,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                encoder_pair_embed_dim * 2,
                encoder_pair_embed_dim,
                bias=False,
            ),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.top_layer_norm.reset_parameters()
        self.mid_layer_norm.reset_parameters()

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        d_tilde=1,
        add_rope=False,
    ):
        # if not self.psm_config.use_memory_efficient_attention:
        #     attn_cls = MultiheadAttentionWithProteinRotaryEmbedding
        # elif self.psm_config.only_use_rotary_embedding_for_protein:
        #     attn_cls = MemEffAttnWithProteinRotaryEmbedding
        # else:
        #     attn_cls = MemEffAttn

        if self.psm_config.use_memory_efficient_attention:
            if self.psm_config.only_use_rotary_embedding_for_protein:
                attn_cls = MemEffAttnWithProteinRotaryEmbedding
            else:
                attn_cls = MemEffAttn
        else:
            if self.psm_config.only_use_rotary_embedding_for_protein:
                attn_cls = MultiheadAttentionWithProteinRotaryEmbedding
            else:
                attn_cls = MultiheadAttention

        # attn_cls = MultiheadAttention

        return attn_cls(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            d_tilde=d_tilde,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=add_rope,
            use_smooth_softmax=self.psm_config.use_smooth_softmax,
            smooth_factor=self.psm_config.smooth_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        x_pair: torch.Tensor = None,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        mixed_attn_bias: Optional[torch.Tensor] = None,
        ifbackprop: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        math_kernel = ifbackprop and pbc_expand_batched is not None

        residual = x
        x = self.top_layer_norm(x)
        if self.psm_config.only_use_rotary_embedding_for_protein:
            x, _ = self.self_attn(
                x,
                key_padding_mask=padding_mask,
                need_weights=False,
                attn_mask=None,
                is_protein=batched_data["is_protein"],
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                attn_bias=mixed_attn_bias,
                math_kernel=math_kernel,
            )
        else:
            x, _ = self.self_attn(
                x,
                key_padding_mask=padding_mask,
                need_weights=False,
                attn_mask=None,
                position_ids=batched_data["position_ids"],
                pbc_expand_batched=pbc_expand_batched,
                attn_bias=mixed_attn_bias,
                math_kernel=math_kernel,
            )
        x = residual + x

        residual = x
        x = self.mid_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        x_p_i = self.pair_proj(x)

        if pbc_expand_batched is not None:
            outcell_index = pbc_expand_batched["outcell_index"]
            _, _, embed_dim = x_p_i.size()

            outcell_index = (
                outcell_index.transpose(1, 0).unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            expand_x_p_i = torch.gather(x_p_i, dim=0, index=outcell_index)

            x_p_j = torch.cat([x_p_i, expand_x_p_i], dim=0)
        else:
            x_p_j = x_p_i

        if x_pair is not None:
            x_pair = x_pair + torch.einsum("lbh,kbh->lkbh", x_p_i, x_p_j)
        else:
            x_pair = torch.einsum("lbh,kbh->lkbh", x_p_i, x_p_i)

        return x, x_pair


class MSAGenEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer.
    """

    def __init__(
        self,
        args,
        psm_config: PSMConfig,
        embedding_dim: int = None,
        ffn_embedding_dim: int = None,
        encoder_pair_embed_dim: int = None,
        num_attention_heads: int = None,
    ):
        super().__init__()

        if embedding_dim is None:
            embedding_dim = psm_config.embedding_dim

        if ffn_embedding_dim is None:
            ffn_embedding_dim = psm_config.ffn_embedding_dim

        if num_attention_heads is None:
            num_attention_heads = psm_config.num_attention_heads

        if encoder_pair_embed_dim is None:
            encoder_pair_embed_dim = psm_config.encoder_pair_embed_dim

        self.psm_config = psm_config

        # Initialize blocks
        self.activation_fn = get_activation_fn(psm_config.activation_fn)
        self.self_attn = self.build_self_attention(
            embedding_dim,
            num_attention_heads,
            dropout=psm_config.dropout,
            add_rope=True,
        )

        self.fc1 = self.build_fc1(
            embedding_dim,
            ffn_embedding_dim,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            embedding_dim,
        )

        # sandwitch layernorm
        self.top_layer_norm = nn.LayerNorm(embedding_dim)
        self.mid_layer_norm = nn.LayerNorm(embedding_dim)

        self.args = args

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.top_layer_norm.reset_parameters()
        self.mid_layer_norm.reset_parameters()

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim, bias=False)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        d_tilde=1,
        add_rope=False,
    ):
        # if not self.psm_config.use_memory_efficient_attention:
        #     attn_cls = MultiheadAttentionWithProteinRotaryEmbedding
        # elif self.psm_config.only_use_rotary_embedding_for_protein:
        #     attn_cls = MemEffAttnWithProteinRotaryEmbedding
        # else:
        #     attn_cls = MemEffAttn

        attn_cls = MultiheadAttention

        return attn_cls(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            d_tilde=d_tilde,
            k_bias=False,
            q_bias=False,
            v_bias=False,
            o_bias=False,
            add_rope=add_rope,
            use_smooth_softmax=self.psm_config.use_smooth_softmax,
            smooth_factor=self.psm_config.smooth_factor,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        batched_data: Dict,
        x_pair: torch.Tensor = None,
        pbc_expand_batched: Optional[Dict[str, torch.Tensor]] = None,
        mixed_attn_bias: Optional[torch.Tensor] = None,
        ifbackprop: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        math_kernel = ifbackprop and pbc_expand_batched is not None

        residual = x
        x = self.top_layer_norm(x)

        x, _ = self.self_attn(
            x,
            key_padding_mask=padding_mask,
            need_weights=False,
            attn_mask=None,
            position_ids=batched_data["position_ids"],
            pbc_expand_batched=pbc_expand_batched,
            attn_bias=mixed_attn_bias,
            math_kernel=math_kernel,
        )

        x = residual + x

        residual = x
        x = self.mid_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x
