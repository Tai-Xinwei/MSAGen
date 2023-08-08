# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
from deepspeed.accelerator import get_accelerator
from einops import rearrange
from torch import Tensor, nn

from megatron import core, get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.model.enums import AttnMaskType, AttnType, LayerType
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.module import MegatronModule
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.model.transformer import (
    CoreAttention,
    FlashSelfAttention,
    ParallelAttention,
)
from megatron.model.utils import attention_mask_func
from sfm.logging import logger
from sfm.modules.FairseqDropout import FairseqDropout

try:
    # FlashAttention (1.x)
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None

try:
    # FlashAttention-2
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

FlashAttentionBuilder = get_accelerator().get_op_builder("FlashAttentionBuilder")
try:
    flash_attn_builder = FlashAttentionBuilder().load()
except TypeError:
    flash_attn_builder = None


class TPMultiheadAttention(MegatronModule):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        config,
        mp_config,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        key_bias=True,
        query_bias=True,
        value_bias=True,
        output_bias=True,
        dropout=0.0,
        num_key_value_heads=None,
        d_tilde=1,
        layer_number=1,
    ):
        super().__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = mp_config.params_dtype
        self.sequence_parallel = mp_config.sequence_parallel
        self.num_attention_heads = num_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_heads
        )
        self.use_gqa = self.num_attention_heads != self.num_key_value_heads

        self.use_flash_attn = (
            args.use_flash_attn
            and attention_type == AttnType.self_attn
            and self.attn_mask_type == AttnMaskType.causal
        )
        if self.use_flash_attn:
            if (
                flash_attn_unpadded_func is None
                and flash_attn_varlen_func is None
                and flash_attn_builder is None
            ):
                raise ImportError(
                    "FlashAttention is not installed, please install with "
                    "pip install flash-attn or or implement your own flash attention"
                )
            assert attention_type == AttnType.self_attn, (
                "FlashAttention code path only supports " "self-attention for now"
            )
            assert self.attn_mask_type == AttnMaskType.causal, (
                "FlashAttention code path only " "supports causal mask for now"
            )
            if rearrange is None:
                raise ImportError(
                    "einops is not installed, please install with pip install einops"
                )

        projection_size = embed_dim
        kv_projection_size = embed_dim

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = core.utils.divide(
            num_heads, world_size
        )

        # Per GQA head and per partition values
        if self.use_gqa:
            kv_projection_size = config.kv_channels * num_key_value_heads
            self.num_key_value_heads_per_partition = core.utils.divide(
                num_key_value_heads, world_size
            )
            self.num_key_value_groups = core.utils.divide(
                num_heads, num_key_value_heads
            )
            assert self.hidden_size_per_attention_head == core.utils.divide(
                kv_projection_size, num_key_value_heads
            )

        self.query = tensor_parallel.ColumnParallelLinear(
            embed_dim,
            projection_size,
            config=mp_config,
            init_method=mp_config.init_method,
            bias=query_bias,
            gather_output=False,
        )

        self.key = tensor_parallel.ColumnParallelLinear(
            embed_dim,
            kv_projection_size,
            config=mp_config,
            init_method=mp_config.init_method,
            bias=key_bias,
            gather_output=False,
        )

        self.value = tensor_parallel.ColumnParallelLinear(
            embed_dim,
            kv_projection_size,
            config=mp_config,
            init_method=mp_config.init_method,
            bias=value_bias,
            gather_output=False,
        )

        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            embed_dim,
            config=mp_config,
            init_method=mp_config.output_layer_init_method,
            bias=output_bias,
            input_is_parallel=True,
            skip_bias_add=True,
        )

        self.core_attention = CoreAttentionBias(
            projection_size,
            self.layer_number,
            mp_config=mp_config,
            config=config,
            attn_mask_type=self.attn_mask_type,
            d_tilde=d_tilde,
        )

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=config.attention_dropout
            )

    def repeat_kv(self, hidden_states, n_rep):
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            slen, batch, num_key_value_heads_per_partition, n_rep, head_dim
        )
        return hidden_states.reshape(
            slen, batch, num_key_value_heads_per_partition * n_rep, head_dim
        )

    def forward(
        self,
        hidden_states,
        key,
        value,
        attention_mask,
        attn_bias=None,
        encoder_output=None,
        inference_params=None,
        rotary_pos_emb=None,
    ):
        # # hidden_states: [sq, b, h]
        # logger.debug(f"hiddens_states shape: {hidden_states.shape}, rot_pos_emb: {rotary_pos_emb.shape}")

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn and not self.use_gqa:
            # Attention heads [sq, b, h] --> [sq, b, (np * hn)]
            query_layer, _ = self.query(hidden_states)
            key_layer, _ = self.key(key)
            value_layer, _ = self.value(value)

            sq, b, _ = query_layer.size()
            query_layer = query_layer.view(
                sq,
                b,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            key_layer = key_layer.view(
                sq,
                b,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            value_layer = value_layer.view(
                sq,
                b,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )

        elif self.attention_type == AttnType.self_attn and self.use_gqa:
            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

            # Attention heads [sq, b, h] --> [sq, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(hidden_states)
            # [sq, b, (np * 2 * hn)] --> [sq, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_key_value_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
            # [sq, b, np, 2 * hn] --> 2 [sq, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_kv_layer, 2
            )

            # Repeat kv
            key_layer = self.repeat_kv(key_layer, self.num_key_value_groups)
            value_layer = self.repeat_kv(value_layer, self.num_key_value_groups)
        else:
            assert not self.use_gqa, "GQA + cross-attn not tested yet"

            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(
                mixed_kv_layer, 2
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2

        # ==================================
        # core attention computation
        # ==================================

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if not self.use_flash_attn:
            # if self.checkpoint_core_attention:
            #     context_layer = self._checkpointed_attention_forward(
            #         query_layer, key_layer, value_layer, attention_mask
            #     )
            # else:
            context_layer = self.core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                attn_bias=attn_bias,
            )
        else:
            q, k, v = [
                rearrange(x, "s b ... -> b s ...").contiguous()
                for x in (query_layer, key_layer, value_layer)
            ]
            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(q, k, v)
            else:
                context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(
                context_layer, "b s h d -> s b (h d)"
            ).contiguous()

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


class CoreAttentionBias(MegatronModule):
    def __init__(
        self,
        embed_dim,
        layer_number,
        mp_config,
        config,
        attn_mask_type=AttnMaskType.padding,
        d_tilde=1.0,
    ):
        super().__init__()
        self.fp16 = mp_config.fp16
        self.bf16 = mp_config.bf16

        self.apply_query_key_layer_scaling = mp_config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = mp_config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = mp_config.sequence_parallel

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = core.utils.divide(embed_dim, world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            embed_dim, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = core.utils.divide(
            config.num_attention_heads, world_size
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            mp_config.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)
        # TODO: add scale factor for muT
        self.norm_factor *= 1.0

    def forward(
        self, query_layer, key_layer, value_layer, attention_mask, attn_bias=None
    ):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        sq = query_layer.shape[0]
        sk = key_layer.shape[0]
        b = query_layer.shape[1]
        np = query_layer.shape[2]

        if attn_bias is not None:
            assert list(attn_bias.shape) == [
                b,
                np,
                sq,
                sk,
            ], f"att_bias shape {attn_bias.shape} is not correct, it should be ({b, np, sq, sk})"
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]),
            query_layer.dtype,
            "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size) + attn_bias

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )

        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
