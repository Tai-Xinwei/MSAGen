# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

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
from sfm.modules.layer_norm import Fp32LayerNorm, LayerNorm

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

try:
    from apex.normalization import FusedLayerNorm as LayerNormTP
except:
    raise ImportError("Please install apex from install/install_megatron.sh")

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
        self.hidden_size_per_attention_head = core.utils.divide(embed_dim, num_heads)
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
            input_is_parallel=False,
            skip_bias_add=False,
        )

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.layer_norm = LayerNorm(embed_dim)

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=config.attention_dropout
            )

        self.scaling = (
            (self.hidden_size_per_attention_head / d_tilde) ** 0.5
        ) / self.hidden_size_per_attention_head  # when d_tilt == 1, match with original transformer scale

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
        query,
        key,
        value,
        key_padding_mask=None,
        attn_bias=None,
    ):
        query_layer, _ = self.query(query)
        key_layer, _ = self.key(key)
        value_layer, _ = self.value(value)

        query_layer *= self.scaling

        tgt_len, bsz, embed_dim = query_layer.size()
        src_len, _, _ = key_layer.size()

        query_layer = query_layer.view(
            tgt_len,
            bsz * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        ).permute(1, 0, 2)

        key_layer = key_layer.view(
            src_len,
            bsz * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        ).permute(1, 0, 2)

        value_layer = value_layer.view(
            src_len,
            bsz * self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        ).permute(1, 0, 2)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if not self.use_flash_attn:
            attn_weights = torch.matmul(query_layer, key_layer.transpose(1, 2))

            assert list(attn_weights.size()) == [
                bsz * self.num_attention_heads_per_partition,
                tgt_len,
                src_len,
            ]

            if attn_bias is not None:
                attn_weights += attn_bias.contiguous().view(
                    bsz * self.num_attention_heads_per_partition, tgt_len, src_len
                )

            if key_padding_mask is not None:
                # don't attend to padding symbols
                attn_weights = attn_weights.view(
                    bsz, self.num_attention_heads_per_partition, tgt_len, src_len
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = attn_weights.view(
                    bsz * self.num_attention_heads_per_partition, tgt_len, src_len
                )

            attn_weights_float = nn.functional.softmax(attn_weights, dim=-1)

            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = self.dropout_module(attn_weights)

            attn = torch.matmul(attn_probs, value_layer)

            assert list(attn.size()) == [
                bsz * self.num_attention_heads_per_partition,
                tgt_len,
                self.hidden_size_per_attention_head,
            ]
            context_layer = (
                attn.permute(1, 0, 2).contiguous().view(tgt_len, bsz, embed_dim)
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

        # merge first for layer_norm, extra communication due to this layer_norm in graphormer attention
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()

        merged_context_layer = torch.zeros(
            (tgt_len, bsz, embed_dim * tp_world_size),
            device=context_layer.device,
            dtype=context_layer.dtype,
        )

        merged_context_layer[
            :, :, embed_dim * tp_rank : embed_dim * (tp_rank + 1)
        ] = context_layer

        dist.all_reduce(merged_context_layer, group=tp_group, op=dist.ReduceOp.SUM)

        # merged_context_layer = context_layer
        context_layer = self.layer_norm(merged_context_layer)
        output, _ = self.dense(context_layer)

        return output, None
