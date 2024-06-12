# -*- coding: utf-8 -*-
import math
import os
import re
from contextlib import contextmanager, nullcontext
from typing import Any, List, Mapping, Optional, Tuple

import torch
import transformer_engine as te
import transformers
import transformers.models
from torch import nn
from torch.nn import functional as F
from transformer_engine.common import recipe
from transformers import (
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from megatron.core import parallel_state, tensor_parallel
from sfm.logging import logger
from sfm.modules.sfmmodule import SFMModule
from sfm.modules.te_modules.te_tensor import (
    TEColumnParallelLinear,
    TERMSNorm,
    TERowParallelLinear,
)

logger.info("Using TEColumnParallelLinear and TERowParallelLinear in tensor parallel")


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        dim: int,
        rotary_percent: float = 1.0,
        base: int = 10000,
        seq_len_interpolation_factor: Optional[int] = None,
        pretrained_max_position_embeddings: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        dim: int
            rotary embedding dimension
        rotary_percent: float
            Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor: int
            if not None, discrete positions will be interpolated by this factor via the trick in
            https://arxiv.org/abs/2306.15595
        pretrained_max_position_embeddings: int
            pre-trained max_position_embeddings before position interpolation
        """
        super().__init__()
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device()
                )
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies

        Parameters
        ----------
        max_seq_len: int
            sequence length of a sample
        offset: int, default = 0
            fixed offset for freqencies
        """
        seq = (
            torch.arange(
                max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            + offset
        )

        if (
            self.pretrained_max_position_embeddings is not None
            and self.seq_len_interpolation_factor is not None
        ):
            if (
                max_seq_len
                > self.pretrained_max_position_embeddings
                * self.seq_len_interpolation_factor
            ):
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb.reshape(emb.size(0), 1, 1, emb.size(1))


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    def __init__(
        self,
        args,
        layer_idx: int = 0,
        **kwargs,
    ):
        if args.fp16:
            logger.info("Using fp16 in transformer layer")
            params_dtype = torch.float16
        elif args.bf16:
            logger.info("Using bf16 in transformer layer")
            params_dtype = torch.bfloat16

        if args.tensor_model_parallel_size > 1:
            tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            tp_group = None

        super().__init__(
            args.hidden_size,
            args.intermediate_size,
            args.num_attention_heads,
            bias=False,
            layernorm_epsilon=args.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=args.num_key_value_heads,
            params_dtype=params_dtype,
            tp_size=args.tensor_model_parallel_size,
            set_parallel_mode=True if args.tensor_model_parallel_size > 1 else False,
            tp_group=tp_group,
        )
        te_rope = RotaryPositionEmbedding(
            args.hidden_size // args.num_attention_heads, base=args.rope_theta
        )
        self.te_rope_emb = te_rope(max_seq_len=args.max_position_embeddings).cuda()

        self.dummy = nn.Linear(1, 1)

    def forward(self, hidden_states, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=self.te_rope_emb,
        )


class TELlamaDecoderLayerMP(TELlamaDecoderLayer, SFMModule):
    def forward(
        self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], **kwargs
    ):
        hidden_states, attention_mask_bool, position_ids = input_tuple
        if (~attention_mask_bool).any():
            temp_attn_maks = (attention_mask_bool.unsqueeze(1).unsqueeze(2),)
        else:
            temp_attn_maks = None

        return (
            super()
            .forward(
                hidden_states,
                attention_mask=temp_attn_maks,
                rotary_pos_emb=self.te_rope_emb,
            )
            .contiguous(),
            attention_mask_bool.contiguous(),
            position_ids.contiguous(),
        )

    def auto_partition_load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        tp_model_size: int,
        tp_rank: int,
        strict: bool = True,
    ):
        keys = list(state_dict.keys())
        new_state_dict = {}
        temp_state_dict = {}
        for key in keys:
            param = state_dict[key]
            if key == "input_layernorm.weight":
                new_state_dict["self_attention.layernorm_qkv.layer_norm_weight"] = param
            elif key == "self_attn.q_proj.weight":
                new_state_dict["self_attention.layernorm_qkv.query_weight"] = param
            elif key == "self_attn.k_proj.weight":
                new_state_dict["self_attention.layernorm_qkv.key_weight"] = param
            elif key == "self_attn.v_proj.weight":
                new_state_dict["self_attention.layernorm_qkv.value_weight"] = param
            elif key == "self_attn.o_proj.weight":
                new_state_dict["self_attention.proj.weight"] = param
            elif key == "post_attention_layernorm.weight":
                new_state_dict["layernorm_mlp.layer_norm_weight"] = param
            elif key == "mlp.gate_proj.weight":
                temp_state_dict["layernorm_mlp.fc1_weight.1"] = param
            elif key == "mlp.up_proj.weight":
                temp_state_dict["layernorm_mlp.fc1_weight.2"] = param
            elif key == "mlp.down_proj.weight":
                new_state_dict["layernorm_mlp.fc2_weight"] = param
            else:
                new_state_dict[key] = param
                logger.warning(f"Check this! Unexpected key: {key}")

        # concat layernorm_mlp.fc1_weight.1 and layernorm_mlp.fc1_weight.2
        if (
            "layernorm_mlp.fc1_weight.1" in temp_state_dict
            and "layernorm_mlp.fc1_weight.2" in temp_state_dict
        ):
            new_state_dict["layernorm_mlp.fc1_weight"] = torch.cat(
                (
                    temp_state_dict["layernorm_mlp.fc1_weight.1"],
                    temp_state_dict["layernorm_mlp.fc1_weight.2"],
                ),
                dim=0,
            )

        del state_dict

        return super().auto_partition_load_state_dict(
            new_state_dict, tp_model_size, tp_rank, strict
        )


class TELlamaModel(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )
        self.args = args
        self.layers = nn.ModuleList([])
        for layer_id in range(config.num_hidden_layers):
            self.layers.append(
                TELlamaDecoderLayer(args),
            )
        self.word_embeddings = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.learnable_cutoff = args.learnable_cutoff
        self.word_embeddings.weight.register_hook(self.freeze_parital_weight_hook)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)
        self.fp8_recipe = recipe.DelayedScaling(
            fp8_format=recipe.Format.HYBRID,
            amax_history_len=16,
            amax_compute_algo="max",
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.word_embeddings(input_ids).transpose(0, 1)
        with te.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=self.fp8_recipe
        ) if self.args.fp8 else nullcontext():
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return (lm_logits.transpose(0, 1),)

    @property
    def emb_weight(self):
        return self.embed_tokens.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    @classmethod
    def to_layers(
        cls, args, config, new_num_tokens=None, load_ckpt=False, layer_id=0, ckp_list=[]
    ):
        cls.pipe_layer = []

        return cls.pipe_layer
