# -*- coding: utf-8 -*-
import math
import os
from contextlib import contextmanager, nullcontext
from typing import Any, List, Mapping, Optional, Tuple

import torch
import transformer_engine as te
import transformers
import transformers.models
from torch import nn
from torch.nn import functional as F
from transformer_engine.common import recipe
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
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
from megatron.model.enums import AttnMaskType, AttnType, LayerType
from megatron.model.language_model import Embedding
from sfm.logging import logger
from sfm.modules.sfmmodule import SFMModule
from sfm.modules.te_modules.te_tensor import (
    TEColumnParallelLinear,
    TERMSNorm,
    TERowParallelLinear,
)
from sfm.utils import PretrainedLayerSpec

logger.info("Using TEColumnParallelLinear and TERowParallelLinear in tensor parallel")


class TELlamaDecoderLayer(te.pytorch.TransformerLayer):
    def __init__(self, args, config):
        if args.fp16:
            logger.info("Using fp16 in transformer layer")
            params_dtype = torch.float16
        elif args.bf16:
            logger.info("Using bf16 in transformer layer")
            params_dtype = torch.bfloat16

        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization="RMSNorm",
            activation="swiglu",
            attn_input_format="bshd",
            num_gqa_groups=config.num_key_value_heads,
            params_dtype=params_dtype,
            tp_size=args.tensor_model_parallel_size,
            set_parallel_mode=True if args.tensor_model_parallel_size > 1 else False,
        )
        te_rope = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads
        )
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()

    def forward(self, hidden_states, attention_mask, **kwargs):
        """
        Custom forward to make sure we only pass relevant arguments to the
        forward pass of the `TransformerLayer`. Also, make sure the output
        format matches the output of the HF's `LlamaDecoderLayer`.
        """
        return (
            super().forward(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=self.te_rope_emb,
            ),
        )


class TELlamaModel(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )
        self.args = args
        self.layers = []
        for layer_id in range(config.num_hidden_layers):
            self.layers.append(
                TELlamaDecoderLayer(args, config),
            )
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.learnable_cutoff = args.learnable_cutoff
        self.embed_tokens.weight.register_hook(self.freeze_parital_weight_hook)

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
        hidden_states = self.embed_tokens(input_ids).transpose(0, 1)
        with te.pytorch.fp8_autocast(
            enabled=True, fp8_recipe=self.fp8_recipe
        ) if self.args.fp8 else nullcontext():
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
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
