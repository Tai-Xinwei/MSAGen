# -*- coding: utf-8 -*-
import math
import os
from typing import Any, List, Mapping, Optional, Tuple

import torch
import transformer_engine.pytorch as te
from torch import nn
from torch.nn import functional as F
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
    def __init__(self, config):
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
        )
        te_rope = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads
        )
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings).cuda()


class TELlamaModel(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )

        self.pipe_layer = []

    @classmethod
    def to_layers(
        cls, args, config, new_num_tokens=None, load_ckpt=False, layer_id=0, ckp_list=[]
    ):
        cls.pipe_layer = []

        return cls.pipe_layer
