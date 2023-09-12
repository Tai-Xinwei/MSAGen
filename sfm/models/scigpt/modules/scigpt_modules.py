# -*- coding: utf-8 -*-
import math
import os
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
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
)

from sfm.logging import logger
from sfm.models.llama2.llama_modules import (
    LlamaDecoderLayerPP,
    LlamaEmbeddingsBase,
    LlamaHead,
    LlamaNorm,
)
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec
from sfm.utils.pipelinemode import pipemode


class SciGPTEmbeddingsPP(LlamaEmbeddingsBase):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 0):
        super().__init__(config, learnable_cutoff=learnable_cutoff)

    def forward(
        self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        input_ids, llm_mask = input_tuple
        assert llm_mask.dtype == torch.bool, "llm_mask must be of type torch.bool"

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            raise ValueError("decoder_input_ids cannot be None")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        device = input_ids.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        mol_idx_mask = input_ids < 0  # B, T

        text_embeds = self.embed_tokens(
            input_ids.masked_fill(mol_idx_mask, 0)
        )  # B, T, hidden_size

        # attention mask
        if llm_mask is None:
            llm_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=text_embeds.device,
            )

        llm_mask = self._prepare_decoder_attention_mask(
            llm_mask,
            (batch_size, seq_length),
            text_embeds,
            past_key_values_length,
            input_ids,
        )

        return text_embeds, llm_mask, position_ids


class SciGPTModelPP(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )

        self.pipe_layer = []

    @classmethod
    def to_layers(
        cls, args, config, learnable_cutoff=0, new_num_tokens=None, load_ckpt=False
    ):
        cls.pipe_layer = []
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                SciGPTEmbeddingsPP,
                config,
                learnable_cutoff=0,
                load_ckpt=args.load_ckpt,
            )
        )
        for i in range(config.num_hidden_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerPP,
                    config,
                    i,
                    load_ckpt=load_ckpt,
                )
            )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaNorm,
                config,
                load_ckpt=load_ckpt,
            )
        )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaHead,
                config,
                learnable_cutoff=learnable_cutoff,
                load_ckpt=load_ckpt,
            )
        )

        return cls.pipe_layer


class Scigpt(LlamaForCausalLM):
    """
    GPT for scientific data.
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, batched_data):
        return super().forward(batched_data["x"])
