# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from sfm.logging import logger
from sfm.models.generalist.modules.hybrid_emb import HybridEmbeddings
from sfm.models.llama2.llama_modules_3dmp import ParallelLlamaMLPAdapter
from sfm.modules.sfmmodule import SFMModule

try:
    from apex.normalization import MixedFusedRMSNorm as LlamaRMSNorm
except:
    logger.error("failed to import apex import hybrid_emb_3dmp.py")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm


class HybridEmbeddingsMP(HybridEmbeddings, SFMModule):
    def __init__(
        self, config: PretrainedConfig, mpllama_config: PretrainedConfig, **kwargs
    ):
        super().__init__(config, if_initialize=False, **kwargs)

        self.mol_rep_layernorm = LlamaRMSNorm(
            config.mfm_hidden_size, eps=config.rms_norm_eps
        )
        if config.btn_adaptor:
            self.mol_adaptor = ParallelLlamaMLPAdapter(
                mpllama_config,
                hidden_size=config.mfm_hidden_size,
                intermediate_size=config.mfm_hidden_size // 4,
                output_size=config.hidden_size,
                hidden_act=config.hidden_act,
            )
        else:
            self.mol_adaptor = ParallelLlamaMLPAdapter(
                mpllama_config,
                hidden_size=config.mfm_hidden_size,
                intermediate_size=config.mfm_hidden_size,
                output_size=config.hidden_size,
                hidden_act=config.hidden_act,
            )

    def forward(self, input_tuple: Tuple):
        mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids = input_tuple

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

        # Merge text and mol embeddings, size [batch_size, seq_length, hidden_size]
        inputs_embeds = self._forward_embedding(
            mol_emb, mol_padding_mask, text_embeds, input_ids
        )

        # attention mask
        if llm_mask is None:
            llm_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        llm_mask = self._prepare_decoder_attention_mask(
            llm_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            input_ids,
        )

        # convert size to [seq_length, batch_size, hidden_size] for decoder layer
        hidden_states = inputs_embeds.to(mol_emb.dtype).transpose(0, 1)

        return (hidden_states, llm_mask, position_ids)
