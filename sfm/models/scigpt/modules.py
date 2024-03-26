# -*- coding: utf-8 -*-
import torch

from sfm.models.llama2.llama_modules import LlamaEmbeddingsBase
from sfm.models.scigpt.config import ScigptConfig
from sfm.utils.pipelinemode import pipemode


class SciGPTEmbeddingsPP(LlamaEmbeddingsBase):
    def __init__(self, config: ScigptConfig, learnable_cutoff: int = 0):
        super().__init__(config, learnable_cutoff=learnable_cutoff)

        self.param_dict = {
            "input_ids": torch.Tensor,
            "llm_mask": torch.Tensor,
        }

    @pipemode
    def forward(self, input_ids, llm_mask):
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

        # B, T, hidden_size
        text_embeds = self.embed_tokens(input_ids)

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
