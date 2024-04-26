# -*- coding: utf-8 -*-
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers import (
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
from transformers.activations import ACT2FN
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GreedySearchOutput
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
)
from transformers.utils import ModelOutput

from sfm.logging import logger
from sfm.models.llama2.llama_modules import (
    LlamaDecoderLayerPP,
    LlamaEmbeddingsBase,
    LlamaHead,
    LlamaNorm,
)
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec
from sfm.utils.pipelinemode import pipemode


class ThreeDimARGenEmbeddingsPP(LlamaEmbeddingsBase):
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


class ThreeDimARGenModelPP(LlamaPreTrainedModel):
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
                ThreeDimARGenEmbeddingsPP,
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


@dataclass
class ThreeDimARGenLanOutputWithPast(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ThreeDimARGenLanGreedySearchOutput(ModelOutput):
    sequences: torch.LongTensor = None
    coordinates: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class ThreeDimARGenLan(LlamaForCausalLM):
    """
    3D Auto-regressive generator.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ntokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, ThreeDimARGenLanOutputWithPast]:
        r"""
        Args:
            label_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]`, or -100 (see `input_ids` docstring), or coordinates (x, y, z). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]` and coordinates.

        Returns:

        Example:

        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if not return_dict:
            output = (logits) + outputs[1:]
            return output

        return ThreeDimARGenLanOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
