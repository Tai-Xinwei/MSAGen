# -*- coding: utf-8 -*-
from typing import Tuple

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


class SciGPTBioEmbeddingsPP(LlamaEmbeddingsBase):
    def __init__(self, config: ScigptConfig, learnable_cutoff: int = 0):
        super().__init__(config, learnable_cutoff=learnable_cutoff)
        self.emb_adapters = AdaptorMLP(config.hidden_size, config.hidden_size * 3)

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

        adaptor_mask = input_ids > 32000
        # B, T, hidden_size
        text_embeds = self.embed_tokens(input_ids)

        ada_embeds = self.emb_adapters(text_embeds)
        text_embeds = torch.where(adaptor_mask.unsqueeze(-1), ada_embeds, text_embeds)

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


class AdaLlamaHead(torch.nn.Module):
    def __init__(self, config: ScigptConfig, learnable_cutoff: int = 32001):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.learnable_cutoff = learnable_cutoff
        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)
        self.head_adapters = AdaptorMLP(config.hidden_size, config.hidden_size * 3)

    @property
    def emb_weight(self):
        return self.lm_head.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        if new_num_tokens == self.config.vocab_size:
            return
        elif new_num_tokens > self.config.vocab_size:
            old_head = self.lm_head.weight
            new_head = torch.nn.Linear(
                self.config.hidden_size,
                new_num_tokens,
                bias=False,
                dtype=old_head.dtype,
                device=old_head.device,
            )

            new_head.weight.data[: old_head.size(0), :] = old_head.data
            self.lm_head = new_head

        else:
            raise ValueError(
                f"new embedding size {new_num_tokens} must be larger than the current one {self.config.vocab_size}"
            )

    def forward(self, input_tuple: Tuple[torch.Tensor]):
        hidden_states = input_tuple[0]

        ada_emb = self.head_adapters(hidden_states)
        ada_logits = self.lm_head(ada_emb)
        lm_logits = self.lm_head(hidden_states)

        torch.argmax(input_tuple[1], dim=-1) > 32000

        return (lm_logits, ada_logits)


class AdaptorMLP(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU,
        drop=0.0,
    ):
        super(AdaptorMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features, bias=False)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
