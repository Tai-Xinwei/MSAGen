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
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec


class LlamaMLPAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.dropout(self.gate_proj(x))) * self.up_proj(x)
        )


class LlamaDecoderLayerPP(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, l: int):
        super().__init__(config)
        self.config = config
        self.l = l
        self.dummy = nn.Linear(1, 1)

    def forward(
        self,
        input_tuple,
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # output_attentions: Optional[bool] = False,
        # use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        hidden_states, attention_mask_bool, position_ids = input_tuple

        attention_mask = torch.zeros_like(
            attention_mask_bool, dtype=hidden_states.dtype, device=hidden_states.device
        )
        attention_mask.masked_fill_(
            ~attention_mask_bool, torch.finfo(hidden_states.dtype).min
        )

        hidden_states = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )[0]

        outputs = (hidden_states, attention_mask_bool, position_ids)

        return outputs


class LlamaNorm(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.dummy = nn.Linear(1, 1)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        hidden_states, attention_mask, position_ids = input_tuple

        hidden_states = self.norm(hidden_states)

        return (hidden_states,)


def lm_logits(embedding, input_tuple):
    """LM logits using word embedding weights."""
    input_ = input_tuple[0]
    logits = F.linear(input_, embedding.emb_weight)
    return logits


class LlamaEmbeddingsPP(nn.Module):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 32001):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.learnable_cutoff = learnable_cutoff
        self.embed_tokens.weight.register_hook(self.freeze_parital_weight_hook)

        # self.weight = self.embed_tokens.weight.data.requires_grad_().cuda()
        # self.weight.grad = torch.zeros_like(self.weight)

        # self.partial_learnable_emb = PartialGradEmbedding(
        #     self.embed_tokens, new_embedding_cutoff=32000
        # )

    @property
    def emb_weight(self):
        return self.embed_tokens.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    def forward(
        self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        mol_emb, mol_padding_mask, llm_mask, input_ids = input_tuple

        # Get text embeddings from language model
        mol_idx_mask = input_ids < 0  # B, T
        ## all freeze
        # with torch.no_grad():
        text_embeds = self.embed_tokens(
            input_ids.masked_fill(mol_idx_mask, 0)
        )  # B, T, hidden_size

        return mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids


class Num_MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(Num_MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LlamaHead(nn.Module):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 32001):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.learnable_cutoff = learnable_cutoff
        # self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)
        # self.weight = self.lm_head.weight.data.requires_grad_().cuda()
        # self.weight.grad = torch.zeros_like(self.weight)
        self.num_head = Num_MLP(config.hidden_size, 4 * config.hidden_size, 1)

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
            new_head = nn.Linear(
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

        lm_logits = self.lm_head(hidden_states)
        num_logits = self.num_head(hidden_states)

        return (lm_logits, num_logits)


class LlamaModelPP(LlamaPreTrainedModel):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )

        self.pipe_layer = []

    @classmethod
    def to_layers(cls, args, config, new_num_tokens=None, load_ckpt=False):
        cls.pipe_layer = []
        for i in range(config.num_hidden_layers):
            cls.pipe_layer.append(
                PretrainedLayerSpec(
                    LlamaDecoderLayerPP,
                    config,
                    i,
                    load_ckpt=load_ckpt,
                    pretrained_ckpt_path=os.path.join(
                        args.llm_model_name_or_path, "model.layers.{}.pt".format(i)
                    ),
                    lora_mode="freeze",
                )
            )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaNorm,
                config,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.llm_model_name_or_path, "model.norm.pt"
                ),
                lora_mode="freeze",
            )
        )
        cls.pipe_layer.append(
            PretrainedLayerSpec(
                LlamaHead,
                config,
                new_num_tokens=new_num_tokens,
                load_ckpt=load_ckpt,
                pretrained_ckpt_path=os.path.join(
                    args.llm_model_name_or_path, "model.lm_head.pt"
                ),
                lora_mode="freeze",
            )
        )

        return cls.pipe_layer


class LlamaForCausalLMPP(LlamaForCausalLM):
    def __init__(self, args, config: LlamaConfig):
        super().__init__(config)
        self.dummy = nn.Parameter(
            torch.zeros(1, dtype=torch.float32), requires_grad=True
        )
