# -*- coding: utf-8 -*-
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import (
    ACT2FN,
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from sfm.data.dec_data.dataset import TextMixedWithEntityData, TokenType
from sfm.data.tamgent2.tokenizer import MolxptTokenizer
from sfm.logging import logger
from sfm.models.tamgent.Qformer import BertConfig, BertLMHeadModel
from sfm.models.tamgent.scheduler import LinearWarmupCosineLRScheduler
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig, ModelOutput
from sfm.pipeline.accelerator.trainer import Model
from sfm.utils.optim.adam import AdamW


@dataclass
class DecDeepFuseConfig(DistributedTrainConfig):
    freeze_text_encoder: bool = True

    llama_model: str = "/blob/shufxi/llama/7B"
    molxpt_model: str = "/blob/shufxi/molxpt"
    max_txt_len_llama: int = 500
    max_txt_len_smiles: int = 2048
    end_sym: str = "\n"

    train_data_path: str = ""
    val_data_path: str = ""

    hidden_size: int = 4096
    entity_hidden_size: int = 1024

    num_heads: int = 16
    entity_num_heads: int = 16  # For now, assume to be less than or equal to num_heads

    num_adapter_layers: int = 2
    adapter_hidden_size: int = 1024
    adapter_activation: str = "gelu"

    vocab_size: int = 0  # Total vocab size, including text and entity
    layer_map_list: str = ""


def merge_from_x(
    x: torch.Tensor,
    token_type_mask: torch.Tensor,
    x_dict: Dict[TokenType, torch.Tensor],
) -> torch.Tensor:
    out = torch.zeros_like(x)
    for token_type, x in x_dict.items():
        out[token_type_mask == token_type.value] = x
    return out


def apply_all_token_types(
    x: torch.Tensor, token_type_mask: torch.Tensor, fn: nn.ModuleDict, **other_input
) -> torch.Tensor:
    x_dict = {k: fn[k](x, token_type_mask, **other_input) for k in TokenType}
    return merge_from_x(x, token_type_mask, x_dict)


class Adapter(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([])
        hidden_dims = (
            [in_dim]
            + [config.adapter_hidden_size] * (config.num_adapter_layers - 1)
            + [out_dim]
        )

        for i in range(config.num_adapter_layes):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != config.num_adapter_layers - 1:
                self.layers.append(ACT2FN[config.adapter_activation])

    def forward(self, x: torch.TensorType) -> torch.Tensor:
        return self.layers(x)


class MLP(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType) -> None:
        super().__init__()
        if token_type == TokenType.Text:
            self.in_proj = nn.Identity()
            self.mlp = LlamaMLP(config)
            self.out_proj = nn.Identity()
            self.norm = LlamaRMSNorm(config)
        else:
            config_copy = deepcopy(config)
            config_copy.hidden_size = config.entity_hidden_size

            self.in_proj = Adapter(
                config, config.hidden_size, config.entity_hidden_size
            )
            self.mlp = LlamaMLP(config)
            self.out_proj = Adapter(
                config, config.entity_hidden_size, config.hidden_size
            )
            self.norm = LlamaRMSNorm(config_copy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        res = x
        x = self.norm(x)
        x = self.mlp(x)
        x = res + x
        x = self.out_proj(x)
        return x


class AttnInProj(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType):
        super.__init__()
        self.config = config
        self.token_type = token_type

        if token_type == TokenType.Text:
            self.layer = nn.Identity()

        else:
            self.layer = Adapter(config, config.hidden_size, config.entity_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return x


class AttnQKVProj(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType) -> None:
        super().__init__()
        self.config = config
        self.token_type = token_type

        if token_type == TokenType.Text:
            self.layer = nn.Linear(config.hidden_size, config.hidden_size)
            self.out_proj = nn.Identity()
            self.norm = LlamaRMSNorm(config)
        else:
            self.layer = nn.Linear(config.entity_hidden_size, config.entity_hidden_size)
            self.out_proj = Adapter(
                config,
                self.config.entity_hidden_size // self.config.entity_num_heads,
                self.config.hidden_size // self.config.num_heads,
            )

            config_copy = deepcopy(config)
            config_copy.hidden_size = config.entity_hidden_size
            self.norm = LlamaRMSNorm(config_copy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = self.norm(x)
        x = self.layer(x)

        if self.token_type == TokenType.Text:
            x = x.reshape(
                bsz,
                seq_len,
                self.config.num_heads,
                self.config.hidden_size // self.config.num_heads,
            ).transpose(1, 2)
        else:
            # The entity may have less attnetion heads than text. Assume that the number of heads is a multiple of the number of entity heads.
            assert self.config.num_heads % self.config.entity_num_heads == 0
            replica = self.config.num_heads // self.config.entity_num_head
            x = x.reshape(
                bsz,
                seq_len,
                self.config.entity_num_heads,
                self.config.entity_hidden_size // self.config.entity_num_heads,
            )
            x = x.repeat(1, 1, replica, 1)
            x = x.transpose(1, 2)

        x = self.out_proj(x)
        return x


class AttnOutputProj(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType) -> None:
        super().__init__()
        self.config = config
        self.token_type = token_type

        if token_type == TokenType.Text:
            self.in_proj = nn.Identity()
            self.layer = nn.Linear(config.hidden_size, config.hidden_size)
            self.out_proj = nn.Identity()

        else:
            self.in_proj = nn.Linear(
                self.config.hidden_size // self.config.num_heads,
                self.config.entity_hidden_size // self.config.entity_num_heads,
            )
            self.layer = nn.Linear(config.entity_hidden_size, config.entity_hidden_size)
            self.out_proj = nn.Linear(config.entity_hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, entity_res: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        x = self.in_proj(x)

        # merge the heads
        if self.token_type != TokenType.Text:
            replica = self.config.num_heads // self.config.entity_num_heads
            x = x.reshape(
                bsz,
                seq_len,
                replica,
                self.config.entity_num_heads,
                self.config.entity_hidden_size // self.config.entity_num_heads,
            )
            x = x.mean(dim=2)

        x = x.reshape(
            bsz, seq_len, -1
        )  # The last dim can be text hidden or entity hidden
        x = self.layer(x)
        x = x + entity_res
        x = self.out_proj(x)
        return x


class FusedAttn(nn.Module):
    """
    We first project all Q, K, V to text space, conduct attntion, then project back to entity space
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self._init_rope()

        self.attn_in_proj_dict = nn.ModuleDict()
        self.q_proj_dict = nn.ModuleDict()
        self.k_proj_dict = nn.ModuleDict()
        self.v_proj_dict = nn.ModuleDict()
        self.o_proj_dict = nn.ModuleDict()

        for token_type in TokenType:
            self.attn_in_proj_dict[token_type] = AttnInProj(config, token_type)
            self.q_proj_dict[token_type] = AttnQKVProj(config, token_type)
            self.k_proj_dict[token_type] = AttnQKVProj(config, token_type)
            self.v_proj_dict[token_type] = AttnQKVProj(config, token_type)
            self.o_proj_dict[token_type] = AttnOutputProj(config, token_type)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        x: torch.tensor,
        token_type_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        x_dict = {k: self.attn_in_proj_dict[k](x, token_type_mask) for k in TokenType}

        query_states_dict = {k: self.q_proj_dict[k](x_dict[k]) for k in TokenType}
        key_states_dict = {k: self.k_proj_dict[k](x_dict[k]) for k in TokenType}
        value_states_dict = {k: self.v_proj_dict[k](x_dict[k]) for k in TokenType}

        query_states = merge_from_x(x, token_type_mask, query_states_dict)
        key_states = merge_from_x(x, token_type_mask, key_states_dict)
        value_states = merge_from_x(x, token_type_mask, value_states_dict)

        # TODO: fater decoding by caching KV
        cos, sin = self.rotary_emb(value_states, seq_len=seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2)

        attn_output_dict = {
            k: self.o_proj_dict[k](attn_output, x_dict[k]) for k in TokenType
        }

        attn_output = merge_from_x(x, token_type_mask, attn_output_dict)

        return attn_output


class TextOnly(nn.Module):
    """
    When the decoder layer is less than LLaMA layer, we use this to only process text

    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.text_layer = LlamaDecoderLayer(config)

    def forward(self, x: torch.Tensor, token_type_mask: torch.Tensor) -> torch.Tensor:
        out = x
        out[token_type_mask == TokenType.TEXT.value] = self.text_layer(
            x[token_type_mask == TokenType.TEXT.value]
        )
        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

        self.attn = FusedAttn(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        token_type_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.attn(
            x, token_type_mask, attention_mask=attention_mask, position_ids=position_ids
        )
        x = self.mlp(x)
        return x


class DecDeepFuse(Model):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layer_map = [int(x) for x in config.layer_map_file.split(",")]

        self.decoder_layers = nn.ModuleList([])
        for llama_layer_id, entity_layer_id in enumerate(self.layer_map):
            if entity_layer_id == -1:
                self.decoder_layers.append(TextOnly(config))
            else:
                self.decoder_layers.append(DecoderLayer(config))
