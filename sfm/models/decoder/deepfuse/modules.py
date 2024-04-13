# -*- coding: utf-8 -*-
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.biogpt.modeling_biogpt import (
    BioGptDecoderLayer,
    BioGptLearnedPositionalEmbedding,
)
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

from sfm.data.dec_data.datasets import MixedTokenData, TokenType
from sfm.models.decoder.deepfuse.config import DecDeepFuseConfig, EntityDecoderType
from sfm.models.decoder.deepfuse.hidden_state import HiddenState


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def make_norm_dict(config: DecDeepFuseConfig) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            TokenType.Text.name: LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            ),
            TokenType.Entity.name: nn.LayerNorm(config.entity_hidden_size),
        }
    )


# True <> -inf
def mask_to_float(mask: torch.BoolTensor, dtype) -> torch.Tensor:
    ret = torch.zeros_like(mask, dtype=dtype, device=mask.device)
    ret[mask] = torch.finfo(dtype).min
    return ret


def mask_to_bool(mask: torch.Tensor) -> torch.BoolTensor:
    return mask <= torch.finfo(mask.dtype).min


class Embed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.embed_tokens = nn.ModuleDict(
            {
                TokenType.Text.name: nn.Embedding(
                    config.vocab_size + config.new_token_count, config.hidden_size
                ),
                TokenType.Entity.name: nn.Embedding(
                    config.entity_vocab_size, config.entity_hidden_size
                ),
            }
        )

        if config.entity_decoder_model_type == EntityDecoderType.BioGPT:
            # Only BioGPT uses the learned positional embedding
            self.embed_positions = BioGptLearnedPositionalEmbedding(
                config.max_position_embeddings, config.entity_hidden_size
            )
        else:
            self.embed_positions = None

    def forward(self, input_tuple) -> Tuple[torch.Tensor]:
        batch = MixedTokenData.from_tuple(input_tuple[:-3])
        attention_mask, position_ids, past_key_value = input_tuple[-3:]

        hidden_tuple = tuple(
            batch.token_seq[batch.token_type_mask == t.value] for t in TokenType
        ) + (batch.token_type_mask,)

        h = HiddenState.from_tuple(hidden_tuple)
        h = h.apply_all_types_mapping(self.embed_tokens)

        bsz, seq_len = batch.token_type_mask.shape
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=h.device)
            position_ids = position_ids.unsqueeze(0)

        if past_key_value is None:
            past_key_values_length = 0
        else:
            past_key_values_length = past_key_value[0].shape[2]

        if attention_mask is None:
            # During training, we use the non_padding_mask
            non_padding_mask = batch.non_padding_mask
            attention_mask = non_padding_mask.to(h.dtype)

            # Only BioGPT uses the learned positional embedding
            if self.embed_positions is not None:
                positions = self.embed_positions(
                    attention_mask, past_key_values_length=past_key_values_length
                )
                x = (
                    h.x_dict[TokenType.Entity]
                    + positions[batch.token_type_mask == TokenType.Entity.value]
                )
                h = h.update_x_dict(TokenType.Entity, x)

            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (bsz, seq_len), x, past_key_values_length
            )

            attention_mask = mask_to_bool(attention_mask)
        elif attention_mask.dim() == 2:
            # The HF generator will pass in a 2D attention mask

            # Only BioGPT uses the learned positional embedding
            if self.embed_positions is not None:
                positions = self.embed_positions(
                    attention_mask, past_key_values_length=past_key_values_length
                )
                x = (
                    h.x_dict[TokenType.Entity]
                    + positions[batch.token_type_mask == TokenType.Entity.value]
                )
                h = h.update_x_dict(TokenType.Entity, x)

            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (bsz, seq_len),
                h.x_dict[TokenType.Text],
                past_key_values_length,
            )
            attention_mask = mask_to_bool(attention_mask)
        else:
            raise Exception(f"Unknown attention mask, shape {attention_mask.shape}")

        return h.to_tuple() + (attention_mask, position_ids)

    # See transformers.models.bart.modeling_bart.BartDecoder
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


class Head(nn.Module):
    """
    The output head, including the final ln and lm_head
    """

    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()

        self.final_norm = make_norm_dict(config)

        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head = nn.ModuleDict(
            {
                TokenType.Text.name: nn.Linear(
                    config.hidden_size,
                    config.vocab_size + config.new_token_count,
                    bias=False,
                ),
                TokenType.Entity.name: nn.Linear(
                    config.entity_hidden_size, config.entity_vocab_size, bias=False
                ),
            }
        )

    def forward(self, x):
        h = HiddenState.from_tuple(x)
        h = h.apply_all_types_mapping(self.final_norm)
        h = h.apply_all_types_mapping(self.lm_head)
        return h.to_tuple()


class Adapter(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, in_dim, out_dim):
        super().__init__()
        self.config = config
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layers = nn.ModuleList([])
        hidden_dims = (
            [in_dim]
            + [config.adapter_hidden_size] * (config.num_adapter_layers - 1)
            + [out_dim]
        )

        for i in range(config.num_adapter_layers):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i != config.num_adapter_layers - 1:
                self.layers.append(ACT2FN[config.adapter_activation])

    def forward(self, x: torch.TensorType) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class BioGPTMLP(nn.Module):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()

        self.fc1 = nn.Linear(config.entity_hidden_size, config.entity_intermediate_size)
        self.fc2 = nn.Linear(config.entity_intermediate_size, config.entity_hidden_size)

        self.activation_fn = ACT2FN[config.entity_hidden_act]

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class MLP(nn.Module):
    def __init__(self, config: DecDeepFuseConfig, token_type: TokenType) -> None:
        super().__init__()
        self.token_type = token_type
        if token_type == TokenType.Text:
            self.mlp = LlamaMLP(config)
        else:
            if config.entity_decoder_model_type == EntityDecoderType.BioGPT:
                self.mlp = BioGPTMLP(config)
            else:
                config_copy = deepcopy(config)
                config_copy.hidden_size = config.entity_hidden_size

                self.mlp = LlamaMLP(config_copy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttnQKVProj(nn.Module):
    def __init__(
        self, config: DecDeepFuseConfig, token_type: TokenType, kind: str, bias: bool
    ):
        super().__init__()
        self.config = config
        self.token_type = token_type
        self.kind = kind

        if token_type == TokenType.Text:
            if kind in ["k", "v"]:
                self.layer = nn.Linear(
                    config.hidden_size,
                    config.num_key_value_heads * config.head_dim,
                    bias=False,
                )
            else:
                self.layer = nn.Linear(
                    config.hidden_size, config.hidden_size, bias=bias
                )
            self.out_adapter = nn.Identity()
        else:
            self.layer = nn.Linear(
                config.entity_hidden_size, config.entity_hidden_size, bias=bias
            )
            self.out_adapter = Adapter(
                config,
                self.config.entity_head_dim,
                self.config.head_dim,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The seq_len here is the number of tokens in the batch
        seq_len, _ = x.shape

        x = self.layer(x)

        if self.token_type != TokenType.Text:
            # First, project on head dim

            x = x.reshape(
                seq_len,
                self.config.entity_num_attention_heads,
                self.config.entity_head_dim,
            )

            x = self.out_adapter(x)

            # Then, duplicate the heads
            replica = (
                self.config.num_attention_heads
                // self.config.entity_num_attention_heads
            )
            x = (
                x[:, None, :, :]
                .expand(-1, replica, -1, -1)
                .reshape(seq_len, self.config.num_attention_heads, self.config.head_dim)
            )

        else:
            if self.kind in ["k", "v"]:
                # repeat k/v heads if n_kv_heads < n_heads
                replica = (
                    self.config.num_attention_heads // self.config.num_key_value_heads
                )
                x = (
                    x.reshape(
                        seq_len, self.config.num_key_value_heads, self.config.head_dim
                    )[:, None, :, :]
                    .expand(-1, replica, -1, -1)
                    .reshape(
                        seq_len, self.config.num_attention_heads, self.config.head_dim
                    )
                )
            else:
                x = x.reshape(
                    seq_len, self.config.num_attention_heads, self.config.head_dim
                )

        return x


class AttnOutputProj(nn.Module):
    def __init__(
        self, config: DecDeepFuseConfig, token_type: TokenType, bias: bool
    ) -> None:
        super().__init__()
        self.config = config
        self.token_type = token_type

        if token_type == TokenType.Text:
            self.adapter = nn.Identity()
            self.layer = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)

        else:
            self.adapter = Adapter(
                config, self.config.head_dim, self.config.entity_head_dim
            )

            self.layer = nn.Linear(
                config.entity_hidden_size, config.entity_hidden_size, bias=bias
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, head, _ = x.shape

        # merge the heads
        if self.token_type != TokenType.Text:
            replica = (
                self.config.num_attention_heads
                // self.config.entity_num_attention_heads
            )
            assert head == self.config.entity_num_attention_heads * replica
            x = x.reshape(
                bsz,
                replica,
                self.config.entity_num_attention_heads,
                self.config.head_dim,
            )
            x = x.mean(dim=1)
            x = self.adapter(x)

        dim = math.prod(x.shape[1:])
        x = x.reshape(bsz, dim)

        x = self.layer(x)
        return x


class FusedAttn(nn.Module):
    """
    We first project all Q, K, V to text space, conduct attntion.
    Then project them back to entity space
    """

    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        self.q_proj_dict = nn.ModuleDict()
        self.k_proj_dict = nn.ModuleDict()
        self.v_proj_dict = nn.ModuleDict()
        self.o_proj_dict = nn.ModuleDict()

        for token_type in TokenType:
            bias = (
                token_type == TokenType.Entity
                and config.entity_decoder_model_type == EntityDecoderType.BioGPT
            )
            self.q_proj_dict[token_type.name] = AttnQKVProj(
                config, token_type, kind="q", bias=bias
            )
            self.k_proj_dict[token_type.name] = AttnQKVProj(
                config, token_type, kind="k", bias=bias
            )
            self.v_proj_dict[token_type.name] = AttnQKVProj(
                config, token_type, kind="v", bias=bias
            )
            self.o_proj_dict[token_type.name] = AttnOutputProj(
                config, token_type, bias=bias
            )

        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self._init_rope()

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
        h: HiddenState,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> HiddenState:
        query_hidden = h.apply_all_types_mapping(self.q_proj_dict)
        key_hidden = h.apply_all_types_mapping(self.k_proj_dict)
        value_hidden = h.apply_all_types_mapping(self.v_proj_dict)

        query_states = query_hidden.to_dense()  # [B, T, H, D]
        key_states = key_hidden.to_dense()
        value_states = value_hidden.to_dense()

        query_states = query_states.transpose(1, 2)  # [B, H, T, D]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[2]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]

        cos, sin = self.rotary_emb(
            value_states,
            position_ids=position_ids,  # seq_len=kv_seq_len for transformers > 4.39
        )
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

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

        attn_output_hidden = HiddenState.from_dense(attn_output, h.token_type_mask)
        attn_output_hidden = attn_output_hidden.apply_all_types_mapping(
            self.o_proj_dict
        )

        return attn_output_hidden, past_key_value

    def freeze_text_model(self):
        for parm_dict in [
            self.q_proj_dict,
            self.k_proj_dict,
            self.k_proj_dict,
            self.o_proj_dict,
        ]:
            for param in parm_dict[TokenType.Text.name].parameters():
                param.requires_grad = False

            for param in self.rotary_emb.parameters():
                param.requires_grad = False

    def freeze_entity_model(self):
        for parm_dict in [
            self.q_proj_dict,
            self.k_proj_dict,
            self.k_proj_dict,
            self.o_proj_dict,
        ]:
            for name, param in parm_dict.named_parameters():
                if "adapter" not in name:
                    param.requires_grad = False


class DeepFuseLayerBase(ABC, nn.Module):
    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__()
        self.config = config

    """
    As PipelineParallel only support Tuple[Tensor] input/output, we need to wrap the function to make it work
    """

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        h = HiddenState.from_tuple(x[:2])
        attention_mask = x[3]
        position_ids = x[4]

        attention_mask_float = mask_to_float(attention_mask, h.dtype)

        ret, _ = self.forward_impl(
            h, attention_mask_float, position_ids, past_key_value=None, use_cache=False
        )

        return ret.to_tuple() + (attention_mask, position_ids)

    # The actual forward function, also used in decoding
    @abstractmethod
    def forward_impl(
        self,
        h: HiddenState,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        use_cache: bool,
    ) -> HiddenState:
        raise NotImplementedError

    # Map the state dict from LLaMA/SciDecoder to DeepFuse
    @abstractmethod
    def map_state_dict(
        self,
        prefix: str,
        llama_state_dict: Dict[str, torch.Tensor],
        entity_layer_id: int,
        entity_decoder_state_dict: Dict[str, torch.Tensor],
    ):
        raise NotImplementedError

    @abstractmethod
    def freeze_text_model(self):
        raise NotImplementedError

    @abstractmethod
    def freeze_entity_model(self):
        raise NotImplementedError


class TextOnly(DeepFuseLayerBase):
    """
    When the decoder layer is less than LLaMA layer, we use this to only process text
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.text_layer = LlamaDecoderLayer(config, layer_idx=0)

    def forward_impl(
        self,
        h: HiddenState,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        use_cache: bool,
    ) -> HiddenState:
        x, attention_mask = h.to_single_type_batch(TokenType.Text, attention_mask)
        x = self.text_layer(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        if use_cache:
            x, present_key_value = x
            h = h.update_single_type_batch(TokenType.Text, x)
            return h, present_key_value
        else:
            return h.update_single_type_batch(TokenType.Text, x[0]), None

    @staticmethod
    def map_state_dict(
        prefix: str,
        llama_state_dict: Dict[str, torch.Tensor],
        entity_layer_id: int,
        entity_decoder_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ret = {}

        for k, v in llama_state_dict.items():
            ret[f"{prefix}.text_layer.{k}"] = v

        return ret

    def freeze_text_model(self):
        for param in self.text_layer.parameters():
            param.requires_grad = False

    def freeze_entity_model(self):
        pass


class MixLayer(DeepFuseLayerBase):
    """
    The layer that mix the text and entity operations
    """

    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__(config)

        self.attn = FusedAttn(config)
        self.mlp = nn.ModuleDict({k.name: MLP(config, k) for k in TokenType})

        self.attn_norm = make_norm_dict(config)
        self.mlp_norm = make_norm_dict(config)

    def forward_impl(
        self,
        h: HiddenState,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        use_cache: bool,
    ) -> HiddenState:
        res = h
        h = h.apply_all_types_mapping(self.attn_norm)
        h, present_key_value = self.attn(
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = h + res

        res = h
        h = h.apply_all_types_mapping(self.mlp_norm)
        h = h.apply_all_types_mapping(self.mlp)
        h = h + res

        return h, present_key_value

    @staticmethod
    def map_state_dict(
        prefix: str,
        llama_state_dict: Dict[str, torch.Tensor],
        entity_layer_id: int,
        entity_decoder_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ret = {}

        # LLaMA params
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mapped_name = f"{prefix}.attn.{name}_dict.Text.layer.weight"
            ret[mapped_name] = llama_state_dict[f"self_attn.{name}.weight"]

        for name in [
            "mlp.gate_proj.weight",
            "mlp.down_proj.weight",
            "mlp.up_proj.weight",
        ]:
            ret[f"{prefix}.mlp.Text.{name}"] = llama_state_dict[name]

        ret[f"{prefix}.attn_norm.Text.weight"] = llama_state_dict[
            "input_layernorm.weight"
        ]
        ret[f"{prefix}.mlp_norm.Text.weight"] = llama_state_dict[
            "post_attention_layernorm.weight"
        ]
        ret[f"{prefix}.attn.rotary_emb.inv_freq"] = llama_state_dict[
            "self_attn.rotary_emb.inv_freq"
        ]

        # Entity decoder params
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            source_name = f"biogpt.layers.{entity_layer_id}.self_attn.{name}.weight"
            mapped_name = f"{prefix}.attn.{name[0]}_proj_dict.Entity.layer.weight"
            ret[mapped_name] = entity_decoder_state_dict[source_name]

            source_name = f"biogpt.layers.{entity_layer_id}.self_attn.{name}.bias"
            mapped_name = f"{prefix}.attn.{name[0]}_proj_dict.Entity.layer.bias"
            ret[mapped_name] = entity_decoder_state_dict[source_name]

        for name in ["fc1", "fc2"]:
            source_name = f"biogpt.layers.{entity_layer_id}.{name}.weight"
            mapped_name = f"{prefix}.mlp.Entity.mlp.{name}.weight"
            ret[mapped_name] = entity_decoder_state_dict[source_name]

            source_name = f"biogpt.layers.{entity_layer_id}.{name}.bias"
            mapped_name = f"{prefix}.mlp.Entity.mlp.{name}.bias"
            ret[mapped_name] = entity_decoder_state_dict[source_name]

        for name in ["self_attn_layer_norm", "final_layer_norm"]:
            norm_name = "attn_norm" if name == "self_attn_layer_norm" else "mlp_norm"

            source_name = f"biogpt.layers.{entity_layer_id}.{name}.weight"

            mapped_name = f"{prefix}.{norm_name}.Entity.weight"
            ret[mapped_name] = entity_decoder_state_dict[source_name]

            source_name = f"biogpt.layers.{entity_layer_id}.{name}.bias"
            mapped_name = f"{prefix}.{norm_name}.Entity.bias"
            ret[mapped_name] = entity_decoder_state_dict[source_name]

        return ret

    def freeze_text_model(self):
        self.attn.freeze_text_model()

        for modele_dict in [self.mlp, self.attn_norm, self.mlp_norm]:
            for param in modele_dict.parameters():
                param.requires_grad = False

    def freeze_entity_model(self):
        self.attn.freeze_entity_model()

        for modele_dict in [self.mlp, self.attn_norm, self.mlp_norm]:
            for name, param in modele_dict.named_parameters():
                if "adapter" not in name:
                    param.requires_grad = False


class SeperateLayer(DeepFuseLayerBase):
    """
    Both text and entity have their own layer
    """

    def __init__(self, config: DecDeepFuseConfig) -> None:
        super().__init__(config)

        self.layers = nn.ModuleDict(
            {k.name: SeperateLayer._to_layer_type(k, config) for k in TokenType}
        )

    @staticmethod
    def _to_layer_type(token_type: TokenType, config: DecDeepFuseConfig) -> nn.Module:
        if token_type == TokenType.Text:
            return LlamaDecoderLayer(config, layer_idx=0)
        else:
            if config.entity_decoder_model_type == EntityDecoderType.BioGPT:
                config_copy = deepcopy(config)
                config_copy.hidden_size = config.entity_hidden_size
                config_copy.intermediate_size = config.entity_intermediate_size
                return BioGptDecoderLayer(config_copy)
            else:
                return LlamaDecoderLayer(config, layer_idx=0)

    def forward_impl(
        self,
        h: HiddenState,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        use_cache: bool,
    ) -> HiddenState:
        if use_cache:
            if past_key_value is None:
                # first time, there is no past key value even the cache is enabled
                past_key_value_dict = None
            else:
                past_key_value_dict = {
                    TokenType.Text.name: (past_key_value[0], past_key_value[1]),
                    TokenType.Entity.name: (past_key_value[2], past_key_value[3]),
                }
            present_key_value_dict = {}
        else:
            past_key_value_dict = None
            present_key_value_dict = None

        for token_type in TokenType:
            x, attention_mask_t = h.to_single_type_batch(token_type, attention_mask)
            past_key_value_for_type = (
                None
                if past_key_value_dict is None
                else past_key_value_dict[token_type.name]
            )

            if type(self.layers[token_type.name]) == BioGptDecoderLayer:
                ret = self.layers[token_type.name](
                    x,
                    attention_mask=attention_mask_t,
                    past_key_value=past_key_value_for_type,
                    use_cache=use_cache,
                )
            else:
                ret = self.layers[token_type.name](
                    x,
                    attention_mask=attention_mask_t,
                    position_ids=position_ids,
                    past_key_value=past_key_value_for_type,
                    use_cache=use_cache,
                )
            if use_cache:
                x, present_key_value = ret
                present_key_value_dict[token_type.name] = present_key_value
            else:
                x = ret[0]

            h = h.update_single_type_batch(token_type, x)

        if use_cache:
            present_key_value = (
                present_key_value_dict[TokenType.Text.name][0],
                present_key_value_dict[TokenType.Text.name][1],
                present_key_value_dict[TokenType.Entity.name][0],
                present_key_value_dict[TokenType.Entity.name][1],
            )
        else:
            present_key_value = None

        return h, present_key_value

    @staticmethod
    def map_state_dict(
        prefix: str,
        llama_state_dict: Dict[str, torch.Tensor],
        entity_layer_id: int,
        entity_decoder_state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ret = {}

        # LLaMA params
        for k, v in llama_state_dict.items():
            ret[f"{prefix}.layers.Text.{k}"] = v

        # Entity decoder params
        for k, v in entity_decoder_state_dict.items():
            source_prefix = f"biogpt.layers.{entity_layer_id}."
            if not k.startswith(source_prefix):
                continue
            source_name = k[len(source_prefix) :]
            target_name = f"{prefix}.layers.Entity.{source_name}"
            ret[target_name] = v
        return ret

    def freeze_text_model(self):
        for param in self.layers.Text.parameters():
            param.requires_grad = False

    def freeze_entity_model(self):
        for param in self.layers.Entity.parameters():
            param.requires_grad = False
