# -*- coding: utf-8 -*-
import math
import os
from abc import abstractmethod
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
    apply_rotary_pos_emb,
    repeat_kv,
)

from sfm.logging import logger
from sfm.utils import PretrainedLayerSpec, TiedPretrainedLayerSpec
from sfm.utils.pipelinemode import pipemode


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


class LlamaMemEffAttention(LlamaAttention):
    # Use this class to replace LlamaAttention for memory efficient attention in V100, A100 should use flash attention instead
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        with torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_mem_efficient=True, enable_flash=False
        ):
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=attention_mask
            )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayerPP(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, l: int, enable_mem_efficient: bool = True):
        super().__init__(config)
        if enable_mem_efficient:
            self.self_attn = LlamaMemEffAttention(config)

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


class LlamaEmbeddingsBase(nn.Module):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 32001):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.learnable_cutoff = learnable_cutoff
        self.embed_tokens.weight.register_hook(self.freeze_parital_weight_hook)

    # Copied from transformers.models.bart.modeling_bart._make_causal_mask
    def _make_causal_mask(
        self,
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), False, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), True)
        # mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [
                    torch.ones(
                        tgt_len, past_key_values_length, dtype=mask.dtype, device=device
                    ),
                    mask,
                ],
                dim=-1,
            )
        assert (
            mask.dtype == torch.bool
        ), f"expected mask to have dtype torch.bool, but got {mask.dtype}"

        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(
        self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
    ):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(
            bsz, 1, tgt_len, src_len
        )  # .to(dtype)

        return expanded_mask.to(torch.bool)

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length,
        input_ids,
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask_bool = None
        if input_shape[-1] > 1:
            combined_attention_mask_bool = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)

            combined_attention_mask_bool = (
                expanded_attn_mask
                if combined_attention_mask_bool is None
                else expanded_attn_mask & combined_attention_mask_bool
            )

        return combined_attention_mask_bool

    @property
    def emb_weight(self):
        return self.embed_tokens.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    @abstractmethod
    def forward():
        raise NotImplementedError


class LlamaEmbeddingsPP(LlamaEmbeddingsBase):
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 32001):
        super().__init__(config, learnable_cutoff=learnable_cutoff)

    def forward(
        self, input_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        mol_emb, mol_padding_mask, llm_mask, input_ids = input_tuple

        mol_idx_mask = input_ids < 0  # B, T

        text_embeds = self.embed_tokens(
            input_ids.masked_fill(mol_idx_mask, 0)
        )  # B, T, hidden_size

        return mol_emb, mol_padding_mask, text_embeds, llm_mask, input_ids


class NumMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(NumMLP, self).__init__()
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
    def __init__(self, config: LlamaConfig, learnable_cutoff: int = 0):
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.learnable_cutoff = learnable_cutoff
        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)
        self.num_head = NumMLP(config.hidden_size, 4 * config.hidden_size, 1)

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
    def to_layers(
        cls, args, config, learnable_cutoff=0, new_num_tokens=None, load_ckpt=False
    ):
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
