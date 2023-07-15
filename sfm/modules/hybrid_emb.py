# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from models.transformers.configuration_utils import PretrainedConfig
from models.transformers.models.llama.modeling_llama import (
    LlamaMLPAdapter,
    LlamaRMSNorm,
    _expand_mask,
    _make_causal_mask,
)


class AdaptorConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        Example:

    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "graphormer_llama_adaptor"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        mfm_hidden_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.mfm_hidden_size = mfm_hidden_size
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class EmbedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, hidden_size: int, num_attention_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        # self.config = config
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # self.max_position_embeddings = max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        # self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # bsz, q_len, _ = hidden_states.size()
        bsz, kv_seq_len, _ = hidden_states.size()
        bszq, q_len, _ = q_states.shape

        if bsz != bszq:
            query_states = (
                self.q_proj(q_states)
                .view(1, -1, self.num_heads, self.head_dim)
                .expand(bsz, -1, -1, -1)
                .transpose(1, 2)
            )
        else:
            query_states = (
                self.q_proj(q_states)
                .view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, kv_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, kv_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            # print(attention_mask); exit()
            # attn_weights = attn_weights + attention_mask
            attn_weights = attn_weights.masked_fill(
                attention_mask.to(torch.bool),
                float("-inf"),
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Hybrid_emb(nn.Module):
    def __init__(self, config: AdaptorConfig):
        super().__init__()
        self.config = config
        self.mode = config.pool_mode

        # self.embed_tokens = torch.nn.Embedding(
        # config.vocab_size, config.hidden_size, config.pad_token_id
        # )

        self.embedding_length = config.embedding_length

        self.mol_rep_layernorm = LlamaRMSNorm(
            config.mfm_hidden_size, eps=config.rms_norm_eps
        )
        if config.btn_adaptor:
            self.mol_adapter = LlamaMLPAdapter(
                hidden_size=config.mfm_hidden_size,
                intermediate_size=config.mfm_hidden_size // 4,
                output_size=config.hidden_size,
                hidden_act=config.hidden_act,
            )
        else:
            self.mol_adapter = LlamaMLPAdapter(
                hidden_size=config.mfm_hidden_size,
                intermediate_size=config.mfm_hidden_size,
                output_size=config.hidden_size,
                hidden_act=config.hidden_act,
                dropout=config.hidden_dropout_prob,
            )

        if self.mode == "qformer" or self.mode == "multimol":
            self.embed_attn = EmbedAttention(
                hidden_size=config.mfm_hidden_size,
                num_attention_heads=32,
                dropout=config.hidden_dropout_prob,
            )
            self.cross_attn = EmbedAttention(
                hidden_size=config.mfm_hidden_size,
                num_attention_heads=32,
                dropout=config.hidden_dropout_prob,
            )
            self.adaptor_embedding = torch.zeros(
                (1, config.embedding_length, config.mfm_hidden_size), requires_grad=True
            )
            self.embedding_length = config.embedding_length

        # self.mol_adapter = MLPAdapter(hidden_size=config.mfm_hidden_size, intermediate_size=config.mfm_hidden_size,
        #   output_size=config.hidden_size, hidden_act=config.hidden_act)
        # self.mol_adapter = nn.Linear(config.mfm_hidden_size, config.hidden_size)

    def _forward_embedding(
        self, mol_rep, mol_padding_mask, token_embed, input_ids: torch.LongTensor = None
    ):
        # TODO (Roger)
        # input_ids: bsz, num_tokens
        B, T = input_ids.shape
        mol_idx_mask = input_ids < 0  # B, T

        # with torch.no_grad():
        #     token_embed = self.embed_tokens(
        #         input_ids.masked_fill(mol_idx_mask, 0)
        #     )  # B, T, hidden_size
        if not torch.any(mol_idx_mask):
            return token_embed

        mol_padding_mask = mol_padding_mask.long().unsqueeze(-1)
        mol_rep = self.mol_rep_layernorm(mol_rep)

        if self.mode == "mean":
            pooled_rep = self.mol_adapter(mol_rep) * (
                1 - mol_padding_mask
            )  # padding_mask: B, nnodes
            pooled_rep = pooled_rep[:, 1:, :].sum(dim=1, keepdim=True)  # B, 1, H
            sum_mask = (1 - mol_padding_mask).sum(dim=1) - 1
            pooled_rep = pooled_rep / sum_mask.unsqueeze(-1)
            pooled_rep = pooled_rep.expand(-1, T, -1)  # B, T, H
            mol_idx_mask = mol_idx_mask.unsqueeze(-1).expand(
                -1, -1, self.config.hidden_size
            )  # B, T, H
            token_embed = torch.where(mol_idx_mask, pooled_rep, token_embed)
        elif self.mode == "cls":
            pooled_rep = (
                self.mol_adapter(mol_rep)[:, 0, :].unsqueeze(1).expand(-1, T, -1)
            )  # B, T, H
            mol_idx_mask = mol_idx_mask.unsqueeze(-1).expand(
                -1, -1, self.config.hidden_size
            )  # B, T, H
            token_embed = torch.where(mol_idx_mask, pooled_rep, token_embed)
        elif self.mode == "full":
            # raise NotImplementedError
            mol_rep = self.mol_adapter(mol_rep)[:, 1:, :]  # B, nnode, H

            # token_embed = token_embed.unsqueeze(1).expand(-1, mol_rep.shape[1], -1, -1) # B, nnode, T, H
            for bidx in range(B):
                mask_idx_list = (mol_idx_mask[bidx] is True).nonzero()
                start = mask_idx_list[0]
                end = mask_idx_list[-1]
                token_embed[bidx, start : end + 1, :] = mol_rep[
                    bidx, : end - start + 1, :
                ]
        elif self.mode == "qformer":
            self.adaptor_embedding = self.adaptor_embedding.to(mol_rep.device).to(
                mol_rep.dtype
            )
            embedding_f, _, _ = self.embed_attn(
                hidden_states=self.adaptor_embedding, q_states=self.adaptor_embedding
            )

            # mol_rep = mol_rep[:, 1:, :]  # B, nnode, H
            attn_mask = (
                mol_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .squeeze(-1)
                .expand(-1, -1, self.embedding_length, -1)
            )  # B, 1, L, nnodes
            embed_mol_rep, _, _ = self.cross_attn(
                hidden_states=mol_rep, q_states=embedding_f, attention_mask=attn_mask
            )
            embed_mol_rep = self.mol_adapter(embed_mol_rep)  # B, nnode, H
            for bidx in range(B):
                mask_idx_list = (mol_idx_mask[bidx] is True).nonzero()
                start = mask_idx_list[0]
                end = mask_idx_list[-1]
                token_embed[bidx, start : end + 1, :] = embed_mol_rep[bidx, :, :]
        elif self.mode == "multimol":
            self.adaptor_embedding = self.adaptor_embedding.to(mol_rep.device).to(
                mol_rep.dtype
            )
            embedding_f, _, _ = self.embed_attn(
                hidden_states=self.adaptor_embedding, q_states=self.adaptor_embedding
            )

            attn_mask = (
                mol_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .squeeze(-1)
                .expand(-1, -1, self.embedding_length, -1)
            )  # B, 1, L, nnodes
            embed_mol_rep, _, _ = self.cross_attn(
                hidden_states=mol_rep, q_states=embedding_f, attention_mask=attn_mask
            )
            embed_mol_rep = self.mol_adapter(embed_mol_rep)  # B, nnode, H

            mol_idx = 0
            mol_idx_mask = torch.where(mol_idx_mask, 1, 0)
            for bidx in range(B):
                mask_idx_list = mol_idx_mask[bidx].nonzero()
                pos = 0
                while pos < len(mask_idx_list):
                    start = mask_idx_list[pos]
                    end = start + self.embedding_length

                    token_embed[bidx, start:end, :] = embed_mol_rep[mol_idx, :, :]

                    mol_idx += 1
                    pos += self.embedding_length
        else:
            raise NotImplementedError

        return token_embed

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
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
            ).bool()

        return combined_attention_mask

    def forward(
        self,
        mol_emb,
        mol_padding_mask,
        text_embeds,
        attention_mask,
        input_ids,
    ):
        if text_embeds is not None:
            batch_size, seq_length, _ = text_embeds.shape
        else:
            raise ValueError("text_embeds cannot be None")

        past_key_values_length = 0

        device = text_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        inputs_embeds = self._forward_embedding(
            mol_emb, mol_padding_mask, text_embeds, input_ids
        )

        # # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length_with_past),
        #         dtype=torch.bool,
        #         device=inputs_embeds.device,
        #     )
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask,
        #     (batch_size, seq_length),
        #     inputs_embeds,
        #     past_key_values_length,
        # )

        hidden_states = inputs_embeds

        return hidden_states, position_ids


class Hybrid_emb_PP(Hybrid_emb):
    def forward(self, input_tuple: Tuple):
        x, padding_mask, input_ids, attention_mask = input_tuple

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

        inputs_embeds = self._forward_embedding(x, padding_mask, input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        return (hidden_states, attention_mask, position_ids)
