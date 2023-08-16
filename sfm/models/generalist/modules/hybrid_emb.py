# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from sfm.logging import logger


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
        mask_text_to_mol_attention (`bool`,  *optional*, defaults to `False`):
            Whether to mask the attention from text tokens to molecule tokens
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
        mask_text_to_mol_attention=False,
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
        self.mask_text_to_mol_attention = mask_text_to_mol_attention
        self.mfm_hidden_size = mfm_hidden_size
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MLPAdapter(nn.Module):
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
            self.act_fn(self.dropout(self.gate_proj(x)) * self.up_proj(x))
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


class QformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 32,
        hidden_dropout_prob: float = 0.1,
        hidden_act: str = "silu",
    ):
        super().__init__()

        self.adaptor_self_attn = EmbedAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=hidden_dropout_prob,
        )
        self.adaptor_cross_attn = EmbedAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=hidden_dropout_prob,
        )
        self.adaptor_mlp = MLPAdapter(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            output_size=hidden_size,
            hidden_act=hidden_act,
            dropout=hidden_dropout_prob,
        )

    def forward(self, hidden_states, query, mask):
        query, _, _ = self.adaptor_self_attn(hidden_states=query, q_states=query)
        query, _, _ = self.adaptor_cross_attn(
            hidden_states=hidden_states, q_states=query, attention_mask=mask
        )
        query = self.adaptor_mlp(query)
        return query


class Qformer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_act: str = "silu",
        num_attention_heads: int = 32,
        hidden_dropout_prob: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()

        self.adaptor_qformer_layers = nn.ModuleList(
            QformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_act=hidden_act,
            )
            for _ in range(num_layers)
        )

    def forward(self, hidden_states, query, mask):
        for layer in self.adaptor_qformer_layers:
            query = layer(hidden_states, query, mask)

        return query


class HybridEmbeddings(nn.Module):
    def __init__(self, config: AdaptorConfig, if_initialize=True):
        super().__init__()
        self.config = config
        self.mode = config.pool_mode

        self.embedding_length = config.embedding_length

        if if_initialize:
            self.mol_rep_layernorm = LlamaRMSNorm(
                config.mfm_hidden_size, eps=config.rms_norm_eps
            )
            if config.btn_adaptor:
                self.mol_adaptor = MLPAdapter(
                    hidden_size=config.mfm_hidden_size,
                    intermediate_size=config.mfm_hidden_size // 4,
                    output_size=config.hidden_size,
                    hidden_act=config.hidden_act,
                    dropout=config.hidden_dropout_prob,
                )
            else:
                self.mol_adaptor = MLPAdapter(
                    hidden_size=config.mfm_hidden_size,
                    intermediate_size=config.mfm_hidden_size,
                    output_size=config.hidden_size,
                    hidden_act=config.hidden_act,
                    dropout=config.hidden_dropout_prob,
                )

            if self.mode == "qformer" or self.mode == "multimol":
                self.qformer = Qformer(
                    config.mfm_hidden_size, config.hidden_act, num_layers=1
                )
                self.adaptor_embedding = nn.Parameter(
                    torch.zeros(1, config.embedding_length, config.mfm_hidden_size)
                )
                self.embedding_length = config.embedding_length

        # self.dummy = nn.Linear(1, 1)

        # self.mol_adapter = MLPAdapter(hidden_size=config.mfm_hidden_size, intermediate_size=config.mfm_hidden_size,
        #   output_size=config.hidden_size, hidden_act=config.hidden_act)
        # self.mol_adapter = nn.Linear(config.mfm_hidden_size, config.hidden_size)

    def _forward_embedding(
        self, mol_rep, mol_padding_mask, token_embed, input_ids: torch.LongTensor = None
    ):
        # TODO (Roger)
        # input_ids: bsz, num_tokens
        mol_rep = mol_rep.transpose(0, 1)[:, 1:, :]
        mol_padding_mask = mol_padding_mask[:, 1:]
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
        # logger.info(f"after layer norm mol_rep, {mol_rep}")

        if self.mode == "mean":
            pooled_rep = self.mol_adaptor(mol_rep) * (
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
                self.mol_adaptor(mol_rep)[:, 0, :].unsqueeze(1).expand(-1, T, -1)
            )  # B, T, H
            mol_idx_mask = mol_idx_mask.unsqueeze(-1).expand(
                -1, -1, self.config.hidden_size
            )  # B, T, H
            token_embed = torch.where(mol_idx_mask, pooled_rep, token_embed)
        elif self.mode == "full":
            mol_rep = self.mol_adaptor(mol_rep)  # [:, 1:, :]  # B, nnode, H
            new_token_embed = torch.empty_like(token_embed)

            for bidx in range(B):
                mask_idx_list = mol_idx_mask[bidx].nonzero()
                start = mask_idx_list[0]
                end = mask_idx_list[-1]
                # # In-place operation
                # token_embed[bidx, start : end + 1, :] = mol_rep[
                #     bidx, : end - start + 1, :
                # ]

                # # Out-of-place operation fix the bug with Tensor Parallelism
                # Create a new tensor with the same shape and device as token_embed
                # Copy the parts of token_embed that are not being replaced
                new_token_embed[bidx, :start, :] = token_embed[bidx, :start, :]
                new_token_embed[bidx, end + 1 :, :] = token_embed[bidx, end + 1 :, :]

                # Copy the mol_rep slice into the new tensor
                new_token_embed[bidx, start : end + 1, :] = mol_rep[
                    bidx, : end - start + 1, :
                ]

            # Assign the new tensor to token_embed
            token_embed = new_token_embed

        elif self.mode == "qformer":
            self.adaptor_embedding = self.adaptor_embedding.to(mol_rep.device).to(
                mol_rep.dtype
            )

            attn_mask = (
                mol_padding_mask.unsqueeze(1)
                .unsqueeze(2)
                .squeeze(-1)
                .expand(-1, -1, self.embedding_length, -1)
            )  # B, 1, L, nnodes

            embed_mol_rep = self.qformer(
                hidden_states=mol_rep, query=self.adaptor_embedding, mask=attn_mask
            )
            embed_mol_rep = self.mol_adaptor(embed_mol_rep)  # B, nnode, H

            cur_mol_idx = 0
            mol_idx_offset = 0
            mol_idx_mask = torch.where(mol_idx_mask, 1, 0)
            for bidx in range(B):
                mask_idx_list = mol_idx_mask[bidx].nonzero()
                pos = 0
                num_mol = 0
                while pos < len(mask_idx_list):
                    start = mask_idx_list[pos]
                    end = start + self.embedding_length
                    cur_mol_idx = -input_ids[bidx, start] - 1
                    token_embed[bidx, start:end, :] = embed_mol_rep[
                        mol_idx_offset + cur_mol_idx, :, :
                    ]

                    num_mol = max(num_mol, cur_mol_idx + 1)
                    pos += self.embedding_length
                mol_idx_offset += num_mol
        else:
            raise NotImplementedError

        return token_embed

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

        # inverted_mask = 1.0 - expanded_mask
        # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
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

        if self.config.mask_text_to_mol_attention and input_shape[-1] > 1:
            mol_token_mask = input_ids < 0
            if attention_mask is not None:
                mol_token_mask &= attention_mask
            mol_token_atten_mask = mol_token_mask.unsqueeze(1).unsqueeze(
                2
            ) & mol_token_mask.unsqueeze(1).unsqueeze(-1)
            mol_token_atten_mask[~mol_token_mask, :] = True
        else:
            mol_token_atten_mask = None

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

        if mol_token_atten_mask is not None:
            combined_attention_mask_bool &= mol_token_atten_mask

        return combined_attention_mask_bool

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


class HybridEmbeddingsPP(HybridEmbeddings):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config, if_initialize=True, **kwargs)
        # self.embed_tokens = torch.nn.Embedding(
        #     config.vocab_size, config.hidden_size, config.pad_token_id
        # )
        # self.weight = self.embed_tokens.weight.data.requires_grad_().cuda()
        # self.weight.grad = torch.zeros_like(self.weight)

        # self.partial_learnable_emb = PartialGradEmbedding(
        # self.embed_tokens, new_embedding_cutoff=32000
        # )

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        if new_num_tokens == self.config.vocab_size:
            return
        elif new_num_tokens > self.config.vocab_size:
            old_embeddings = self.embed_tokens.weight
            new_embeddings = torch.nn.Embedding(
                new_num_tokens,
                self.config.hidden_size,
                self.config.pad_token_id,
                dtype=old_embeddings.dtype,
                device=old_embeddings.device,
            )
            # random init new embeddings with normal
            new_embeddings.weight.data.normal_(mean=0.0, std=1.0)

            new_embeddings.weight.data[
                : self.config.vocab_size, :
            ] = old_embeddings.data
            self.embed_tokens = new_embeddings

        else:
            raise ValueError(
                f"new embedding size {new_num_tokens} must be larger than the current one {self.config.vocab_size}"
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

        # Merge text and mol embeddings
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

        hidden_states = inputs_embeds.to(mol_emb.dtype)

        return (hidden_states, llm_mask, position_ids)
