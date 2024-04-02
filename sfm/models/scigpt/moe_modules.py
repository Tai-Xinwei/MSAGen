# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers.models.mixtral.modeling_mixtral import (
    MixtralFlashAttention2,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)

from sfm.models.scigpt.moe_config import ScigptMoeConfig
from sfm.utils.pipelinemode import pipemode


class ScigptMoeEmbeddingsPP(nn.Module):
    def __init__(self, config: ScigptMoeConfig):
        super().__init__()
        self.config = config
        self.learnable_cutoff = config.learnable_cutoff

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.embed_tokens.weight.register_hook(self.freeze_parital_weight_hook)

        self.param_dict = {
            "input_ids": torch.Tensor,
        }

    @pipemode
    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        text_embeds = self.embed_tokens(input_ids)
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids.device
        ).expand(bsz, -1)

        gate_logits = torch.zeros(
            self.config.num_hidden_layers, bsz, seq_len, self.config.num_local_experts
        ).to(self.embed_tokens.weight)

        return text_embeds, position_ids, gate_logits

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad


class ScigptMoeDecoderLayerPP(nn.Module):
    def __init__(self, config: ScigptMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = MixtralFlashAttention2(config, layer_idx=layer_idx)
        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layer_norm = MixtralRMSNorm(config.hidden_size)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size)

        if config.compile_layers:
            # No need & cannot compile attn as it is already FlashAttn
            self.block_sparse_moe = torch.compile(self.block_sparse_moe)
            self.input_layer_norm = torch.compile(self.input_layer_norm)
            self.post_attention_layernorm = torch.compile(self.post_attention_layernorm)

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "position_ids": torch.Tensor,
            "gate_logits": torch.Tensor,
        }

    @pipemode
    def forward(self, hidden_states, position_ids, gate_logits):
        residual = hidden_states
        hidden_states = self.input_layer_norm(hidden_states)

        hidden_states, _, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,  # default to causal attention
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        gate_logits[self.layer_idx] = router_logits

        return hidden_states, position_ids, gate_logits


class ScigptMoeNormPP(nn.Module):
    def __init__(self, config: ScigptMoeConfig):
        super().__init__()
        self.config = config
        self.norm = MixtralRMSNorm(config.hidden_size)

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "position_ids": torch.Tensor,
            "gate_logits": torch.Tensor,
        }

    @pipemode
    def forward(self, hidden_states, position_ids, gate_logits):
        return self.norm(hidden_states), gate_logits


class ScigptMoeHeadPP(nn.Module):
    def __init__(self, config: ScigptMoeConfig):
        super().__init__()
        self.config = config
        self.learnable_cutoff = config.learnable_cutoff
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "gate_logits": torch.Tensor,
        }

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

    @pipemode
    def forward(self, hidden_states, gate_logits):
        return self.lm_head(hidden_states), gate_logits
