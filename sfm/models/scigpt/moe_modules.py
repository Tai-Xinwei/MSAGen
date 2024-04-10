# -*- coding: utf-8 -*-
from typing import Mapping

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


class SafeMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0 and self.training:
                # When no tokens are selected, we need to mock the current_state
                # Or there will be no gradient for the expert
                # Then the PP will fail
                # see https://github.com/microsoft/DeepSpeed/issues/5066
                top_x_ = torch.zeros(1).to(hidden_states.device).to(torch.int32)
                top_x_list = top_x_.tolist()
                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                fake_state = expert_layer(current_state * 0)
                final_hidden_states.index_add_(
                    0, top_x_, fake_state.to(hidden_states.dtype)
                )
            else:
                # in torch it is faster to index using lists than torch tensors
                top_x_list = top_x.tolist()
                idx_list = idx.tolist()

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
                current_hidden_states = (
                    expert_layer(current_state)
                    * routing_weights[top_x_list, idx_list, None]
                )

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype)
                )

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class ScigptMoeDecoderLayerPP(nn.Module):
    def __init__(self, config: ScigptMoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = MixtralFlashAttention2(config, layer_idx=layer_idx)
        self.block_sparse_moe = SafeMixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size)

        if config.compile_layers:
            # Moe And Flash attn cannot be compiled yet.
            self.input_layernorm = torch.compile(self.input_layernorm)
            self.post_attention_layernorm = torch.compile(self.post_attention_layernorm)

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "position_ids": torch.Tensor,
            "gate_logits": torch.Tensor,
        }

    def state_dict(self):
        state_dict = super().state_dict()
        if self.config.compile_layers:
            return {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
        else:
            return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.config.compile_layers:
            state_dict_compiled = {}
            for k, v in state_dict.items():
                if "layernorm" in k:
                    fields = k.split(".")
                    fields = fields[0] + "._orig_mod." + ".".join(fields[1:])
                    state_dict_compiled[fields] = v
                else:
                    state_dict_compiled[k] = v
            state_dict = state_dict_compiled
        return super().load_state_dict(state_dict, strict, assign)

    @pipemode
    def forward(self, hidden_states, position_ids, gate_logits):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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

    @pipemode
    def forward(self, hidden_states, gate_logits):
        return self.lm_head(hidden_states), gate_logits
