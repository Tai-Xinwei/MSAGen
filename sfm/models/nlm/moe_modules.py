# -*- coding: utf-8 -*-

from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.models.mixtral.modeling_mixtral import (
    MixtralFlashAttention2,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)

from sfm.logging import logger
from sfm.models.nlm.moe_config import MoeModelConfig
from sfm.utils.pipelinemode import pipemode

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as RMSNorm

    logger.info("using apex fused RMSNorm")
except ImportError:
    logger.info("using MixtralRMSNorm")
    RMSNorm = MixtralRMSNorm


def load_balancing_loss_func(
    gate_scores: torch.Tensor,
    n_exp: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    # see transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
    # see also https://github.com/huggingface/transformers/issues/29503
    # https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942
    # So modified the code to match the SwithTransformers paper

    bsz, seq_len, n_exp = gate_scores.shape  # [B, L, E]

    gate_scores = gate_scores.view(bsz * seq_len, n_exp)  # [B*L, E]

    _, selected_experts = torch.topk(gate_scores, top_k, dim=-1)  # [B*L, k]

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, n_exp
    )  # [B*L, k, n_exp]

    n_tok = bsz * seq_len
    if attention_mask is not None:
        n_tok = attention_mask.sum().item()
        expert_mask[attention_mask.reshape(-1), :, :] = 0
        gate_scores[attention_mask.reshape(-1), :] = 0

    tokens_per_expert = expert_mask.float().sum(dim=0) / n_tok  # [k, E]

    router_prob_per_expert = gate_scores.sum(dim=0) / n_tok
    router_prob_per_expert = router_prob_per_expert.unsqueeze(0)  # [1, E]

    all_expert_loss = tokens_per_expert * router_prob_per_expert  # [k, n_exp]

    loss = all_expert_loss.sum(dim=-1).mean() * n_exp

    return loss


class MoeEmbeddingsPP(nn.Module):
    def __init__(self, config: MoeModelConfig):
        super().__init__()
        self.config = config
        self.learnable_cutoff = config.learnable_cutoff
        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

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
        position_ids = (
            torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            .expand(bsz, -1)
            .contiguous()
        )

        lb_loss = torch.tensor([0.0]).to(text_embeds)

        return text_embeds, position_ids, lb_loss

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad


class SafeMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.config.router_jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.config.router_jitter_noise,
                1.0 + self.config.router_jitter_noise,
            )
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )

        routing_scores = routing_weights.reshape(batch_size, sequence_length, -1)

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

        idxes = []
        top_xes = []
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            idxes.append(idx)
            top_xes.append(top_x)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = idxes[expert_idx], top_xes[expert_idx]

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

        return final_hidden_states, routing_scores


class MoeDecoderLayerPP(nn.Module):
    def __init__(self, config: MoeModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

        self.self_attn = MixtralFlashAttention2(config, layer_idx=layer_idx)
        self.block_sparse_moe = SafeMixtralSparseMoeBlock(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "position_ids": torch.Tensor,
            "lb_loss": torch.Tensor,
        }

    def forward_attn(self, hidden_states, position_ids):
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
        return hidden_states

    def forward_moe(self, hidden_states):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_scores = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        lb_loss = load_balancing_loss_func(router_scores)
        return hidden_states, lb_loss

    @pipemode
    def forward(self, hidden_states, position_ids, lb_loss):
        # hidden_states = self.forward_attn(hidden_states, position_ids)
        hidden_states = checkpoint(
            self.forward_attn, hidden_states, position_ids, use_reentrant=False
        )

        # hidden_states, lb = self.forward_moe(hidden_states)
        hidden_states, lb = checkpoint(
            self.forward_moe, hidden_states, use_reentrant=False
        )

        lb_loss = lb_loss + lb

        return hidden_states, position_ids, lb_loss


class MoeNormPP(nn.Module):
    def __init__(self, config: MoeModelConfig):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "position_ids": torch.Tensor,
            "lb_loss": torch.Tensor,
        }

    @pipemode
    def forward(self, hidden_states, position_ids, lb_loss):
        return self.norm(hidden_states), lb_loss


class MoeHeadPP(nn.Module):
    def __init__(self, config: MoeModelConfig):
        super().__init__()
        self.config = config
        self.learnable_cutoff = config.learnable_cutoff
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight.register_hook(self.freeze_parital_weight_hook)
        self.dummy = torch.nn.Linear(1, 1)  # Make DeepSpeed happy

        self.param_dict = {
            "hidden_states": torch.Tensor,
            "lb_loss": torch.Tensor,
        }

    @property
    def emb_weight(self):
        return self.lm_head.weight

    def freeze_parital_weight_hook(self, grad):
        grad[: self.learnable_cutoff, :] = 0
        return grad

    @pipemode
    def forward(self, hidden_states, lb_loss):
        return self.lm_head(hidden_states), lb_loss
