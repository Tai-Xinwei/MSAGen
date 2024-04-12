# -*- coding: utf-8 -*-
from typing import Optional

import torch
from torch import nn

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.models.nlm.moe_config import MoeModelConfig


def load_balancing_loss_func(
    gate_scores: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    # see transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
    _, selected_experts = torch.topk(gate_scores, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(gate_scores, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = gate_scores.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(attention_mask.device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(attention_mask.device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            gate_scores * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class LmMoeCriterion(nn.Module):
    def __init__(self, config: MoeModelConfig, reduction="mean"):
        super().__init__()
        self.config = config
        self.lm_loss_func = AutoregressiveCriterion(config, reduction)

    def forward(self, output, label, gate_scores):
        lm_loss, _ = self.lm_loss_func(output, label)

        lb_loss = load_balancing_loss_func(
            gate_scores=gate_scores,
            num_experts=self.config.num_local_experts,
            top_k=self.config.num_experts_per_tok,
        )

        loss = lm_loss + self.config.router_aux_loss_coef * lb_loss
        log_loss = {
            "lm_loss": lm_loss.item(),
            "lb_loss": lb_loss.item(),
        }

        return loss, log_loss
