# -*- coding: utf-8 -*-
from typing import Optional

import torch
from torch import nn

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.models.nlm.moe_config import MoeModelConfig


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

    num_layers, bsz, seq_len, n_exp = gate_scores.shape

    gate_scores = gate_scores.view(num_layers, bsz * seq_len, n_exp)  # [L, B*H, E]

    gate_scores = gate_scores.transpose(1, 0)  # [B*H, L, E]

    _, selected_experts = torch.topk(gate_scores, top_k, dim=-1)  # [B*H, L, k]

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, n_exp
    )  # [B*H, L, k, n_exp]

    n_tok = bsz * seq_len
    if attention_mask is not None:
        n_tok = attention_mask.sum().item()
        expert_mask[attention_mask.reshape(-1), :, :, :] = 0
        gate_scores[attention_mask.reshape(-1), :] = 0

    tokens_per_expert = expert_mask.float().sum(dim=0) / n_tok  # [L, k, n_exp]

    router_prob_per_expert = gate_scores.sum(dim=0) / n_tok
    router_prob_per_expert = router_prob_per_expert.unsqueeze(1)  # [L, 1, n_exp]

    all_expert_loss = tokens_per_expert * router_prob_per_expert  # [L, k, n_exp]

    # sum over experts inside one layer, then avergae over layers and top_k
    loss = all_expert_loss.sum(dim=-1).mean() * n_exp

    return loss


class LmMoeCriterion(nn.Module):
    def __init__(self, config: MoeModelConfig, reduction="mean"):
        super().__init__()
        self.config = config
        self.lm_loss_func = AutoregressiveCriterion(config, reduction)

    @torch.compile
    def forward(self, output, label, gate_scores):
        lm_loss, _ = self.lm_loss_func(output, label)

        lb_loss = load_balancing_loss_func(
            gate_scores=gate_scores,
            n_exp=self.config.num_local_experts,
            top_k=self.config.n_exp_per_tok,
        )

        loss = lm_loss + self.config.router_aux_loss_coef * lb_loss
        log_loss = {
            "lm_loss": lm_loss.item(),
            "lb_loss": lb_loss.item(),
        }

        return loss, log_loss
