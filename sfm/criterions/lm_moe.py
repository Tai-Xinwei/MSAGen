# -*- coding: utf-8 -*-
from torch import nn
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.models.nlm.moe_config import MoeModelConfig


class LmMoeCriterion(nn.Module):
    def __init__(self, config: MoeModelConfig, reduction="mean"):
        super().__init__()
        self.config = config
        self.lm_loss_func = AutoregressiveCriterion(config, reduction)

    def forward(self, output, label, gate_logits):
        lm_loss, _ = self.lm_loss_func(output, label)
        num_layers, bsz, seq_len, num_experts = gate_logits.shape
        gate_logits_tuple = tuple(
            gate_logits[i].reshape(bsz * seq_len, num_experts)
            for i in range(num_layers)
        )
        lb_loss = load_balancing_loss_func(
            gate_logits=gate_logits_tuple,
            num_experts=self.config.num_local_experts,
            top_k=self.config.num_experts_per_tok,
        )

        loss = lm_loss + self.config.router_aux_loss_coef * lb_loss
        log_loss = {
            "lm_loss": lm_loss.item(),
            "lb_loss": lb_loss.item(),
        }

        return loss, log_loss
