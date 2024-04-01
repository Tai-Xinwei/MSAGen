# -*- coding: utf-8 -*-
from torch import nn
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.models.scigpt.moe_config import ScigptMoeConfig


class LmMoeCriterion(nn.Module):
    def __init__(self, config: ScigptMoeConfig, reduction="mean"):
        super().__init__()
        self.lm_loss_func = AutoregressiveCriterion(config, reduction)

    def forward(self, output, label, gate_logits):
        lm_loss, _ = self.lm_loss_func(output, label)
        lb_loss = load_balancing_loss_func(
            gate_logits=gate_logits,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
        )

        loss = lm_loss + self.config.config.router_aux_loss_coef * lb_loss
        log_loss = {
            "lm_loss": lm_loss.item(),
            "lb_loss": lb_loss.item(),
        }

        return loss, log_loss
