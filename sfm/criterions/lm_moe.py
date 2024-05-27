# -*- coding: utf-8 -*-
from typing import Optional

import torch
from torch import nn

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.models.nlm.moe_config import MoeModelConfig


class LmMoeCriterion(nn.Module):
    def __init__(self, config: MoeModelConfig, reduction="mean"):
        super().__init__()
        self.config = config
        self.lm_loss_func = AutoregressiveCriterion(config, reduction)

    def forward(self, output, label, lb_loss):
        lm_loss, _ = self.lm_loss_func(output, label)

        lb_loss = lb_loss[0] / self.config.num_hidden_layers

        loss = lm_loss + self.config.router_aux_loss_coef * lb_loss
        log_loss = {
            "lm_loss": lm_loss.item(),
            "lb_loss": lb_loss.item(),
        }

        return loss, log_loss
