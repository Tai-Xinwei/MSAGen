# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

# from sklearn.metrics import roc_auc_score
import torch.distributed as dist
import torch.nn as nn


class DiffMAE3dCriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.typeloss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)
        self.nodeloss = nn.L1Loss(reduction=reduction)
        self.ernergyloss = nn.L1Loss(reduction=reduction)

        self.args = args
        self.atom_loss_coeff = self.args.atom_loss_coeff
        self.pos_loss_coeff = self.args.pos_loss_coeff
        self.y_2d_loss_coeff = self.args.y_2d_loss_coeff

    def forward(self, batch_data, logits, node_output, y_pred):
        with torch.no_grad():
            node_mask = batch_data["node_mask"].squeeze(-1).bool()
            targets = batch_data["x"][:, :, 0][node_mask]
            y_2d_target = batch_data["y"]

        # energy gap loss
        y_2d_loss = (
            self.ernergyloss(
                y_pred[:, 0, :].to(torch.float32).squeeze(),
                y_2d_target.to(torch.float32),
            )
            * self.y_2d_loss_coeff
        )

        # atom type loss
        logits = logits[:, 1:, :][node_mask]
        type_loss = (
            self.typeloss(
                logits.view(-1, logits.size(-1)).to(torch.float32),
                targets.view(-1),
            )
            * self.atom_loss_coeff
        )

        # pos mae loss
        node_output = node_output[node_mask]
        ori_pos = batch_data["pos"][node_mask]
        pos_loss = (
            self.nodeloss(node_output.to(torch.float32), ori_pos.to(torch.float32)).sum(
                dim=-1
            )
            * self.pos_loss_coeff
        )

        loss = type_loss + pos_loss + y_2d_loss
        return loss
