# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
# from sklearn.metrics import roc_auc_score
import torch.distributed as dist
import torch.nn as nn


class MAE2d_criterions(nn.Module):
    def __init__(self, args, datasetmean, datasetstd) -> None:
        super().__init__()
        self.l1 = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.l2 = nn.SmoothL1Loss(reduction="mean")
        self.args = args
        self.datasetmean = datasetmean
        self.datasetstd = datasetstd

    def forward(self, targets, logits):
        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        return loss
