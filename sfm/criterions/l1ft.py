# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sfm.logging import logger


class L1Criterions(nn.Module):
    def __init__(self, reduction="mean", data_mean=0.0, data_std=1.0) -> None:
        super().__init__()
        self.l1 = nn.L1Loss(reduction=reduction)
        self.datasetmean = data_mean
        self.datasetstd = data_std

    def forward(self, batch_data: Dict, logits: Tensor):
        with torch.no_grad():
            y = batch_data["y"]
        logits = logits[:, 0, :].reshape(-1).float()
        # logger.info("logits: {}, y: {}".format(logits, y))
        loss = self.l1(
            logits, (y[: logits.size(0)] - self.datasetmean) / self.datasetstd
        )

        return loss


class L1CriterionsPP(L1Criterions):
    def forward(self, output: Tensor, label: Tuple):
        x, y, _ = label
        logits = output
        logits = logits[:, 0, :].reshape(-1)

        loss = self.l1(
            logits.float(), (y[: logits.size(0)] - self.datasetmean) / self.datasetstd
        )

        return loss


class BinaryCriterions(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, output, label):
        x, y, _ = label
        logits = output
        logits = logits[:, 0, :]
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1).float(), y.reshape(-1).float(), reduction=self.reduction
        )

        return loss
