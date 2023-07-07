# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
# from sklearn.metrics import roc_auc_score
import torch.distributed as dist
import torch.nn as nn


class L1_criterions(nn.Module):
    def __init__(self, args, reduction="mean", data_mean=0.0, data_std=1.0) -> None:
        super().__init__()
        # self.l1 = nn.CrossEntropyLoss(reduction='reduction, ignore_index=0)
        self.l1 = nn.L1Loss(reduction=reduction)
        self.args = args
        self.datasetmean = data_mean
        self.datasetstd = data_std

    def forward(self, batch_data, logits, node_output):
        with torch.no_grad():
            y = batch_data["y"]
        logits = logits[:, 0, :].reshape(-1)
        # print(y.shape, logits.shape)

        loss = self.l1(
            logits, (y[: logits.size(0)] - self.datasetmean) / self.datasetstd
        )

        return loss


class L1_criterions_PP(nn.Module):
    def __init__(self, args, reduction="mean", data_mean=0.0, data_std=1.0) -> None:
        super().__init__()
        # self.l1 = nn.CrossEntropyLoss(reduction='reduction, ignore_index=0)
        self.l1 = nn.L1Loss(reduction=reduction)
        self.args = args
        self.datasetmean = data_mean
        self.datasetstd = data_std

    def forward(self, output, label):
        x, y, _ = label
        logits = output
        logits = logits[:, 0, :].reshape(-1)
        # print(y.shape, logits.shape)
        # print(y)

        loss = self.l1(
            logits, (y[: logits.size(0)] - self.datasetmean) / self.datasetstd
        )
        # loss = self.l1(logits + self.datasetmean / self.datasetstd, y[: logits.size(0)] / self.datasetstd)
        # loss = self.l1(logits, y[: logits.size(0)])

        return loss


class Binary_criterions(nn.Module):
    def __init__(self, args, reduction="mean", data_mean=0.0, data_std=1.0) -> None:
        super().__init__()
        # self.l1 = nn.CrossEntropyLoss(reduction='reduction, ignore_index=0)
        # self.l1 = nn.L1Loss(reduction=reduction)
        self.args = args
        self.reduction = reduction
        # self.datasetmean = data_mean
        # self.datasetstd = data_std

    def forward(self, output, label):
        x, y, _ = label
        logits = output
        logits = logits[:, 0, :]
        # print(y.shape, logits.shape)
        # print(y, logits) ; exit()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1).float(), y.reshape(-1).float(), reduction=self.reduction
        )
        # loss = self.l1(logits, y[: logits.size(0)])

        return loss
