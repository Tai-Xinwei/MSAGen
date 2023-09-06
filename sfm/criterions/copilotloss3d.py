# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn

from megatron.core import tensor_parallel
from sfm.criterions.copilotloss import CopilotCriterionsNumPP


def CopilotCriterionsMP(output, targets):
    # Shape of output: [bs, seq, H]
    # Shape of labels: [bs, seq]
    labels, loss_mask = targets[0], targets[1]
    # get union mask of labels mask and loss mask
    # labels_mask = labels != IGNORE_INDEX
    # loss_mask = loss_mask | labels_mask

    labels = labels[..., 1:]
    loss_mask = loss_mask[..., 1:].contiguous()

    # [b s h] => [s b h]
    output = output.transpose(0, 1).contiguous()
    logits = output[:-1, :, :].float()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    # logger.info(f"logits, {logits}, labels, {labels}")

    losses = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
    # [s b] => [b, s]

    losses = losses.transpose(0, 1).contiguous().view(-1)
    loss_mask = loss_mask.view(-1)
    # logger.info(f"losses, {losses}")
    if loss_mask.sum() == 0:
        loss = torch.tensor(0.0).to(losses.device)
    else:
        loss = torch.sum(losses * loss_mask) / loss_mask.sum()
    return loss


class CopilotCriterionsNumMP(CopilotCriterionsNumPP):
    def __init__(self, config, vocab_size=32001, reduction="mean") -> None:
        super().__init__(config, vocab_size, reduction)
        self.l_num = nn.L1Loss(reduction=reduction)
        self.global_step = 0
        self.wandb_log = True

    def forward(self, output, label):
        labels, loss_mask = label[0].to(torch.int64), label[1]
        num_labels = label[2]

        loss_mask = loss_mask[..., 1:].contiguous()

        logits = output[0]
        num_logits = output[1].transpose(0, 1).contiguous()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # transpose and contiguous to enable model parallelism
        shift_logits = shift_logits.transpose(0, 1).contiguous()
        shift_labels = shift_labels.transpose(0, 1).contiguous()
        shift_labels = shift_labels.to(shift_logits.device)

        loss = (
            tensor_parallel.vocab_parallel_cross_entropy(shift_logits, shift_labels)
            .transpose(0, 1)
            .contiguous()
            .view(-1)
        )
        loss_mask = loss_mask.view(-1)

        if loss_mask.sum() == 0:
            loss = torch.tensor(0.0).to(loss.device)
        else:
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        num_idx = num_labels != -100
        if num_idx.any():
            num_labels = num_labels[num_idx]
            num_logits = num_logits[num_idx].view(-1).contiguous()
            num_loss = self.l_num(num_logits, num_labels)
        else:
            num_loss = torch.tensor(0.0).to(loss.device)

        loss_log = {"lm_loss": loss, "num_loss": num_loss}
        total_loss = loss + num_loss

        return (total_loss, loss_log)
