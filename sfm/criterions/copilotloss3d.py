# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from megatron.core import tensor_parallel


def CopilotCriterionsMP(output, targets):
    # Shape of output: [seq, bs, H]
    # Shape of labels: [bs, seq]
    labels, loss_mask = targets[0], targets[1]
    # get union mask of labels mask and loss mask
    # labels_mask = labels != IGNORE_INDEX
    # loss_mask = loss_mask | labels_mask

    labels = labels[..., 1:]
    loss_mask = loss_mask[..., 1:].contiguous()

    logits = output[:-1, :, :].float()
    # args = get_args()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    # logger.info(f"logits, {logits}, labels, {labels}")

    losses = tensor_parallel.vocab_parallel_cross_entropy(logits.contiguous(), labels)
    # [s b] => [b, s]

    losses = losses.transpose(0, 1).contiguous().view(-1)
    loss_mask = loss_mask.view(-1)
    # logger.info(f"losses, {losses}")
    if loss_mask.sum() == 0:
        loss = torch.tensor(0.0).to(losses.device)
    else:
        loss = torch.sum(losses * loss_mask) / loss_mask.sum()
    return loss
