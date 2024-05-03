# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core import parallel_state, tensor_parallel
from sfm.logging import logger


class AutoregressiveThreeDCriterion(nn.Module):
    def __init__(self, config, reduction="mean") -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.config = config
        self.vocab_size = config.vocab_size

    def forward(self, model_output, label):
        if type(label) == tuple:
            labels = label[0]
            loss_mask = label[1]
        else:
            labels = label
            loss_mask = torch.ones_like(labels).bool()

        loss_mask = loss_mask[..., 1:].contiguous().transpose(0, 1)

        logits = model_output[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # transpose and contiguous to enable model parallelism
        shift_logits = shift_logits.transpose(0, 1).contiguous()
        shift_labels = shift_labels.transpose(0, 1).contiguous()
        shift_labels = shift_labels.to(shift_logits.device)
        loss = tensor_parallel.vocab_parallel_cross_entropy(shift_logits, shift_labels)

        instruct_mask = shift_labels != -100

        loss_mask = instruct_mask & loss_mask
        # logger.info(f"{loss_mask}")
        loss_mask = loss_mask.view(-1)

        loss = loss.contiguous().view(-1)
        if loss_mask.sum() == 0:
            loss = torch.tensor(0.0).to(loss.device)
        else:
            loss = loss.masked_fill_(~loss_mask, 0.0)
            loss = torch.sum(loss) / (loss_mask).sum()

        log_loss = {"lm_loss": loss}
        return (loss, log_loss)


class AutoregressiveCriterion(nn.Module):
    def __init__(self, config, reduction="mean") -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.config = config
        self.vocab_size = config.vocab_size

    def forward(self, model_output, label_tuple):
        labels, _ = label_tuple[0]
        logits = model_output[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.cross_entropy(shift_logits, shift_labels)

        log_loss = {"lm_loss": loss}

        return (loss, log_loss)
