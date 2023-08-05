# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import torch.nn as nn


class CopilotCriterions(nn.Module):
    def __init__(self, config, vocab_size=32001, reduction="mean") -> None:
        super().__init__()
        self.l1 = nn.CrossEntropyLoss(reduction=reduction)
        self.config = config
        self.vocab_size = vocab_size

    def forward(self, output, label):
        labels = label
        logits = output

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.l1(shift_logits, shift_labels)

        return loss


class CopilotCriterionsPP(CopilotCriterions):
    def forward(self, output, label):
        labels = label[0]
        logits = output

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.l1(shift_logits, shift_labels)

        return loss


class CopilotCriterionsMP(nn.Module):
    def __init__(self, config, vocab_size=32001, reduction="mean") -> None:
        super().__init__()
        self.l1 = nn.CrossEntropyLoss(reduction=reduction)
        self.config = config
        self.vocab_size = vocab_size

    def forward(self, output, label):
        labels = label[0]
        logits = output

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.l1(shift_logits, shift_labels)

        return loss
