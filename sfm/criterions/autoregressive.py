# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import torch.nn as nn

from sfm.logging import logger


class AutoregressiveCriterion(nn.Module):
    def __init__(self, config, reduction="mean") -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.config = config
        self.vocab_size = config.vocab_size

    def forward(self, output, label):
        if type(label) == tuple:
            labels = label[0]
        elif type(label) == dict:
            labels = label["x"]
        else:
            labels = label

        logits = output

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.cross_entropy(shift_logits, shift_labels)

        return (loss, {})


class Bio0AutoregressiveCriterion(nn.Module):
    def __init__(self, config, reduction="mean") -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        self.config = config
        self.vocab_size = config.vocab_size

    def forward(self, output, label):
        if type(label) == tuple:
            labels = label[0]
        elif type(label) == dict:
            labels = label["x"]
        else:
            labels = label

        lm_logits, special_logits = output
        special_logits_mask = labels > 32000
        lm_logits = torch.where(special_logits_mask, special_logits, lm_logits)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.cross_entropy(shift_logits, shift_labels)

        return (loss, {})
