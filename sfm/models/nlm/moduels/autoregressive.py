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

    def forward(self, model_output, lable_tuple):
        labels = lable_tuple[0][0]
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
