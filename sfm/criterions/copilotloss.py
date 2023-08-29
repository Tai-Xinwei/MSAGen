# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import torch.nn as nn

from sfm.logging import logger

import wandb  # isort:skip


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
        logits = output[0]
        # logger.info(f"labels, {labels.shape}, logits, {logits.shape}")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.l1(shift_logits, shift_labels)

        return loss


class CopilotCriterionsNumPP(CopilotCriterions):
    def __init__(self, config, vocab_size=32001, reduction="mean") -> None:
        super().__init__(config, vocab_size, reduction)
        self.l_num = nn.L1Loss(reduction=reduction)
        self.l_bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.global_step = 0
        # TODO
        self.mlp_bce = False
        self.wandb_log = True

    def forward(self, output, label):
        labels = label[0].to(torch.int64)
        num_labels = label[2]

        # logits, num_logits = output
        logits = output[0]
        num_logits = output[1]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        loss = self.l1(shift_logits, shift_labels)

        num_idx = num_labels != -100
        if num_idx.any():
            num_labels = num_labels[num_idx]
            num_logits = num_logits[num_idx].view(-1)
            num_loss = self.l_num(num_logits, num_labels)
        else:
            num_loss = torch.tensor(0.0).to(loss.device)

        if self.mlp_bce:
            lm_binary_label_mask = (labels == 8241) | (labels == 3782)
            # left shift
            lm_binary_label_mask_shift = torch.cat(
                [
                    lm_binary_label_mask[:, 1:],
                    torch.zeros_like(lm_binary_label_mask[:, 0]).unsqueeze(-1),
                ],
                dim=-1,
            )
            lm_binary_logits = num_logits[lm_binary_label_mask_shift]
            lm_binary_label = torch.zeros_like(lm_binary_logits)
            lm_binary_label_dict_idx = labels[lm_binary_label_mask]
            lm_binary_label[lm_binary_label_dict_idx == 8241] = 1
            lm_binary_label[lm_binary_label_dict_idx == 3782] = 0
            lm_binary_loss = self.l_bce(lm_binary_logits, lm_binary_label)
        else:
            lm_binary_loss = torch.tensor(0.0).to(loss.device)

        # For local single A100 training
        self.global_step += 1

        if self.global_step % 10 == 0:
            logger.info(
                f"lm_loss, {loss}, num_loss, {num_loss} bce_loss, {lm_binary_loss}"
            )
            # # TODO: this should use wandb inside, check pfm/pretrain_pfm.sh
            # if self.wandb_log:
            #     wandb.log(
            #         {"lm_loss": loss, "num_loss": num_loss, "bce_loss": lm_binary_loss}
            #     )

        return loss + num_loss + lm_binary_loss
