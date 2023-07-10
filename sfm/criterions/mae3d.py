# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import torch.nn as nn


class MAE3d_criterions(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.l1 = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.l2 = nn.L1Loss(reduction="mean")
        self.args = args

    def forward(self, batch_data, logits, node_output):
        with torch.no_grad():
            node_mask = batch_data["node_mask"].squeeze(-1).bool()
            targets = batch_data["x"][:, :, 0][node_mask]

        logits = logits[:, 1:, :][node_mask]

        type_loss = (
            self.l1(
                logits.view(-1, logits.size(-1)).to(torch.float32),
                targets.view(-1),
            )
            * self.args.atom_loss_coeff
        )

        node_output = node_output[node_mask]
        ori_pos = batch_data["pos"][node_mask]
        pos_loss = (
            self.l2(node_output.to(torch.float32), ori_pos.to(torch.float32)).sum(
                dim=-1
            )
            * self.args.pos_loss_coeff
        )

        loss = type_loss + pos_loss

        return loss


class MAE3d_criterions_PP(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.l1 = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)
        self.l2 = nn.L1Loss(reduction=reduction)
        self.args = args

    def tensors_decode(self, value_tensor, shape_tensor):
        x_len = shape_tensor[0] * shape_tensor[1] * shape_tensor[2]
        x = value_tensor[:x_len].view(shape_tensor[0], shape_tensor[1], shape_tensor[2])
        node_output = value_tensor[x_len:].view(
            shape_tensor[3], shape_tensor[4], shape_tensor[5]
        )

        return x, node_output

    def forward(self, output, label):
        x, pos, node_mask = label
        # value_tensor, shape_tensor = output
        # logits, node_output = self.tensors_decode(value_tensor, shape_tensor)
        logits, node_output = output
        node_mask = node_mask.squeeze(-1).bool()

        targets = x[:, :, 0][node_mask]
        logits = logits[:, 1:, :][node_mask]
        ori_pos = pos[node_mask]
        node_output = node_output[node_mask]

        loss1 = (
            self.l1(
                logits.view(-1, logits.size(-1)).to(torch.float32),
                targets.view(-1),
            )
            * self.args.atom_loss_coeff
        )

        node_output_loss = (
            self.l2(node_output.to(torch.float32), ori_pos.to(torch.float32)).sum(
                dim=-1
            )
            * self.args.pos_loss_coeff
        )

        loss = loss1 + node_output_loss  # .detach_().requires_grad_(True)

        return loss


class copilot_criterions_PP(nn.Module):
    def __init__(self, args, config, reduction="mean") -> None:
        super().__init__()
        self.l1 = nn.CrossEntropyLoss(reduction=reduction)

        self.args = args
        self.config = config

    # def tensors_decode(self, value_tensor, shape_tensor):
    #     x_len = shape_tensor[0]*shape_tensor[1]*shape_tensor[2]
    #     x = value_tensor[:x_len].view(shape_tensor[0], shape_tensor[1], shape_tensor[2])
    #     node_output = value_tensor[x_len:].view(shape_tensor[3], shape_tensor[4], shape_tensor[5])

    #     return x, node_output

    def forward(self, output, label):
        # print(type(output))
        labels = label[0]
        label[1]
        logits = output

        # value_tensor, shape_tensor = output
        # logits, node_output = self.tensors_decode(value_tensor, shape_tensor)
        # labels = torch.where(input_ids > 0, labels, -100)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        # print("shift_logits", shift_logits.shape, shift_labels.shape)
        loss = self.l1(shift_logits, shift_labels)
        # print("loss", loss)

        return loss
