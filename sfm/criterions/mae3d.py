# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed as dist
import torch.nn as nn

from sfm.logging import logger


class MAE3dCriterions(nn.Module):
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


class MAE3dCriterionsPP(nn.Module):
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


class ProteinMAE3dCriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)
        self.loss_pos = nn.MSELoss(reduction=reduction)
        self.args = args

    def forward(self, batch_data, logits, node_output, mask_pos, mask_aa):
        if mask_aa.any():
            with torch.no_grad():
                aa_seq = batch_data["x"][mask_aa.squeeze(-1).bool()]

            logits = logits[:, :, :][mask_aa.squeeze(-1).bool()]

            type_loss = (
                self.loss_type(
                    logits.view(-1, logits.size(-1)).to(torch.float32),
                    aa_seq.view(-1),
                )
                * self.args.atom_loss_coeff
            )
            # compute type accuracy
            type_acc = (
                logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq
            ).sum().to(torch.float32) / aa_seq.view(-1).size(0)
        else:
            type_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
            type_acc = 0.0

        pos_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        loss = type_loss + pos_loss

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "loss_pos": pos_loss,
            "type_acc": type_acc,
        }


class ProteinPMLMBPE(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_pairtype = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)
        self.loss_bpe = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)

        self.args = args
        self.num_aa_type = args.num_residues

    def forward(
        self, batch_data, logits, bpe_logits, diag_mask, mask_aa, pair_mask_aa_0
    ):
        with torch.no_grad():
            aa_seq = batch_data["x"]
            if "bpe" in batch_data:
                bpe_seq = batch_data["bpe"]
            else:
                bpe_seq = None

            paired_seq = aa_seq.unsqueeze(-1) * self.num_aa_type + aa_seq.unsqueeze(-2)

            # pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
            pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
            pair_mask_aa = pair_mask_aa & pair_mask_aa_0.bool()
            aa_seq = aa_seq[mask_aa.squeeze(-1).bool()]

            # logits [mask_L, vocab^2]
            paired_seq = paired_seq[pair_mask_aa.squeeze(-1).bool()]

            diag_logits = logits[diag_mask]

        type_loss = self.loss_pairtype(
            logits.view(-1, logits.size(-1)).to(torch.float32),
            paired_seq.view(-1),
        )
        if bpe_seq is not None:
            bpe_loss = self.loss_bpe(
                bpe_logits.view(-1, bpe_logits.size(-1)).to(torch.float32),
                bpe_seq.view(-1),
            )

            loss = type_loss + bpe_loss
        else:
            bpe_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
            loss = type_loss

        with torch.no_grad():
            # compute type accuracy
            type_acc = (
                (logits.view(-1, logits.size(-1)).argmax(dim=-1) == paired_seq)
                .to(torch.float32)
                .mean()
            )

            # compuate diag accuracy
            diag_logits = diag_logits[
                ...,
                torch.arange(self.num_aa_type) * self.num_aa_type
                + torch.arange(self.num_aa_type),
            ]
            diag_type_acc = (
                (diag_logits.view(-1, diag_logits.size(-1)).argmax(dim=-1) == aa_seq)
                .to(torch.float32)
                .mean()
            )
            if bpe_seq is not None:
                bpe_acc = (
                    (
                        bpe_logits.view(-1, bpe_logits.size(-1)).argmax(dim=-1)
                        == bpe_seq.view(-1)
                    )
                    .to(torch.float32)
                    .mean()
                )
            else:
                bpe_acc = torch.tensor([0.0], device=logits.device)

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "bpe_loss": bpe_loss,
            "type_acc": type_acc,
            "diag_type_acc": diag_type_acc,
            "bpe_acc": bpe_acc,
        }


class ProteinPMLMMSA(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_pairtype = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)
        self.loss_mlm = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)

        self.args = args
        self.num_aa_type = args.num_residues

    def forward(
        self, batch_data, logits, mlm_logits, diag_mask, mask_aa, pair_mask_aa_0
    ):
        with torch.no_grad():
            aa_seq = batch_data["x"]

            paired_seq = aa_seq.unsqueeze(-1) * self.num_aa_type + aa_seq.unsqueeze(-2)

            # pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
            pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
            pair_mask_aa = pair_mask_aa & pair_mask_aa_0.bool()

            # logits [mask_L, vocab^2]
            paired_seq = paired_seq[pair_mask_aa.squeeze(-1).bool()]

            aa_seq = aa_seq[mask_aa.squeeze(-1).bool()]

        mlm_logits = mlm_logits[mask_aa.squeeze(-1).bool()]

        pair_loss = self.loss_pairtype(
            logits.view(-1, logits.size(-1)).to(torch.float32),
            paired_seq.view(-1),
        )

        mlm_loss = self.loss_mlm(
            mlm_logits.view(-1, mlm_logits.size(-1)).to(torch.float32),
            aa_seq.view(-1),
        )

        loss = 0.125 * pair_loss + mlm_loss

        with torch.no_grad():
            # compute type accuracy
            type_acc = (
                (logits.view(-1, logits.size(-1)).argmax(dim=-1) == paired_seq)
                .to(torch.float32)
                .mean()
            )

            mlm_acc = (
                (
                    mlm_logits.view(-1, mlm_logits.size(-1)).argmax(dim=-1)
                    == aa_seq.view(-1)
                )
                .to(torch.float32)
                .mean()
            )

        return loss, {
            "total_loss": loss,
            "loss_type": pair_loss,
            "loss_mlm": mlm_loss,
            "type_acc": type_acc,
            "mlm_acc": mlm_acc,
        }


class ProteinPMLM(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_pairtype = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)

        self.args = args
        self.num_aa_type = args.num_residues

    def forward(
        self, batch_data, logits, bpe_logits, diag_mask, mask_aa, pair_mask_aa_0
    ):
        with torch.no_grad():
            aa_seq = batch_data["x"]

            paired_seq = aa_seq.unsqueeze(-1) * self.num_aa_type + aa_seq.unsqueeze(-2)

            # pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
            pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
            pair_mask_aa = pair_mask_aa & pair_mask_aa_0.bool()
            aa_seq = aa_seq[mask_aa.squeeze(-1).bool()]

            # logits [mask_L, vocab^2]
            paired_seq = paired_seq[pair_mask_aa.squeeze(-1).bool()]

            diag_logits = logits[diag_mask]

        type_loss = self.loss_pairtype(
            logits.view(-1, logits.size(-1)).to(torch.float32),
            paired_seq.view(-1),
        )

        loss = type_loss

        with torch.no_grad():
            # compute type accuracy
            type_acc = (
                (logits.view(-1, logits.size(-1)).argmax(dim=-1) == paired_seq)
                .to(torch.float32)
                .mean()
            )

            # compuate diag accuracy
            diag_logits = diag_logits[
                ...,
                torch.arange(self.num_aa_type) * self.num_aa_type
                + torch.arange(self.num_aa_type),
            ]
            diag_type_acc = (
                (diag_logits.view(-1, diag_logits.size(-1)).argmax(dim=-1) == aa_seq)
                .to(torch.float32)
                .mean()
            )

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "type_acc": type_acc,
            "diag_type_acc": diag_type_acc,
        }


class ProteinMLM(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)
        self.args = args
        self.num_aa_type = args.num_residues

    def forward(self, batch_data, logits, node_output, mask_pos, mask_aa):
        with torch.no_grad():
            aa_seq = batch_data["x"][mask_aa.squeeze(-1).bool()]

        logits = logits[:, :, :][mask_aa.squeeze(-1).bool()]

        type_loss = (
            self.loss_type(
                logits.view(-1, logits.size(-1)).to(torch.float32),
                aa_seq.view(-1),
            )
            * self.args.atom_loss_coeff
        )

        with torch.no_grad():
            # compute type accuracy
            type_acc = (
                logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq
            ).sum().to(torch.float32) / aa_seq.view(-1).size(0)

        loss = type_loss

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "diag_type_acc": type_acc,
        }
