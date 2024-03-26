# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

# from sklearn.metrics import roc_auc_score
import torch.nn as nn

from sfm.models.tox.modules.physics import compute_PDEloss


class DiffMAE3dCriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.typeloss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=0)
        self.nodeloss = nn.L1Loss(reduction=reduction)
        self.ernergyloss = nn.L1Loss(reduction=reduction)

        self.args = args
        self.atom_loss_coeff = self.args.atom_loss_coeff
        self.pos_loss_coeff = self.args.pos_loss_coeff
        self.y_2d_loss_coeff = self.args.y_2d_loss_coeff

    def forward(self, batch_data, logits, node_output, y_pred):
        with torch.no_grad():
            node_mask = batch_data["node_mask"].squeeze(-1).bool()
            targets = batch_data["x"][:, :, 0][node_mask]
            y_2d_target = batch_data["y"]

        # energy gap loss
        y_2d_loss = (
            self.ernergyloss(
                y_pred[:, 0, :].to(torch.float32).squeeze(),
                y_2d_target.to(torch.float32),
            )
            * self.y_2d_loss_coeff
        )

        # atom type loss
        logits = logits[:, 1:, :][node_mask]
        type_loss = (
            self.typeloss(
                logits.view(-1, logits.size(-1)).to(torch.float32),
                targets.view(-1),
            )
            * self.atom_loss_coeff
        )

        # pos mae loss
        node_output = node_output[node_mask]
        ori_pos = batch_data["pos"][node_mask]
        pos_loss = (
            self.nodeloss(node_output.to(torch.float32), ori_pos.to(torch.float32)).sum(
                dim=-1
            )
            * self.pos_loss_coeff
        )

        loss = type_loss + pos_loss + y_2d_loss
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
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        self.loss_pos = nn.MSELoss(reduction="mean")
        self.loss_angle = nn.MSELoss(reduction="mean")
        self.loss_dist = nn.L1Loss(reduction="mean")
        self.args = args
        # self.pos_loss_coeff = 0.2
        # self.type_loss_coeff = 1.0

    def forward_score(
        self,
        batch_data,
        logits,
        node_output,
        angle_output,
        mask_pos,
        mask_aa,
        ang_score,
        ang_score_norm,
    ):
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

        # mask_pos =  mask_aa | mask_pos
        if mask_pos.any():
            with torch.no_grad():
                ori_pos = batch_data["pos"]
                bsz, _, _ = ori_pos.size()
                pos_mask = ori_pos == float("inf")
                ori_pos = ori_pos.masked_fill(pos_mask, 0.0)

            ori_pos = ori_pos[mask_pos.squeeze(-1)]
            node_output = node_output[mask_pos.squeeze(-1)]

            pos_loss = (
                self.loss_pos(
                    node_output.to(torch.float32), ori_pos.to(torch.float32)
                ).sum(dim=-1)
                * self.args.pos_loss_coeff  # / bsz
            )
        else:
            pos_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        # mask_angle = mask_aa | mask_pos
        mask_angle = mask_pos
        if mask_angle.any():
            with torch.no_grad():
                ori_angle = batch_data["ang"]
                angle_mask = ori_angle == float("inf")

            mask_angle = mask_angle & (~angle_mask)
            ang_score = ang_score[mask_angle.squeeze(-1)]
            angle_output = angle_output[mask_angle.squeeze(-1)]

            angle_loss = ((ang_score - angle_output) ** 2 / ang_score_norm).mean()
        else:
            angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        coeff_pos = (
            0.5
            * (type_loss.detach().item() + angle_loss.detach().item())
            / (
                type_loss.detach().item()
                + pos_loss.detach().item()
                + angle_loss.detach().item()
            )
        )
        coeff_ang = (
            0.5
            * (type_loss.detach().item() + pos_loss.detach().item())
            / (
                type_loss.detach().item()
                + pos_loss.detach().item()
                + angle_loss.detach().item()
            )
        )
        coeff_type = 1 - coeff_pos - coeff_ang

        # loss = coeff_pos * pos_loss + coeff_type * type_loss + 0.5 * coeff_ang * angle_loss

        # loss = type_loss + 0.2 * pos_loss / 100 + 0.5 * angle_loss / (3.1415926 * 3.1415926)
        loss = (
            coeff_pos * pos_loss / 100
            + coeff_type * type_loss
            + coeff_ang * angle_loss / (3.1415926 * 3.1415926)
        )

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "loss_pos": pos_loss,
            "loss_angle": angle_loss,
            "type_acc": type_acc,
        }

    def forward(
        self,
        batch_data,
        logits,
        node_output,
        angle_output,
        mask_pos,
        mask_aa,
        delta_ang=None,
        ang_score_norm=None,
    ):
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

        # mask_pos =  mask_aa | mask_pos
        if mask_pos.any():
            with torch.no_grad():
                ori_pos = batch_data["pos"]
                bsz, _, _ = ori_pos.size()
                pos_mask = ori_pos == float("inf")
                ori_pos = ori_pos.masked_fill(pos_mask, 0.0)

            ori_pos = ori_pos[mask_pos.squeeze(-1)]
            node_output = node_output[mask_pos.squeeze(-1)]

            pos_loss = (
                self.loss_pos(
                    node_output.to(torch.float32), ori_pos.to(torch.float32)
                ).sum(dim=-1)
                * self.args.pos_loss_coeff  # / bsz
            )
        else:
            pos_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        # mask_angle = mask_aa | mask_pos
        mask_angle = mask_pos
        if mask_angle.any():
            with torch.no_grad():
                ori_angle = batch_data["ang"]
                angle_mask = ori_angle == float("inf")
                ori_angle = ori_angle.to(angle_output.dtype)

            mask_angle = mask_angle & (~angle_mask)
            ori_angle = ori_angle[mask_angle.squeeze(-1)]
            angle_output = angle_output[mask_angle.squeeze(-1)]

            angle_loss = (
                self.loss_angle(
                    angle_output.to(torch.float32), ori_angle.to(torch.float32)
                ).sum(dim=-1)
                * self.args.pos_loss_coeff  # / bsz
            )
        else:
            angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        coeff_pos = (
            0.5
            * (type_loss.detach().item() + angle_loss.detach().item())
            / (
                type_loss.detach().item()
                + pos_loss.detach().item()
                + angle_loss.detach().item()
            )
        )
        coeff_ang = (
            0.5
            * (type_loss.detach().item() + pos_loss.detach().item())
            / (
                type_loss.detach().item()
                + pos_loss.detach().item()
                + angle_loss.detach().item()
            )
        )
        1 - coeff_pos - coeff_ang

        # loss1
        # loss = coeff_pos * pos_loss + coeff_type * type_loss + 0.5 * coeff_ang * angle_loss

        # loss2
        # loss = (
        #     coeff_pos * pos_loss / 100
        #     + coeff_type * type_loss
        #     + coeff_ang * angle_loss / (3.1415926 * 3.1415926)
        # )

        # loss3
        # loss = type_loss + pos_loss / 100 + angle_loss / (3.1415926 * 3.1415926)

        # loss4
        loss = type_loss + pos_loss / 10 + angle_loss / (3.1415926)

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "loss_pos": pos_loss,
            "loss_angle": angle_loss,
            "type_acc": type_acc,
        }

    def forward_v3(
        self,
        batch_data,
        logits,
        node_output,
        angle_output,
        mask_pos,
        mask_aa,
        angle,
        pos,
        padding_mask,
    ):
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

        mask_pos = mask_aa | mask_pos
        if mask_pos.any():
            with torch.no_grad():
                ori_pos = batch_data["pos"]
                bsz, _, _ = ori_pos.size()
                pos_mask = ori_pos == float("inf")
                # pos_mask = pos_mask | padding_mask.bool()
                ori_pos = ori_pos.masked_fill(pos_mask, 0.0)

            pos = pos.masked_fill(pos_mask, 0.0)
            pos_pred = pos - node_output
            pos_pred = pos_pred.masked_fill(pos_mask, 0.0)

            delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
            ori_dist = delta_pos0.norm(dim=-1)

            delta_pos = pos_pred.unsqueeze(1) - pos_pred.unsqueeze(2)
            dist = delta_pos.norm(dim=-1)

            dist_mask = ~(
                padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
            )

            ori_dist = ori_dist[dist_mask]
            dist = dist[dist_mask]

            ori_pos = ori_pos[mask_pos.squeeze(-1)]
            node_output = node_output[mask_pos.squeeze(-1)]
            pos = pos[mask_pos.squeeze(-1)]

            pos_loss = self.loss_pos(
                node_output.to(torch.float32), (pos - ori_pos).to(torch.float32)
            )
            dist_loss = self.loss_dist(
                ori_dist.to(torch.float32), dist.to(torch.float32)
            )
        else:
            pos_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
            dist_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        # mask_angle = mask_aa | mask_pos
        mask_angle = mask_pos
        if mask_angle.any():
            with torch.no_grad():
                ori_angle = batch_data["ang"]
                angle_mask = ori_angle == float("inf")
                ori_angle = ori_angle.to(angle_output.dtype)

            mask_angle = mask_angle & (~angle_mask)
            ori_angle = ori_angle[mask_angle.squeeze(-1)]
            angle = angle[mask_angle.squeeze(-1)]
            angle_output = angle_output[mask_angle.squeeze(-1)]

            angle_loss = (
                self.loss_angle(
                    angle_output.to(torch.float32),
                    (angle - ori_angle).to(torch.float32),
                ).sum(dim=-1)
                * self.args.pos_loss_coeff  # / bsz
            )
        else:
            angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        loss = type_loss + angle_loss / (3.1415926) + dist_loss / 10 + pos_loss / 10

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "loss_pos": pos_loss,
            "loss_dist": dist_loss,
            "loss_angle": angle_loss,
            "type_acc": type_acc,
        }


class ProteinMAEDistCriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        self.loss_pos = nn.MSELoss(reduction="mean")
        self.loss_angle = nn.MSELoss(reduction="mean")
        self.loss_dist = nn.MSELoss(reduction="mean")
        self.args = args
        self.num_aa_type = args.num_residues
        self.mode = args.mode

    def forward(
        self,
        batch_data,
        logits,
        pair_output,
        pair_mask_aa_0,
        angle_output,
        mask_pos,
        mask_aa,
        ang_score,
        ang_score_norm,
        padding_mask,
        backbone=None,
    ):
        if mask_aa.any():
            with torch.no_grad():
                aa_seq = batch_data["x"]
                # paired_seq = aa_seq.unsqueeze(-1) * self.num_aa_type + aa_seq.unsqueeze(
                #     -2
                # )
                # pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
                # pair_mask_aa = pair_mask_aa & pair_mask_aa_0.bool()

                # # logits [mask_L, vocab^2]
                # paired_seq = paired_seq[pair_mask_aa.squeeze(-1).bool()]
                aa_seq = aa_seq[mask_aa.squeeze(-1).bool()]

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
                (logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq)
                .to(torch.float32)
                .mean()
            )

            # type_loss = self.loss_type(
            #     pair_output.view(-1, pair_output.size(-1)).to(torch.float32),
            #     paired_seq.view(-1),
            # )

            # type_acc = (
            #     (
            #         pair_output.view(-1, pair_output.size(-1)).argmax(dim=-1)
            #         == paired_seq
            #     )
            #     .to(torch.float32)
            #     .mean()
            # )

        else:
            type_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
            type_acc = 0.0

        with torch.no_grad():
            ori_pos = batch_data["pos"][:, :, :3, :]
            batch_data["pos_mask"].bool()

            ori_pos = ori_pos.mean(dim=2, keepdim=False)
            delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
            ori_dist = delta_pos0.norm(dim=-1)

        # pred_pos = backbone.mean(dim=2, keepdim=False)

        dist_mask = ~(
            padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
        )
        dist_filter = (ori_dist > 0.2) & (ori_dist < 20.0)
        dist_mask = dist_mask & dist_filter

        ori_dist = ori_dist[dist_mask]
        dist = pair_output[dist_mask]
        # dist = pred_pos.unsqueeze(1) - pred_pos.unsqueeze(2)
        # dist = dist.norm(dim=-1)
        # dist = dist[dist_mask]

        dist_loss = self.loss_dist(ori_dist.to(torch.float32), dist.to(torch.float32))

        # # mask_angle = mask_pos.squeeze(-1)
        # # if mask_angle.any():
        with torch.no_grad():
            ori_angle = batch_data["ang"][:, :, :3]
            angle_mask = batch_data["ang_mask"][:, :, :3].bool()
        # mask_angle = mask_angle & angle_mask
        mask_angle = angle_mask & (~padding_mask.bool().unsqueeze(-1))

        if self.mode == "score":
            ang_score = ang_score[:, :, :3]
            ang_score = ang_score[mask_angle.squeeze(-1)]
            angle_output = angle_output.to(torch.float32) * torch.sqrt(
                ang_score_norm.to(torch.float32)
            )
            angle_output = angle_output[mask_angle.squeeze(-1)]
            angle_loss = ((ang_score - angle_output) ** 2).mean()
        elif self.mode == "x0":
            angle_output = angle_output[mask_angle.squeeze(-1)]
            ori_angle = ori_angle[mask_angle.squeeze(-1)]
            angle_loss = self.loss_angle(
                angle_output.to(torch.float32), ori_angle.to(torch.float32)
            )
        # # else:
        # angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        loss = type_loss + angle_loss / torch.pi + dist_loss

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "loss_dist": dist_loss,
            "loss_angle": angle_loss,
            "type_acc": type_acc,
        }


class ProteinMAEDistPDECriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        self.loss_pos = nn.MSELoss(reduction="mean")
        self.loss_angle = nn.MSELoss(reduction="mean")
        self.loss_dist = nn.L1Loss(reduction="mean")
        self.args = args
        self.lamb_pde = args.lamb_pde
        self.mode = args.mode
        self.num_aa_type = args.num_residues

    # add the PDE loss for angle diffusion
    def forward(
        self,
        batch_data,
        logits,
        pair_output,
        angle_output,
        mask_pos,
        mask_aa,
        ang_score,
        ang_score_norm,
        q_output,
        q_output_mtq,
        q_output_ptq,
        q_score,
        q_score_norm,
        padding_mask,
        pair_mask_aa,
        time_pos,
        q_point,
        nabla_phi_term,
        laplace_phi_term,
        hp,
        hm,
    ):
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
                (logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq)
                .to(torch.float32)
                .mean()
            )
        else:
            type_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
            type_acc = 0.0

        # # if not mask_pos.any():
        # with torch.no_grad():
        #     ori_pos = batch_data["pos"]
        #     bsz, _, _ = ori_pos.size()
        #     pos_mask = ori_pos == float("inf")
        #     # pos_mask = pos_mask | padding_mask.bool()
        #     ori_pos = ori_pos.masked_fill(pos_mask, 0.0)

        # delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
        # ori_dist = delta_pos0.norm(dim=-1)

        # dist_mask = ~(
        #     padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
        # )
        # dist_filter = ori_dist < 2.0
        # dist_mask = dist_mask & dist_filter

        # ori_dist = ori_dist[dist_mask]
        # dist = pair_output[dist_mask].squeeze(-1)

        # dist_loss = self.loss_dist(ori_dist.to(torch.float32), dist.to(torch.float32))
        # # else:
        #     dist_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        mask_angle = mask_pos.squeeze(-1)
        if mask_angle.any():
            with torch.no_grad():
                ori_angle = batch_data["ang"][:, :, :3]
                angle_mask = batch_data["ang_mask"][:, :, :3].bool()
            mask_angle = mask_angle & angle_mask

            if self.mode == "score":
                ang_score = ang_score[:, :, :3]
                ang_score = ang_score[mask_angle.squeeze(-1)]
                angle_output = angle_output.to(torch.float32) * torch.sqrt(
                    ang_score_norm.to(torch.float32)
                )
                angle_output = angle_output[mask_angle.squeeze(-1)]
                angle_loss = ((ang_score - angle_output) ** 2).mean()
            elif self.mode == "x0":
                angle_output = angle_output[mask_angle.squeeze(-1)]
                ori_angle = ori_angle[mask_angle.squeeze(-1)]
                angle_loss = self.loss_angle(
                    angle_output.to(torch.float32), ori_angle.to(torch.float32)
                )
        else:
            angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        # (q, t, mixtureGaussian, q_output, q_output_mtq, q_output_ptq, padding_mask)
        q_pde_loss = compute_PDEloss(
            q_output,
            time_pos / self.args.t_timesteps,
            q_point[:, :, :3],
            nabla_phi_term[:, :, :3],
            laplace_phi_term,
            q_output_mtq,
            q_output_ptq,
            padding_mask,
            hp,
            hm,
        )

        # loss = type_loss + angle_loss / 100 + self.lamb_pde * q_pde_loss + dist_loss
        loss = type_loss + angle_loss + self.lamb_pde * q_pde_loss  # + 10 * dist_loss

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            # "loss_dist": dist_loss,
            "loss_angle": angle_loss,
            "q_pde_loss": q_pde_loss,
            "type_acc": type_acc,
        }

    # add the PDE loss for angle diffusion
    # pair loss
    def forward_v1(
        self,
        batch_data,
        logits,
        pair_output,
        angle_output,
        mask_pos,
        mask_aa,
        ang_score,
        ang_score_norm,
        q_output,
        q_output_mtq,
        q_output_ptq,
        q_score,
        q_score_norm,
        padding_mask,
        pair_mask_aa_0,
        time_pos,
        q_point,
        nabla_phi_term,
        laplace_phi_term,
        hp,
        hm,
    ):
        if mask_aa.any():
            with torch.no_grad():
                aa_seq = batch_data["x"]
                paired_seq = aa_seq.unsqueeze(-1) * self.num_aa_type + aa_seq.unsqueeze(
                    -2
                )
                pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
                pair_mask_aa = pair_mask_aa & pair_mask_aa_0.bool()

                # logits [mask_L, vocab^2]
                paired_seq = paired_seq[pair_mask_aa.squeeze(-1).bool()]
                aa_seq = aa_seq[mask_aa.squeeze(-1).bool()]

            logits = logits[:, :, :][mask_aa.squeeze(-1).bool()]

            # type_loss = (
            #     self.loss_type(
            #         logits.view(-1, logits.size(-1)).to(torch.float32),
            #         aa_seq.view(-1),
            #     )
            #     * self.args.atom_loss_coeff
            # )
            # # compute type accuracy
            # type_acc = (
            #     (logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq)
            #     .to(torch.float32)
            #     .mean()
            # )

            type_loss = self.loss_type(
                pair_output.view(-1, pair_output.size(-1)).to(torch.float32),
                paired_seq.view(-1),
            )

            type_acc = (
                (
                    pair_output.view(-1, pair_output.size(-1)).argmax(dim=-1)
                    == paired_seq
                )
                .to(torch.float32)
                .mean()
            )

        else:
            type_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
            type_acc = 0.0

        # # if not mask_pos.any():
        # with torch.no_grad():
        #     ori_pos = batch_data["pos"]
        #     bsz, _, _ = ori_pos.size()
        #     pos_mask = ori_pos == float("inf")
        #     # pos_mask = pos_mask | padding_mask.bool()
        #     ori_pos = ori_pos.masked_fill(pos_mask, 0.0)

        # delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
        # ori_dist = delta_pos0.norm(dim=-1)

        # dist_mask = ~(
        #     padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
        # )
        # dist_filter = ori_dist < 2.0
        # dist_mask = dist_mask & dist_filter

        # ori_dist = ori_dist[dist_mask]
        # dist = pair_output[dist_mask].squeeze(-1)

        # dist_loss = self.loss_dist(ori_dist.to(torch.float32), dist.to(torch.float32))
        # # else:
        #     dist_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        mask_angle = mask_pos.squeeze(-1)
        if mask_angle.any():
            with torch.no_grad():
                ori_angle = batch_data["ang"][:, :, :3]
                angle_mask = batch_data["ang_mask"][:, :, :3].bool()
            mask_angle = mask_angle & angle_mask

            if self.mode == "score":
                ang_score = ang_score[:, :, :3]
                ang_score = ang_score[mask_angle.squeeze(-1)]
                angle_output = angle_output.to(torch.float32) * torch.sqrt(
                    ang_score_norm.to(torch.float32)
                )
                angle_output = angle_output[mask_angle.squeeze(-1)]
                angle_loss = ((ang_score - angle_output) ** 2).mean()
            elif self.mode == "x0":
                angle_output = angle_output[mask_angle.squeeze(-1)]
                ori_angle = ori_angle[mask_angle.squeeze(-1)]
                angle_loss = self.loss_angle(
                    angle_output.to(torch.float32), ori_angle.to(torch.float32)
                )
        else:
            angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        # (q, t, mixtureGaussian, q_output, q_output_mtq, q_output_ptq, padding_mask)
        q_pde_loss = compute_PDEloss(
            q_output,
            time_pos / self.args.t_timesteps,
            q_point[:, :, :3],
            nabla_phi_term[:, :, :3],
            laplace_phi_term,
            q_output_mtq,
            q_output_ptq,
            padding_mask,
            hp,
            hm,
        )

        # loss = type_loss + angle_loss / 100 + self.lamb_pde * q_pde_loss + dist_loss
        loss = type_loss + angle_loss + self.lamb_pde * q_pde_loss  # + 10 * dist_loss

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            # "loss_dist": dist_loss,
            "loss_angle": angle_loss,
            "q_pde_loss": q_pde_loss,
            "type_acc": type_acc,
        }
