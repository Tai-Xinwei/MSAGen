# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

# from sklearn.metrics import roc_auc_score
import torch.nn as nn

from sfm.logging import logger
from sfm.models.tox.modules.physics import (
    VESDE,
    compute_pde_control_loss,
    compute_PDE_q_loss,
    compute_terminal_ism_loss,
)


class ProteinMAEDistCriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        self.loss_pos = nn.MSELoss(reduction="mean")
        self.loss_angle = nn.MSELoss(reduction="mean")
        self.loss_dist = nn.MSELoss(reduction="mean")
        self.args = args
        self.num_aa_type = args.num_residues
        self.diffmode = args.diffmode

    def forward(
        self,
        batch_data,
        output_dict,
    ):
        logits = output_dict["x"]
        pair_output = output_dict["x_pair"]
        angle_output = output_dict["angle_output"]
        ang_epsilon = output_dict["ang_epsilon"]  # add agn_epsilon target
        mask_pos = output_dict["mask_pos"]
        mask_aa = output_dict["mask_aa"]
        padding_mask = output_dict["padding_mask"]
        output_dict["backbone"]

        if mask_aa.any():
            with torch.no_grad():
                aa_seq = batch_data["x"]
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
        else:
            type_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
            type_acc = 0.0

        with torch.no_grad():
            ori_pos = batch_data["pos"][:, :, :3, :]
            batch_data["pos_mask"].bool()

            ori_pos = ori_pos.mean(dim=2, keepdim=False)
            delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
            ori_dist = delta_pos0.norm(dim=-1)

        pair_mask_aa = self._set_dist_mask(batch_data["x"])

        dist_mask = ~(
            padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
        )
        dist_filter = (ori_dist > 0.2) & (ori_dist < 20.0)
        dist_mask = dist_mask & dist_filter & pair_mask_aa

        ori_dist = ori_dist[dist_mask]
        dist = pair_output[dist_mask]
        dist_loss = self.loss_dist(ori_dist.to(torch.float32), dist.to(torch.float32))

        mask_angle = mask_pos.squeeze(-1)
        if mask_angle.any():
            with torch.no_grad():
                ori_angle = batch_data["ang"][:, :, :3]
                angle_mask = batch_data["ang_mask"][:, :, :3].bool()
            mask_angle = mask_angle & angle_mask
            mask_angle = angle_mask & (~padding_mask.bool().unsqueeze(-1))

            if self.diffmode == "epsilon":
                ang_epsilon = ang_epsilon[:, :, :3]
                ang_epsilon = ang_epsilon[mask_angle]
                epsilon_pred = angle_output[mask_angle]
                angle_loss = ((ang_epsilon - epsilon_pred) ** 2).mean()
            elif self.diffmode == "x0":
                angle_output = angle_output[mask_angle]
                ori_angle = ori_angle[mask_angle]
                angle_loss = self.loss_angle(
                    angle_output.to(torch.float32), ori_angle.to(torch.float32)
                )
            else:
                raise ValueError(f"diffmode {self.diffmode} not supported")
        else:
            angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        loss = type_loss + 10 * angle_loss + dist_loss

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "loss_dist": dist_loss,
            "loss_angle": angle_loss,
            "type_acc": type_acc,
        }

    @torch.compile
    def _set_dist_mask(self, residue_seq):
        """
        compute the mask for distance loss in the complete doc mode
        """
        B, L = residue_seq.shape
        pair_mask_aa = torch.zeros(
            (B, L, L), device=residue_seq.device, dtype=torch.int8
        )
        for i in range(B):
            mask_start_idx = (residue_seq[i] == 0).nonzero(as_tuple=True)[0]
            mask_end_idx = (residue_seq[i] == 2).nonzero(as_tuple=True)[0]

            for j in range(len(mask_end_idx)):
                s_idx = mask_start_idx[j]
                e_idx = mask_end_idx[j]
                pair_mask_aa[i, s_idx:e_idx, s_idx:e_idx] = 1.0

            if len(mask_start_idx) > len(mask_end_idx):
                s_idx = mask_start_idx[-1]
                pair_mask_aa[i, s_idx:, s_idx:] = 1.0

        return pair_mask_aa.bool()


class ProteinMAEDistPDECriterions(nn.Module):
    def __init__(self, args, reduction="mean") -> None:
        super().__init__()
        self.loss_type = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.05)
        self.loss_pos = nn.MSELoss(reduction="mean")
        self.loss_angle = nn.MSELoss(reduction="mean")
        self.loss_dist = nn.L1Loss(reduction="mean")
        self.args = args
        self.lamb_ism = args.lamb_ism
        self.lamb_pde_q = args.lamb_pde_q
        self.lamb_pde_control = args.lamb_pde_control
        self.diffmode = args.diffmode
        self.vesde = VESDE()

    # add the PDE loss for angle diffusion
    def forward(
        self,
        batch_data,
        output_dict,
    ):
        # whether to use the PDE loss
        if_ism_loss = False if self.lamb_ism == 0 else True
        if_pde_q_loss = False if self.lamb_pde_q == 0 else True
        if_pde_control_loss = False if self.lamb_pde_control == 0 else True

        # need mask when computing loss
        unified_angle_mask = output_dict["unified_angle_mask"]

        """----------------------type loss----------------------"""
        mask_aa = output_dict["mask_aa"]
        logits = output_dict["x"]
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

        """----------------------dist loss----------------------"""
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

        # self.loss_dist(ori_dist.to(torch.float32), dist.to(torch.float32))
        # else:
        #     dist_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        """----------------------angle loss----------------------"""
        if unified_angle_mask.any():
            angle_output = output_dict["angle_output"]
            ang_epsilon = output_dict["ang_epsilon"]

            if self.diffmode == "epsilon":
                # retrieve the noise
                angle_epsilon_masked = ang_epsilon[unified_angle_mask].to(torch.float32)
                epsilon_pre_masked = angle_output[unified_angle_mask].to(torch.float32)
                angle_loss = ((epsilon_pre_masked - angle_epsilon_masked) ** 2).mean()
            elif self.diffmode == "x0":
                ori_angle = batch_data["ang"][:, :, :3]
                angle_output_masked = angle_output[unified_angle_mask]
                ori_angle_masked = ori_angle[unified_angle_mask]
                angle_loss = self.loss_angle(
                    angle_output_masked.to(torch.float32),
                    ori_angle_masked.to(torch.float32),
                )
        else:
            angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        """---------------------- terminal ism loss----------------------"""
        if unified_angle_mask.any() and if_ism_loss:
            sigma_1 = output_dict["sigma_1"]
            dt = output_dict["dt"]
            angle_output1 = output_dict["angle_output1"]
            angle_output2 = output_dict["angle_output2"]

            # nabla_phi and laplace_phi have been masked
            angle_output1 = torch.where(
                unified_angle_mask, angle_output1, torch.zeros_like(angle_output1)
            )
            angle_output2 = torch.where(
                unified_angle_mask, angle_output2, torch.zeros_like(angle_output2)
            )

            terminal_ism_loss = compute_terminal_ism_loss(
                angle_output1,
                angle_output2,
                sigma_t=sigma_1,
                k=dt,
            )
        else:
            terminal_ism_loss = torch.tensor(
                [0.0], device=logits.device, requires_grad=True
            )

        """----------------------pde q loss----------------------"""
        if unified_angle_mask.any() and if_pde_q_loss:
            nabla_phi_term = output_dict["nabla_phi_term"]
            laplace_phi_term = output_dict["laplace_phi_term"]
            hp = output_dict["hp"]
            hm = output_dict["hm"]
            q_output = output_dict["q_output"]
            q_output_ptq = output_dict["q_output_ptq"]
            q_output_mtq = output_dict["q_output_mtq"]
            ang_sigma = output_dict["ang_sigma"]

            # nabla_phi and laplace_phi have been masked
            q_output = torch.where(
                unified_angle_mask, q_output, torch.zeros_like(q_output)
            ).to(torch.float32)
            q_output_mtq = torch.where(
                unified_angle_mask, q_output_mtq, torch.zeros_like(q_output_mtq)
            ).to(torch.float32)
            q_output_ptq = torch.where(
                unified_angle_mask, q_output_ptq, torch.zeros_like(q_output_ptq)
            ).to(torch.float32)

            pde_q_loss = compute_PDE_q_loss(
                self.vesde,
                q_output,
                nabla_phi_term,
                laplace_phi_term,
                q_output_mtq,
                q_output_ptq,
                hp,
                hm,
                ang_sigma,
                is_clip=False,
            )
        else:
            pde_q_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        """----------------------pde control loss----------------------"""
        if unified_angle_mask.any() and if_pde_control_loss:
            angle_output_single_time = output_dict["angle_output_single_time"]
            ang_spsilon_single_time = output_dict["ang_epsilon_single_time"]

            epsilon_pred_single_time_masked = torch.where(
                unified_angle_mask,
                angle_output_single_time,
                torch.zeros_like(angle_output_single_time),
            ).to(torch.float32)
            ang_epsilon_single_time_masked = torch.where(
                unified_angle_mask,
                ang_spsilon_single_time,
                torch.zeros_like(ang_spsilon_single_time),
            ).to(torch.float32)
            pde_control_loss = compute_pde_control_loss(
                epsilon_pred_single_time_masked,
                ang_epsilon_single_time_masked,
            )
        else:
            pde_control_loss = torch.tensor(
                [0.0], device=logits.device, requires_grad=True
            )

        """----------------------total loss----------------------"""
        loss = (
            type_loss
            + angle_loss
            + self.lamb_ism * terminal_ism_loss
            + self.lamb_pde_q * pde_q_loss
            + self.lamb_pde_control * pde_control_loss
            # + 10 * dist_loss
        )

        return loss, {
            "total_loss": loss,
            "loss_type": type_loss,
            "loss_angle": angle_loss,
            "loss_ism": terminal_ism_loss,
            "pde_q_loss": pde_q_loss,
            "pde_control_loss": pde_control_loss,
            # "loss_dist": dist_loss,
            "type_acc": type_acc,
        }

    # # add the PDE loss for angle diffusion
    # # pair loss
    # def forward_v1(
    #     self,
    #     batch_data,
    #     logits,
    #     pair_output,
    #     angle_output,
    #     mask_pos,
    #     mask_aa,
    #     ang_score,
    #     ang_score_norm,
    #     q_output,
    #     q_output_mtq,
    #     q_output_ptq,
    #     q_score,
    #     q_score_norm,
    #     padding_mask,
    #     pair_mask_aa_0,
    #     time_pos,
    #     q_point,
    #     nabla_phi_term,
    #     laplace_phi_term,
    #     hp,
    #     hm,
    # ):
    #     if mask_aa.any():
    #         with torch.no_grad():
    #             aa_seq = batch_data["x"]
    #             paired_seq = aa_seq.unsqueeze(-1) * self.num_aa_type + aa_seq.unsqueeze(
    #                 -2
    #             )
    #             pair_mask_aa = mask_aa.unsqueeze(1).bool() & mask_aa.unsqueeze(2).bool()
    #             pair_mask_aa = pair_mask_aa & pair_mask_aa_0.bool()

    #             # logits [mask_L, vocab^2]
    #             paired_seq = paired_seq[pair_mask_aa.squeeze(-1).bool()]
    #             aa_seq = aa_seq[mask_aa.squeeze(-1).bool()]

    #         logits = logits[:, :, :][mask_aa.squeeze(-1).bool()]

    #         # type_loss = (
    #         #     self.loss_type(
    #         #         logits.view(-1, logits.size(-1)).to(torch.float32),
    #         #         aa_seq.view(-1),
    #         #     )
    #         #     * self.args.atom_loss_coeff
    #         # )
    #         # # compute type accuracy
    #         # type_acc = (
    #         #     (logits.view(-1, logits.size(-1)).argmax(dim=-1) == aa_seq)
    #         #     .to(torch.float32)
    #         #     .mean()
    #         # )

    #         type_loss = self.loss_type(
    #             pair_output.view(-1, pair_output.size(-1)).to(torch.float32),
    #             paired_seq.view(-1),
    #         )

    #         type_acc = (
    #             (
    #                 pair_output.view(-1, pair_output.size(-1)).argmax(dim=-1)
    #                 == paired_seq
    #             )
    #             .to(torch.float32)
    #             .mean()
    #         )

    #     else:
    #         type_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
    #         type_acc = 0.0

    #     # # if not mask_pos.any():
    #     # with torch.no_grad():
    #     #     ori_pos = batch_data["pos"]
    #     #     bsz, _, _ = ori_pos.size()
    #     #     pos_mask = ori_pos == float("inf")
    #     #     # pos_mask = pos_mask | padding_mask.bool()
    #     #     ori_pos = ori_pos.masked_fill(pos_mask, 0.0)

    #     # delta_pos0 = ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
    #     # ori_dist = delta_pos0.norm(dim=-1)

    #     # dist_mask = ~(
    #     #     padding_mask.bool().unsqueeze(1) | padding_mask.bool().unsqueeze(2)
    #     # )
    #     # dist_filter = ori_dist < 2.0
    #     # dist_mask = dist_mask & dist_filter

    #     # ori_dist = ori_dist[dist_mask]
    #     # dist = pair_output[dist_mask].squeeze(-1)

    #     # dist_loss = self.loss_dist(ori_dist.to(torch.float32), dist.to(torch.float32))
    #     # # else:
    #     #     dist_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

    #     mask_angle = mask_pos.squeeze(-1)
    #     if mask_angle.any():
    #         with torch.no_grad():
    #             ori_angle = batch_data["ang"][:, :, :3]
    #             angle_mask = batch_data["ang_mask"][:, :, :3].bool()
    #         mask_angle = mask_angle & angle_mask

    #         if self.mode == "score":
    #             ang_score = ang_score[:, :, :3]
    #             ang_score = ang_score[mask_angle.squeeze(-1)]
    #             angle_output = angle_output.to(torch.float32) * torch.sqrt(
    #                 ang_score_norm.to(torch.float32)
    #             )
    #             angle_output = angle_output[mask_angle.squeeze(-1)]
    #             angle_loss = ((ang_score - angle_output) ** 2).mean()
    #         elif self.mode == "x0":
    #             angle_output = angle_output[mask_angle.squeeze(-1)]
    #             ori_angle = ori_angle[mask_angle.squeeze(-1)]
    #             angle_loss = self.loss_angle(
    #                 angle_output.to(torch.float32), ori_angle.to(torch.float32)
    #             )
    #     else:
    #         angle_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

    #     # (q, t, mixtureGaussian, q_output, q_output_mtq, q_output_ptq, padding_mask)
    #     pde_q_loss = compute_PDE_qloss(
    #         self.vesde,
    #         q_output,
    #         time_pos / self.args.t_timesteps,
    #         q_point[:, :, :3],
    #         nabla_phi_term[:, :, :3],
    #         laplace_phi_term,
    #         q_output_mtq,
    #         q_output_ptq,
    #         padding_mask,
    #         hp,
    #         hm,
    #     )

    #     loss = type_loss + angle_loss / 10 + self.lamb_pde * pde_q_loss

    #     return loss, {
    #         "total_loss": loss,
    #         "loss_type": type_loss,
    #         # "loss_dist": dist_loss,
    #         "loss_angle": angle_loss,
    #         "pde_q_loss": pde_q_loss,
    #         # "pde_control_loss": compute_pde_control_los,
    #         "type_acc": type_acc,
    #     }
