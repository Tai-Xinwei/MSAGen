# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

from sfm.data.prot_data.internal_dataset import InternalToCRAB
from sfm.logging import logger


class CaDistogramLoss(nn.Module):
    def __init__(self, args, reduction="mean"):
        super().__init__()
        self.args = args
        self.reduction = reduction
        self.distogram_min_bin = args.distogram_min_bin  # 2.3125
        self.distogram_max_bin = args.distogram_max_bin  # 21.6875
        self.distogram_n_bins = args.distogram_n_bins  # 64
        self.linear = nn.Linear(2 * args.embedding_dim, args.distogram_n_bins)
        self.register_buffer(
            "boundaries",
            torch.linspace(
                self.distogram_min_bin,
                self.distogram_max_bin,
                self.distogram_n_bins - 1,
            )
            ** 2,
            persistent=False,
        )

    def forward(self, batched_data: dict):
        # input logits: [B, L, L, C]
        input = batched_data["output"]["logits"]  # [B, L, C]
        # get the [B, L, L, C] tensor logits, if OOM, use summation.
        logits = torch.cat(
            [input.unsqueeze(-2), input.unsqueeze(-3)], dim=-1
        )  # [B, L, L, 2*C]
        logits = self.linear(logits)  # [B, L, L, n_bins]
        logits = logits + logits.transpose(-2, -3)
        ca_coords = batched_data["crab"]["A"][:, 1, :]  # [B, L, 3]
        padding_mask = batched_data["crab"]["padding_mask"]  # [B, L]
        dists = torch.sum(
            (ca_coords.unsqueeze(-2) - ca_coords.unsqueeze(-3)) ** 2, dim=-1
        )  # [B, L, L]
        true_bins = torch.bucketize(dists, self.boundaries)  # [B, L, L]
        dists_mask = ~(
            padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)
        )  # [B, L, L]
        loss = F.cross_entropy(
            logits.view(-1, self.distogram_n_bins), true_bins.view(-1), reduction="none"
        ).view(
            true_bins.shape
        )  # [B, L, L]
        # how many valid distances are there, avoid division by zero, but should not happen
        denom = 1e-6 + torch.sum(dists_mask, dim=(-1, -2))  # [B, ]
        # FP16-friendly sum. Equivalent to:
        # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) / (eps + torch.sum(square_mask, dim=(-1, -2))))
        loss = loss * dists_mask  # [B, L, L]
        loss = torch.sum(loss, dim=-1)  # [B, L]
        loss /= denom[..., None]  # [B, L]
        loss = torch.sum(loss, dim=-1)  # [B, ]
        # return loss that averages over the B
        return torch.mean(loss)


class RebuiltCaDistogramLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.distogram_min_bin = args.distogram_min_bin  # 2.3125
        self.distogram_min_bin = 2.3125
        # self.distogram_max_bin = args.distogram_max_bin  # 21.6875
        self.distogram_max_bin = 21.6875
        # self.distogram_n_bins = args.distogram_n_bins  # 64
        self.distogram_n_bins = 64
        # self.linear = nn.Linear(2 * args.embedding_dim, args.distogram_n_bins)
        # self.register_buffer(
        #     "boundaries",
        #     torch.linspace(
        #         self.distogram_min_bin,
        #         self.distogram_max_bin,
        #         self.distogram_n_bins - 1,
        #     )
        #     ** 2,
        #     persistent=False,
        # )
        self.internal2crab = InternalToCRAB(eps=args.eps)

    def logits2internal(self, batched_data: dict) -> dict:
        mask_str = batched_data["mask"]["mask_str"]  # [B, L]

        unit_circles_ba = batched_data["output"]["ba_logits"]  # [B, L, 6]
        unit_circles_da = batched_data["output"]["da_logits"]  # [B, L, 6]
        bl_logits = batched_data["output"]["bl_logits"]  # [B, L, 3]

        norm_ba_C_N_CA = torch.norm(unit_circles_ba[:, :, :2], dim=-1) + self.args.eps
        norm_ba_N_CA_C = torch.norm(unit_circles_ba[:, :, 2:4], dim=-1) + self.args.eps
        norm_ba_CA_C_N = torch.norm(unit_circles_ba[:, :, 4:], dim=-1) + self.args.eps

        angle_ba_C_N_CA = 0.5 * (
            torch.acos(unit_circles_ba[:, :, 0] / norm_ba_C_N_CA)
            + (torch.pi - torch.asin(unit_circles_ba[:, :, 1] / norm_ba_C_N_CA))
        )
        angle_ba_N_CA_C = 0.5 * (
            torch.acos(unit_circles_ba[:, :, 2] / norm_ba_N_CA_C)
            + (torch.pi - torch.asin(unit_circles_ba[:, :, 3] / norm_ba_N_CA_C))
        )
        angle_ba_CA_C_N = 0.5 * (
            torch.acos(unit_circles_ba[:, :, 4] / norm_ba_CA_C_N)
            + (torch.pi - torch.asin(unit_circles_ba[:, :, 5] / norm_ba_CA_C_N))
        )

        norm_da_CA_C_N_CA = (
            torch.norm(unit_circles_da[:, :, :2], dim=-1) + self.args.eps
        )
        norm_da_C_N_CA_C = (
            torch.norm(unit_circles_da[:, :, 2:4], dim=-1) + self.args.eps
        )
        norm_da_N_CA_C_N = torch.norm(unit_circles_da[:, :, 4:], dim=-1) + self.args.eps

        angle_da_CA_C_N_CA = 0.5 * (
            torch.acos(unit_circles_da[:, :, 0] / norm_da_CA_C_N_CA)
            + (torch.pi - torch.asin(unit_circles_da[:, :, 1] / norm_da_CA_C_N_CA))
        )
        angle_da_C_N_CA_C = 0.5 * (
            torch.acos(unit_circles_da[:, :, 2] / norm_da_C_N_CA_C)
            + (torch.pi - torch.asin(unit_circles_da[:, :, 3] / norm_da_C_N_CA_C))
        )
        angle_da_N_CA_C_N = 0.5 * (
            torch.acos(unit_circles_da[:, :, 4] / norm_da_N_CA_C_N)
            + (torch.pi - torch.asin(unit_circles_da[:, :, 5] / norm_da_N_CA_C_N))
        )

        bl_N_CA = torch.where(
            mask_str, bl_logits[:, :, 0], batched_data["internal"]["bl_N_CA"]
        )
        bl_CA_C = torch.where(
            mask_str, bl_logits[:, :, 1], batched_data["internal"]["bl_CA_C"]
        )
        bl_C_N = torch.where(
            mask_str, bl_logits[:, :, 2], batched_data["internal"]["bl_C_N"]
        )

        angle_ba_C_N_CA = torch.where(
            mask_str, angle_ba_C_N_CA, batched_data["internal"]["ba_C_N_CA"]
        )
        angle_ba_N_CA_C = torch.where(
            mask_str, angle_ba_N_CA_C, batched_data["internal"]["ba_N_CA_C"]
        )
        angle_ba_CA_C_N = torch.where(
            mask_str, angle_ba_CA_C_N, batched_data["internal"]["ba_CA_C_N"]
        )

        angle_da_CA_C_N_CA = torch.where(
            mask_str, angle_da_CA_C_N_CA, batched_data["internal"]["da_CA_C_N_CA"]
        )
        angle_da_C_N_CA_C = torch.where(
            mask_str, angle_da_C_N_CA_C, batched_data["internal"]["da_C_N_CA_C"]
        )
        angle_da_N_CA_C_N = torch.where(
            mask_str, angle_da_N_CA_C_N, batched_data["internal"]["da_N_CA_C_N"]
        )

        batched_data["rebuilt"] = {
            "bl_N_CA": bl_N_CA,
            "bl_CA_C": bl_CA_C,
            "bl_C_N": bl_C_N,
            "ba_C_N_CA": angle_ba_C_N_CA,
            "ba_N_CA_C": angle_ba_N_CA_C,
            "ba_CA_C_N": angle_ba_CA_C_N,
            "da_CA_C_N_CA": angle_da_CA_C_N_CA,
            "da_C_N_CA_C": angle_da_C_N_CA_C,
            "da_N_CA_C_N": angle_da_N_CA_C_N,
        }

        return batched_data

    def forward(self, batched_data: dict):
        padding_mask = batched_data["crab"]["padding_mask"]  # [B, L]
        ca_coords = batched_data["crab"]["A"][:, :, 1, :]  # [B, L, 3]
        ca_coords = ca_coords.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        batched_data = self.logits2internal(batched_data)
        batched_data_rebuilt = self.internal2crab(batched_data)

        ca_coords_rebuilt = batched_data_rebuilt["crab_rebuilt"]["A"][
            :, :, 1, :
        ]  # [B, L, 3]
        ca_coords_rebuilt = ca_coords_rebuilt.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        dists = torch.norm(
            ca_coords.unsqueeze(-2) - ca_coords.unsqueeze(-3), dim=-1
        )  # [B, L, L]
        dists_rebuilt = torch.norm(
            ca_coords_rebuilt.unsqueeze(-2) - ca_coords_rebuilt.unsqueeze(-3), dim=-1
        )  # [B, L, L]

        padd_mask = ~(
            padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(-2)
        )  # [B, L, L]
        dist_mask = (dists > self.distogram_min_bin) & (dists < self.distogram_max_bin)

        # simple FAPE-like loss
        # distance = torch.norm(ca_coords - ca_coords_rebuilt, dim=-1)
        # distance = torch.clamp(distance, max=10.0) * 0.1
        # distance = distance * ~padding_mask
        # disto_loss = distance.mean()

        disto_loss = F.mse_loss(
            ca_coords_rebuilt[~padding_mask], ca_coords[~padding_mask], reduction="mean"
        )

        disto_loss = (
            F.mse_loss(
                dists[padd_mask & dist_mask],
                dists_rebuilt[padd_mask & dist_mask],
                reduction="mean",
            )
            * 0.5
        )  # for symmetry
        return {"disto_loss": disto_loss}


class InternalSeqMAELoss(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, batched_data: dict):
        seq = batched_data["crab"]["R"]  # [B, L]
        mask_seq = batched_data["mask"]["mask_seq"]  # [B, L]

        # if mask_seq.any():
        seq_label = seq[mask_seq.squeeze(-1)]  # [L, B, D] vs [B, L]
        seq_logits = batched_data["output"]["seq_logits"][mask_seq.squeeze(-1)]
        seq_type_loss = F.cross_entropy(seq_logits, seq_label, reduction="mean")
        with torch.no_grad():
            seq_type_acc = (seq_logits.argmax(dim=-1) == seq_label).float().mean()
        # else:
        #     seq_type_loss = torch.tensor(
        #         [0.0], device=logits.device, requires_grad=True
        #     )
        #     seq_type_acc = 0.0

        return {"seq_type_loss": seq_type_loss, "seq_type_acc": seq_type_acc}


class InternalStruMAELoss(nn.Module):
    def __init__(self, args):
        super().__init__()

    def angle2unit_circle(self, angles: torch.Tensor) -> torch.Tensor:
        # angles: [B, L] --> [B, L, 2]
        angles = self.inf2zero(angles)
        return torch.cat(
            [torch.cos(angles).unsqueeze(-1), torch.sin(angles).unsqueeze(-1)], dim=-1
        )

    def inf2zero(self, x: torch.Tensor) -> torch.Tensor:
        return x.masked_fill(torch.isinf(x), 0.0)

    def forward(self, batched_data: dict):
        mask_str = batched_data["mask"]["mask_str"]  # [B, L]
        # some internal coordinates have N_res -1 valid values.
        # TODO: add these feature in dataset?

        # if mask_str.any():
        bl_label = torch.cat(
            [
                batched_data["internal"]["bl_N_CA"].unsqueeze(-1),
                batched_data["internal"]["bl_CA_C"].unsqueeze(-1),
                batched_data["internal"]["bl_C_N"].unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, L, 3]
        bl_label = bl_label[mask_str.squeeze(-1)]
        bl_logits = batched_data["output"]["bl_logits"][mask_str.squeeze(-1)]
        bl_loss = F.mse_loss(bl_logits, bl_label, reduction="mean")

        ba_label = torch.cat(
            [
                self.angle2unit_circle(batched_data["internal"]["ba_C_N_CA"]),
                self.angle2unit_circle(batched_data["internal"]["ba_N_CA_C"]),
                self.angle2unit_circle(batched_data["internal"]["ba_CA_C_N"]),
            ],
            dim=-1,
        )  # [B, L, 6]
        ba_label = ba_label[mask_str.squeeze(-1)]
        ba_logits = batched_data["output"]["ba_logits"][mask_str.squeeze(-1)]
        ba_norm = torch.norm(ba_logits.view(-1, 3, 2), dim=-1)
        ba_logits = ba_logits.view(-1, 3, 2) / ba_norm.unsqueeze(-1)
        ba_loss = F.mse_loss(ba_logits.view(-1, 6), ba_label, reduction="mean")
        ba_norm_loss = torch.mean(torch.abs(ba_norm - 1))

        da_label = torch.cat(
            [
                self.angle2unit_circle(batched_data["internal"]["da_CA_C_N_CA"]),
                self.angle2unit_circle(batched_data["internal"]["da_C_N_CA_C"]),
                self.angle2unit_circle(batched_data["internal"]["da_N_CA_C_N"]),
            ],
            dim=-1,
        )  # [B, L, 6]
        da_label = da_label[mask_str.squeeze(-1)]
        da_logits = batched_data["output"]["da_logits"][mask_str.squeeze(-1)]
        da_norm = torch.norm(da_logits.view(-1, 3, 2), dim=-1)
        da_logits = da_logits.view(-1, 3, 2) / da_norm.unsqueeze(-1)
        da_loss = F.mse_loss(da_logits.view(-1, 6), da_label, reduction="mean")
        da_norm_loss = torch.mean(torch.abs(da_norm - 1))

        # else:
        #     bl_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
        #     ba_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
        #     ba_norm_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
        #     da_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)
        #     da_norm_loss = torch.tensor([0.0], device=logits.device, requires_grad=True)

        return {
            "bl_loss": bl_loss,
            "ba_loss": ba_loss,
            "ba_norm_loss": ba_norm_loss,
            "da_loss": da_loss,
            "da_norm_loss": da_norm_loss,
        }
