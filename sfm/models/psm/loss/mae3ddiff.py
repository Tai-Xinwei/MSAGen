# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from json import decoder

import torch
import torch.nn as nn

from sfm.logging import logger
from sfm.models.psm.psm_config import (
    DiffusionTrainingLoss,
    ForceLoss,
    PSMConfig,
    StressLoss,
)


class NoiseTolerentL1Loss(nn.Module):
    def __init__(self, noise_tolerance: float = 1.0, reduction: str = "none"):
        super().__init__()
        self.reduction = reduction
        self.noise_tolerance = noise_tolerance

    def forward(self, input, target):
        diff = torch.abs(input - target)
        diff = torch.where(
            diff < self.noise_tolerance,
            diff,
            2 * (torch.sqrt(diff * self.noise_tolerance) - 0.5 * self.noise_tolerance),
        )

        if self.reduction == "mean":
            return torch.mean(diff)
        elif self.reduction == "sum":
            return torch.sum(diff)
        elif self.reduction == "none":
            return diff
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def svd_superimpose(P, Q, mask=None):
    """
    P has shape (B, N, 3)
    Q has shape (B, N, 3)
    """
    B = P.shape[0]
    mask = mask.unsqueeze(-1) if mask is not None else None
    weights = torch.ones_like(mask, dtype=P.dtype)
    if mask is not None:
        P = torch.where(mask, 0.0, P)
        Q = torch.where(mask, 0.0, Q)
        weights = torch.where(mask, 0.0, weights)

    P_centroid = (P * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    Q_centroid = (Q * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # replace nan to 0, weights.sum(dim=1, keepdim=True) could be 0
    P_centroid[torch.isnan(P_centroid)] = 0.0
    Q_centroid[torch.isnan(Q_centroid)] = 0.0

    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    if mask is not None:
        P_centered = torch.where(mask, 0.0, P_centered)
        Q_centered = torch.where(mask, 0.0, Q_centered)

    # Find rotation matrix by Kabsch algorithm
    H = torch.einsum("bni,bnj->bij", weights * P_centered, Q_centered)
    U, S, Vt = torch.linalg.svd(H.float())

    # ensure right-handedness
    d = torch.sign(torch.linalg.det(torch.einsum("bki,bjk->bij", Vt, U)))
    # Trick for torch.vmap
    diag_values = torch.cat(
        [
            torch.ones((B, 1), dtype=P.dtype, device=P.device),
            torch.ones((B, 1), dtype=P.dtype, device=P.device),
            d[:, None] * torch.ones((B, 1), dtype=P.dtype, device=P.device),
        ],
        dim=-1,
    )
    M = torch.eye(3, dtype=P.dtype, device=P.device)[None] * diag_values[:, None]
    R = torch.einsum("bki,bkh,bjh->bij", Vt, M, U)

    T = Q_centroid - torch.einsum("bik,bjk->bji", R, P_centroid)
    return R, T


class DiffMAE3dCriterions(nn.Module):
    def __init__(
        self,
        args: PSMConfig,
        molecule_energy_mean: float = 0.0,
        molecule_energy_std: float = 1.0,
        periodic_energy_mean: float = 0.0,
        periodic_energy_std: float = 1.0,
        molecule_energy_per_atom_mean: float = 0.0,
        molecule_energy_per_atom_std: float = 1.0,
        periodic_energy_per_atom_mean: float = 0.0,
        periodic_energy_per_atom_std: float = 1.0,
        molecule_force_mean: float = 0.0,
        molecule_force_std: float = 1.0,
        periodic_force_mean: float = 0.0,
        periodic_force_std: float = 1.0,
        periodic_stress_mean: float = 0.0,
        periodic_stress_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.args = args

        self.diffusion_mode = args.diffusion_mode
        self.seq_only = args.seq_only

        self.energy_loss = nn.L1Loss(reduction="none")

        if self.args.force_loss_type == ForceLoss.L1:
            self.force_loss = nn.L1Loss(reduction="none")
        elif self.args.force_loss_type == ForceLoss.L2:
            self.force_loss = nn.MSELoss(reduction="none")
        elif self.args.force_loss_type == ForceLoss.MSE:
            self.force_loss = nn.MSELoss(reduction="none")
        elif self.args.force_loss_type == ForceLoss.SmoothL1:
            self.force_loss = nn.SmoothL1Loss(reduction="none")
        elif self.args.force_loss_type == ForceLoss.NoiseTolerentL1:
            self.force_loss = NoiseTolerentL1Loss(noise_tolerance=3.0, reduction="none")
        else:
            raise ValueError(f"Invalid force loss type: {self.args.force_loss_type}")

        if self.args.stress_loss_type == StressLoss.L1:
            self.stress_loss = nn.L1Loss(reduction="none")
        elif self.args.stress_loss_type == StressLoss.L2:
            self.stress_loss = nn.MSELoss(reduction="none")
        elif self.args.stress_loss_type == StressLoss.MSE:
            self.stress_loss = nn.MSELoss(reduction="none")
        elif self.args.stress_loss_type == StressLoss.SmoothL1:
            self.stress_loss = nn.SmoothL1Loss(reduction="none")
        elif self.args.stress_loss_type == StressLoss.NoiseTolerentL1:
            self.stress_loss = NoiseTolerentL1Loss(
                noise_tolerance=3.0, reduction="none"
            )
        else:
            raise ValueError(f"Invalid stress loss type: {self.args.stress_loss_type}")

        if self.args.diffusion_training_loss == DiffusionTrainingLoss.L1:
            self.noise_loss = nn.L1Loss(reduction="none")
        elif self.args.diffusion_training_loss == DiffusionTrainingLoss.MSE:
            self.noise_loss = nn.MSELoss(reduction="none")
        elif self.args.diffusion_training_loss == DiffusionTrainingLoss.SmoothL1:
            self.noise_loss = nn.SmoothL1Loss(reduction="none", beta=2.0)
        elif self.args.diffusion_training_loss == DiffusionTrainingLoss.L2:
            self.noise_loss = nn.MSELoss(reduction="none")
        else:
            raise ValueError(
                f"Invalid diffusion training loss type: {self.args.diffusion_training_loss}"
            )

        self.aa_mlm_loss = nn.CrossEntropyLoss(reduction="mean")
        self.contact_loss = nn.L1Loss(reduction="mean")

        self.molecule_energy_mean = molecule_energy_mean
        self.molecule_energy_std = molecule_energy_std
        self.periodic_energy_mean = periodic_energy_mean
        self.periodic_energy_std = periodic_energy_std
        self.molecule_energy_per_atom_mean = molecule_energy_per_atom_mean
        self.molecule_energy_per_atom_std = molecule_energy_per_atom_std
        self.periodic_energy_per_atom_mean = periodic_energy_per_atom_mean
        self.periodic_energy_per_atom_std = periodic_energy_per_atom_std
        self.molecule_force_mean = molecule_force_mean
        self.molecule_force_std = molecule_force_std
        self.periodic_force_mean = periodic_force_mean
        self.periodic_force_std = periodic_force_std
        self.periodic_stress_mean = periodic_stress_mean
        self.periodic_stress_std = periodic_stress_std

        self.material_force_loss_ratio = args.material_force_loss_ratio
        self.material_energy_loss_ratio = args.material_energy_loss_ratio
        self.molecule_force_loss_ratio = args.molecule_force_loss_ratio
        self.molecule_energy_loss_ratio = args.molecule_energy_loss_ratio

        # if args.AutoGradForce:
        #     # self.material_force_loss_ratio = 5.0
        #     # self.molecule_force_loss_ratio = 10.0
        #     # self.material_energy_loss_ratio = 0.5
        #     # self.molecule_energy_loss_ratio = 1.0

        #     self.material_force_loss_ratio = 2.0
        #     self.molecule_force_loss_ratio = 2.0
        #     self.material_energy_loss_ratio = 1.0
        #     self.molecule_energy_loss_ratio = 1.0

        #     logger.info("overriding force and energy loss ratio in autograd mode:")
        #     logger.info(f"{self.material_force_loss_ratio=}")
        #     logger.info(f"{self.molecule_force_loss_ratio=}")
        #     logger.info(f"{self.material_energy_loss_ratio=}")
        #     logger.info(f"{self.molecule_energy_loss_ratio=}")

        self.hard_dist_loss_raito = args.hard_dist_loss_raito
        self.if_total_energy = args.if_total_energy
        self.all_atom = args.all_atom

        self.epsilon = 1e-5
        self.diffusion_rescale_coeff = args.diffusion_rescale_coeff

    def _reduce_energy_loss(
        self, energy_loss, loss_mask, is_molecule, is_periodic, use_per_atom_energy=True
    ):
        num_samples = torch.sum(loss_mask.long())
        energy_loss = (
            energy_loss.clone()
        )  # energy_loss cloned since it will be resued multiple times
        if num_samples > 0:
            # multiply the loss by std of energy labels
            # note that this works only when using MAE loss
            # for example, with MSE loss, we need to multiply squre of the std
            if use_per_atom_energy:
                mol_loss_scale = self.molecule_energy_per_atom_std
                if (
                    hasattr(self.args, "energy_per_atom_label_scale")
                    and not self.training
                ):
                    # energy_per_atom label were scaled so we scale the loss back when evaluating
                    mol_loss_scale /= self.args.energy_per_atom_label_scale
                energy_loss[is_molecule] = energy_loss[is_molecule] * mol_loss_scale
                energy_loss[is_periodic] = (
                    energy_loss[is_periodic] * self.periodic_energy_per_atom_std
                )
            else:
                energy_loss[is_molecule] = (
                    energy_loss[is_molecule] * self.molecule_energy_std
                )
                energy_loss[is_periodic] = (
                    energy_loss[is_periodic] * self.periodic_energy_std
                )
            energy_loss = torch.mean(energy_loss[loss_mask])
        else:
            energy_loss = torch.tensor(
                0.0, device=energy_loss.device, requires_grad=True
            )
        return energy_loss, num_samples

    def _reduce_force_or_noise_loss(
        self,
        force_or_noise_loss,
        sample_mask,
        token_mask,
        is_molecule,
        is_periodic,
        molecule_loss_factor=1.0,
        periodic_loss_factor=1.0,
        is_protein=None,
    ):
        if len(sample_mask.shape) == (len(token_mask.shape) - 1):
            sample_mask = sample_mask & token_mask.any(dim=-1)
        elif len(sample_mask.shape) == len(token_mask.shape):
            sample_mask = sample_mask & token_mask
        elif (len(sample_mask.shape) - 1) == len(token_mask.shape):
            sample_mask = sample_mask & token_mask.unsqueeze(-1)
        else:
            raise ValueError(
                f"sample_mask and token_mask have incompatible shapes: {sample_mask.shape} and {token_mask.shape}"
            )

        num_samples = torch.sum(sample_mask.long())
        force_or_noise_loss = (
            force_or_noise_loss.clone()
        )  # force_or_noise_loss cloned since it will be resued multiple times
        if num_samples > 0:
            # multiply the loss by std (across all atoms and all 3 coordinates) of force labels
            # note that this works only when using MAE loss
            # for example, with MSE loss, we need to multiply squre of the std
            # for noise loss, the factors should be 1.0
            force_or_noise_loss[is_molecule] = (
                force_or_noise_loss[is_molecule] * molecule_loss_factor
            )
            force_or_noise_loss[is_periodic] = (
                force_or_noise_loss[is_periodic] * periodic_loss_factor
            )

            if self.all_atom:
                force_or_noise_loss = force_or_noise_loss.masked_fill(
                    ~token_mask.unsqueeze(-1).unsqueeze(-1), 0.0
                )
            else:
                force_or_noise_loss = force_or_noise_loss.masked_fill(
                    ~token_mask.unsqueeze(-1), 0.0
                )

            if len(sample_mask.shape) == 1:
                force_or_noise_loss = torch.sum(
                    force_or_noise_loss[sample_mask], dim=[1, 2]
                ) / (3.0 * torch.sum(token_mask[sample_mask], dim=-1))
            elif len(sample_mask.shape) == 2:
                # TODO: need to average over tokens in one sample first then all smaples
                if is_protein is not None:
                    force_or_noise_loss[~is_protein] = (
                        force_or_noise_loss[~is_protein] * 4
                    )

                force_or_noise_loss = torch.sum(
                    force_or_noise_loss[sample_mask], dim=[0, 1]
                ) / (3.0 * torch.sum(token_mask[sample_mask], dim=-1))
            elif len(sample_mask.shape) == 3:
                # TODO: need to average over tokens in one sample first then all smaples
                if is_protein is not None:
                    force_or_noise_loss[~is_protein.unsqueeze(-1)] = (
                        force_or_noise_loss[~is_protein.unsqueeze(-1)] * 4
                    )
                force_or_noise_loss = torch.sum(
                    force_or_noise_loss[sample_mask], dim=[0, 1]
                ) / (3.0 * torch.sum(sample_mask))
            else:
                raise ValueError(
                    f"sample_mask has an unexpected shape: {sample_mask.shape}"
                )

            force_or_noise_loss = force_or_noise_loss.mean()
        else:
            force_or_noise_loss = torch.tensor(
                0.0, device=force_or_noise_loss.device, requires_grad=True
            )
        return force_or_noise_loss, num_samples

    def _reduce_stress_loss(self, stress_loss, sample_mask, stress_std):
        stress_loss = stress_loss.clone()
        num_samples = int(torch.sum(sample_mask.long()))
        if num_samples > 0:
            stress_loss = stress_loss.masked_fill(~sample_mask[:, None, None], 0.0)
            stress_loss = stress_loss.sum() / (9.0 * num_samples) * stress_std
        else:
            stress_loss = torch.tensor(
                0.0, device=stress_loss.device, requires_grad=True
            )
        return stress_loss, num_samples

    def calculate_pos_pred(self, model_output):
        noise_pred = model_output["noise_pred"]
        sqrt_one_minus_alphas_cumprod_t = model_output[
            "sqrt_one_minus_alphas_cumprod_t"
        ]
        sqrt_alphas_cumprod_t = model_output["sqrt_alphas_cumprod_t"]
        pos_pred = (
            model_output["pos"] - sqrt_one_minus_alphas_cumprod_t * noise_pred
        ) / sqrt_alphas_cumprod_t
        return pos_pred

    @torch.no_grad()
    def _alignment_x0(self, model_output, pos_pred, atomic_numbers):
        pos_label = model_output["ori_pos"]
        B = pos_label.shape[0]

        if self.all_atom:
            R, T = svd_superimpose(
                pos_label.view(B, -1, 3).float(),
                pos_pred.view(B, -1, 3).float(),
                model_output["padding_mask"].unsqueeze(-1).repeat(1, 1, 37).view(B, -1)
                | model_output["protein_mask"].any(dim=-1).view(B, -1)
                | atomic_numbers.eq(156).unsqueeze(-1).repeat(1, 1, 37).view(B, -1)
                # | ((~model_output["is_protein"]) & model_output["is_complex"].unsqueeze(-1))
                # | atomic_numbers.eq(2).unsqueeze(-1).repeat(1, 1, 37).view(B, -1),
            )
        else:
            R, T = svd_superimpose(
                pos_label.float(),
                pos_pred.float(),
                model_output["padding_mask"]
                | model_output["protein_mask"].any(dim=-1)
                | atomic_numbers.eq(156)
                # | ((~model_output["is_protein"]) & model_output["is_complex"].unsqueeze(-1))
                # | atomic_numbers.eq(2),
            )
        # | ((~model_output["is_protein"]) & model_output["is_complex"].unsqueeze(-1))
        return R, T

    def dist_loss(self, model_output, R, T, pos_pred, atomic_numbers):
        # calculate aligned pred pos
        # pos_pred = torch.einsum("bij,bkj->bki", R.float(), pos_pred.float()) + T.float()

        # smooth lddt loss
        if self.all_atom:
            pos_label = (
                model_output["ori_pos"][:, :, 1, :].float()
                * self.diffusion_rescale_coeff
            )
            pos_pred = pos_pred[:, :, 1, :] * self.diffusion_rescale_coeff
            B, L = pos_label.shape[:2]

            filter_mask = (
                model_output["padding_mask"]
                | model_output["protein_mask"].all(dim=(-1, -2))
                | atomic_numbers.eq(156)
                | atomic_numbers.eq(2)
            )
        else:
            pos_label = model_output["ori_pos"].float() * self.diffusion_rescale_coeff
            pos_pred = pos_pred * self.diffusion_rescale_coeff
            B, L = pos_label.shape[:2]

            # make is_protein mask contain ligand in complex data
            # is_protein = model_output["is_protein"].any(dim=-1).unsqueeze(-1).repeat(1, L)
            filter_mask = (
                model_output["padding_mask"]
                | model_output["protein_mask"].any(dim=-1)
                | atomic_numbers.eq(156)
                | atomic_numbers.eq(2)
            )

        is_protein = model_output["is_protein"] & (~filter_mask)

        is_complex = model_output["is_complex"]
        is_ligand = (
            is_complex.unsqueeze(-1) & (~model_output["is_protein"]) & (~filter_mask)
        )

        delta_pos_label = (pos_label.unsqueeze(1) - pos_label.unsqueeze(2)).norm(dim=-1)
        delta_pos_pred = (pos_pred.unsqueeze(1) - pos_pred.unsqueeze(2)).norm(dim=-1)
        pair_protein_mask = is_protein.unsqueeze(1) & is_protein.unsqueeze(2)
        # pair_ligand_mask = is_ligand.unsqueeze(1) & is_ligand.unsqueeze(2)

        dist_mask_protein_near = (
            (delta_pos_label < 5) & (delta_pos_label > 0.1) & pair_protein_mask
        )
        dist_mask_protein_mid = (
            (delta_pos_label >= 5) & (delta_pos_label < 10) & pair_protein_mask
        )
        dist_mask_protein_far = (
            (delta_pos_label >= 10) & (delta_pos_label < 15) & pair_protein_mask
        )
        # dist_mask_ligand = (
        #     (delta_pos_label < 5) & (delta_pos_label > 0.1) & pair_ligand_mask
        # )

        dist_mask_protein = (
            dist_mask_protein_near | dist_mask_protein_mid | dist_mask_protein_far
        )
        dist_mask = dist_mask_protein  # | dist_mask_ligand

        if dist_mask.any():
            delta = torch.abs(delta_pos_label - delta_pos_pred)
            delta1 = delta[dist_mask]
            error = 0.25 * (
                torch.sigmoid(0.5 - delta1)
                + torch.sigmoid(1 - delta1)
                + torch.sigmoid(2 - delta1)
                + torch.sigmoid(4 - delta1)
            )
            lddt = error.mean()
            num_pddt_loss = 1
        else:
            lddt = torch.tensor(1.0, device=pos_label.device, requires_grad=True)
            num_pddt_loss = 0

        dist_map = model_output["dist_map"] if "dist_map" in model_output else None
        if dist_map is not None and dist_mask.any():
            mask_temp = torch.ones(L, L, dtype=torch.bool, device=pos_label.device)

            # Create upper and lower triangular masks
            upper_mask = torch.triu(mask_temp, diagonal=6)
            lower_mask = torch.tril(mask_temp, diagonal=-6)

            # Combine the masks
            mask_temp = upper_mask | lower_mask

            protein_contact_mask = (
                (delta_pos_label <= 32)
                # & (delta_pos_label >= 4)
                & pair_protein_mask
                # & mask_temp
            )
            protein_contact_mask = protein_contact_mask & dist_mask
            delta = dist_map.squeeze(-1) - delta_pos_label
            if protein_contact_mask.any():
                contact_loss = torch.abs(delta[protein_contact_mask]).mean()
                num_contact_losss = 1
            else:
                contact_loss = torch.tensor(
                    0.0, device=pos_label.device, requires_grad=True
                )
                num_contact_losss = 0

            contact_acc = 0.0
        else:
            contact_loss = torch.tensor(
                0.0, device=pos_label.device, requires_grad=True
            )
            contact_acc = 0.0
            num_contact_losss = 0

        # # hard distance loss
        time_step = model_output["time_step"]
        if time_step is not None and dist_mask.any():
            time_coefficient = (
                (1 - time_step) * torch.exp(-time_step / 0.1)
            ).unsqueeze(-1)

            if dist_mask_protein_near.any():
                hard_dist_loss = (delta * time_coefficient)[
                    dist_mask_protein_near
                ].mean() * 2
            else:
                hard_dist_loss = None

            if dist_mask_protein_mid.any() and hard_dist_loss is not None:
                hard_dist_loss = (
                    hard_dist_loss
                    + (delta * time_coefficient)[dist_mask_protein_mid].mean()
                )
            elif dist_mask_protein_mid.any() and hard_dist_loss is None:
                hard_dist_loss = (delta * time_coefficient)[
                    dist_mask_protein_mid
                ].mean()

            if dist_mask_protein_far.any() and hard_dist_loss is not None:
                hard_dist_loss = (
                    hard_dist_loss
                    + (delta * time_coefficient)[dist_mask_protein_far].mean() * 0.25
                )
            elif dist_mask_protein_far.any() and hard_dist_loss is None:
                hard_dist_loss = (delta * time_coefficient)[
                    dist_mask_protein_far
                ].mean() * 0.25

            protein_ligand_mask = (is_protein.unsqueeze(1) & is_ligand.unsqueeze(2)) | (
                is_ligand.unsqueeze(1) & is_protein.unsqueeze(2)
            )
            inter_dist_mask_near = (
                (delta_pos_label <= 4) & (delta_pos_label > 0.1) & protein_ligand_mask
            )
            inter_dist_mask_far = (
                (delta_pos_label < 8) & (delta_pos_label > 4) & protein_ligand_mask
            )

            if inter_dist_mask_near.any():
                # time_coefficien_inter = (
                #     (1 - time_step**2) * torch.exp(-time_step)
                # ).unsqueeze(-1)
                time_coefficien_inter = (
                    (1 - time_step) * torch.exp(-time_step / 0.1)
                ).unsqueeze(-1)
                inter_dist_loss = (delta * time_coefficien_inter)[
                    inter_dist_mask_near
                ].mean()

                if inter_dist_mask_far.any():
                    inter_dist_loss += (delta * time_coefficien_inter)[
                        inter_dist_mask_far
                    ].mean() * 0.5

                num_inter_dist_loss = 1
            else:
                inter_dist_loss = torch.tensor(
                    0.0, device=pos_label.device, requires_grad=True
                )
                num_inter_dist_loss = 0
        else:
            hard_dist_loss = None
            inter_dist_loss = torch.tensor(
                0.0, device=pos_label.device, requires_grad=True
            )
            num_inter_dist_loss = 0

        return (
            1 - lddt,
            num_pddt_loss,
            hard_dist_loss,
            inter_dist_loss,
            num_inter_dist_loss,
            contact_loss,
            contact_acc,
            num_contact_losss,
        )

    def _rescale_autograd_force(
        self,
        force_pred,
        sample_mask,
        is_molecule,
        is_periodic,
        use_per_atom_energy=True,
    ):
        num_samples = torch.sum(sample_mask.long())
        if num_samples > 0:
            if use_per_atom_energy:
                mol_loss_scale = self.molecule_energy_per_atom_std
                if (
                    hasattr(self.args, "energy_per_atom_label_scale")
                    and not self.training
                ):
                    # energy_per_atom label were scaled so we scale the loss back when evaluating
                    mol_loss_scale /= self.args.energy_per_atom_label_scale
                force_pred[is_molecule] = (
                    force_pred[is_molecule] * mol_loss_scale / self.molecule_force_std
                )
                force_pred[is_periodic] = (
                    force_pred[is_periodic]
                    * self.periodic_energy_per_atom_std
                    / self.periodic_force_std
                )
            else:
                force_pred[is_molecule] = (
                    force_pred[is_molecule]
                    * self.molecule_energy_std
                    / self.molecule_force_std
                )
                force_pred[is_periodic] = (
                    force_pred[is_periodic]
                    * self.periodic_energy_std
                    / self.periodic_force_std
                )
            return force_pred
        else:
            return force_pred

    def forward(self, model_output, batched_data):
        energy_per_atom_label = batched_data["energy_per_atom"]
        total_energy_label = batched_data["energy"]
        atomic_numbers = batched_data["token_id"]
        adj = batched_data["adj"]

        noise_label = model_output["noise"]
        force_label = model_output["force_label"]
        stress_label = model_output["stress_label"]
        pos_label = model_output["ori_pos"]
        force_pred = model_output["forces"]
        autograd_force_pred = (
            model_output["autograd_forces"]
            if "autograd_forces" in model_output
            else None
        )
        autograd_stress_pred = (
            model_output["autograd_stress"]
            if "autograd_stress" in model_output
            else None
        )
        stress_pred = (
            model_output["stress_pred"] if "stress_pred" in model_output else None
        )
        energy_per_atom_pred = model_output["energy_per_atom"]
        total_energy_pred = model_output["total_energy"]
        noise_pred = model_output["noise_pred"]
        non_atom_mask = model_output["non_atom_mask"]
        clean_mask = model_output["clean_mask"]
        aa_mask = model_output["aa_mask"]
        is_protein = model_output["is_protein"]
        is_molecule = model_output["is_molecule"]
        is_periodic = model_output["is_periodic"]
        is_complex = model_output["is_complex"]
        is_seq_only = model_output["is_seq_only"]
        diff_loss_mask = model_output["diff_loss_mask"]
        protein_mask = model_output["protein_mask"]
        sqrt_one_minus_alphas_cumprod_t = model_output[
            "sqrt_one_minus_alphas_cumprod_t"
        ]
        sqrt_alphas_cumprod_t = model_output["sqrt_alphas_cumprod_t"]
        dist_map = model_output["dist_map"] if "dist_map" in model_output else None

        n_graphs = energy_per_atom_label.size()[0]
        if clean_mask is None:
            clean_mask = torch.zeros(
                n_graphs, dtype=torch.bool, device=energy_per_atom_label.device
            )

        # energy and force loss only apply on total clean samples
        if self.all_atom:
            total_clean = clean_mask.all(dim=(-1, -2))
        else:
            total_clean = clean_mask.all(dim=-1)

        energy_mask = total_clean & batched_data["has_energy"]
        force_mask = total_clean & batched_data["has_forces"]
        stress_mask = total_clean & batched_data["has_stress"]

        if not self.seq_only:
            # diffussion loss
            if self.diffusion_mode == "epsilon":
                if not is_seq_only.all():
                    pos_pred = self.calculate_pos_pred(model_output)
                    if (
                        self.args.align_x0_in_diffusion_loss and not is_periodic.any()
                    ):  # and not is_periodic.any():
                        try:
                            R, T = self._alignment_x0(
                                model_output, pos_pred, atomic_numbers
                            )
                        except:
                            logger.warning("error happens in calcualte R, T")
                            R, T = torch.eye(
                                3, device=pos_pred.device, dtype=pos_pred.dtype
                            ).unsqueeze(0).repeat(n_graphs, 1, 1), torch.zeros_like(
                                pos_pred, device=pos_pred.device, dtype=pos_pred.dtype
                            )
                    else:
                        R, T = torch.eye(
                            3, device=pos_pred.device, dtype=pos_pred.dtype
                        ).unsqueeze(0).repeat(n_graphs, 1, 1), torch.zeros_like(
                            pos_pred, device=pos_pred.device, dtype=pos_pred.dtype
                        )

                    if is_protein.any():
                        # align pred pos and calculate smooth lddt loss for protein
                        (
                            smooth_lddt_loss,
                            num_pddt_loss,
                            hard_dist_loss,
                            inter_dist_loss,
                            num_inter_dist_loss,
                            contact_loss,
                            contact_acc,
                            num_contact_losss,
                        ) = self.dist_loss(model_output, R, T, pos_pred, atomic_numbers)
                        if hard_dist_loss is None:
                            hard_dist_loss = torch.tensor(
                                0.0, device=smooth_lddt_loss.device, requires_grad=True
                            )
                    else:
                        smooth_lddt_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        hard_dist_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        inter_dist_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        num_pddt_loss = 0
                        num_inter_dist_loss = 0

                    if self.args.align_x0_in_diffusion_loss and not is_periodic.any():
                        # noise pred loss
                        aligned_noise_pred = (
                            sqrt_alphas_cumprod_t * pos_label
                            + sqrt_one_minus_alphas_cumprod_t
                            * (noise_label - noise_pred)
                        )

                        # aligned_noise_pred = torch.einsum(
                        #     "bij,bkj->bki", R.float(), aligned_noise_pred.float()
                        # )
                        # unreduced_noise_loss = self.noise_loss(
                        #     aligned_noise_pred.to(noise_label.dtype),
                        #     (pos_label) * sqrt_alphas_cumprod_t,
                        # )

                        aligned_pos_label = (
                            torch.einsum("bij,bkj->bki", R.float(), pos_label.float())
                            + T.float()
                        )
                        unreduced_noise_loss = (
                            self.noise_loss(
                                aligned_noise_pred.to(noise_label.dtype),
                                (aligned_pos_label) * sqrt_alphas_cumprod_t,
                            )
                            * self.diffusion_rescale_coeff
                        )

                        unreduced_noise_loss = (
                            unreduced_noise_loss / sqrt_one_minus_alphas_cumprod_t
                        ) * self.diffusion_rescale_coeff
                    else:
                        unreduced_noise_loss = (
                            self.noise_loss(
                                noise_pred.to(noise_label.dtype), noise_label
                            )
                            * self.diffusion_rescale_coeff
                        )
                else:
                    unreduced_noise_loss = (
                        self.noise_loss(noise_pred.to(noise_label.dtype), noise_label)
                        * self.diffusion_rescale_coeff
                    )
                    smooth_lddt_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    hard_dist_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    inter_dist_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    num_pddt_loss = 0
                    num_inter_dist_loss = 0
                contact_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                bond_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                num_contact_losss = 0
                num_bond_loss = 0
            elif self.diffusion_mode == "x0":
                # x0 pred loss, noise pred is x0 pred here
                unreduced_noise_loss = self.noise_loss(
                    noise_pred.to(noise_label.dtype), pos_label
                )
                contact_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                num_contact_losss = 0
            elif self.diffusion_mode == "edm":
                weight_pos_edm = model_output["weight_edm"]
                if not is_seq_only.all():
                    if self.args.align_x0_in_diffusion_loss and not is_periodic.any():
                        # try:
                        R, T = self._alignment_x0(
                            model_output, noise_pred, atomic_numbers
                        )
                        # except:
                        #     logger.warning("error happens in calcualte R, T")
                        #     R, T = torch.eye(
                        #         3, device=noise_pred.device, dtype=noise_pred.dtype
                        #     ).unsqueeze(0).repeat(n_graphs, 1, 1), torch.zeros_like(
                        #         noise_pred,
                        #         device=noise_pred.device,
                        #         dtype=noise_pred.dtype,
                        #     )
                    else:
                        R, T = torch.eye(
                            3, device=noise_pred.device, dtype=noise_pred.dtype
                        ).unsqueeze(0).repeat(n_graphs, 1, 1), torch.zeros_like(
                            noise_pred, device=noise_pred.device, dtype=noise_pred.dtype
                        )

                    if is_protein.any():
                        # align pred pos and calculate smooth lddt loss for protein
                        (
                            smooth_lddt_loss,
                            num_pddt_loss,
                            hard_dist_loss,
                            inter_dist_loss,
                            num_inter_dist_loss,
                            contact_loss,
                            contact_acc,
                            num_contact_losss,
                        ) = self.dist_loss(
                            model_output, R, T, noise_pred, atomic_numbers
                        )
                        if hard_dist_loss is None:
                            hard_dist_loss = torch.tensor(
                                0.0, device=smooth_lddt_loss.device, requires_grad=True
                            )
                    else:
                        smooth_lddt_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        hard_dist_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        inter_dist_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        contact_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        num_pddt_loss = 0
                        num_inter_dist_loss = 0
                        num_contact_losss = 0

                    if (
                        is_molecule.any()
                        or (
                            is_complex
                            & ((atomic_numbers > 2) & (atomic_numbers < 129)).any(
                                dim=-1
                            )
                        ).any()
                    ):
                        molecule_mask = (
                            ((atomic_numbers > 2) & (atomic_numbers < 129))
                            & (~is_periodic).unsqueeze(-1)
                            & (~protein_mask.any(dim=-1))
                        )
                        bond_loss_mask = (
                            adj
                            & molecule_mask.unsqueeze(-1)
                            & molecule_mask.unsqueeze(1)
                        )
                        if bond_loss_mask.any():
                            ori_pos = model_output["ori_pos"]
                            pair_pos_label = (
                                ori_pos.unsqueeze(1) - ori_pos.unsqueeze(2)
                            ).norm(dim=-1)
                            pair_pos_pred = (
                                noise_pred.unsqueeze(1) - noise_pred.unsqueeze(2)
                            ).norm(dim=-1)

                            bond_loss = (
                                (
                                    pair_pos_label[bond_loss_mask]
                                    - pair_pos_pred[bond_loss_mask]
                                )
                                .abs()
                                .mean()
                            )
                            num_bond_loss = 1
                        else:
                            bond_loss = torch.tensor(
                                0.0, device=noise_label.device, requires_grad=True
                            )
                            num_bond_loss = 0
                    else:
                        bond_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        num_bond_loss = 0

                    if self.args.align_x0_in_diffusion_loss and not is_periodic.any():
                        if self.all_atom:
                            pos_label = torch.einsum(
                                "bij,blzj->blzi", R.to(pos_label.dtype), pos_label
                            ) + T.to(pos_label.dtype).unsqueeze(1)
                        else:
                            pos_label = torch.einsum(
                                "bij,blj->bli", R.to(pos_label.dtype), pos_label
                            ) + T.to(pos_label.dtype)

                    if weight_pos_edm is not None:
                        if self.args.diffusion_training_loss in [
                            DiffusionTrainingLoss.L1,
                            DiffusionTrainingLoss.SmoothL1,
                            DiffusionTrainingLoss.L2,
                        ]:
                            unreduced_noise_loss = (
                                weight_pos_edm.sqrt()
                                * self.noise_loss(
                                    noise_pred.to(pos_label.dtype), pos_label
                                )
                            )
                        elif (
                            self.args.diffusion_training_loss
                            == DiffusionTrainingLoss.MSE
                        ):
                            unreduced_noise_loss = weight_pos_edm * self.noise_loss(
                                noise_pred.to(pos_label.dtype), pos_label
                            )
                    else:
                        unreduced_noise_loss = self.noise_loss(
                            noise_pred.to(pos_label.dtype), pos_label
                        )
                else:
                    unreduced_noise_loss = self.noise_loss(
                        noise_pred.to(pos_label.dtype), pos_label
                    )
                    smooth_lddt_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    hard_dist_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    inter_dist_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    contact_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    bond_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    num_bond_loss = 0
                    num_pddt_loss = 0
                    num_inter_dist_loss = 0
                    num_contact_losss = 0
            else:
                raise ValueError(f"Invalid diffusion mode: {self.diffusion_mode}")

            if not is_seq_only.all():
                if self.all_atom:
                    noise_loss, num_noise_sample = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask) & (~is_seq_only.unsqueeze(-1).unsqueeze(-1)),
                        diff_loss_mask & ~protein_mask.all(dim=(-1, -2)),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        molecule_noise_loss,
                        num_molecule_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask)
                        & is_molecule.unsqueeze(-1).unsqueeze(-1)
                        & (~is_complex.unsqueeze(-1)).unsqueeze(-1),
                        diff_loss_mask & ~protein_mask.all(dim=(-1, -2)),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        periodic_noise_loss,
                        num_periodic_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,  # unreduced_periodic_noise_loss,
                        (~clean_mask) & is_periodic.unsqueeze(-1).unsqueeze(-1),
                        diff_loss_mask & ~protein_mask.all(dim=(-1, -2)),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        protein_noise_loss,
                        num_protein_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask)
                        & is_protein.unsqueeze(-1)
                        & (~is_seq_only.unsqueeze(-1).unsqueeze(-1))
                        & (~is_complex.unsqueeze(-1).unsqueeze(-1)),
                        diff_loss_mask & ~protein_mask.all(dim=(-1, -2)),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        complex_noise_loss,
                        num_complex_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask)
                        & is_complex.unsqueeze(-1).unsqueeze(-1)
                        & (~is_seq_only.unsqueeze(-1).unsqueeze(-1)),
                        diff_loss_mask
                        & ~protein_mask.all(dim=(-1, -2))
                        & atomic_numbers.ne(2),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                        is_protein=is_protein,
                    )
                else:
                    noise_loss, num_noise_sample = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask) & (~is_seq_only.unsqueeze(-1)),
                        diff_loss_mask & ~protein_mask.any(dim=-1),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        molecule_noise_loss,
                        num_molecule_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask)
                        & is_molecule.unsqueeze(-1)
                        & (~is_complex.unsqueeze(-1)),
                        diff_loss_mask & ~protein_mask.any(dim=-1),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        periodic_noise_loss,
                        num_periodic_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,  # unreduced_periodic_noise_loss,
                        (~clean_mask) & is_periodic.unsqueeze(-1),
                        diff_loss_mask & ~protein_mask.any(dim=-1),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        protein_noise_loss,
                        num_protein_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask)
                        & is_protein
                        & (~is_seq_only.unsqueeze(-1))
                        & (~is_complex.unsqueeze(-1)),
                        diff_loss_mask & ~protein_mask.any(dim=-1),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                    )
                    (
                        complex_noise_loss,
                        num_complex_noise_sample,
                    ) = self._reduce_force_or_noise_loss(
                        unreduced_noise_loss,
                        (~clean_mask)
                        & is_complex.unsqueeze(-1)
                        & (~is_seq_only.unsqueeze(-1)),
                        diff_loss_mask
                        & ~protein_mask.any(dim=-1)
                        & atomic_numbers.ne(2),
                        is_molecule,
                        is_periodic,
                        1.0,
                        1.0,
                        is_protein=is_protein,
                    )
            else:
                noise_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                molecule_noise_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                periodic_noise_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                protein_noise_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                complex_noise_loss = torch.tensor(
                    0.0, device=noise_label.device, requires_grad=True
                )
                num_noise_sample = 0
                num_molecule_noise_sample = 0
                num_periodic_noise_sample = 0
                num_protein_noise_sample = 0
                num_complex_noise_sample = 0

            if self.args.diffusion_training_loss == DiffusionTrainingLoss.L2:
                molecule_noise_loss = torch.sqrt(molecule_noise_loss + self.epsilon)
                periodic_noise_loss = torch.sqrt(periodic_noise_loss + self.epsilon)
                protein_noise_loss = torch.sqrt(protein_noise_loss + self.epsilon)
                complex_noise_loss = torch.sqrt(complex_noise_loss + self.epsilon)

            # energy loss
            if self.if_total_energy:
                unreduced_energy_loss = self.energy_loss(
                    total_energy_pred.to(torch.float32),
                    total_energy_label.to(torch.float32),
                )
            else:
                unreduced_energy_loss = self.energy_loss(
                    energy_per_atom_pred.to(torch.float32),
                    energy_per_atom_label.to(torch.float32),
                )
            energy_loss, num_energy_sample = self._reduce_energy_loss(
                unreduced_energy_loss,
                energy_mask,
                is_molecule,
                is_periodic,
                use_per_atom_energy=True,
            )
            molecule_energy_loss, num_molecule_energy_sample = self._reduce_energy_loss(
                unreduced_energy_loss,
                energy_mask & is_molecule,
                is_molecule,
                is_periodic,
                use_per_atom_energy=True,
            )
            periodic_energy_loss, num_periodic_energy_sample = self._reduce_energy_loss(
                unreduced_energy_loss,
                energy_mask & is_periodic,
                is_molecule,
                is_periodic,
                use_per_atom_energy=True,
            )

            # force loss
            unreduced_force_loss = self.force_loss(
                force_pred.to(dtype=force_label.dtype), force_label
            )
            force_loss, num_force_sample = self._reduce_force_or_noise_loss(
                unreduced_force_loss,
                force_mask,
                ~non_atom_mask,
                is_molecule,
                is_periodic,
                self.molecule_force_std,
                self.periodic_force_std,
            )
            (
                molecule_force_loss,
                num_molecule_force_sample,
            ) = self._reduce_force_or_noise_loss(
                unreduced_force_loss,
                force_mask & is_molecule,
                ~non_atom_mask,
                is_molecule,
                is_periodic,
                self.molecule_force_std,
                self.periodic_force_std,
            )
            (
                periodic_force_loss,
                num_periodic_force_sample,
            ) = self._reduce_force_or_noise_loss(
                unreduced_force_loss,
                force_mask & is_periodic,
                ~non_atom_mask,
                is_molecule,
                is_periodic,
                self.molecule_force_std,
                self.periodic_force_std,
            )

            if autograd_force_pred is not None:
                unreduced_autograd_force_loss = self.force_loss(
                    autograd_force_pred.to(dtype=force_label.dtype), force_label
                )
                (
                    autograd_force_loss,
                    num_autograd_force_sample,
                ) = self._reduce_force_or_noise_loss(
                    unreduced_autograd_force_loss,
                    force_mask,
                    ~non_atom_mask,
                    is_molecule,
                    is_periodic,
                    self.molecule_force_std,
                    self.periodic_force_std,
                )
                (
                    molecule_autograd_force_loss,
                    num_molecule_autograd_force_sample,
                ) = self._reduce_force_or_noise_loss(
                    unreduced_autograd_force_loss,
                    force_mask & is_molecule,
                    ~non_atom_mask,
                    is_molecule,
                    is_periodic,
                    self.molecule_force_std,
                    self.periodic_force_std,
                )
                (
                    periodic_autograd_force_loss,
                    num_periodic_autograd_force_sample,
                ) = self._reduce_force_or_noise_loss(
                    unreduced_autograd_force_loss,
                    force_mask & is_periodic,
                    ~non_atom_mask,
                    is_molecule,
                    is_periodic,
                    self.molecule_force_std,
                    self.periodic_force_std,
                )
            else:
                autograd_force_loss = torch.tensor(
                    0.0, device=energy_per_atom_label.device, requires_grad=True
                )
                num_autograd_force_sample = 0

            if autograd_stress_pred is not None:
                unreduced_autograd_stress_loss = self.stress_loss(
                    autograd_stress_pred, stress_label
                )
                (
                    periodic_autograd_stress_loss,
                    num_periodic_autograd_stress_sample,
                ) = self._reduce_stress_loss(
                    unreduced_autograd_stress_loss,
                    stress_mask & is_periodic,
                    self.periodic_stress_std,
                )

            if stress_pred is not None:
                unreduced_stress_loss = self.stress_loss(stress_pred, stress_label)
                (
                    periodic_stress_loss,
                    num_periodic_stress_sample,
                ) = self._reduce_stress_loss(
                    unreduced_stress_loss,
                    stress_mask & is_periodic,
                    self.periodic_stress_std,
                )

        else:
            energy_loss = torch.tensor(
                0.0, device=energy_per_atom_label.device, requires_grad=True
            )
            molecule_energy_loss = torch.tensor(
                0.0, device=energy_per_atom_label.device, requires_grad=True
            )
            periodic_energy_loss = torch.tensor(
                0.0, device=energy_per_atom_label.device, requires_grad=True
            )
            force_loss = torch.tensor(
                0.0, device=force_label.device, requires_grad=True
            )
            molecule_force_loss = torch.tensor(
                0.0, device=force_label.device, requires_grad=True
            )
            periodic_force_loss = torch.tensor(
                0.0, device=force_label.device, requires_grad=True
            )
            noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            molecule_noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            periodic_noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            protein_noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            contact_loss = torch.tensor(0.0, device=dist_map.device, requires_grad=True)

            autograd_force_loss = torch.tensor(
                0.0, device=energy_per_atom_label.device, requires_grad=True
            )
            num_energy_sample = 0
            num_molecule_energy_sample = 0
            num_periodic_energy_sample = 0
            num_force_sample = 0
            num_molecule_force_sample = 0
            num_periodic_force_sample = 0
            num_noise_sample = 0
            num_molecule_noise_sample = 0
            num_periodic_noise_sample = 0
            num_protein_noise_sample = 0
            num_complex_noise_sample = 0
            num_contact_losss = 0
            num_autograd_force_sample = 0

        # mlm loss
        if aa_mask.any():
            logits = model_output["aa_logits"][aa_mask]
            aa_mlm_loss = self.aa_mlm_loss(
                logits,
                atomic_numbers[aa_mask],
            )
            aa_acc = (
                (
                    logits.view(-1, logits.size(-1)).argmax(dim=-1)
                    == atomic_numbers[aa_mask]
                )
                .to(torch.float32)
                .mean()
            )
            num_aa_mask_token = torch.sum(aa_mask.to(dtype=aa_mlm_loss.dtype))

            decoder_aa_logits = model_output["decoder_aa_logits"]
            if decoder_aa_logits is not None:
                decoder_logits = model_output["decoder_aa_logits"][aa_mask]
                decoder_mlm_loss = self.aa_mlm_loss(
                    decoder_logits,
                    atomic_numbers[aa_mask],
                )
                decoder_aa_acc = (
                    (
                        decoder_logits.view(-1, decoder_logits.size(-1)).argmax(dim=-1)
                        == atomic_numbers[aa_mask]
                    )
                    .to(torch.float32)
                    .mean()
                )
                num_decoder_aa_mask_token = torch.sum(
                    aa_mask.to(dtype=decoder_mlm_loss.dtype)
                )
            else:
                decoder_mlm_loss = torch.tensor(
                    0.0, device=atomic_numbers.device, requires_grad=True
                )
                decoder_aa_acc = 0.0
                num_decoder_aa_mask_token = 0.0
        else:
            aa_mlm_loss = torch.tensor(
                0.0, device=atomic_numbers.device, requires_grad=True
            )
            decoder_mlm_loss = torch.tensor(
                0.0, device=atomic_numbers.device, requires_grad=True
            )

            aa_acc = 0.0
            decoder_aa_acc = 0.0
            num_aa_mask_token = 0.0

            num_decoder_aa_mask_token = 0.0

        if not is_seq_only.all():
            loss = torch.tensor(0.0, device=atomic_numbers.device, requires_grad=True)

            if torch.any(torch.isnan(protein_noise_loss)) or torch.any(
                torch.isinf(protein_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in protein_noise_loss: {protein_noise_loss}"
                )
                protein_noise_loss = torch.tensor(
                    0.0, device=protein_noise_loss.device, requires_grad=True
                )
                num_protein_noise_sample = 0
            else:
                loss = loss + 4.0 * protein_noise_loss

            if torch.any(torch.isnan(complex_noise_loss)) or torch.any(
                torch.isinf(complex_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in complex_noise_loss: {complex_noise_loss}"
                )
                complex_noise_loss = torch.tensor(
                    0.0, device=complex_noise_loss.device, requires_grad=True
                )
                num_complex_noise_sample = 0
            else:
                loss = loss + 4.0 * complex_noise_loss

            if torch.any(torch.isnan(periodic_noise_loss)) or torch.any(
                torch.isinf(periodic_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in periodic_noise_loss: {periodic_noise_loss}"
                )
                periodic_noise_loss = torch.tensor(
                    0.0, device=periodic_noise_loss.device, requires_grad=True
                )
                num_periodic_noise_sample = 0
            else:
                loss = loss + periodic_noise_loss

            if torch.any(torch.isnan(molecule_noise_loss)) or torch.any(
                torch.isinf(molecule_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in molecule_noise_loss: {molecule_noise_loss}"
                )
                molecule_noise_loss = torch.tensor(
                    0.0, device=molecule_noise_loss.device, requires_grad=True
                )
                num_molecule_noise_sample = 0
            else:
                loss = loss + molecule_noise_loss

            if torch.any(torch.isnan(aa_mlm_loss)) or torch.any(
                torch.isinf(aa_mlm_loss)
            ):
                logger.error(f"NaN or inf detected in aa_mlm_loss: {aa_mlm_loss}")
                aa_mlm_loss = torch.tensor(
                    0.0, device=aa_mlm_loss.device, requires_grad=True
                )
                num_aa_mask_token = 0
            else:
                loss = loss + aa_mlm_loss

            if torch.any(torch.isnan(decoder_mlm_loss)) or torch.any(
                torch.isinf(decoder_mlm_loss)
            ):
                logger.error(
                    f"NaN or inf detected in decoder_mlm_loss: {decoder_mlm_loss}"
                )
                decoder_mlm_loss = torch.tensor(
                    0.0, device=decoder_mlm_loss.device, requires_grad=True
                )
                num_aa_mask_token = 0
            else:
                loss = loss + decoder_mlm_loss

            if torch.any(torch.isnan(contact_loss)) or torch.any(
                torch.isinf(contact_loss)
            ):
                logger.error(f"NaN or inf detected in contact_loss: {contact_loss}")
                contact_loss = torch.tensor(
                    0.0, device=contact_loss.device, requires_grad=True
                )
                num_aa_mask_token = 0
            else:
                loss = loss + contact_loss

            if torch.any(torch.isnan(periodic_energy_loss)) or torch.any(
                torch.isinf(periodic_energy_loss)
            ):
                logger.error(
                    f"NaN or inf detected in periodic_energy_loss: {periodic_energy_loss}"
                )
                periodic_energy_loss = torch.tensor(
                    0.0, device=periodic_energy_loss.device, requires_grad=True
                )
                num_periodic_energy_sample = 0
            else:
                loss = loss + periodic_energy_loss

            if torch.any(torch.isnan(molecule_energy_loss)) or torch.any(
                torch.isinf(molecule_energy_loss)
            ):
                logger.error(
                    f"NaN or inf detected in molecule_energy_loss: {molecule_energy_loss}"
                )
                molecule_energy_loss = torch.tensor(
                    0.0, device=molecule_energy_loss.device, requires_grad=True
                )
                num_molecule_energy_sample = 0
            else:
                loss = loss + molecule_energy_loss

            if self.args.AutoGradForce:
                if torch.any(torch.isnan(periodic_force_loss)) or torch.any(
                    torch.isinf(periodic_force_loss)
                ):
                    logger.error(
                        f"NaN or inf detected in periodic_force_loss: {periodic_force_loss}"
                    )
                    periodic_force_loss = torch.tensor(
                        0.0, device=periodic_force_loss.device, requires_grad=True
                    )
                    num_periodic_force_sample = 0
                else:
                    loss = loss + periodic_force_loss

                if torch.any(torch.isnan(molecule_force_loss)) or torch.any(
                    torch.isinf(molecule_force_loss)
                ):
                    logger.error(
                        f"NaN or inf detected in molecule_force_loss: {molecule_force_loss}"
                    )
                    molecule_force_loss = torch.tensor(
                        0.0, device=molecule_force_loss.device, requires_grad=True
                    )
                    num_molecule_force_sample = 0
                else:
                    loss = loss + molecule_force_loss

            if torch.any(torch.isnan(smooth_lddt_loss)) or torch.any(
                torch.isinf(smooth_lddt_loss)
            ):
                logger.error(
                    f"NaN or inf detected in smooth_lddt_loss: {smooth_lddt_loss}"
                )
                smooth_lddt_loss = torch.tensor(
                    0.0, device=smooth_lddt_loss.device, requires_grad=True
                )
            else:
                loss = loss + smooth_lddt_loss

            if self.args.use_hard_dist_loss:
                # (1.0 / (20 * hard_dist_loss.item() ** 1.2) if num_pddt_loss > 0 else 0)

                if torch.any(torch.isnan(hard_dist_loss)) or torch.any(
                    torch.isinf(hard_dist_loss)
                ):
                    logger.error(
                        f"NaN or inf detected in hard_dist_loss: {hard_dist_loss}"
                    )
                    hard_dist_loss = torch.tensor(
                        0.0, device=hard_dist_loss.device, requires_grad=True
                    )
                else:
                    loss = loss + self.hard_dist_loss_raito * hard_dist_loss

                if torch.any(torch.isnan(inter_dist_loss)) or torch.any(
                    torch.isinf(inter_dist_loss)
                ):
                    logger.error(
                        f"NaN or inf detected in inter_dist_loss: {inter_dist_loss}"
                    )
                    inter_dist_loss = torch.tensor(
                        0.0, device=inter_dist_loss.device, requires_grad=True
                    )
                else:
                    loss = loss + (self.hard_dist_loss_raito * inter_dist_loss)

            if self.args.use_bond_loss:
                if torch.any(torch.isnan(bond_loss)) or torch.any(
                    torch.isinf(bond_loss)
                ):
                    logger.error(f"NaN or inf detected in bond_loss: {bond_loss}")
                    bond_loss = torch.tensor(
                        0.0, device=bond_loss.device, requires_grad=True
                    )
                else:
                    loss = loss + bond_loss

            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                logger.error(
                    f"NaN or inf detected in loss: {loss}, molecule_energy_loss: {molecule_energy_loss}, molecule_force_loss: {molecule_force_loss}, periodic_energy_loss: {periodic_energy_loss}, periodic_force_loss: {periodic_force_loss}, molecule_noise_loss: {molecule_noise_loss}, periodic_noise_loss: {periodic_noise_loss}, protein_noise_loss: {protein_noise_loss}, complex_noise_loss: {complex_noise_loss}, noise_loss: {noise_loss}, aa_mlm_loss: {aa_mlm_loss}"
                )
                loss = torch.tensor(
                    0.0, device=atomic_numbers.device, requires_grad=True
                )

            if autograd_force_pred is not None:
                if torch.any(torch.isnan(autograd_force_loss)) or torch.any(
                    torch.isinf(autograd_force_loss)
                ):
                    logger.error(
                        f"NaN or inf detected in autograd_force_loss: {autograd_force_loss}"
                    )
                    autograd_force_loss = torch.tensor(
                        0.0, device=autograd_force_loss.device, requires_grad=True
                    )
                else:
                    loss = loss + autograd_force_loss

            if autograd_stress_pred is not None:
                if torch.any(torch.isnan(periodic_autograd_stress_loss)) or torch.any(
                    torch.isinf(periodic_autograd_stress_loss)
                ):
                    logger.error(
                        f"NaN or inf detected in periodic_autograd_stress_loss: {periodic_autograd_stress_loss}"
                    )
                    periodic_autograd_stress_loss = torch.tensor(
                        0.0,
                        device=periodic_autograd_stress_loss.device,
                        requires_grad=True,
                    )
                else:
                    loss = (
                        loss
                        + periodic_autograd_stress_loss * self.args.stress_loss_factor
                    )

            if stress_pred is not None:
                if torch.any(torch.isnan(periodic_stress_loss)) or torch.any(
                    torch.isinf(periodic_stress_loss)
                ):
                    logger.error(
                        f"NaN or inf detected in periodic_stress_loss: {periodic_stress_loss}"
                    )
                    periodic_stress_loss = torch.tensor(
                        0.0, device=periodic_stress_loss.device, requires_grad=True
                    )
                else:
                    loss = loss + periodic_stress_loss * self.args.stress_loss_factor

        elif not torch.any(torch.isnan(aa_mlm_loss)):
            loss = aa_mlm_loss
        else:
            raise ValueError("aa_mlm_loss is NaN")

        # for loss exist in every sample of the batch, no extra number of samples are recorded (will use batch size in loss reduction)
        # for loss does not exist in every example of the batch, use a tuple, where the first element is the averaged loss value
        # and the second element is the number of samples (or token numbers) in the batch with that loss considered
        logging_output = {
            "total_loss": loss.to(torch.float32),
            "energy_loss": (float(energy_loss.detach()), int(num_energy_sample)),
            "molecule_energy_loss": (
                float(molecule_energy_loss.detach()),
                int(num_molecule_energy_sample),
            ),
            "periodic_energy_loss": (
                float(periodic_energy_loss.detach()),
                int(num_periodic_energy_sample),
            ),
            "force_loss": (float(force_loss), int(num_force_sample)),
            "molecule_force_loss": (
                float(molecule_force_loss.detach()),
                int(num_molecule_force_sample),
            ),
            "periodic_force_loss": (
                float(periodic_force_loss.detach()),
                int(num_periodic_force_sample),
            ),
            "noise_loss": (float(noise_loss), int(num_noise_sample)),
            "molecule_noise_loss": (
                float(molecule_noise_loss.detach()),
                int(num_molecule_noise_sample),
            ),
            "periodic_noise_loss": (
                float(periodic_noise_loss.detach()),
                int(num_periodic_noise_sample),
            ),
            "protein_noise_loss": (
                float(protein_noise_loss.detach()),
                int(num_protein_noise_sample),
            ),
            "complex_noise_loss": (
                float(complex_noise_loss.detach()),
                int(num_complex_noise_sample),
            ),
            "aa_mlm_loss": (float(aa_mlm_loss.detach()), int(num_aa_mask_token)),
            "decoder_mlm_loss": (
                float(decoder_mlm_loss.detach()),
                int(num_decoder_aa_mask_token),
            ),
            "aa_acc": (float(aa_acc), int(num_aa_mask_token)),
            "decoder_aa_acc": (float(decoder_aa_acc), int(num_decoder_aa_mask_token)),
            "contact_loss": (float(contact_loss.detach()), int(num_contact_losss)),
            "bond_loss": (float(bond_loss), int(num_bond_loss)),
            "smooth_lddt_loss": (float(smooth_lddt_loss.detach()), int(num_pddt_loss)),
            "hard_dist_loss": (float(hard_dist_loss.detach()), int(num_pddt_loss)),
            "inter_dist_loss": (
                float(inter_dist_loss.detach()),
                int(num_inter_dist_loss),
            ),
        }

        if autograd_force_pred is not None:
            logging_output.update(
                {
                    "autograd_force_loss": (
                        autograd_force_loss,
                        num_autograd_force_sample,
                    ),
                    "molecule_autograd_force_loss": (
                        molecule_autograd_force_loss,
                        num_molecule_autograd_force_sample,
                    ),
                    "periodic_autograd_force_loss": (
                        periodic_autograd_force_loss,
                        num_periodic_autograd_force_sample,
                    ),
                }
            )

        if autograd_stress_pred is not None:
            logging_output.update(
                {
                    "periodic_autograd_stress_loss": (
                        periodic_autograd_stress_loss,
                        num_periodic_autograd_stress_sample,
                    )
                }
            )
        if stress_pred is not None:
            logging_output.update(
                {
                    "periodic_stress_loss": (
                        periodic_stress_loss,
                        num_periodic_stress_sample,
                    )
                }
            )

        def _reduce_matched_result(model_output, metric_name, min_or_max: str):
            metric = model_output[metric_name]
            matched_mask = ~metric.isnan()
            num_matched_samples = int(torch.sum(matched_mask.long()))
            matched_rate = float(torch.mean(matched_mask.float()))
            if num_matched_samples > 0:
                mean_metric = float(
                    torch.sum(metric[matched_mask]) / num_matched_samples
                )
                metric_tuple = (mean_metric, num_matched_samples)
            else:
                metric_tuple = (0.0, 0)
            matched_rate_tuple = (matched_rate, metric.numel())
            torch_min_or_max_func = torch.min if min_or_max == "min" else torch.max
            if metric.size()[-1] > 1:
                any_matched_mask = ~metric.isnan().any(dim=-1)
                num_any_matched_samples = int(torch.sum(any_matched_mask))
                best_metric = torch_min_or_max_func(metric, dim=-1)[0]
                if num_any_matched_samples > 0:
                    best_metric = float(
                        torch.sum(best_metric[any_matched_mask])
                        / num_any_matched_samples
                    )
                    best_metric_tuple = (best_metric, num_any_matched_samples)
                else:
                    best_metric_tuple = (0.0, 0)
                any_matched_rate_tuple = (
                    float(torch.mean(any_matched_mask.float())),
                    int(metric.size()[0]),
                )
            else:
                best_metric_tuple = metric_tuple
                any_matched_rate_tuple = matched_rate_tuple
            return (
                metric_tuple,
                matched_rate_tuple,
                best_metric_tuple,
                any_matched_rate_tuple,
            )

        if "rmsd" in model_output:
            (
                logging_output["rmsd"],
                logging_output["matched_rate"],
                logging_output["min_rmsd"],
                logging_output["any_matched_rate"],
            ) = _reduce_matched_result(model_output, "rmsd", "min")
        if "relaxed_rmsd" in model_output:
            (
                logging_output["relaxed_rmsd"],
                logging_output["relaxed_matched_rate"],
                logging_output["relaxed_min_rmsd"],
                logging_output["relaxed_any_matched_rate"],
            ) = _reduce_matched_result(model_output, "relaxed_rmsd", "min")
        if "tm_score" in model_output:
            (
                logging_output["tm_score"],
                _,
                logging_output["max_tm_score"],
                _,
            ) = _reduce_matched_result(model_output, "tm_score", "max")
        if "lddt" in model_output:
            (
                logging_output["lddt"],
                _,
                logging_output["max_lddt"],
                _,
            ) = _reduce_matched_result(model_output, "lddt", "max")
        if "p1" in model_output:
            (
                logging_output["p1"],
                _,
                logging_output["min_p1"],
                _,
            ) = _reduce_matched_result(model_output, "p1", "min")
        if "relaxed_p1" in model_output:
            (
                logging_output["relaxed_p1"],
                _,
                logging_output["relaxed_min_p1"],
                _,
            ) = _reduce_matched_result(model_output, "relaxed_p1", "min")

        return loss, logging_output


class PolicyGradientLoss(nn.Module):
    def __init__(self, args):
        super(PolicyGradientLoss, self).__init__()
        self.args = args

    def forward(self, model_output, batched_data):
        advantage = model_output["reward"] - model_output["value_per_atom"].detach()
        log_prob = model_output["log_prob"]
        model_output["old_log_prob"]
        kl = model_output["kl"]

        value_loss = model_output["value_loss"]
        alternate_counter = model_output["alternate_counter"]

        # ratio = torch.exp(log_prob - old_log_prob)
        # ratio = torch.clamp(
        #     ratio, 1.0 - self.args.ratio_clip, 1.0 + self.args.ratio_clip
        # )
        ratio = log_prob

        policy_loss = torch.mean(
            -self.args.reward_weight * advantage * ratio + self.args.kl_weight * kl
        )

        logging_output = {
            "policy_loss": -torch.mean(advantage * log_prob),
            "value_loss": value_loss,
            "kl": torch.mean(kl),
        }

        loss = value_loss if alternate_counter > 0 else policy_loss

        return loss, logging_output


class DiffProteaCriterions(DiffMAE3dCriterions):
    def __init__(
        self,
        args: PSMConfig,
        molecule_energy_mean: float = 0.0,
        molecule_energy_std: float = 1.0,
        periodic_energy_mean: float = 0.0,
        periodic_energy_std: float = 1.0,
        molecule_energy_per_atom_mean: float = 0.0,
        molecule_energy_per_atom_std: float = 1.0,
        periodic_energy_per_atom_mean: float = 0.0,
        periodic_energy_per_atom_std: float = 1.0,
        molecule_force_mean: float = 0.0,
        molecule_force_std: float = 1.0,
        periodic_force_mean: float = 0.0,
        periodic_force_std: float = 1.0,
        periodic_stress_mean: float = 0.0,
        periodic_stress_std: float = 1.0,
    ) -> None:
        super(DiffProteaCriterions, self).__init__(
            args,
            molecule_energy_mean,
            molecule_energy_std,
            periodic_energy_mean,
            periodic_energy_std,
            molecule_energy_per_atom_mean,
            molecule_energy_per_atom_std,
            periodic_energy_per_atom_mean,
            periodic_energy_per_atom_std,
            molecule_force_mean,
            molecule_force_std,
            periodic_force_mean,
            periodic_force_std,
            periodic_stress_mean,
            periodic_stress_std,
        )

        self.aa_diff_loss = nn.L1Loss(reduction="none")

    def _reduce_aa_diff_noise_loss(
        self,
        force_or_noise_loss,
        sample_mask,
        token_mask,
        is_molecule,
        is_periodic,
        molecule_loss_factor=1.0,
        periodic_loss_factor=1.0,
    ):
        if len(sample_mask.shape) == (len(token_mask.shape) - 1):
            sample_mask = sample_mask & token_mask.any(dim=-1)
        elif len(sample_mask.shape) == len(token_mask.shape):
            sample_mask = sample_mask & token_mask
        else:
            raise ValueError(
                f"sample_mask and token_mask have incompatible shapes: {sample_mask.shape} and {token_mask.shape}"
            )

        num_samples = torch.sum(sample_mask.long())
        force_or_noise_loss = (
            force_or_noise_loss.clone()
        )  # force_or_noise_loss cloned since it will be resued multiple times
        if num_samples > 0:
            # multiply the loss by std (across all atoms and all 3 coordinates) of force labels
            # note that this works only when using MAE loss
            # for example, with MSE loss, we need to multiply squre of the std
            # for noise loss, the factors should be 1.0
            force_or_noise_loss[is_molecule] = (
                force_or_noise_loss[is_molecule] * molecule_loss_factor
            )
            force_or_noise_loss[is_periodic] = (
                force_or_noise_loss[is_periodic] * periodic_loss_factor
            )
            force_or_noise_loss = force_or_noise_loss.masked_fill(
                ~token_mask.unsqueeze(-1), 0.0
            )
            if len(sample_mask.shape) == 1:
                force_or_noise_loss = torch.sum(
                    force_or_noise_loss[sample_mask], dim=[1, 2]
                ) / (160.0 * torch.sum(token_mask[sample_mask], dim=-1))
            elif len(sample_mask.shape) == 2:
                # TODO: need to average over tokens in one sample first then all smaples
                force_or_noise_loss = torch.sum(
                    force_or_noise_loss[sample_mask], dim=[0, 1]
                ) / (160.0 * torch.sum(token_mask[sample_mask], dim=-1))
            else:
                raise ValueError(
                    f"sample_mask has an unexpected shape: {sample_mask.shape}"
                )
            force_or_noise_loss = force_or_noise_loss.mean()
        else:
            force_or_noise_loss = torch.tensor(
                0.0, device=force_or_noise_loss.device, requires_grad=True
            )
        return force_or_noise_loss, num_samples

    def forward(self, model_output, batched_data):
        energy_per_atom_label = batched_data["energy_per_atom"]
        batched_data["energy"]
        atomic_numbers = batched_data["token_id"]
        noise_label = model_output["noise"]
        model_output["force_label"]
        pos_label = model_output["ori_pos"]
        model_output["forces"]
        model_output["energy_per_atom"]
        model_output["total_energy"]
        noise_pred = model_output["noise_pred"]
        model_output["non_atom_mask"]
        clean_mask = model_output["clean_mask"]
        aa_mask = model_output["aa_mask"]
        is_protein = model_output["is_protein"]
        is_molecule = model_output["is_molecule"]
        is_periodic = model_output["is_periodic"]
        is_complex = model_output["is_complex"]
        is_seq_only = model_output["is_seq_only"]
        diff_loss_mask = model_output["diff_loss_mask"]
        protein_mask = model_output["protein_mask"]
        sqrt_one_minus_alphas_cumprod_t = model_output[
            "sqrt_one_minus_alphas_cumprod_t"
        ]
        sqrt_alphas_cumprod_t = model_output["sqrt_alphas_cumprod_t"]

        n_graphs = energy_per_atom_label.size()[0]
        if clean_mask is None:
            clean_mask = torch.zeros(
                n_graphs, dtype=torch.bool, device=energy_per_atom_label.device
            )

        # energy and force loss only apply on total clean samples
        total_clean = clean_mask.all(dim=-1)
        total_clean & batched_data["has_energy"]
        total_clean & batched_data["has_forces"]

        if not self.seq_only:
            # diffussion loss
            if self.diffusion_mode == "protea":
                if not is_seq_only.all():
                    pos_pred = self.calculate_pos_pred(model_output)
                    if (
                        self.args.align_x0_in_diffusion_loss and not is_periodic.any()
                    ):  # and not is_periodic.any():
                        R, T = self._alignment_x0(
                            model_output, pos_pred, atomic_numbers
                        )
                    else:
                        R, T = torch.eye(
                            3, device=pos_pred.device, dtype=pos_pred.dtype
                        ).unsqueeze(0).repeat(n_graphs, 1, 1), torch.zeros_like(
                            pos_pred, device=pos_pred.device, dtype=pos_pred.dtype
                        )

                    if is_protein.any():
                        # align pred pos and calculate smooth lddt loss for protein
                        (
                            smooth_lddt_loss,
                            hard_dist_loss,
                            inter_dist_loss,
                            num_inter_dist_loss,
                        ) = self.dist_loss(model_output, R, T, pos_pred, atomic_numbers)
                        num_pddt_loss = 1
                        if hard_dist_loss is None:
                            hard_dist_loss = torch.tensor(
                                0.0, device=smooth_lddt_loss.device, requires_grad=True
                            )
                    else:
                        smooth_lddt_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        hard_dist_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        inter_dist_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        num_pddt_loss = 0
                        num_inter_dist_loss = 0

                    if self.args.align_x0_in_diffusion_loss and not is_periodic.any():
                        # noise pred loss
                        aligned_noise_pred = (
                            sqrt_alphas_cumprod_t * pos_label
                            + sqrt_one_minus_alphas_cumprod_t
                            * (noise_label - noise_pred)
                        )

                        aligned_pos_label = (
                            torch.einsum("bij,bkj->bki", R.float(), pos_label.float())
                            # + T.float()
                        )
                        unreduced_noise_loss = self.noise_loss(
                            aligned_noise_pred.to(noise_label.dtype),
                            (aligned_pos_label) * sqrt_alphas_cumprod_t,
                        )
                        unreduced_noise_loss = (
                            unreduced_noise_loss / sqrt_one_minus_alphas_cumprod_t
                        )
                    else:
                        unreduced_noise_loss = self.noise_loss(
                            noise_pred.to(noise_label.dtype), noise_label
                        )
                    unreduced_aa_diff_loss = self.aa_diff_loss(
                        model_output["aa_logits"], batched_data["noise_1d"]
                    )

                else:
                    unreduced_noise_loss = self.noise_loss(
                        noise_pred.to(noise_label.dtype), noise_label
                    )
                    smooth_lddt_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    hard_dist_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    inter_dist_loss = torch.tensor(
                        0.0, device=noise_label.device, requires_grad=True
                    )
                    num_pddt_loss = 0
                    num_inter_dist_loss = 0
            else:
                raise ValueError(f"Invalid diffusion mode: {self.diffusion_mode}")

            noise_loss, num_noise_sample = self._reduce_force_or_noise_loss(
                unreduced_noise_loss,
                (~clean_mask) & (~is_seq_only.unsqueeze(-1)),
                diff_loss_mask & ~protein_mask.any(dim=-1),
                is_molecule,
                is_periodic,
                1.0,
                1.0,
            )
            (
                molecule_noise_loss,
                num_molecule_noise_sample,
            ) = self._reduce_force_or_noise_loss(
                unreduced_noise_loss,
                (~clean_mask) & is_molecule.unsqueeze(-1) & (~is_complex.unsqueeze(-1)),
                diff_loss_mask & ~protein_mask.any(dim=-1),
                is_molecule,
                is_periodic,
                1.0,
                1.0,
            )
            (
                periodic_noise_loss,
                num_periodic_noise_sample,
            ) = self._reduce_force_or_noise_loss(
                unreduced_noise_loss,
                (~clean_mask) & is_periodic.unsqueeze(-1),
                diff_loss_mask & ~protein_mask.any(dim=-1),
                is_molecule,
                is_periodic,
                1.0,
                1.0,
            )
            (
                protein_noise_loss,
                num_protein_noise_sample,
            ) = self._reduce_force_or_noise_loss(
                unreduced_noise_loss,
                (~clean_mask)
                & is_protein
                & (~is_seq_only.unsqueeze(-1))
                & (~is_complex.unsqueeze(-1)),
                diff_loss_mask & ~protein_mask.any(dim=-1),
                is_molecule,
                is_periodic,
                1.0,
                1.0,
            )
            (
                complex_noise_loss,
                num_complex_noise_sample,
            ) = self._reduce_force_or_noise_loss(
                unreduced_noise_loss,
                (~clean_mask) & is_complex.unsqueeze(-1) & (~is_seq_only.unsqueeze(-1)),
                diff_loss_mask & ~protein_mask.any(dim=-1) & atomic_numbers.ne(2),
                is_molecule,
                is_periodic,
                1.0,
                1.0,
            )

            if self.args.diffusion_training_loss == DiffusionTrainingLoss.L2:
                molecule_noise_loss = torch.sqrt(molecule_noise_loss + self.epsilon)
                periodic_noise_loss = torch.sqrt(periodic_noise_loss + self.epsilon)
                protein_noise_loss = torch.sqrt(protein_noise_loss + self.epsilon)
                complex_noise_loss = torch.sqrt(complex_noise_loss + self.epsilon)

            (
                aa_diff_loss,
                num_aa_diff_token,
            ) = self._reduce_aa_diff_noise_loss(
                unreduced_aa_diff_loss,
                aa_mask,
                diff_loss_mask,
                is_molecule,
                is_periodic,
                1.0,
                1.0,
            )
        else:
            noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            molecule_noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            periodic_noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            protein_noise_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )
            aa_diff_loss = torch.tensor(
                0.0, device=noise_label.device, requires_grad=True
            )

            num_noise_sample = 0
            num_molecule_noise_sample = 0
            num_periodic_noise_sample = 0
            num_protein_noise_sample = 0
            num_complex_noise_sample = 0
            num_aa_diff_token = 0

        if not is_seq_only.all():
            loss = torch.tensor(0.0, device=atomic_numbers.device, requires_grad=True)

            if torch.any(torch.isnan(protein_noise_loss)) or torch.any(
                torch.isinf(protein_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in protein_noise_loss: {protein_noise_loss}"
                )
                protein_noise_loss = torch.tensor(
                    0.0, device=protein_noise_loss.device, requires_grad=True
                )
                num_protein_noise_sample = 0
            else:
                loss = loss + 4.0 * protein_noise_loss

            if torch.any(torch.isnan(complex_noise_loss)) or torch.any(
                torch.isinf(complex_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in complex_noise_loss: {complex_noise_loss}"
                )
                complex_noise_loss = torch.tensor(
                    0.0, device=complex_noise_loss.device, requires_grad=True
                )
                num_complex_noise_sample = 0
            else:
                loss = loss + 4.0 * complex_noise_loss

            if torch.any(torch.isnan(periodic_noise_loss)) or torch.any(
                torch.isinf(periodic_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in periodic_noise_loss: {periodic_noise_loss}"
                )
                periodic_noise_loss = torch.tensor(
                    0.0, device=periodic_noise_loss.device, requires_grad=True
                )
                num_periodic_noise_sample = 0
            else:
                loss = loss + periodic_noise_loss

            if torch.any(torch.isnan(molecule_noise_loss)) or torch.any(
                torch.isinf(molecule_noise_loss)
            ):
                logger.error(
                    f"NaN or inf detected in molecule_noise_loss: {molecule_noise_loss}"
                )
                molecule_noise_loss = torch.tensor(
                    0.0, device=molecule_noise_loss.device, requires_grad=True
                )
                num_molecule_noise_sample = 0
            else:
                loss = loss + molecule_noise_loss

            if torch.any(torch.isnan(aa_diff_loss)) or torch.any(
                torch.isinf(aa_diff_loss)
            ):
                logger.error(f"NaN or inf detected in noise_loss: {noise_loss}")
                aa_diff_loss = torch.tensor(
                    0.0, device=noise_loss.device, requires_grad=True
                )
                num_aa_diff_token = 0
            else:
                loss = loss + aa_diff_loss

            if torch.any(torch.isnan(smooth_lddt_loss)) or torch.any(
                torch.isinf(smooth_lddt_loss)
            ):
                logger.error(
                    f"NaN or inf detected in smooth_lddt_loss: {smooth_lddt_loss}"
                )
                smooth_lddt_loss = torch.tensor(
                    0.0, device=smooth_lddt_loss.device, requires_grad=True
                )
            else:
                loss = loss + smooth_lddt_loss

            if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                logger.error(
                    f"NaN or inf detected in loss: {loss}, molecule_noise_loss: {molecule_noise_loss}, periodic_noise_loss: {periodic_noise_loss}, protein_noise_loss: {protein_noise_loss}, complex_noise_loss: {complex_noise_loss}, noise_loss: {noise_loss}, aa_diff_loss: {aa_diff_loss}"
                )
                loss = torch.tensor(
                    0.0, device=atomic_numbers.device, requires_grad=True
                )
        else:
            loss = aa_diff_loss

        # for loss exist in every sample of the batch, no extra number of samples are recorded (will use batch size in loss reduction)
        # for loss does not exist in every example of the batch, use a tuple, where the first element is the averaged loss value
        # and the second element is the number of samples (or token numbers) in the batch with that loss considered
        logging_output = {
            "total_loss": loss.to(torch.float32),
            "noise_loss": (float(noise_loss), int(num_noise_sample)),
            "molecule_noise_loss": (
                float(molecule_noise_loss.detach()),
                int(num_molecule_noise_sample),
            ),
            "periodic_noise_loss": (
                float(periodic_noise_loss.detach()),
                int(num_periodic_noise_sample),
            ),
            "protein_noise_loss": (
                float(protein_noise_loss.detach()),
                int(num_protein_noise_sample),
            ),
            "complex_noise_loss": (
                float(complex_noise_loss.detach()),
                int(num_complex_noise_sample),
            ),
            "aa_diff_loss": (float(aa_diff_loss.detach()), int(num_aa_diff_token)),
            "smooth_lddt_loss": (float(smooth_lddt_loss.detach()), int(num_pddt_loss)),
            "hard_dist_loss": (float(hard_dist_loss.detach()), int(num_pddt_loss)),
            "inter_dist_loss": (
                float(inter_dist_loss.detach()),
                int(num_inter_dist_loss),
            ),
        }

        def _reduce_matched_result(model_output, metric_name, min_or_max: str):
            metric = model_output[metric_name]
            matched_mask = ~metric.isnan()
            num_matched_samples = int(torch.sum(matched_mask.long()))
            matched_rate = float(torch.mean(matched_mask.float()))
            if num_matched_samples > 0:
                mean_metric = float(
                    torch.sum(metric[matched_mask]) / num_matched_samples
                )
                metric_tuple = (mean_metric, num_matched_samples)
            else:
                metric_tuple = (0.0, 0)
            matched_rate_tuple = (matched_rate, metric.numel())
            torch_min_or_max_func = torch.min if min_or_max == "min" else torch.max
            if metric.size()[-1] > 1:
                any_matched_mask = ~metric.isnan().any(dim=-1)
                num_any_matched_samples = int(torch.sum(any_matched_mask))
                best_metric = torch_min_or_max_func(metric, dim=-1)[0]
                if num_any_matched_samples > 0:
                    best_metric = float(
                        torch.sum(best_metric[any_matched_mask])
                        / num_any_matched_samples
                    )
                    best_metric_tuple = (best_metric, num_any_matched_samples)
                else:
                    best_metric_tuple = (0.0, 0)
                any_matched_rate_tuple = (
                    float(torch.mean(any_matched_mask.float())),
                    int(metric.size()[0]),
                )
            else:
                best_metric_tuple = metric_tuple
                any_matched_rate_tuple = matched_rate_tuple
            return (
                metric_tuple,
                matched_rate_tuple,
                best_metric_tuple,
                any_matched_rate_tuple,
            )

        if "rmsd" in model_output:
            (
                logging_output["rmsd"],
                logging_output["matched_rate"],
                logging_output["min_rmsd"],
                logging_output["any_matched_rate"],
            ) = _reduce_matched_result(model_output, "rmsd", "min")
        if "tm_score" in model_output:
            (
                logging_output["tm_score"],
                _,
                logging_output["max_tm_score"],
                _,
            ) = _reduce_matched_result(model_output, "tm_score", "max")
        if "lddt" in model_output:
            (
                logging_output["lddt"],
                _,
                logging_output["max_lddt"],
                _,
            ) = _reduce_matched_result(model_output, "lddt", "max")

        return loss, logging_output
