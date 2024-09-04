# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import deprecated

from sfm.logging import logger
from sfm.models.psm.psm_config import DiffusionTrainingLoss, ForceLoss, PSMConfig


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
    ) -> None:
        super().__init__()
        self.args = args

        self.diffusion_mode = args.diffusion_mode
        self.seq_only = args.seq_only

        self.energy_loss = nn.L1Loss(reduction="none")

        if self.args.force_loss_type == ForceLoss.L1:
            self.force_loss = nn.L1Loss(reduction="none")
        elif self.args.force_loss_type == ForceLoss.MSE:
            self.force_loss = nn.MSELoss(reduction="none")
        elif self.args.force_loss_type == ForceLoss.SmoothL1:
            self.force_loss = nn.SmoothL1Loss(reduction="none")
        elif self.args.force_loss_type == ForceLoss.NoiseTolerentL1:
            self.force_loss = NoiseTolerentL1Loss(noise_tolerance=3.0, reduction="none")
        else:
            raise ValueError(f"Invalid force loss type: {self.args.force_loss_type}")

        if self.args.diffusion_training_loss == DiffusionTrainingLoss.L1:
            self.noise_loss = nn.L1Loss(reduction="none")
        elif self.args.diffusion_training_loss == DiffusionTrainingLoss.MSE:
            self.noise_loss = nn.MSELoss(reduction="none")
        elif self.args.diffusion_training_loss == DiffusionTrainingLoss.SmoothL1:
            self.noise_loss = nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(
                f"Invalid diffusion training loss type: {self.args.diffusion_training_loss}"
            )

        self.aa_mlm_loss = nn.CrossEntropyLoss(reduction="mean")

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

        self.material_force_loss_ratio = args.material_force_loss_ratio
        self.material_energy_loss_ratio = args.material_energy_loss_ratio
        self.molecule_force_loss_ratio = args.molecule_force_loss_ratio
        self.molecule_energy_loss_ratio = args.molecule_energy_loss_ratio

        self.hard_dist_loss_raito = args.hard_dist_loss_raito
        self.if_total_energy = args.if_total_energy

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
                ) / (3.0 * torch.sum(token_mask[sample_mask], dim=-1))
            elif len(sample_mask.shape) == 2:
                # TODO: need to average over tokens in one sample first then all smaples
                force_or_noise_loss = torch.sum(
                    force_or_noise_loss[sample_mask], dim=[0, 1]
                ) / (3.0 * torch.sum(token_mask[sample_mask], dim=-1))
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
    def _alignment_x0(self, model_output, pos_pred):
        pos_label = model_output["ori_pos"]

        R, T = svd_superimpose(
            pos_pred.float(),
            pos_label.float(),
            model_output["padding_mask"] | model_output["protein_mask"].any(dim=-1),
        )

        return R, T

    def dist_loss(self, model_output, R, T, pos_pred):
        # calculate aligned pred pos
        pos_pred = torch.einsum("bij,bkj->bki", R.float(), pos_pred.float()) + T.float()

        # smooth lddt loss
        pos_label = model_output["ori_pos"]
        B, L = pos_label.shape[:2]

        # make is_protein mask contain ligand in complex data
        is_protein = model_output["is_protein"].any(dim=-1).unsqueeze(-1).repeat(1, L)
        is_protein = is_protein & (~model_output["padding_mask"])

        delta_pos_label = (pos_label.unsqueeze(1) - pos_label.unsqueeze(2)).norm(dim=-1)
        delta_pos_pred = (pos_pred.unsqueeze(1) - pos_pred.unsqueeze(2)).norm(dim=-1)
        pair_protein_mask = is_protein.unsqueeze(1) & is_protein.unsqueeze(2)
        dist_mask = (delta_pos_label < 15) & (delta_pos_label > 0.1) & pair_protein_mask
        delta = torch.abs(delta_pos_label - delta_pos_pred)
        delta1 = delta[dist_mask]
        error = 0.25 * (
            torch.sigmoid(0.5 - delta1)
            + torch.sigmoid(1 - delta1)
            + torch.sigmoid(2 - delta1)
            + torch.sigmoid(4 - delta1)
        )
        lddt = error.mean()

        # # hard distance loss
        time_step = model_output["time_step"]

        time_coefficient = ((1 - time_step) * torch.exp(-time_step / 0.4)).unsqueeze(-1)
        hard_dist_loss = (delta * time_coefficient)[dist_mask].mean()

        # time_mask = time_step < 0.1
        # pair_time_mask = time_mask.unsqueeze(1) & time_mask.unsqueeze(2)
        # hard_dist_mask = dist_mask & pair_time_mask
        # if hard_dist_mask.any():
        #     hard_dist_loss = delta[hard_dist_mask].mean()
        # else:
        #     hard_dist_loss = torch.tensor(0.0, device=delta.device, requires_grad=True)

        return 1 - lddt, hard_dist_loss

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
        noise_label = model_output["noise"]
        force_label = model_output["force_label"]
        pos_label = model_output["ori_pos"]
        force_pred = model_output["forces"]
        autograd_force_pred = (
            model_output["autograd_forces"]
            if "autograd_forces" in model_output
            else None
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

        n_graphs = energy_per_atom_label.size()[0]
        if clean_mask is None:
            clean_mask = torch.zeros(
                n_graphs, dtype=torch.bool, device=energy_per_atom_label.device
            )

        # energy and force loss only apply on total clean samples
        total_clean = clean_mask.all(dim=-1)
        energy_mask = total_clean & batched_data["has_energy"]
        force_mask = total_clean & batched_data["has_forces"]

        if not self.seq_only:
            # diffussion loss
            if self.diffusion_mode == "epsilon":
                if not is_seq_only.any():
                    pos_pred = self.calculate_pos_pred(model_output)
                    if self.args.align_x0_in_diffusion_loss:
                        R, T = self._alignment_x0(model_output, pos_pred)
                    else:
                        R, T = torch.eye(
                            3, device=pos_pred.device, dtype=pos_pred.dtype
                        ).unsqueeze(0).repeat(n_graphs, 1, 1), torch.zeros_like(
                            pos_pred, device=pos_pred.device, dtype=pos_pred.dtype
                        )

                    if is_protein.any():
                        # align pred pos and calculate smooth lddt loss for protein
                        smooth_lddt_loss, hard_dist_loss = self.dist_loss(
                            model_output, R, T, pos_pred
                        )
                        num_pddt_loss = 1
                    else:
                        smooth_lddt_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        hard_dist_loss = torch.tensor(
                            0.0, device=noise_label.device, requires_grad=True
                        )
                        num_pddt_loss = 0

                    if self.args.align_x0_in_diffusion_loss:
                        # noise pred loss
                        aligned_noise_pred = (
                            sqrt_alphas_cumprod_t * pos_label
                            + sqrt_one_minus_alphas_cumprod_t
                            * (noise_label - noise_pred)
                        )
                        aligned_noise_pred = torch.einsum(
                            "bij,bkj->bki", R, aligned_noise_pred.float()
                        )
                        unreduced_noise_loss = self.noise_loss(
                            aligned_noise_pred.to(noise_label.dtype),
                            pos_label * sqrt_alphas_cumprod_t,
                        )
                        unreduced_noise_loss = (
                            unreduced_noise_loss / sqrt_one_minus_alphas_cumprod_t
                        )
                    else:
                        unreduced_noise_loss = self.noise_loss(
                            noise_pred.to(noise_label.dtype), noise_label
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
                    num_pddt_loss = 0

            elif self.diffusion_mode == "x0":
                # x0 pred loss, noise pred is x0 pred here
                unreduced_noise_loss = self.noise_loss(
                    noise_pred.to(noise_label.dtype), pos_label
                )
            else:
                raise ValueError(f"Invalid diffusion mode: {self.diffusion_mode}")

            noise_loss, num_noise_sample = self._reduce_force_or_noise_loss(
                unreduced_noise_loss,
                ~clean_mask,
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
                (~clean_mask) & is_molecule.unsqueeze(-1),
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
                diff_loss_mask & ~protein_mask.any(dim=-1),
                is_molecule,
                is_periodic,
                1.0,
                1.0,
            )

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
        else:
            aa_mlm_loss = torch.tensor(
                0.0, device=atomic_numbers.device, requires_grad=True
            )
            aa_acc = 0.0
            num_aa_mask_token = 0.0

        if not self.seq_only:
            loss = (
                self.molecule_energy_loss_ratio * molecule_energy_loss
                + self.molecule_force_loss_ratio * molecule_force_loss
                + self.material_energy_loss_ratio * periodic_energy_loss
                + self.material_force_loss_ratio * periodic_force_loss
                + autograd_force_loss
                + noise_loss
                + aa_mlm_loss
                + smooth_lddt_loss
            )
            if self.args.use_hard_dist_loss:
                loss += self.hard_dist_loss_raito * hard_dist_loss
        else:
            loss = aa_mlm_loss

        # for loss exist in every sample of the batch, no extra number of samples are recorded (will use batch size in loss reduction)
        # for loss does not exist in every example of the batch, use a tuple, where the first element is the averaged loss value
        # and the second element is the number of samples (or token numbers) in the batch with that loss considered
        logging_output = {
            "total_loss": loss,
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
            "aa_acc": (float(aa_acc), int(num_aa_mask_token)),
            "smooth_lddt_loss": (float(smooth_lddt_loss.detach()), int(num_pddt_loss)),
            "hard_dist_loss": (float(hard_dist_loss.detach()), int(num_pddt_loss)),
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
