# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn

from sfm.logging import logger
from sfm.models.psm.psm_config import DiffusionTrainingLoss


class DiffMAE3dCriterions(nn.Module):
    def __init__(
        self,
        args,
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
        self.force_loss = nn.MSELoss(reduction="none")
        self.noise_loss = (
            nn.L1Loss(reduction="none")
            if self.args.diffusion_training_loss == DiffusionTrainingLoss.L1
            else nn.MSELoss(reduction="none")
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

        self.energy_loss_ratio = args.energy_loss_ratio
        self.force_loss_ratio = args.force_loss_ratio

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
                energy_loss[is_molecule] = (
                    energy_loss[is_molecule] * self.molecule_energy_per_atom_std
                )
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

    def forward(self, model_output, batched_data):
        energy_per_atom_label = batched_data["energy_per_atom"]
        atomic_numbers = batched_data["token_id"]
        noise_label = model_output["noise"]
        force_label = model_output["force_label"]

        if self.diffusion_mode == "x0":
            pos_label = batched_data["ori_pos"]
        force_pred = model_output["forces"]
        energy_per_atom_pred = model_output["energy_per_atom"]
        noise_pred = model_output["noise_pred"]
        non_atom_mask = model_output["non_atom_mask"]
        clean_mask = model_output["clean_mask"]
        aa_mask = model_output["aa_mask"]
        is_protein = model_output["is_protein"]
        is_molecule = model_output["is_molecule"]
        is_periodic = model_output["is_periodic"]
        is_seq_only = model_output["is_seq_only"]
        diff_loss_mask = model_output["diff_loss_mask"]
        protein_mask = model_output["protein_mask"]
        # sqrt_one_minus_alphas_cumprod_t = model_output["sqrt_one_minus_alphas_cumprod_t"]

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
            # energy loss and force loss
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

            if self.diffusion_mode == "epsilon":
                # noise pred loss
                unreduced_noise_loss = self.noise_loss(
                    noise_pred.to(noise_label.dtype), noise_label
                )
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
                ~clean_mask & is_molecule.unsqueeze(-1),
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
                ~clean_mask & is_periodic.unsqueeze(-1),
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
                ~clean_mask & is_protein.unsqueeze(-1) & ~is_seq_only.unsqueeze(-1),
                diff_loss_mask & ~protein_mask.any(dim=-1),
                is_molecule,
                is_periodic,
                1.0,
                1.0,
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

        def calculate_energy_loss_ratio(energy_loss_mag):
            return np.clip(1.0 - (energy_loss_mag - 1.0) / 1000, 0.001, 1.0)

        if not self.seq_only:
            loss = (
                self.energy_loss_ratio * energy_loss
                + self.force_loss_ratio * force_loss
                + noise_loss
                + aa_mlm_loss
            )
        else:
            loss = aa_mlm_loss

        # for loss exist in every sample of the batch, no extra number of samples are recorded (will use batch size in loss reduction)
        # for loss does not exist in every example of the batch, use a tuple, where the first element is the averaged loss value
        # and the second element is the number of samples (or token numbers) in the batch with that loss considered
        logging_output = {
            "total_loss": loss,
            "energy_loss": (float(energy_loss), int(num_energy_sample)),
            "molecule_energy_loss": (
                float(molecule_energy_loss),
                int(num_molecule_energy_sample),
            ),
            "periodic_energy_loss": (
                float(periodic_energy_loss),
                int(num_periodic_energy_sample),
            ),
            "force_loss": (float(force_loss), int(num_force_sample)),
            "molecule_force_loss": (
                float(molecule_force_loss),
                int(num_molecule_force_sample),
            ),
            "periodic_force_loss": (
                float(periodic_force_loss),
                int(num_periodic_force_sample),
            ),
            "noise_loss": (float(noise_loss), int(num_noise_sample)),
            "molecule_noise_loss": (
                float(molecule_noise_loss),
                int(num_molecule_noise_sample),
            ),
            "periodic_noise_loss": (
                float(periodic_noise_loss),
                int(num_periodic_noise_sample),
            ),
            "protein_noise_loss": (
                float(protein_noise_loss),
                int(num_protein_noise_sample),
            ),
            "aa_mlm_loss": (float(aa_mlm_loss), int(num_aa_mask_token)),
            "aa_acc": (float(aa_acc), int(num_aa_mask_token)),
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

        return loss, logging_output
