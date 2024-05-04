# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

# from sklearn.metrics import roc_auc_score
import torch.nn as nn

from sfm.models.psm.psm_config import DiffusionTrainingLoss


class DiffMAE3dCriterions(nn.Module):
    def __init__(
        self,
        args,
    ) -> None:
        super().__init__()
        self.args = args

        self.energy_loss = nn.L1Loss(reduction="none")
        self.force_loss = nn.L1Loss(reduction="none")
        self.noise_loss = (
            nn.L1Loss(reduction="none")
            if self.args.diffusion_training_loss == DiffusionTrainingLoss.L1
            else nn.MSELoss(reduction="none")
        )
        self.aa_mlm_loss = nn.CrossEntropyLoss(reduction="mean")

    def _reduce_energy_loss(self, energy_loss, loss_mask):
        num_samples = torch.sum(loss_mask.long())
        if num_samples > 0:
            energy_loss = torch.mean(energy_loss[loss_mask])
        else:
            energy_loss = torch.tensor(
                0.0, device=energy_loss.device, requires_grad=True
            )
        return energy_loss, num_samples

    def _reduce_force_or_noise_loss(self, force_or_noise_loss, sample_mask, token_mask):
        sample_mask &= token_mask.any(dim=-1)
        num_samples = torch.sum(sample_mask.long())
        if num_samples > 0:
            force_or_noise_loss = force_or_noise_loss.masked_fill(
                ~token_mask.unsqueeze(-1), 0.0
            )
            force_or_noise_loss = torch.sum(
                force_or_noise_loss[sample_mask], dim=[1, 2]
            ) / (3.0 * torch.sum(token_mask[sample_mask], dim=-1))
            force_or_noise_loss = force_or_noise_loss.mean()
        else:
            force_or_noise_loss = torch.tensor(
                0.0, device=force_or_noise_loss.device, requires_grad=True
            )
        return force_or_noise_loss, num_samples

    def forward(self, model_output, batched_data):
        force_label = batched_data["forces"]
        energy_label = batched_data["energy"]
        atomic_numbers = batched_data["token_id"]
        noise_label = model_output["noise"]
        force_pred = model_output["forces"]
        energy_pred = model_output["energy"]
        noise_pred = model_output["noise_pred"]
        non_atom_mask = model_output["non_atom_mask"]
        clean_mask = model_output["clean_mask"]
        aa_mask = model_output["aa_mask"]
        is_protein = model_output["is_protein"]
        is_molecule = model_output["is_molecule"]
        is_periodic = model_output["is_periodic"]
        diff_loss_mask = model_output["diff_loss_mask"]
        protein_mask = model_output["protein_mask"]

        n_graphs = energy_label.size()[0]
        if clean_mask is None:
            clean_mask = torch.zeros(
                n_graphs, dtype=torch.bool, device=energy_label.device
            )
        padding_mask = atomic_numbers.eq(0)
        energy_mask = clean_mask & ~is_protein
        force_mask = clean_mask & is_periodic

        # energy loss and force loss
        unreduced_energy_loss = self.energy_loss(
            energy_pred.to(torch.float32),
            energy_label.to(torch.float32),
        )
        energy_loss, num_energy_sample = self._reduce_energy_loss(
            unreduced_energy_loss, energy_mask
        )
        molecule_energy_loss, num_molecule_energy_sample = self._reduce_energy_loss(
            unreduced_energy_loss, energy_mask & is_molecule
        )
        periodic_energy_loss, num_periodic_energy_sample = self._reduce_energy_loss(
            unreduced_energy_loss, energy_mask & is_periodic
        )

        unreduced_force_loss = self.force_loss(
            force_pred.to(dtype=force_label.dtype), force_label
        )
        force_loss, num_force_sample = self._reduce_force_or_noise_loss(
            unreduced_force_loss, force_mask, ~non_atom_mask
        )
        (
            molecule_force_loss,
            num_molecule_force_sample,
        ) = self._reduce_force_or_noise_loss(
            unreduced_force_loss, force_mask & is_molecule, ~non_atom_mask
        )
        (
            periodic_force_loss,
            num_periodic_force_sample,
        ) = self._reduce_force_or_noise_loss(
            unreduced_force_loss, force_mask & is_periodic, ~non_atom_mask
        )

        # noise pred loss
        unreduced_noise_loss = self.noise_loss(
            noise_pred.to(noise_label.dtype), noise_label
        )
        noise_loss, num_noise_sample = self._reduce_force_or_noise_loss(
            unreduced_noise_loss,
            ~clean_mask,
            diff_loss_mask & ~protein_mask.any(dim=-1),
        )
        (
            molecule_noise_loss,
            num_molecule_noise_sample,
        ) = self._reduce_force_or_noise_loss(
            unreduced_noise_loss,
            ~clean_mask & is_molecule,
            diff_loss_mask & ~protein_mask.any(dim=-1),
        )
        (
            periodic_noise_loss,
            num_periodic_noise_sample,
        ) = self._reduce_force_or_noise_loss(
            unreduced_noise_loss,
            ~clean_mask & is_periodic,
            diff_loss_mask & ~protein_mask.any(dim=-1),
        )
        protein_noise_loss, num_protein_noise_sample = self._reduce_force_or_noise_loss(
            unreduced_noise_loss,
            ~clean_mask & is_protein,
            diff_loss_mask & ~protein_mask.any(dim=-1),
        )

        # mlm loss
        if aa_mask.any():
            aa_mask = aa_mask & ~padding_mask
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

        loss = energy_loss + force_loss + noise_loss + aa_mlm_loss

        # for loss exist in every sample of the batch, no extra number of samples are recorded (will use batch size in loss reduction)
        # for loss does not exist in every example of the batch, use a tuple, where the first element is the averaged loss value
        # and the second element is the number of samples (or token numbers) in the batch with that loss considered
        logging_output = {
            "total_loss": loss,
            "energy_loss": (energy_loss, num_energy_sample),
            "molecule_energy_loss": (molecule_energy_loss, num_molecule_energy_sample),
            "periodic_energy_loss": (periodic_energy_loss, num_periodic_energy_sample),
            "force_loss": (force_loss, num_force_sample),
            "molecule_force_loss": (molecule_force_loss, num_molecule_force_sample),
            "periodic_force_loss": (periodic_force_loss, num_periodic_force_sample),
            "noise_loss": (noise_loss, num_noise_sample),
            "molecule_noise_loss": (molecule_noise_loss, num_molecule_noise_sample),
            "periodic_noise_loss": (periodic_noise_loss, num_periodic_noise_sample),
            "protein_noise_loss": (protein_noise_loss, num_protein_noise_sample),
            "aa_mlm_loss": (aa_mlm_loss, num_aa_mask_token),
            "aa_acc": (aa_acc, num_aa_mask_token),
        }

        return loss, logging_output
