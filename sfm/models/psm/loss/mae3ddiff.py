# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

# from sklearn.metrics import roc_auc_score
import torch.distributed as dist
import torch.nn as nn

from sfm.models.psm.psm_config import DiffusionTrainingLoss, PSMConfig


class DiffMAE3dCriterions(nn.Module):
    def __init__(
        self,
        args,
    ) -> None:
        super().__init__()
        self.args = args

        self.energy_loss = nn.L1Loss(reduction="mean")
        self.force_loss = nn.L1Loss(reduction="none")
        self.noise_loss = (
            nn.L1Loss(reduction="none")
            if self.args.diffusion_training_loss == DiffusionTrainingLoss.L1
            else nn.MSELoss(reduction="none")
        )
        self.aa_mlm_loss = nn.CrossEntropyLoss(reduction="mean")

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
        # is_molecule = model_output["is_molecule"]
        is_periodic = model_output["is_periodic"]

        n_graphs = energy_label.size()[0]
        if clean_mask is None:
            clean_mask = torch.zeros(
                n_graphs, dtype=torch.bool, device=energy_label.device
            )
        n_clean_graphs = torch.sum(clean_mask.to(dtype=torch.long))
        n_corrupted_graphs = n_graphs - n_clean_graphs
        padding_mask = atomic_numbers.eq(0)
        energy_mask = clean_mask & ~is_protein
        force_mask = clean_mask & is_periodic

        if n_clean_graphs > 0:
            energy_loss = self.energy_loss(
                energy_pred[energy_mask].to(torch.float32),
                energy_label[energy_mask].to(torch.float32),
            )
            if force_mask.any():
                force_loss = (
                    self.force_loss(force_pred.to(dtype=force_label.dtype), force_label)
                    .masked_fill(non_atom_mask.unsqueeze(-1), 0.0)
                    .sum(dim=[1, 2])
                    / (batched_data["num_atoms"] * 3)
                )[force_mask].mean()
            else:
                force_loss = torch.tensor(
                    [0.0], device=atomic_numbers.device, requires_grad=True
                )
        else:
            energy_loss = torch.tensor(
                [0.0], device=atomic_numbers.device, requires_grad=True
            )
            force_loss = torch.tensor(
                [0.0], device=atomic_numbers.device, requires_grad=True
            )

        if n_corrupted_graphs > 0:
            protein_mask = model_output["protein_mask"]
            noise_loss = (
                self.noise_loss(noise_pred.to(dtype=noise_label.dtype), noise_label)
                .masked_fill(padding_mask.unsqueeze(-1) | protein_mask, 0.0)
                .sum(dim=[1, 2])
                / (
                    torch.sum(
                        (~(padding_mask | protein_mask.any(dim=-1))).to(
                            dtype=noise_label.dtype
                        ),
                        dim=-1,
                        keepdim=True,
                    )
                    * 3
                )
            )[~clean_mask].mean()
        else:
            noise_loss = torch.tensor(
                [0.0], device=atomic_numbers.device, requires_grad=True
            )

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
        else:
            aa_mlm_loss = torch.tensor(
                [0.0], device=atomic_numbers.device, requires_grad=True
            )
            aa_acc = 0.0

        loss = energy_loss + force_loss + noise_loss + aa_mlm_loss
        # print(f"loss: {loss}, energy_loss: {energy_loss}, force_loss: {force_loss}, noise_loss: {noise_loss}, aa_mlm_loss: {aa_mlm_loss}, aa_acc: {aa_acc}")

        # TODO: log losses by periodic, molecule and protein systems, respectively
        logging_output = {
            "total_loss": loss,
            "energy_loss": energy_loss,
            "force_loss": force_loss,
            "noise_loss": noise_loss,
            "aa_mlm_loss": aa_mlm_loss,
            "aa_acc": aa_acc,
        }

        return loss, logging_output
