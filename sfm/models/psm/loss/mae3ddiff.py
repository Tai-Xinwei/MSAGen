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

        self.energy_loss = nn.L1Loss(reduction="mean")
        self.force_loss = nn.L1Loss(reduction="none")
        self.noise_loss = (
            nn.L1Loss(reduction="none")
            if self.args.diffusion_training_loss == DiffusionTrainingLoss.L1
            else nn.MSELoss(reduction="none")
        )

    def forward(self, model_output, batched_data):
        force_label = batched_data["forces"]
        energy_label = batched_data["y"]
        atomic_numbers = batched_data["token_id"]
        noise_label = model_output["noise"]
        force_pred = model_output["forces"]
        energy_pred = model_output["energy"]
        noise_pred = model_output["noise_pred"]
        non_atom_mask = model_output["non_atom_mask"]
        clean_mask = model_output["clean_mask"]

        n_graphs = energy_label.size()[0]
        if clean_mask is None:
            clean_mask = torch.zeros(
                n_graphs, dtype=torch.bool, device=energy_label.device
            )
        n_clean_graphs = torch.sum(clean_mask.to(dtype=torch.long))
        n_corrupted_graphs = n_graphs - n_clean_graphs
        padding_mask = atomic_numbers.eq(0)
        if n_clean_graphs > 0:
            energy_loss = self.energy_loss(
                energy_pred[clean_mask], energy_label[clean_mask]
            )
            force_loss = (
                self.force_loss(force_pred.to(dtype=force_label.dtype), force_label)
                .masked_fill(non_atom_mask.unsqueeze(-1), 0.0)
                .sum(dim=[1, 2])
                / (batched_data["num_atoms"] * 3)
            )[clean_mask].mean()
        else:
            energy_loss = 0.0
            force_loss = 0.0

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
            noise_loss = 0.0

        loss = energy_loss + force_loss + noise_loss

        # TODO: log losses by periodic, molecule and protein systems, respectively
        logging_output = {
            "total_loss": loss,
            "energy_loss": energy_loss,
            "force_loss": force_loss,
            "noise_loss": noise_loss,
        }

        return loss, logging_output
