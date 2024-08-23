# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import DiffusionTrainingLoss


class RLreward(DiffMAE3dCriterions):
    def __init__(self, args, reward_model: str = "rmsd") -> None:
        super().__init__(args)

        self.reward_model = reward_model

    def forward(self, model_output, batched_data):
        pos_pred = batched_data["pred_pos"]
        pos_label = batched_data["ori_pos"]

        protein_mask = model_output["protein_mask"]
        loss_mask = (protein_mask.any(dim=-1) | model_output["padding_mask"]).unsqueeze(
            -1
        )

        if self.reward_model == "rmsd":
            R, T = self._alignment_x0(model_output, pos_pred)
            pos_label = torch.einsum("bij,bkj->bki", R, pos_label.float())  # + T

            unreduced_noise_loss = torch.sqrt(((pos_pred - pos_label) ** 2))
        elif self.reward_model == "lddt":
            # use smoothed lddt as reward
            is_protein = batched_data["is_protein"]

            dist_label = (pos_label.unsqueeze(1) - pos_label.unsqueeze(2)).norm(dim=-1)
            dist_pred = (pos_pred.unsqueeze(1) - pos_pred.unsqueeze(2)).norm(dim=-1)

            pair_protein_mask = is_protein.unsqueeze(1) & is_protein.unsqueeze(2)
            dist_mask = (dist_label < 15) & (dist_label > 0.1) & pair_protein_mask
            delta = torch.abs(dist_label - dist_pred)
            delta1 = delta[dist_mask]
            unreduced_noise_loss = 0.25 * (
                torch.sigmoid(0.5 - delta1)
                + torch.sigmoid(1 - delta1)
                + torch.sigmoid(2 - delta1)
                + torch.sigmoid(4 - delta1)
            )
        else:
            raise ValueError(f"reward model {self.reward_model} not supported")

        unreduced_noise_loss = unreduced_noise_loss.masked_fill(loss_mask, 0.0)
        unreduced_noise_loss = unreduced_noise_loss.sum(dim=-1).sum(dim=-1)
        selected_count = (~loss_mask).sum(dim=-1).sum(dim=-1)
        unreduced_noise_loss = unreduced_noise_loss / selected_count.float()
        unreduced_noise_loss = (
            1.0 - unreduced_noise_loss
            if self.reward_model == "lddt"
            else unreduced_noise_loss
        )
        unreduced_noise_loss[selected_count == 0] = 0.0

        return torch.exp(-unreduced_noise_loss)
