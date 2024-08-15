# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import DiffusionTrainingLoss


class RLreward(DiffMAE3dCriterions):
    def __init__(self, args) -> None:
        super().__init__(args)

    def forward(self, model_output, batched_data):
        pos_pred = batched_data["pred_pos"]
        pos_label = batched_data["ori_pos"]

        protein_mask = model_output["protein_mask"]

        R, T = self._alignment_x0(model_output, batched_data, pos_pred)

        pos_pred = torch.einsum("bij,bkj->bki", R, pos_pred.float())
        unreduced_noise_loss = self.noise_loss(pos_pred.to(pos_label.dtype), pos_label)

        protein_mask = protein_mask.any(dim=-1).unsqueeze(-1)
        unreduced_noise_loss = unreduced_noise_loss.masked_fill(protein_mask, 0.0)
        unreduced_noise_loss = unreduced_noise_loss.sum(dim=-1).sum(dim=-1)
        selected_count = (~protein_mask).sum(dim=-1).sum(dim=-1)
        unreduced_noise_loss = unreduced_noise_loss / selected_count.float()
        unreduced_noise_loss[selected_count == 0] = 0.0

        # unreduced_noise_loss = unreduced_noise_loss.mean(dim=(1, 2))

        return torch.exp(-unreduced_noise_loss)
