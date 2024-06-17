# -*- coding: utf-8 -*-
from typing import Dict, Union

import torch
import torch.nn as nn

from sfm.models.psm.psmmodel import PSMConfig
from sfm.utils.register import Register


class PSMFinetuneBaseModule(nn.Module):
    def __init__(self, args: PSMConfig):
        super().__init__()
        self.args = args

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        return loss, logging_output


PSM_FT_REGISTER: Union[Dict[str, PSMFinetuneBaseModule.__class__], Register] = Register(
    "psm_finetine_module_register"
)


@PSM_FT_REGISTER.register("homo_lumo_gap_head")
class HomoLumoGapHead(PSMFinetuneBaseModule):
    def __init__(self, args: PSMConfig):
        super().__init__(args)

        embedding_dim = args.encoder_embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1, bias=True),
        )

    def update_batched_data(self, samples, batched_data):
        batched_data["homo_lumo_gap"] = torch.cat([x["homo_lumo_gap"] for x in samples])
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # TODO: check result_dict
        decoder_x_output = result_dict["decoder_x_output"]
        out = self.head(decoder_x_output)
        result_dict["homo_lumo_gap"] = (
            out.squeeze(-1).masked_fill(result_dict["non_atom_mask"], 0.0).sum(dim=-1)
            / result_dict["num_atoms"]
        )
        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        head_loss = nn.L1Loss()(
            model_output["homo_lumo_gap"], batched_data["homo_lumo_gap"]
        )
        loss += head_loss
        logging_output["homo_lumo_gap_loss"] = head_loss
        logging_output["total_loss"] = loss
        return loss, logging_output
