# -*- coding: utf-8 -*-
from typing import Dict, Union

import torch
import torch.nn as nn

from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.psmmodel import PSMConfig
from sfm.utils.register import Register

kcalmol_to_ev = 0.0433634


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


@PSM_FT_REGISTER.register("md_energy_force_head")
class MDEnergyForceHead(PSMFinetuneBaseModule):
    def __init__(self, args: PSMConfig):
        super().__init__(args)

        self.energy_loss_weight = args.energy_loss_weight
        self.force_loss_weight = args.force_loss_weight

        embedding_dim = args.encoder_embed_dim
        self.energy_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1, bias=True),
        )
        self.force_head = EquivariantVectorOutput(embedding_dim)

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        decoder_x_output = result_dict["decoder_x_output"]
        decoder_vec_output = result_dict["decoder_vec_output"]
        energy_out = self.energy_head(decoder_x_output).squeeze(-1)
        result_dict["pred_energy"] = energy_out.masked_fill(
            result_dict["non_atom_mask"], 0.0
        ).sum(dim=-1)
        force_out = self.force_head(decoder_x_output, decoder_vec_output).squeeze(-1)
        expanded_mask = result_dict["non_atom_mask"].unsqueeze(-1).expand_as(force_out)
        result_dict["pred_forces"] = force_out.masked_fill(expanded_mask, 0.0)
        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        e_pred = model_output["pred_energy"]
        e_true = batched_data["energy"]
        f_pred = model_output["pred_forces"]
        f_true = batched_data["forces"]

        e_loss = torch.mean(torch.abs(e_pred - e_true))
        f_loss = torch.mean(torch.abs(f_pred - f_true))

        if self.args.loss_unit == "kcal/mol":
            e_loss /= kcalmol_to_ev
            f_loss /= kcalmol_to_ev

        size = e_true.shape[0]

        loss = self.energy_loss_weight * e_loss + self.force_loss_weight * f_loss
        logging_output = {
            "loss": loss,
            "energy_loss": (e_loss, size),
            "force_loss": (f_loss, size),
        }
        return loss, logging_output
