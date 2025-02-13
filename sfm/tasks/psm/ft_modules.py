# -*- coding: utf-8 -*-
from math import log
from typing import Dict, Union

import torch
import torch.nn as nn
from sympy import comp

from sfm.models.psm.equivariant.geomformer import EquivariantVectorOutput
from sfm.models.psm.invariant.dit_encoder import DiTBlock
from sfm.models.psm.modules.autograd import GradientHead
from sfm.models.psm.modules.confidence_model import (
    compute_pde,
    compute_plddt,
    lddt_loss,
    pde_loss,
)
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
        self.auto_grad = args.AutoGradForce

        embedding_dim = args.encoder_embed_dim
        self.energy_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1, bias=True),
        )

        if not self.auto_grad:
            self.force_head = EquivariantVectorOutput(embedding_dim)
        else:
            self.force_head = GradientHead()

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        decoder_x_output = result_dict["decoder_x_output"]
        decoder_vec_output = result_dict["decoder_vec_output"]
        energy_out = self.energy_head(decoder_x_output).squeeze(-1)
        energy_out = energy_out.masked_fill(result_dict["non_atom_mask"], 0.0)
        result_dict["pred_energy"] = energy_out.sum(dim=-1)
        if not self.auto_grad:
            force_out = self.force_head(decoder_x_output, decoder_vec_output).squeeze(
                -1
            )
        else:
            force_out = self.force_head(
                energy_out,
                result_dict["non_atom_mask"],
                result_dict["pos"],
                result_dict["is_periodic"],
                result_dict["is_molecule"],
            )

        expanded_mask = result_dict["non_atom_mask"].unsqueeze(-1).expand_as(force_out)
        result_dict["pred_forces"] = force_out.masked_fill(expanded_mask, 0.0)
        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        e_pred = model_output["pred_energy"]
        e_true = batched_data["energy"]
        f_pred = model_output["pred_forces"]
        f_true = batched_data["forces"]

        if self.args.loss_unit != "ev":
            e_true /= kcalmol_to_ev
            f_true /= kcalmol_to_ev

        e_loss = torch.mean(torch.abs(e_pred - e_true))
        f_loss = torch.mean(torch.abs(f_pred - f_true))
        # f_loss_mse = torch.mean(torch.abs(f_pred - f_true)**2)

        size = e_true.shape[0]

        loss = self.energy_loss_weight * e_loss + self.force_loss_weight * f_loss

        logging_output = {
            "loss": loss,
            "energy_loss": (e_loss, size),
            "force_loss": (f_loss, size),
            # "force_loss_mse": (f_loss_mse, size),
        }
        return loss, logging_output


@PSM_FT_REGISTER.register("md_energy_force_multi_head")
class MDEnergyForceMultiHead(PSMFinetuneBaseModule):
    def __init__(self, args: PSMConfig):
        super().__init__(args)

        self.energy_loss_weight = args.energy_loss_weight
        self.force_loss_weight = args.force_loss_weight
        self.auto_grad = args.AutoGradForce

        embedding_dim = args.encoder_embed_dim

        # Initialize multiple energy and force heads based on dataset_name_list
        self.energy_heads = nn.ModuleDict()
        self.force_heads = nn.ModuleDict()

        for dataset_name in args.dataset_name_list.split(","):
            # Create energy head for each dataset_name
            if dataset_name not in ["deshaw_400", "deshaw_650"]:
                self.energy_heads[dataset_name] = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim, bias=True),
                    nn.SiLU(),
                    nn.Linear(embedding_dim, 1, bias=True),
                )

                # If dataset_name is not "deshaw_400" or "deshaw_650", create force head
                if not self.auto_grad:
                    self.force_heads[dataset_name] = EquivariantVectorOutput(
                        embedding_dim
                    )
                else:
                    self.force_heads[dataset_name] = GradientHead()

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        decoder_x_output = result_dict["decoder_x_output"]
        decoder_vec_output = result_dict["decoder_vec_output"]
        dataset_name = result_dict["data_name"][0]

        # Use the correct energy head based on dataset_name
        energy_head = (
            dataset_name
            if dataset_name not in ["deshaw_400", "deshaw_650"]
            else "deshaw_120"
        )
        energy_out = self.energy_heads[energy_head](decoder_x_output).squeeze(-1)
        energy_out = energy_out.masked_fill(result_dict["non_atom_mask"], 0.0)
        result_dict["pred_energy"] = energy_out.sum(dim=-1)

        # If dataset_name is not "deshaw_400" or "deshaw_650", process force head
        if dataset_name not in ["deshaw_400", "deshaw_650"]:
            if not self.auto_grad:
                force_out = self.force_heads[dataset_name](
                    decoder_x_output, decoder_vec_output
                ).squeeze(-1)
            else:
                force_out = self.force_heads[dataset_name](
                    energy_out,
                    result_dict["non_atom_mask"],
                    result_dict["pos"],
                    result_dict["is_periodic"],
                    result_dict["is_molecule"],
                )

            expanded_mask = (
                result_dict["non_atom_mask"].unsqueeze(-1).expand_as(force_out)
            )
            result_dict["pred_forces"] = force_out.masked_fill(expanded_mask, 0.0)

        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        e_pred = model_output["pred_energy"]
        e_true = batched_data["energy"]
        f_pred = model_output.get("pred_forces", None)
        f_true = batched_data.get("forces", None)

        if self.args.loss_unit != "ev":
            e_true /= kcalmol_to_ev
            if f_true is not None:
                f_true /= kcalmol_to_ev

        e_loss = torch.mean(torch.abs(e_pred - e_true))
        loss = self.energy_loss_weight * e_loss

        if f_pred is not None and f_true is not None:
            f_loss = torch.mean(torch.abs(f_pred - f_true))
            loss += self.force_loss_weight * f_loss
        else:
            f_loss = None

        size = e_true.shape[0]

        logging_output = {
            "loss": loss,
            "energy_loss": (e_loss, size),
        }

        if f_loss is not None:
            logging_output["force_loss"] = (f_loss, size)

        return loss, logging_output


@PSM_FT_REGISTER.register("plddt_confidence_head")
class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, args: PSMConfig, no_bins: int = 50, c_hidden: int = 128):
        super(PerResidueLDDTCaPredictor, self).__init__()
        psm_config = PSMConfig(args)

        self.no_bins = no_bins
        self.c_in = args.encoder_embed_dim
        self.c_hidden = c_hidden
        self.n_head = 8

        self.linear_s = nn.Linear(self.c_in, self.c_hidden)
        self.linear_c = nn.Linear(self.c_in, self.c_hidden)
        self.linear_z = nn.Linear(args.encoder_pair_embed_dim, self.c_hidden)

        self.dist_proj = nn.Sequential(
            nn.Linear(1, self.c_hidden),
            nn.SiLU(),
            nn.Linear(self.c_hidden, 1),
        )

        self.layers = nn.ModuleList([])
        for _ in range(4):
            self.layers.extend(
                [
                    DiTBlock(
                        args,
                        psm_config,
                        embedding_dim=c_hidden,
                        ffn_embedding_dim=c_hidden,
                        num_attention_heads=self.n_head,
                    )
                ]
            )

        self.proj_s = nn.Linear(self.c_hidden, self.no_bins)
        self.proj_z = nn.Linear(self.c_hidden, 64)

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s = self.linear_s(result_dict["decoder_x_output_sample"])
        c = self.linear_c(result_dict["encoder_output"])
        # z = self.linear_z(result_dict["x_pair"])

        pos_pred = result_dict["pred_pos_sample"]

        dist = (pos_pred.unsqueeze(1) - pos_pred.unsqueeze(2)).norm(dim=-1).to(s.dtype)
        dist = self.dist_proj(dist.unsqueeze(-1)).squeeze(-1)
        attn_bias = dist.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        for layer in self.layers:
            s = layer(
                s,
                c,
                result_dict["padding_mask"],
                result_dict,
                pbc_expand_batched=None,
                mixed_attn_bias=attn_bias,
            )

        z = torch.einsum("blh,bkh->blkh", s, s)

        s = self.proj_s(s)
        z = self.proj_z(z)

        result_dict["plddt_logits"] = s
        result_dict["pde_logits"] = z

        with torch.no_grad():
            result_dict["plddt"] = compute_plddt(s)
            # calculate mean pLDDT score corresponding to the mask
            result_dict["mean_plddt"] = result_dict["plddt"][
                (~result_dict["padding_mask"])
                & (
                    ~result_dict["protein_mask"].any(dim=-1)
                )  # result_dict["is_protein"]
            ].mean()

            result_dict["pde"], result_dict["mean_pde"] = compute_pde(
                z,
                (~result_dict["padding_mask"])
                & (~result_dict["protein_mask"].any(dim=-1)),
            )

        return result_dict

    def update_loss(self, loss, logging_output, model_output, batched_data):
        pos_pred = model_output["pred_pos_sample"]
        pos_orig = model_output["orig_pos_sample"]

        resolution = torch.ones(
            batched_data["is_protein"].shape[0],
            device=pos_pred.device,
            dtype=pos_pred.dtype,
        )

        lddt_loss_output, lddt_label, lddt_acc, lddt_stats = lddt_loss(
            model_output["plddt_logits"],
            pos_pred,
            pos_orig,
            (~model_output["padding_mask"])
            & (~model_output["protein_mask"].any(dim=-1)),
            batched_data["is_protein"],
            resolution,
        )

        pde_loss_output, pde_label, pde_acc = pde_loss(
            model_output["pde_logits"],
            pos_pred,
            pos_orig,
            (~model_output["padding_mask"])
            & (~model_output["protein_mask"].any(dim=-1)),
            batched_data["is_protein"],
            resolution,
        )

        loss += lddt_loss_output
        loss += pde_loss_output

        logging_output["lddt_loss"] = lddt_loss_output
        logging_output["lddt_acc"] = lddt_acc
        logging_output["lddt_label"] = lddt_label

        logging_output["pde_loss"] = pde_loss_output
        logging_output["pde_acc"] = pde_acc
        logging_output["pde_label"] = pde_label
        logging_output["pde_pred"] = model_output["mean_pde"]

        logging_output = {**logging_output, **lddt_stats}
        return loss, logging_output


# @PSM_FT_REGISTER.register("protein_understanding_head")
# class PerResidueLDDTCaPredictor(nn.Module):
#     def __init__(self, args, no_bins=50, c_hidden=128):
#         super(PerResidueLDDTCaPredictor, self).__init__()

#         self.no_bins = no_bins
#         self.c_in = args.encoder_embed_dim
#         self.c_hidden = c_hidden


#         self.n_sequence = (
#             2 if args.task_name in ["yeast_ppi", "human_ppi", "ppi_affinity"] else 1
#         )
#         self.n_classes = n_classes
#         self.head = torch.nn.Sequential(
#             torch.nn.Dropout(args.head_dropout),
#             torch.nn.Linear(
#                 args.encoder_embed_dim * self.n_sequence, args.encoder_embed_dim
#             ),
#             torch.nn.GELU(),
#             nn.LayerNorm(args.encoder_embed_dim),
#             torch.nn.Linear(args.encoder_embed_dim, n_classes),
#         )
#         self.return_residue_emb = (
#             True if args.task_name == "secondary_structure" else False
#         )

#     def update_batched_data(self, samples, batched_data):
#         return batched_data

#     def forward(self, result_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         s = result_dict["decoder_x_output"]
#         s = self.layer_norm(s)
#         s = self.linear_1(s)
#         s = self.relu(s)
#         s = self.linear_2(s)
#         s = self.relu(s)
#         s = self.linear_3(s)
#         result_dict["plddt_logits"] = s
#         with torch.no_grad():
#             result_dict["plddt"] = compute_plddt(s)
#             # calculate mean pLDDT score corresponding to the mask
#             result_dict["mean_plddt"] = result_dict["plddt"][
#                 result_dict["is_protein"]
#             ].mean()
#         return result_dict

#     def update_loss(self, loss, logging_output, model_output, batched_data):
#         pos_pred = model_output["pred_pos_sample"]
#         pos_orig = model_output["orig_pos_sample"]

#         resolution = torch.ones(
#             batched_data["is_protein"].shape[0],
#             device=pos_pred.device,
#             dtype=pos_pred.dtype,
#         )
#         lddt_loss_output, lddt_label, lddt_acc, lddt_stats = lddt_loss(
#             model_output["plddt_logits"],
#             pos_pred,
#             pos_orig,
#             batched_data["is_protein"],
#             resolution,
#         )
#         loss += lddt_loss_output
#         logging_output["lddt_loss"] = lddt_loss_output
#         logging_output["lddt_acc"] = lddt_acc
#         logging_output["lddt_label"] = lddt_label
#         logging_output = {**logging_output, **lddt_stats}
#         return loss, logging_output
