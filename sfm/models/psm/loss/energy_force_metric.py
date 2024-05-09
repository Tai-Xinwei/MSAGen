# -*- coding: utf-8 -*-
from collections import defaultdict

import torch
import torch.nn as nn

from .utility import HATREE_TO_KCAL, get_energy_from_h, get_pyscf_obj_from_dataset


class EnergyError(nn.Module):
    def __init__(self, loss_weight, metric="mae"):
        super().__init__(loss_weight)
        self.loss_weight = loss_weight
        self.metric = metric
        self.name = "energy_loss"

    def forward(self, batch_data, error_dict={}, metric=None):
        error_dict["loss"] = error_dict.get("loss", 0)
        metric = self.metric if metric is None else metric

        if metric == "mae":
            loss = torch.mean(
                torch.abs(batch_data["energy"] - batch_data["pred_energy"])
            )
        elif metric == "mse":
            loss = torch.mean((batch_data["energy"] - batch_data["pred_energy"]) ** 2)
        elif metric == "rmse":
            loss = torch.sqrt(
                torch.mean((batch_data["energy"] - batch_data["pred_energy"]) ** 2)
            )
        elif metric == "Huber" or metric == "huber":
            # apply mask
            loss = torch.nn.HuberLoss(reduction="mean")(
                batch_data["pred_energy"], batch_data["energy"]
            )
        else:
            raise ValueError(f"loss not support metric: {metric}")

        error_dict["loss"] += self.loss_weight * loss
        error_dict[f"energy_loss_{metric}"] = loss
        return error_dict


class ForcesError(nn.Module):
    def __init__(self, loss_weight, metric="mae"):
        super().__init__(loss_weight)

        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "force_loss"

    def forward(self, batch_data, error_dict={}, metric=None):
        error_dict["loss"] = error_dict.get("loss", 0)
        metric = self.metric if metric is None else metric

        if metric == "mae":
            loss = torch.mean(
                torch.abs(batch_data["forces"] - batch_data["pred_forces"])
            )
        elif metric == "mse":
            loss = torch.mean((batch_data["forces"] - batch_data["pred_forces"]) ** 2)
        elif metric == "rmse":
            loss = torch.sqrt(
                torch.mean((batch_data["forces"] - batch_data["pred_forces"]) ** 2)
            )
        elif metric == "Huber" or metric == "huber":
            # apply mask
            loss = torch.nn.HuberLoss(reduction="mean")(
                batch_data["pred_forces"], batch_data["forces"]
            )
        else:
            raise ValueError(f"loss not support metric: {metric}")

        error_dict["loss"] += self.loss_weight * loss
        error_dict[f"forces_loss_{metric}"] = loss.detach()


class HamiltonianError(nn.Module):
    def __init__(self, loss_weight, metric="mae", sparse=False, sparse_coeff=1e-5):
        super().__init__(loss_weight)
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "hamiltonian_loss"
        self.sparse = sparse
        self.sparse_coeff = sparse_coeff

    def forward(self, batch_data, error_dict={}, metric=None):
        error_dict["loss"] = error_dict.get("loss", 0)
        metric = self.metric if metric is None else metric

        mask = torch.cat((batch_data["diag_mask"], batch_data["non_diag_mask"]))
        # predict = torch.cat((predictions['hamiltonian_diagonal_blocks'],predictions['hamiltonian_non_diagonal_blocks']*mask_l1.unsqueeze(-1).unsqueeze(-1)))
        predict = torch.cat(
            (
                batch_data["pred_hamiltonian_diagonal_blocks"],
                batch_data["pred_hamiltonian_non_diagonal_blocks"],
            )
        )  # no distance norm
        target = torch.cat(
            (batch_data["diag_hamiltonian"], batch_data["non_diag_hamiltonian"])
        )  # the label is ground truth minus initial guess
        if self.sparse:
            # target geq to sparse coeff is considered as non-zero
            sparse_mask = torch.abs(target).ge(self.sparse_coeff).float()
            target = target * sparse_mask
        diff = (predict - target) * mask

        weight = mask.numel() / mask.sum()

        if metric == "mae":
            loss = weight * torch.mean(torch.abs(diff))
            (torch.abs((predict - target)) / (torch.abs(target) + 1e-5))[mask == 1]

        elif metric == "mse":
            loss = weight * torch.mean(diff**2)
        elif metric == "rmse":
            loss = torch.sqrt(weight * torch.mean(diff**2))
        elif (metric == "maemse") or (metric == "msemae"):
            mae = weight * torch.mean(torch.abs(diff))
            mse = weight * torch.mean(diff**2)
            error_dict["hami_loss_mae"] = mae.detach()
            error_dict["hami_loss_mse"] = mse.detach()
            loss = mae + mse
        elif metric == "Huber" or metric == "huber":
            # apply mask
            loss = torch.nn.HuberLoss(reduction="mean")(predict * mask, target * mask)
        else:
            raise ValueError(f"loss not support metric: {metric}")
            # print("mean of predict and target",torch.mean(torch.abs(predict)),torch.mean(torch.abs(target)))
            # print("max of predict and target",torch.max(predict),torch.max(target))
            # print("min of predict and target",torch.min(predict),torch.min(target))

        error_dict["loss"] += loss * self.loss_weight
        error_dict[f"hami_loss_{metric}"] = loss.detach()


class EnergyHamiError(nn.Module):
    def __init__(
        self,
        loss_weight,
        trainer=None,
        metric="mae",
        basis="def2-svp",
        transform_h=False,
    ):
        super().__init__(loss_weight)

        self.trainer = trainer
        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "energy_hami_loss"
        self.basis = basis
        self.transform_h = transform_h

    def _batch_energy_hami(self, batch_data):
        batch_size = batch_data["energy"].shape[0]
        energy = batch_data["energy"]
        self.trainer.model.hami_model.build_final_matrix(
            batch_data
        )  # construct full_hamiltonian
        full_hami = batch_data["pred_hamiltonian"]
        hami_energy = torch.zeros_like(energy)
        for i in range(batch_size):
            start, end = batch_data["ptr"][i], batch_data["ptr"][i + 1]
            pos = batch_data["pos"][start:end].detach().cpu().numpy()
            atomic_numbers = (
                batch_data["atomic_numbers"][start:end].detach().cpu().numpy()
            )
            mol, mf = get_pyscf_obj_from_dataset(
                pos, atomic_numbers, basis=self.basis, gpu=True
            )
            dm0 = mf.init_guess_by_minao()
            init_h = mf.get_fock(dm=dm0)

            f_hi = full_hami[i].detach().cpu().numpy() / HATREE_TO_KCAL + init_h
            # if self.transform_h:
            #     f_hi = transform_h_into_pyscf(f_hi, mol)
            hami_energy[i] = get_energy_from_h(mf, f_hi)
            hami_energy[i] *= HATREE_TO_KCAL
        return hami_energy

    def forward(self, batch_data, error_dict={}, metric=None):
        metric = self.metric if metric is None else metric

        predict = self._batch_energy_hami(batch_data)
        target = batch_data["pyscf_energy"]  # batch_data['energy']
        diff = predict - target
        print(f"pyscf energy using NN pred:{predict}, gt is {target}")

        if metric == "mae":
            loss = torch.mean(torch.abs(diff))
        elif metric == "mse":
            loss = torch.mean(diff**2)
        elif metric == "rmse":
            loss = torch.sqrt(torch.mean(diff**2))
        elif (metric == "maemse") or (metric == "msemae"):
            mae = torch.mean(torch.abs(diff))
            mse = torch.mean(diff**2)
            error_dict["energy_hami_loss_mae"] = mae.detach()
            error_dict["energy_hami_loss_mse"] = mse.detach()
            loss = mae + mse
        elif metric == "Huber" or metric == "huber":
            loss = torch.nn.HuberLoss(reduction="mean")(predict, target)
        else:
            raise ValueError(f"loss not support metric: {metric}")
        error_dict["loss"] += loss.detach()
        error_dict[f"real_world_pyscf_fockenergy_{metric}"] = loss.detach()


class HomoLumoHamiError(nn.Module):
    def __init__(self, loss_weight, metric="mae", basis="def2-svp", transform_h=False):
        super().__init__(loss_weight)

        self.metric = metric
        self.loss_weight = loss_weight
        self.name = "homo_lumo_hami_loss"
        self.basis = basis
        self.transform_h = transform_h

    def forward(self, batch_data, error_dict={}, metric=None):
        # TODO：merge this later when yunyang complete this
        # Note: only related to hamiltonian prediction
        pass
