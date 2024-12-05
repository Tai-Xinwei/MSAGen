# -*- coding: utf-8 -*-
import torch
from torch import Tensor

from sfm.models.psm.psm_config import PSMConfig


class GradientHead(torch.nn.Module):
    def __init__(
        self,
        psm_config: PSMConfig,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_energy_std=1.0,
        periodic_energy_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
        periodic_stress_mean=0.0,
        periodic_stress_std=1.0,
        supervise_total_energy: bool = False,
    ):
        super(GradientHead, self).__init__()
        self.psm_config = psm_config
        self.molecule_energy_per_atom_std = molecule_energy_per_atom_std
        self.periodic_energy_per_atom_std = periodic_energy_per_atom_std
        self.molecule_energy_std = molecule_energy_std
        self.periodic_energy_std = periodic_energy_std
        self.molecule_force_std = molecule_force_std
        self.periodic_force_std = periodic_force_std
        self.periodic_stress_mean = periodic_stress_mean
        self.periodic_stress_std = periodic_stress_std
        self.supervise_total_energy = supervise_total_energy
        self.strain = None

    def wrap_input(self, batched_data):
        pos: Tensor = batched_data["pos"]
        pos.requires_grad_(True)
        batch_size = pos.size(0)
        device = pos.device
        dtype = pos.dtype
        if (
            self.psm_config.supervise_autograd_stress
            and batched_data["has_stress"].any()
        ):
            cell = batched_data["cell"]

            self.strain = torch.zeros([batch_size, 3, 3], device=device, dtype=dtype)
            self.strain.requires_grad_(True)
            strain_augment = self.strain.unsqueeze(1).expand(-1, pos.size(1), -1, -1)

            cell = torch.matmul(
                cell, torch.eye(3, device=device, dtype=dtype)[None, :, :] + self.strain
            )
            pos = torch.einsum(
                "bki, bkij -> bkj",
                pos,
                (torch.eye(3, device=device)[None, None, :, :] + strain_augment),
            )
            batched_data["cell"] = cell
            batched_data["pos"] = pos

    def forward(
        self,
        energy_per_atom,
        non_atom_mask,
        pos,
        cell,
        is_periodic,
        is_molecule,
        has_stress,
    ):
        energy_per_atom = energy_per_atom.masked_fill(non_atom_mask, 0.0)
        energy_per_atom = torch.where(
            is_periodic.unsqueeze(-1),
            energy_per_atom
            * (
                self.periodic_energy_std
                if self.supervise_total_energy
                else self.periodic_energy_per_atom_std
            ),
            energy_per_atom,
        )
        energy_per_atom = torch.where(
            is_molecule.unsqueeze(-1),
            energy_per_atom
            * (
                self.molecule_force_std
                if self.supervise_total_energy
                else self.molecule_energy_per_atom_std
            ),
            energy_per_atom,
        )
        energy = energy_per_atom.sum(dim=-1, keepdim=True)
        grad_outputs = [torch.ones_like(energy)]

        if self.psm_config.supervise_autograd_stress and has_stress.any():
            grad = torch.autograd.grad(
                outputs=energy,
                inputs=[pos, self.strain],
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=True,
            )
        else:
            grad = torch.autograd.grad(
                outputs=energy,
                inputs=pos,
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=True,
            )

        force_grad = grad[0]
        forces = torch.neg(force_grad)

        forces = torch.where(
            is_periodic.unsqueeze(-1).unsqueeze(-1),
            forces / self.periodic_force_std,
            forces,
        )
        forces = torch.where(
            is_molecule.unsqueeze(-1).unsqueeze(-1),
            forces / self.molecule_force_std,
            forces,
        )

        if self.psm_config.supervise_autograd_stress and has_stress.any():
            stress_grad = grad[1]
            volume = torch.abs(torch.linalg.det(cell))
            stress = (
                1 / volume[:, None, None] * stress_grad * 160.21766208
                - self.periodic_stress_mean
                * torch.eye(3, device=stress_grad.device, dtype=stress_grad.dtype)
            ) / self.periodic_stress_std
        else:
            stress = None

        self.strain = None

        return forces, stress
