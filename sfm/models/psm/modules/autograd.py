# -*- coding: utf-8 -*-
from abc import ABC, ABCMeta, abstractmethod

import torch


class GradientHead(torch.nn.Module):
    def __init__(
        self,
        molecule_energy_per_atom_std=1.0,
        periodic_energy_per_atom_std=1.0,
        molecule_force_std=1.0,
        periodic_force_std=1.0,
    ):
        super(GradientHead, self).__init__()
        self.molecule_energy_per_atom_std = molecule_energy_per_atom_std
        self.periodic_energy_per_atom_std = periodic_energy_per_atom_std
        self.molecule_force_std = molecule_force_std
        self.periodic_force_std = periodic_force_std  # should be 1.0 ,the rescailing is done in the loss function

    def forward(
        self,
        energy,
        pos,
    ):
        grad_outputs = [torch.ones_like(energy)]

        grad = torch.autograd.grad(
            outputs=energy,
            inputs=pos,
            grad_outputs=grad_outputs,
            create_graph=self.training,
        )

        force_grad = grad[0]

        if force_grad is not None:
            forces = torch.neg(force_grad)

        return forces
