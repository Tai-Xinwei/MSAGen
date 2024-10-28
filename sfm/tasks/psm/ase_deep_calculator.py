# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from torch.utils.data import DataLoader

from sfm.data.psm_data.dataset import MatterSimDataset
from sfm.data.psm_data.unifieddataset import BatchedDataDataset
from sfm.utils.move_to_device import move_to_device


class DeepCalculator(Calculator):
    """
    Deep calculator based on ase Calculator
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model,
        args,
        extra_collate_fn=None,
        atoms_type: str = "periodic",  # choices=["periodic", "molecule"]
        stress_weight: float = 1.0,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
        """
        super().__init__(**kwargs)
        self.model = model
        self.args = args
        self.compute_stress = False  # TODO: support for stress
        self.stress_weight = stress_weight
        self.extra_collate_fn = extra_collate_fn
        self.atoms_type = atoms_type

        self.device = next(model.parameters()).device if device is None else device

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:
        """

        all_changes = [
            "positions",
            "numbers",
            "cell",
            "pbc",
            "initial_charges",
            "initial_magmoms",
        ]

        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if self.atoms_type == "periodic":
            dataset = MatterSimDataset(
                self.args, self.args.data_path, split="train", atoms_list=[atoms]
            )
        elif self.atoms_type == "molecule":
            raise NotImplementedError("molecule type is not supported yet")
        else:
            raise NotImplementedError(f"Unknown atoms_type: {self.atoms_type}")
        batched_dataset = BatchedDataDataset(
            self.args, [dataset], len(dataset), extra_collate_fn=self.extra_collate_fn
        )
        data_loader = DataLoader(
            batched_dataset,
            sampler=None,
            batch_size=1,
            collate_fn=batched_dataset.collate,
            drop_last=False,
        )

        for data in data_loader:
            data = move_to_device(data, self.device)
            result = self.model(data, skip_sample=True)

        energy = (
            result["energy_per_atom"] * dataset.energy_per_atom_std
            + dataset.energy_per_atom_mean
        ) * result["num_atoms"]
        force_key = (
            "forces"
            if not self.args.use_autograd_force_for_relaxation_and_md
            else "autograd_forces"
        )
        forces = result[force_key]
        forces = forces * dataset.force_std + dataset.force_mean
        self.results.update(
            energy=energy.detach().cpu().numpy()[0],
            free_energy=energy.detach().cpu().numpy()[0],
            forces=forces.detach().cpu().numpy()[0],
        )

        if self.compute_stress:
            self.results.update(
                stress=self.stress_weight
                * full_3x3_to_voigt_6_stress(
                    result["stresses"].detach().cpu().numpy()[0]
                )
            )
        else:
            self.results.update(
                stress=self.stress_weight
                * full_3x3_to_voigt_6_stress(np.zeros([3, 3], dtype=np.float32))
            )
