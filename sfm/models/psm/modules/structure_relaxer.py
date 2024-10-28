# -*- coding: utf-8 -*-
import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, Union

import numpy as np
import torch
from ase import Atoms
from ase.build import make_supercell
from ase.constraints import ExpCellFilter
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from torch import Tensor
from torch.nn import Module
from tqdm import tqdm

from sfm.logging.loggers import logger as sfm_logger
from sfm.models.psm.psm_config import PSMConfig
from sfm.tasks.psm.ase_deep_calculator import DeepCalculator
from sfm.utils.register import Register


class BaseStructureRelaxer(ABC, metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def relax(self, atoms: Atoms, model: Module):
        pass


RELAXER_REGISTER: Union[Dict[str, BaseStructureRelaxer.__class__], Register] = Register(
    "relaxer_register"
)


@RELAXER_REGISTER.register("periodic")
class PeriodicStructureRelaxer(BaseStructureRelaxer):
    def __init__(self, psm_config: PSMConfig) -> None:
        self.psm_config = psm_config

    def _remove_unit_cell_virtual_atoms(self, batched_data: Dict[str, Tensor]):
        # note that we need to handle edge_input and spatial_pos for molecule
        # but we do not handle these for periodic systems and proteins

        original_tensors = {}

        original_tensors["is_stable_periodic"] = batched_data[
            "is_stable_periodic"
        ].clone()
        batched_data["is_stable_periodic"] = torch.zeros_like(
            batched_data["is_stable_periodic"], dtype=torch.bool
        )

        original_tensors["token_id"] = batched_data["token_id"].clone()
        unit_cell_virtual_node_mask = batched_data["token_id"] >= 128
        batched_data["token_id"] = batched_data["token_id"].masked_fill(
            unit_cell_virtual_node_mask >= 128, 0
        )
        batched_data["token_id"] = batched_data["token_id"][:, :-8]

        original_tensors["num_atoms"] = batched_data["num_atoms"].clone()

        original_tensors["pos"] = batched_data["pos"].clone()
        batched_data["pos"] = batched_data["pos"].masked_fill(
            unit_cell_virtual_node_mask.unsqueeze(-1), 0.0
        )
        batched_data["pos"] = batched_data["pos"][:, :-8, :]

        original_tensors["forces"] = batched_data["forces"].clone()
        batched_data["forces"] = batched_data["forces"].masked_fill(
            unit_cell_virtual_node_mask.unsqueeze(-1), 0.0
        )
        batched_data["forces"] = batched_data["forces"][:, :-8, :]

        original_tensors["node_attr"] = batched_data["node_attr"].clone()
        batched_data["node_attr"] = batched_data["node_attr"].masked_fill(
            unit_cell_virtual_node_mask.unsqueeze(-1), 0
        )
        batched_data["node_attr"] = batched_data["node_attr"][:, :-8, :]

        original_tensors["attn_bias"] = batched_data["attn_bias"].clone()
        batched_data["attn_bias"][:, 1:, 1:] = batched_data["attn_bias"][
            :, 1:, 1:
        ].masked_fill(unit_cell_virtual_node_mask.unsqueeze(-1), 0.0)
        batched_data["attn_bias"][:, 1:, 1:] = batched_data["attn_bias"][
            :, 1:, 1:
        ].masked_fill(unit_cell_virtual_node_mask.unsqueeze(1), 0.0)
        batched_data["attn_bias"] = batched_data["attn_bias"][:, :-8, :-8]

        original_tensors["in_degree"] = batched_data["in_degree"].clone()
        batched_data["in_degree"] = batched_data["in_degree"].masked_fill(
            unit_cell_virtual_node_mask, 0
        )
        batched_data["in_degree"] = batched_data["in_degree"][:, :-8]

        original_tensors["out_degree"] = batched_data["out_degree"].clone()
        batched_data["out_degree"] = batched_data["out_degree"].masked_fill(
            unit_cell_virtual_node_mask, 0
        )
        batched_data["out_degree"] = batched_data["out_degree"][:, :-8]

        original_tensors["init_pos"] = batched_data["init_pos"].clone()
        batched_data["init_pos"] = torch.zeros_like(batched_data["init_pos"])[:, :-8]

        if "adj" in batched_data:
            original_tensors["adj"] = batched_data["adj"].clone()
            batched_data["adj"] = batched_data["adj"].masked_fill(
                unit_cell_virtual_node_mask.unsqueeze(-1), False
            )
            batched_data["adj"] = batched_data["adj"].masked_fill(
                unit_cell_virtual_node_mask.unsqueeze(1), False
            )
            batched_data["adj"] = batched_data["adj"][:, :-8, :-8]

        if "attn_edge_type" in batched_data:
            original_tensors["attn_edge_type"] = batched_data["attn_edge_type"].clone()
            batched_data["attn_edge_type"] = batched_data["attn_edge_type"].masked_fill(
                unit_cell_virtual_node_mask.unsqueeze(-1).unsqueeze(-1), 0
            )
            batched_data["attn_edge_type"] = batched_data["attn_edge_type"].masked_fill(
                unit_cell_virtual_node_mask.unsqueeze(1).unsqueeze(-1), 0
            )
            batched_data["attn_edge_type"] = batched_data["attn_edge_type"][
                :, :-8, :-8, :
            ]

        original_tensors["sqrt_one_minus_alphas_cumprod_t"] = batched_data[
            "sqrt_one_minus_alphas_cumprod_t"
        ].clone()
        batched_data["sqrt_one_minus_alphas_cumprod_t"] = batched_data[
            "sqrt_one_minus_alphas_cumprod_t"
        ][:, :-8]

        return batched_data, original_tensors

    def _relax_one_step(self, batched_data: Dict[str, Tensor], model: Module, i: int):
        force_key = (
            "autograd_forces"
            if self.psm_config.use_autograd_force_for_relaxation_and_md
            else "forces"
        )
        pos = batched_data["pos"]
        outputs = model(
            batched_data,
            skip_sample=True,
            time_step=torch.zeros([pos.size()[0]], dtype=pos.dtype, device=pos.device)
            .unsqueeze(-1)
            .repeat(1, pos.shape[1]),
            clean_mask=torch.ones([pos.size()[0]], dtype=torch.bool, device=pos.device)
            .unsqueeze(-1)
            .repeat(1, pos.shape[1]),
        )
        if force_key == "autograd_forces" and force_key not in outputs:
            sfm_logger.warning(
                "No autograd_forces found in model outputs, will use forces instead."
            )
            force_key = "forces"
        forces: Tensor = outputs[force_key]
        padding_mask = batched_data["token_id"].eq(0)
        forces = forces.masked_fill(forces.isnan(), 0.0)
        batched_data["pos"] = (
            batched_data["pos"] + self.psm_config.structure_relax_step_size * forces
        )
        batched_data["pos"] = (
            batched_data["pos"].masked_fill(padding_mask.unsqueeze(-1), 0.0).detach()
        )
        return batched_data, forces

    def relax(self, atoms: Atoms, model: Module):
        if not os.path.exists(self.psm_config.sampled_structure_output_path):
            os.makedirs(self.psm_config.sampled_structure_output_path)
        self.calculator = DeepCalculator(
            model, args=self.psm_config, atoms_type="periodic", device="cuda"
        )
        task_atoms = make_supercell(
            atoms,
            [
                [self.psm_config.relax_initial_cell_matrix[0], 0, 0],
                [0, self.psm_config.relax_initial_cell_matrix[1], 0],
                [0, 0, self.psm_config.relax_initial_cell_matrix[2]],
            ],
        )
        relaxed_atoms = task_atoms.copy()
        prev_atoms = relaxed_atoms.copy()
        try:
            for frac in tqdm(
                np.arange(
                    self.psm_config.relax_lower_deformation,
                    self.psm_config.relax_upper_deformation
                    + self.psm_config.relax_deformation_step,
                    self.psm_config.relax_deformation_step,
                )
            ):
                _atoms: Atoms = prev_atoms
                pres = frac / 160.2176621
                _atoms.calc = self.calculator
                ecf = ExpCellFilter(_atoms, scalar_pressure=pres)
                sfm_logger.info(f"before relaxation: {_atoms.get_positions()}")
                FIRE_logfile = os.path.join(
                    self.psm_config.sampled_structure_output_path,
                    f"relaxation-{frac}.log",
                )
                FIRE_trajfile = os.path.join(
                    self.psm_config.sampled_structure_output_path,
                    f"trajectory-{frac}.traj",
                )
                traj = Trajectory(FIRE_trajfile, "w", _atoms)
                qn = FIRE(ecf, logfile=FIRE_logfile)
                qn.attach(traj)
                qn.run(
                    fmax=self.psm_config.relax_fmax,
                    steps=self.psm_config.relax_ase_steps,
                )

                sfm_logger.info(f"after relaxation: {_atoms.get_positions()}")

            prev_atoms = _atoms.copy()
            return prev_atoms
        except Exception as e:
            sfm_logger.info(f"{e}")
            return atoms
