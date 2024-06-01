# -*- coding: utf-8 -*-
import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import Atom, Mol, RWMol, rdDistGeom
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolfiles import MolToMolFile
from rdkit.Chem.rdmolops import SanitizeMol
from rdkit.Geometry.rdGeometry import Point3D
from torch import Tensor

from sfm.logging import logger
from sfm.utils.register import Register

bond_dict = {
    0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
    1: rdkit.Chem.rdchem.BondType.SINGLE,
    2: rdkit.Chem.rdchem.BondType.DOUBLE,
    3: rdkit.Chem.rdchem.BondType.TRIPLE,
    4: rdkit.Chem.rdchem.BondType.AROMATIC,
}


class BaseConverter(ABC, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def convert(
        self,
        batched_data: Dict[str, Tensor],
        poses: Tensor,
        sampled_structure_output_path: Optional[str] = None,
        sampled_structure_output_prefix: Optional[str] = "",
        sample_index: Optional[int] = -1,
    ) -> List[Optional[Any]]:
        return []

    @abstractmethod
    def match(
        self,
        sampled_structure: Optional[Any],
        original_structure: Optional[Any],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
    ) -> float:
        # if matched successfully, return RMSD.
        # otherwise, return nan
        return 0.0


CONVERTER_REGISTER: Union[Dict[str, BaseConverter.__class__], Register] = Register(
    "converter_register"
)


@CONVERTER_REGISTER.register("molecule")
class MoleculeConverter(BaseConverter):
    def __init__(self) -> None:
        super().__init__()
        self.periodic_table = Chem.GetPeriodicTable()

    def _add_bond_to_mol(self, mol: RWMol, edge_index, edge_attr, num_edges):
        edge_index = edge_index[:num_edges].reshape(num_edges // 2, 2, -1)[
            :, 0
        ]  # deduplicate the bidirectional edges
        bond_order = edge_attr[:num_edges].reshape(num_edges // 2, 2, -1)[:, 0, 0]
        num_edges = num_edges // 2
        for i in range(num_edges):
            mol.AddBond(
                int(edge_index[i, 0]),
                int(edge_index[i, 1]),
                bond_dict[int(bond_order[i]) + 1],
            )  # +1 because ogb label bond orders from 0

    def _add_atoms_to_mol(self, mol: RWMol, atomic_numbers: Tensor):
        for atomic_number in atomic_numbers:
            mol.AddAtom(Atom(self.periodic_table.GetElementSymbol(int(atomic_number))))

    def _add_pos_to_mol(self, mol: RWMol, pos: Tensor):
        rdDistGeom.EmbedMolecule(
            mol,
            coordMap={
                i: Point3D(float(pos[i][0]), float(pos[i][1]), float(pos[i][2]))
                for i in range(mol.GetNumAtoms())
            },
        )

    def convert(
        self, batched_data: Dict[str, Tensor], poses: Tensor
    ) -> List[Optional[Mol]]:
        num_atoms = batched_data["num_atoms"]
        batch_size = num_atoms.size()[0]
        is_molecule = batched_data["is_molecule"]
        all_num_edges = batched_data["num_edges"]
        all_edge_index = batched_data["edge_index"]
        all_edge_attr = batched_data["edge_attr"]
        all_index = batched_data["idx"]
        structures: List[Optional[Mol]] = []
        for i in range(batch_size):
            if is_molecule[i]:
                atomic_numbers = (
                    batched_data["node_attr"][i][: num_atoms[i], 0] - 1
                )  # minus 1 due to padding
                num_edges = all_num_edges[i]
                edge_index = all_edge_index[i]
                edge_attr = all_edge_attr[i]
                index = int(all_index[i])
                pos = poses[i]
                mol = RWMol(Mol())
                try:
                    self._add_atoms_to_mol(mol, atomic_numbers)
                    self._add_bond_to_mol(mol, edge_index, edge_attr, num_edges)
                    SanitizeMol(mol)
                    self._add_pos_to_mol(mol, pos)
                    mol = mol.GetMol()
                except Exception as e:
                    logger.info(
                        f"Failed to generate moelcule from sampled structure for index {index}. {e}"
                    )
                    mol = None
                structures.append(mol)
        return structures

    def match(
        self,
        sampled_structure: Optional[Mol],
        original_structure: Optional[Mol],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
    ) -> float:
        if sampled_structure is None or original_structure is None:
            return np.nan
        rmsd = np.nan
        try:
            rmsd = AlignMol(sampled_structure, original_structure)
            if sampled_structure_output_path is not None:
                MolToMolFile(
                    sampled_structure,
                    f"{sampled_structure_output_path}/sampled_{idx}_{sample_index}.mol",
                )
                MolToMolFile(
                    original_structure,
                    f"{sampled_structure_output_path}/original_{idx}_{sample_index}.mol",
                )
        except:
            logger.info(
                f"Failed to align molecules for sampled structure with index {idx}."
            )
        if rmsd is None:
            rmsd = np.nan
        return rmsd


class SampledStructureConverter:
    def __init__(self, sampled_structure_output_path: Optional[str]) -> None:
        self.sampled_structure_output_path = sampled_structure_output_path
        if self.sampled_structure_output_path is not None and not os.path.exists(
            self.sampled_structure_output_path
        ):
            os.system(f"mkdir {self.sampled_structure_output_path}")

    def convert_and_match(
        self,
        batched_data: Dict[str, Tensor],
        original_pos: Tensor,
        sample_index: int,
    ) -> Tensor:
        batch_size = batched_data["is_molecule"].size()[0]
        device = batched_data["is_molecule"].device
        all_rmsds = torch.zeros(batch_size, dtype=torch.float, device=device)
        for system_tag in ["molecule", "periodic", "protein"]:
            is_mask = batched_data[f"is_{system_tag}"]
            if torch.any(is_mask):
                indexes = batched_data["idx"][is_mask]
                sampled_structures = CONVERTER_REGISTER[system_tag]().convert(
                    batched_data, batched_data["pos"]
                )
                original_structures = CONVERTER_REGISTER[system_tag]().convert(
                    batched_data, original_pos
                )
                rmsds = []
                for sampled_structure, original_structure, index in zip(
                    sampled_structures, original_structures, indexes
                ):
                    rmsds.append(
                        CONVERTER_REGISTER[system_tag]().match(
                            sampled_structure,
                            original_structure,
                            int(index),
                            self.sampled_structure_output_path,
                            sample_index,
                        )
                    )
                all_rmsds[is_mask] = torch.tensor(
                    rmsds, dtype=torch.float, device=device
                )
        return all_rmsds
