# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rdkit
import torch
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from rdkit.Chem import Mol
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolfiles import MolToMolFile
from torch import Tensor

from sfm.data.mol_data.utils.molecule import xyz2mol
from sfm.logging import logger
from sfm.utils.register import Register

bond_dict = {
    0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
    1: rdkit.Chem.rdchem.BondType.SINGLE,
    2: rdkit.Chem.rdchem.BondType.DOUBLE,
    3: rdkit.Chem.rdchem.BondType.TRIPLE,
    4: rdkit.Chem.rdchem.BondType.AROMATIC,
}

AA1TO3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

VOCAB = {
    # "<pad>": 0,  # padding
    # "1"-"127": 1-127, # atom type
    # "<cell_corner>": 128, use for pbc material
    "L": 130,
    "A": 131,
    "G": 132,
    "V": 133,
    "S": 134,
    "E": 135,
    "R": 136,
    "T": 137,
    "I": 138,
    "D": 139,
    "P": 140,
    "K": 141,
    "Q": 142,
    "N": 143,
    "F": 144,
    "Y": 145,
    "M": 146,
    "H": 147,
    "W": 148,
    "C": 149,
    "X": 150,
    "B": 151,
    "U": 152,
    "Z": 153,
    "O": 154,
    "-": 155,
    ".": 156,
    "<mask>": 157,
    "<cls>": 158,
    "<eos>": 159,
    # "<unk>": 160,
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

    def _get_bond_orders(self, edge_index, edge_attr, num_edges):
        edge_index = edge_index[:num_edges].reshape(num_edges // 2, 2, -1)[:, 0]
        bond_order = edge_attr[:num_edges].reshape(num_edges // 2, 2, -1)[:, 0, 0]
        num_edges = num_edges // 2
        return [
            (int(edge_index[i, 0]), int(edge_index[i, 1]), int(bond_order[i]) + 1)
            for i in range(num_edges)
        ]

    def convert(
        self, batched_data: Dict[str, Tensor], poses: Tensor
    ) -> List[Optional[Mol]]:
        is_molecule = batched_data["is_molecule"].cpu().numpy()
        index = batched_data["idx"].cpu().numpy()
        num_atoms = batched_data["num_atoms"].cpu().numpy()
        atm_nums = (batched_data["node_attr"][:, :, 0] - 1).squeeze(-1).cpu().numpy()
        coords = poses.cpu().numpy()
        num_edges = batched_data["num_edges"].cpu().numpy()
        edge_index = batched_data["edge_index"].cpu().numpy()
        edge_attr = batched_data["edge_attr"].cpu().numpy()

        structures: List[Optional[Mol]] = []
        batch_size = num_atoms.shape[0]
        for i in range(batch_size):
            if is_molecule[i]:
                try:
                    mol = xyz2mol(
                        atoms=atm_nums[i][: num_atoms[i]].tolist(),
                        coords=coords[i][: num_atoms[i]].tolist(),
                        bond_orders=self._get_bond_orders(
                            edge_index[i], edge_attr[i], num_edges[i]
                        ),
                        charge=None,
                        check_charge=False,
                    )
                except Exception as e:
                    print(
                        f"Failed to generate moelcule from sampled structure for index {index[i]}. {e}"
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
            return {"rmsd": np.nan}

        if sampled_structure_output_path is not None:
            MolToMolFile(
                sampled_structure,
                f"{sampled_structure_output_path}/sampled_{idx}_{sample_index}.mol",
            )
            MolToMolFile(
                original_structure,
                f"{sampled_structure_output_path}/original_{idx}_{sample_index}.mol",
            )

        rmsd = np.nan
        try:
            assert sampled_structure.GetNumAtoms() == original_structure.GetNumAtoms()
            rmsd = AlignMol(
                sampled_structure,
                original_structure,
                atomMap=[(i, i) for i in range(original_structure.GetNumAtoms())],
            )
        except Exception as ex:
            print(
                f"Failed to align molecules for sampled structure with index {idx}. {ex}"
            )
        if rmsd is None:
            rmsd = np.nan
        return {"rmsd": rmsd}


@CONVERTER_REGISTER.register("periodic")
class PeriodicConverter(BaseConverter):
    def convert(self, batched_data: Dict[str, Tensor], poses: Tensor):
        num_atoms = batched_data["num_atoms"].cpu().numpy()
        batch_size = num_atoms.shape[0]
        structures: List[Optional[Atoms]] = []
        atomic_numbers = batched_data["node_attr"][:, :-8, 0] - 1
        pbcs = batched_data["pbc"].cpu().numpy()
        for i in range(batch_size):
            try:
                pos = poses[i].cpu()
                pos = pos[: num_atoms[i] + 8]
                atomic_number = atomic_numbers[i][: num_atoms[i]].cpu().numpy()
                pos = pos - pos[-8, :].unsqueeze(0)
                cell = torch.cat(
                    [
                        pos[-4, :].unsqueeze(0),
                        pos[-6, :].unsqueeze(0),
                        pos[-7, :].unsqueeze(0),
                    ],
                    dim=0,
                )
                atoms = Atoms(
                    numbers=atomic_number,
                    positions=pos[:-8, :].numpy(),
                    cell=cell,
                    pbc=pbcs[i],
                )
                structures.append(atoms)
            except Exception as e:
                logger.warning(f"{e}")
                structures.append(None)
        return structures

    def match(
        self,
        sampled_structure: Optional[Atoms],
        original_structure: Optional[Atoms],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
    ) -> float:
        if sampled_structure is None or original_structure is None:
            return {"rmsd": np.nan}
        rmsd = np.nan
        try:
            sampled_structure.write(
                f"{sampled_structure_output_path}/sampled_{idx}_{sample_index}.cif",
                format="cif",
            )
            original_structure.write(
                f"{sampled_structure_output_path}/original_{idx}_{sample_index}.cif",
                format="cif",
            )
            sampled_pymatgen_structure = Structure.from_ase_atoms(sampled_structure)
            original_pymatgen_structure = Structure.from_ase_atoms(original_structure)
            matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
            rmsd = matcher.get_rms_dist(
                sampled_pymatgen_structure, original_pymatgen_structure
            )
            if rmsd is not None:
                rmsd = rmsd[0]
        except Exception as e:
            logger.info(
                f"Failed to align material for sampled structure with index {idx} {e}."
            )
        if rmsd is None:
            rmsd = np.nan
        return {"rmsd": rmsd}


@CONVERTER_REGISTER.register("protein")
class ProteinConverter(BaseConverter):
    def convert(self, batched_data: Dict[str, Tensor], poses: Tensor):
        num_residues = batched_data["num_atoms"].cpu().numpy()
        batch_size = num_residues.shape[0]
        structures: List[Optional[List[str]]] = []
        token_ids = batched_data["token_id"]
        for i in range(batch_size):
            try:
                pos = poses[i].cpu()
                pos = pos[: num_residues[i]]
                residue_ids = token_ids[i][: num_residues[i]].cpu().numpy()
                pdb_lines = []
                sequence = []
                for token_id in residue_ids:
                    for key in VOCAB:
                        if VOCAB[key] == int(token_id):
                            sequence.append(key)
                for i, (x, y, z) in enumerate(pos):
                    atomidx = i + 1
                    resname = AA1TO3.get(sequence[i], "UNK")
                    resnumb = i + 1
                    pdb_lines.append(
                        f"ATOM  {atomidx:>5d}  CA  {resname}  {resnumb:>4d}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  \n"
                    )
                pdb_lines.append("TER\n")
                pdb_lines.append("END\n")

                structures.append(pdb_lines)
            except Exception as e:
                logger.warning(f"{e}")
                structures.append(None)
        return structures

    def match(
        self,
        sampled_structure: Optional[List[str]],
        original_structure: Optional[List[str]],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
    ) -> float:
        if sampled_structure is None or original_structure is None:
            return {"rmsd": np.nan, "tm_score": np.nan}
        tm_score = np.nan
        rmsd = np.nan
        try:
            sampled_path = (
                f"{sampled_structure_output_path}/sampled_{idx}_{sample_index}.pdb"
            )
            original_path = (
                f"{sampled_structure_output_path}/original_{idx}_{sample_index}.pdb"
            )
            with open(sampled_path, "w") as out_file:
                out_file.writelines(sampled_structure)
            with open(original_path, "w") as out_file:
                out_file.writelines(original_structure)

            lines = []
            with tempfile.TemporaryFile() as tmp:
                proc = subprocess.Popen(
                    ["./TMscore", sampled_path, original_path], stdout=tmp, stderr=tmp
                )
                proc.wait()
                tmp.seek(0)
                lines = [_.decode("utf-8") for _ in tmp.readlines()]
            for i, line in enumerate(lines):
                cols = line.split()
                if line.startswith("RMSD") and len(cols) > 5:
                    rmsd = float(cols[5])
                elif line.startswith("TM-score") and len(cols) > 2:
                    tm_score = float(cols[2])
        except Exception as e:
            logger.info(
                f"Failed to calculate TM-score for sampled structure with index {idx} {e}."
            )
        return {"rmsd": rmsd, "tm_score": tm_score}


class SampledStructureConverter:
    def __init__(self, sampled_structure_output_path: Optional[str]) -> None:
        self.sampled_structure_output_path = sampled_structure_output_path
        if self.sampled_structure_output_path is not None and not os.path.exists(
            self.sampled_structure_output_path
        ):
            os.system(f"mkdir -p {self.sampled_structure_output_path}")

        if not os.path.exists("TMscore"):
            with tempfile.TemporaryFile() as tmp:
                subprocess.Popen(
                    ["wget", "https://zhanggroup.org/TM-score/TMscoreLinux.zip"],
                    stdout=tmp,
                    stderr=tmp,
                )
                if not os.path.exists("TMscore"):
                    subprocess.Popen(
                        ["unzip", "TMscoreLinux.zip"], stdout=tmp, stderr=tmp
                    )
                subprocess.Popen(["rm", "TMscoreLinux.zip*"], stdout=tmp, stderr=tmp)

    def convert_and_match(
        self,
        batched_data: Dict[str, Tensor],
        original_pos: Tensor,
        sample_index: int,
    ) -> Tensor:
        batch_size = batched_data["is_molecule"].size()[0]
        all_results = [None for _ in range(batch_size)]
        for system_tag in ["molecule", "periodic", "protein"]:
            is_mask = batched_data[f"is_{system_tag}"]
            indexes_in_batch = is_mask.nonzero().squeeze(-1)
            if torch.any(is_mask):
                indexes = batched_data["idx"][is_mask]
                sampled_structures = CONVERTER_REGISTER[system_tag]().convert(
                    batched_data, batched_data["pos"]
                )
                original_structures = CONVERTER_REGISTER[system_tag]().convert(
                    batched_data, original_pos
                )
                for sampled_structure, original_structure, index, index_in_batch in zip(
                    sampled_structures, original_structures, indexes, indexes_in_batch
                ):
                    all_results[index_in_batch] = CONVERTER_REGISTER[
                        system_tag
                    ]().match(
                        sampled_structure,
                        original_structure,
                        int(index),
                        self.sampled_structure_output_path,
                        sample_index,
                    )
        return all_results
