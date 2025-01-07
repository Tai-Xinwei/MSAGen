# -*- coding: utf-8 -*-
import collections
import os
import subprocess
import tempfile
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rdkit
import torch
from ase import Atoms
from Bio.SVDSuperimposer import SVDSuperimposer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolfiles import MolToMolFile
from torch import Tensor
from torch.nn import Module

from sfm.data.mol_data.utils.molecule import xyz2mol
from sfm.logging import logger
from sfm.models.psm.modules.structure_relaxer import RELAXER_REGISTER
from sfm.models.psm.psm_config import PSMConfig
from sfm.utils.register import Register

bond_dict = {
    0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
    1: rdkit.Chem.rdchem.BondType.SINGLE,
    2: rdkit.Chem.rdchem.BondType.DOUBLE,
    3: rdkit.Chem.rdchem.BondType.TRIPLE,
    4: rdkit.Chem.rdchem.BondType.AROMATIC,
}

VOCAB2AA = {
    130: "LEU",  # "L"
    131: "ALA",  # "A"
    132: "GLY",  # "G"
    133: "VAL",  # "V"
    134: "SER",  # "S"
    135: "GLU",  # "E"
    136: "ARG",  # "R"
    137: "THR",  # "T"
    138: "ILE",  # "I"
    139: "ASP",  # "D"
    140: "PRO",  # "P"
    141: "LYS",  # "K"
    142: "GLN",  # "Q"
    143: "ASN",  # "N"
    144: "PHE",  # "F"
    145: "TYR",  # "Y"
    146: "MET",  # "M"
    147: "HIS",  # "H"
    148: "TRP",  # "W"
    149: "CYS",  # "C"
    150: "UNK",  # "X"
    # other_code: "UNK",
}
NUM2SYM = {_: Chem.GetPeriodicTable().GetElementSymbol(_ + 1) for _ in range(118)}


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
        relaxed_sampled_structure: Optional[Any],
        original_structure: Optional[Any],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
        given_protein: bool = False,
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
                        f"Failed to generate molecule from sampled structure for index {index[i]}. {e}"
                    )
                    mol = None
                structures.append(mol)
        return structures

    def match(
        self,
        sampled_structure: Optional[Mol],
        relaxed_sampled_structure: Optional[Mol],
        original_structure: Optional[Mol],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
        given_protein: bool = False,
    ) -> float:
        if relaxed_sampled_structure is not None:
            logger.warning(
                "Matching behavior with relaxed structurs for molecule is not defined yet."
            )

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

    def get_space_group(self, structure: Structure, symprec: float = 0.1) -> str:
        try:
            return structure.get_space_group_info(symprec=symprec)[0]
        except TypeError:
            # space group analysis failed, most likely due to overlapping atoms
            return "P1"

    def match(
        self,
        sampled_structure: Optional[Atoms],
        relaxed_sampled_structure: Optional[Atoms],
        original_structure: Optional[Atoms],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
        given_protein: bool = False,
    ) -> float:
        if sampled_structure is None or original_structure is None:
            return {"rmsd": np.nan}
        rmsd = np.nan
        relaxed_rmsd = np.nan
        p1 = np.nan
        relaxed_p1 = np.nan
        try:
            sampled_structure.write(
                f"{sampled_structure_output_path}/sampled_{idx}_{sample_index}.cif",
                format="cif",
            )
            if relaxed_sampled_structure is not None:
                relaxed_sampled_structure.write(
                    f"{sampled_structure_output_path}/relaxed_sampled_{idx}_{sample_index}.cif",
                    format="cif",
                )
            original_structure.write(
                f"{sampled_structure_output_path}/original_{idx}_{sample_index}.cif",
                format="cif",
            )
            sampled_pymatgen_structure = Structure.from_ase_atoms(sampled_structure)
            p1 = (
                1.0 if self.get_space_group(sampled_pymatgen_structure) == "P1" else 0.0
            )
            original_pymatgen_structure = Structure.from_ase_atoms(original_structure)
            matcher = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)
            rmsd = matcher.get_rms_dist(
                sampled_pymatgen_structure, original_pymatgen_structure
            )
            if relaxed_sampled_structure is not None:
                relaxed_sampled_pymatgen_structure = Structure.from_ase_atoms(
                    relaxed_sampled_structure
                )
                relaxed_p1 = (
                    1.0
                    if self.get_space_group(relaxed_sampled_pymatgen_structure) == "P1"
                    else 0.0
                )
                relaxed_rmsd = matcher.get_rms_dist(
                    relaxed_sampled_pymatgen_structure, original_pymatgen_structure
                )
        except Exception as e:
            logger.info(
                f"Failed to align material for sampled structure with index {idx} {e}."
            )
        if rmsd is None:
            rmsd = np.nan
        elif isinstance(rmsd, tuple):
            rmsd = rmsd[0]

        ret = {
            "rmsd": rmsd,
            "p1": p1,
        }

        if relaxed_sampled_structure is not None:
            if relaxed_rmsd is None:
                relaxed_rmsd = np.nan
            elif isinstance(relaxed_rmsd, tuple):
                relaxed_rmsd = relaxed_rmsd[0]
            ret.update({"relaxed_rmsd": relaxed_rmsd, "relaxed_p1": relaxed_p1})
        return ret


@CONVERTER_REGISTER.register("protein")
class ProteinConverter(BaseConverter):
    def convert(self, batched_data: Dict[str, Tensor], poses: Tensor):
        num_residues = batched_data["num_atoms"].cpu().numpy()
        batch_size = num_residues.shape[0]
        structures: List[Optional[List[str]]] = []
        token_ids = batched_data["token_id"]
        keys = batched_data.get("key", ["TEMP"] * batch_size)
        for i in range(batch_size):
            try:
                pos = poses[i].cpu()
                pos = pos[: num_residues[i]]
                residue_ids = token_ids[i][: num_residues[i]].cpu().numpy()
                pdb_lines = [f"HEADER    {keys[i]}\n"]
                atomidx = 0
                chainid = "A"
                resnumb = 0
                for i, (x, y, z) in enumerate(pos):
                    record = "ATOM  "
                    symbol = "C"
                    atomname = " CA "
                    resnumb += 1
                    resname = VOCAB2AA.get(residue_ids[i], "UNK")
                    if np.isnan(x):
                        # Process missing residues in ground truth protein
                        continue
                    atomidx += 1
                    pdb_lines.append(
                        f"{record:<6s}{atomidx:>5d} {atomname:4s} {resname:3s} "
                        f"{chainid}{resnumb:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
                        f"  1.00  0.00          {symbol}  \n"
                    )
                pdb_lines.append("TER\n")
                pdb_lines.append("END\n")
                structures.append(pdb_lines)
            except Exception as e:
                logger.warning(f"Failed to sample for protein {keys[i]}, {e}")
                structures.append(None)
        return structures

    def match(
        self,
        sampled_structure: Optional[List[str]],
        relaxed_sampled_structure: Optional[List[str]],
        original_structure: Optional[List[str]],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
        given_protein: bool = False,
    ) -> float:
        if relaxed_sampled_structure is not None:
            logger.warning(
                "Matching behavior with relaxed structurs for protein is not defined yet."
            )
        rmsd, tm_score, lddt = np.nan, np.nan, np.nan
        try:
            assert (
                sampled_structure and sampled_structure[0][:6] == "HEADER"
            ), f"Wrong sample structure {sampled_structure[0]}"
            assert (
                original_structure and original_structure[0][:6] == "HEADER"
            ), f"Wrong original structure {original_structure[0]}"
            assert (
                sampled_structure[0] == original_structure[0]
            ), f"Wrong name for sample {sampled_structure[0]}"
            key = sampled_structure[0].split()[1]
            sampled_path = os.path.join(
                sampled_structure_output_path, f"{key}-{sample_index+1}.pdb"
            )
            with open(sampled_path, "w") as out_file:
                out_file.writelines(sampled_structure)
            original_path = os.path.join(
                sampled_structure_output_path, f"{key}-native.pdb"
            )
            with open(original_path, "w") as out_file:
                out_file.writelines(original_structure)
            lines = []
            lines.extend(
                subprocess.run(
                    f"TMscore {sampled_path} {original_path}",
                    shell=True,
                    capture_output=True,
                    text=True,
                ).stdout.split("\n")
            )
            lines.extend(
                subprocess.run(
                    f"lddt -c {sampled_path} {original_path}",
                    shell=True,
                    capture_output=True,
                    text=True,
                ).stdout.split("\n")
            )
            for line in lines:
                cols = line.split()
                if line.startswith("RMSD") and len(cols) > 5:
                    rmsd = float(cols[5])
                elif line.startswith("TM-score") and len(cols) > 2:
                    tm_score = float(cols[2])
                elif line.startswith("Global LDDT") and len(cols) > 3:
                    lddt = float(cols[3])
            logger.success(
                f"Sample={idx:3d}-{key:7s}, Model={sample_index+1}, "
                f"RMSD={rmsd:6.3f}, TM-score={tm_score:6.4f}, LDDT={lddt:6.4f}."
            )
        except Exception as e:
            logger.warning(f"Failed to evaluate sample {idx}, {e}.")
        return {"rmsd": rmsd, "tm_score": tm_score, "lddt": lddt}


@CONVERTER_REGISTER.register("complex")
class ComplexConverter(BaseConverter):
    def convert(self, batched_data: Dict[str, Tensor], poses: Tensor):
        def _num2str(num: int) -> str:
            if 0 <= num - 1 < 26:
                return chr(ord("A") + num - 1 - 0)
            elif 26 <= num - 1 < 52:
                return chr(ord("a") + num - 1 - 26)
            elif 52 <= num - 1 < 62:
                return chr(ord("0") + num - 1 - 52)
            else:
                raise ValueError("More than 62 chains.")

        num_atoms = batched_data["num_atoms"].cpu().numpy()
        batch_size = num_atoms.shape[0]
        structures: List[Optional[List[str]]] = []
        keys = batched_data.get("key", ["TEMP"] * batch_size)
        for i in range(batch_size):
            pdb_lines = [f"HEADER    {keys[i]}\n"]
            try:
                pos = poses[i][: num_atoms[i]].cpu().numpy()
                tok = batched_data["token_id"][i][: num_atoms[i]].cpu().numpy()
                msk = batched_data["is_protein"][i][: num_atoms[i]].cpu().numpy()
                ids = batched_data["chain_ids"][i][: num_atoms[i]].cpu().numpy()
                pdb_lines = [f"HEADER    {keys[i]}\n"]
                atomidx = 0
                chainid = "A"
                resnumb = 0
                ligatom = collections.defaultdict(int)
                for idx, (x, y, z) in enumerate(pos):
                    if _num2str(ids[idx]) != chainid:
                        pdb_lines.append("TER\n")
                        chainid = _num2str(ids[idx])
                        resnumb = 0
                        ligatom = collections.defaultdict(int)
                    if msk[idx]:
                        record = "ATOM  "
                        symbol = "C"
                        atomname = " CA "
                        resnumb += 1
                        resname = VOCAB2AA.get(tok[idx], "UNK")
                    else:
                        record = "HETATM"
                        symbol = NUM2SYM.get(tok[idx] - 2, "X")
                        ligatom[symbol] += 1
                        atomname = f"{symbol.upper():>2s}{ligatom[symbol]:<2d}"
                        resnumb = resnumb
                        resname = "LIG"
                    if np.isnan(x):
                        # Process missing residues in ground truth protein
                        continue
                    atomidx += 1
                    pdb_lines.append(
                        f"{record:<6s}{atomidx:>5d} {atomname:4s} {resname:3s} "
                        f"{chainid}{resnumb:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}"
                        f"  1.00  0.00          {symbol}  \n"
                    )
                pdb_lines.append("END\n")
            except Exception as e:
                logger.warning(f"Failed to sample for protein {keys[i]}, {e}")
        structures.append(pdb_lines)
        return structures

    def match(
        self,
        sampled_structure: Optional[List[str]],
        relaxed_sampled_structure: Optional[List[str]],
        original_structure: Optional[List[str]],
        idx: int,
        sampled_structure_output_path: Optional[str] = None,
        sample_index: Optional[int] = -1,
        given_protein: bool = False,
    ) -> float:
        if relaxed_sampled_structure is not None:
            logger.warning(
                "Matching behavior with relaxed structurs for complex is not defined yet."
            )

        def _get_xyz(atomlines: list):
            protpos, ligpos = [], []
            for line in atomlines:
                if line[:6] not in ("ATOM  ", "HETATM"):
                    continue
                xyz = float(line[30:38]), float(line[38:46]), float(line[46:54])
                key = line[12:16] + line[21:26]
                if line[:6] == "ATOM  ":
                    protpos.append((key, xyz))
                else:
                    ligpos.append((key, xyz))
            return protpos, ligpos

        def _calc_rmsd(x1, x2, ref1=None, ref2=None):
            if ref1 is None or ref2 is None:
                ref1, ref2 = x1, x2
            sup = SVDSuperimposer()
            sup.set(ref1, ref2)
            sup.run()
            rot, tran = sup.get_rotran()
            x2_t = np.dot(x2, rot) + tran
            rmsd = np.sqrt(((((x1 - x2_t) ** 2)) * 3).mean())
            return rmsd

        pocket_aligned_rmsd, tm_score = np.nan, np.nan
        # try:
        assert (
            sampled_structure and sampled_structure[0][:6] == "HEADER"
        ), f"Wrong sample structure {sampled_structure[0]}"
        assert (
            original_structure and original_structure[0][:6] == "HEADER"
        ), f"Wrong original structure {original_structure[0]}"
        assert (
            sampled_structure[0] == original_structure[0]
        ), f"Wrong name for sample {sampled_structure[0]}"
        key = sampled_structure[0].split()[1]
        sampled_path = os.path.join(
            sampled_structure_output_path, f"{key}-{sample_index+1}.pdb"
        )
        with open(sampled_path, "w") as out_file:
            out_file.writelines(sampled_structure)
        original_path = os.path.join(sampled_structure_output_path, f"{key}-native.pdb")
        with open(original_path, "w") as out_file:
            out_file.writelines(original_structure)

        # extract positions for common protein residues and ligand atoms
        sampled_protein, sampled_ligand = _get_xyz(sampled_structure)
        original_protein, original_ligand = _get_xyz(original_structure)

        commprt = set([_[0] for _ in sampled_protein]) & set(
            [_[0] for _ in original_protein]
        )
        commlig = set([_[0] for _ in sampled_ligand]) & set(
            [_[0] for _ in original_ligand]
        )
        smplprt = np.array([_[1] for _ in sampled_protein if _[0] in commprt])
        smpllig = np.array([_[1] for _ in sampled_ligand if _[0] in commlig])
        origprt = np.array([_[1] for _ in original_protein if _[0] in commprt])
        origlig = np.array([_[1] for _ in original_ligand if _[0] in commlig])

        # calculate Kabsch RMSD between sampled ligand and original ligand
        kabsch_rmsd = _calc_rmsd(smpllig, origlig)
        # calculate pocket aligned RMSD
        dist = np.linalg.norm(smplprt[:, None, :] - smpllig[None, :, :], axis=-1)

        if given_protein:
            mask = np.min(dist, axis=-1) < 1000  # 1000 Angstrom
        else:
            mask = np.min(dist, axis=-1) < 10  # 10 Angstrom

        smpl_pocket, orig_pocket = smplprt[mask], origprt[mask]
        assert len(smpl_pocket) >= 1, f"Cannot find pocket atoms for {key}."
        pocket_aligned_rmsd = _calc_rmsd(smpllig, origlig, smpl_pocket, orig_pocket)
        pocket_ref_rmsd = _calc_rmsd(smpl_pocket, orig_pocket)
        # calculate TM-score on protein
        with (
            tempfile.NamedTemporaryFile() as predpdb,
            tempfile.NamedTemporaryFile() as natipdb,
        ):
            with open(predpdb.name, "w") as fp:
                fp.writelines([_ for _ in sampled_structure if _[:6] != "HETATM"])
            with open(natipdb.name, "w") as fp:
                fp.writelines([_ for _ in original_structure if _[:6] != "HETATM"])
            lines = []
            lines.extend(
                subprocess.run(
                    f"TMscore {predpdb.name} {natipdb.name}",
                    shell=True,
                    capture_output=True,
                    text=True,
                ).stdout.split("\n")
            )
            for line in lines:
                cols = line.split()
                if line.startswith("TM-score") and len(cols) > 2:
                    tm_score = float(cols[2])

        logger.success(
            f"Sample={idx:3d}-{key:7s}, Model={sample_index+1}, "
            f"TM-score={tm_score:6.4f}, "
            f"Kabsch-RMSD={kabsch_rmsd:6.3f}, "
            f"Pocket-RMSD={pocket_ref_rmsd:6.3f}, "
            f"Pocket-aligned-RMSD={pocket_aligned_rmsd:6.3f}."
        )
        # except Exception as e:
        #     logger.warning(f"Failed to evaluate sample {idx}, {e}.")
        return {"rmsd": pocket_aligned_rmsd, "tm_score": tm_score}


class SampledStructureConverter:
    def __init__(
        self,
        sampled_structure_output_path: Optional[str],
        psm_config: PSMConfig,
        model: Module,
    ) -> None:
        self.sampled_structure_output_path = sampled_structure_output_path
        if self.sampled_structure_output_path is not None:
            os.makedirs(self.sampled_structure_output_path, exist_ok=True)
        exitcode, output = subprocess.getstatusoutput("which TMscore")
        if exitcode != 0:
            raise ValueError(f"Program 'TMscore' not installed, {output}.")
        exitcode, output = subprocess.getstatusoutput("which lddt")
        if exitcode != 0:
            raise ValueError(f"Program 'lddt' not installed, {output}.")
        self.psm_config = psm_config
        self.given_protein = self.psm_config.sample_ligand_only
        self.model = model
        self.relaxers = {
            "protein": None,
            "periodic": None,
            "molecule": None,
            "complex": None,
        }
        if self.psm_config.relax_after_sampling_structure:
            for key in self.relaxers:
                if key in RELAXER_REGISTER:
                    self.relaxers[key] = RELAXER_REGISTER[key](psm_config)
                else:
                    logger.warning(f"No relaxer for {key} systems.")

    def convert_and_match(
        self,
        batched_data: Dict[str, Tensor],
        original_pos: Tensor,
        sample_index: int,
    ) -> Tensor:
        batch_size = batched_data["is_molecule"].size()[0]
        all_results = [None for _ in range(batch_size)]
        for system_tag in ["molecule", "periodic", "protein", "complex"]:
            is_mask = batched_data[f"is_{system_tag}"]
            if system_tag == "protein":
                is_mask = batched_data["is_protein"].any(dim=1) & (
                    ~batched_data["is_complex"]
                )
            indexes_in_batch = is_mask.nonzero().squeeze(-1)
            if torch.any(is_mask):
                indexes = batched_data["idx"][is_mask]
                sampled_structures = CONVERTER_REGISTER[system_tag]().convert(
                    batched_data, batched_data["pos"]
                )
                original_structures = CONVERTER_REGISTER[system_tag]().convert(
                    batched_data, original_pos
                )
                relaxed_sampled_structures = (
                    [
                        self.relaxers[system_tag].relax(
                            atoms=sampled_structure, model=self.model
                        )
                        for sampled_structure in sampled_structures
                    ]
                    if self.relaxers[system_tag] is not None
                    else [None for _ in sampled_structures]
                )
                for (
                    sampled_structure,
                    relaxed_sampled_structure,
                    original_structure,
                    index,
                    index_in_batch,
                ) in zip(
                    sampled_structures,
                    relaxed_sampled_structures,
                    original_structures,
                    indexes,
                    indexes_in_batch,
                ):
                    all_results[index_in_batch] = CONVERTER_REGISTER[
                        system_tag
                    ]().match(
                        sampled_structure,
                        relaxed_sampled_structure,
                        original_structure,
                        int(index),
                        self.sampled_structure_output_path,
                        sample_index,
                        self.given_protein,
                    )
        return all_results
