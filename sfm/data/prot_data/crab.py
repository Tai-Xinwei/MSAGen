# -*- coding: utf-8 -*-
# import some necessary packages
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import Bio.PDB
import Bio.PDB.Atom
import Bio.PDB.Chain
import Bio.PDB.Residue
import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBIO, MMCIFParser, PDBParser
from Bio.PDB.mmcifio import MMCIFIO

"""
Structure Entity: Level S
    |
    --> Chain Entity: Level C
        |
        --> Residue Entity: Level R
                |
                |--> Atom Entity: Level A
                # --> Bond Entity: Level B, for future use, not used yet

CRAB representation
A complex can be represented as a collection of chains, which are collections of residues, which are collections of atoms and bonds.


C tensor: (B, N_res), padded sequence of amino acid residues.
R tensor: (B, N_res), padded chain level mapping of residues.
A tensor: (B, N_res, N_atom, 3), padded coordinates of atoms, the last dimension means cartesian coordinates (x, y, z).
# B tensor: (B, N_bond, 3), sparse matrix recording bonds between atoms, the last dimension means (atom_idx1, atom_idx2, bondtype). Maybe only need to store bonds from non-protein ligands.

Template from BioPython:
1. IO: pdb, cif, lmdb input/output. Ref: Alphafold2, Chroma. https://biopython.org/DIST/docs/tutorial/Tutorial-1.83.html
2. Differenciable internal coordinate <--> Cartesian coordinate conversion, nn.Module

"""


# define the amino acid sequence
AA20_1 = list("ACDEFGHIKLMNPQRSTVWY")
AA20_3 = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]
assert len(AA20_1) == len(AA20_3)
AA20_3_TO_1 = {k: v for k, v in zip(AA20_3, AA20_1)}
AA20_1_TO_3 = {k: v for k, v in zip(AA20_1, AA20_3)}

ATOMS_BB = ["N", "CA", "C", "O"]
CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"


# statistics compiled from https://github.com/biopython/biopython/blob/master/Bio/PDB/ic_data.py
amino_acid_dict = {
    "A": torch.tensor([1.46120, 1.52579, 1.33094]),
    "C": torch.tensor([1.45964, 1.52388, 1.33066]),
    "D": torch.tensor([1.46141, 1.52681, 1.33107]),
    "E": torch.tensor([1.46058, 1.52593, 1.33073]),
    "F": torch.tensor([1.46000, 1.52457, 1.33064]),
    "G": torch.tensor([1.45534, 1.51677, 1.33072]),
    "H": torch.tensor([1.46046, 1.52435, 1.33078]),
    "I": torch.tensor([1.46029, 1.52636, 1.33089]),
    "K": torch.tensor([1.46080, 1.52595, 1.33064]),
    "L": torch.tensor([1.46035, 1.52503, 1.33088]),
    "M": torch.tensor([1.46091, 1.52499, 1.33086]),
    "N": torch.tensor([1.46048, 1.52569, 1.33090]),
    "P": torch.tensor([1.46668, 1.52595, 1.33171]),
    "Q": torch.tensor([1.46078, 1.52562, 1.33069]),
    "R": torch.tensor([1.46051, 1.52541, 1.33064]),
    "S": torch.tensor([1.46027, 1.52539, 1.33098]),
    "T": torch.tensor([1.45979, 1.52573, 1.33079]),
    "V": torch.tensor([1.46011, 1.52633, 1.33078]),
    "W": torch.tensor([1.46029, 1.52458, 1.33076]),
    "Y": torch.tensor([1.45996, 1.52434, 1.33050]),
    "<pad>": torch.tensor([torch.inf, torch.inf, torch.inf]),
}

amino_acid_dict_angle = {
    "A": torch.tensor([111.05839, 116.69299, 121.43998]),
    "C": torch.tensor([110.93113, 116.58489, 121.51876]),
    "D": torch.tensor([111.03743, 116.74172, 121.56274]),
    "E": torch.tensor([111.15787, 116.74841, 121.45065]),
    "F": torch.tensor([110.81237, 116.58552, 121.57766]),
    "G": torch.tensor([113.13469, 116.65649, 121.38416]),
    "H": torch.tensor([111.09335, 116.66524, 121.56887]),
    "I": torch.tensor([109.82832, 116.66184, 121.58807]),
    "K": torch.tensor([111.08063, 116.70069, 121.55854]),
    "L": torch.tensor([110.90523, 116.73371, 121.48377]),
    "M": torch.tensor([110.97184, 116.72073, 121.38091]),
    "N": torch.tensor([111.52947, 116.69676, 121.61726]),
    "P": torch.tensor([112.64045, 116.77550, 120.33021]),
    "Q": torch.tensor([111.10228, 116.71033, 121.45675]),
    "R": torch.tensor([111.01141, 116.68310, 121.45120]),
    "S": torch.tensor([111.24019, 116.64870, 121.49507]),
    "T": torch.tensor([110.71436, 116.64426, 121.53022]),
    "V": torch.tensor([109.79444, 116.61840, 121.61608]),
    "W": torch.tensor([110.90325, 116.66104, 121.52771]),
    "Y": torch.tensor([110.93364, 116.57590, 121.64133]),
    "<pad>": torch.tensor([torch.inf, torch.inf, torch.inf]),
}


class BondLengthCalculator(nn.Module):
    """Returns the bond length.
    Inputs:
    * c1: (batch, 3) or (3,)
    * c2: (batch, 3) or (3,)
    """

    def __init__(self):
        super(BondLengthCalculator, self).__init__()

    def forward(self, c1: torch.Tensor, c2: torch.Tensor):
        """Calculate the bond length between two atoms."""
        return torch.norm(c1 - c2, dim=-1)


class BondAngleCalculator(nn.Module):
    """Returns the angle in radians.
    Inputs:
    * c1: (batch, 3) or (3,)
    * c2: (batch, 3) or (3,)
    * c3: (batch, 3) or (3,)
    """

    def __init__(self):
        super(BondAngleCalculator, self).__init__()

    def forward(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor):
        """Calculate the bond angle between three atoms."""
        u1 = c2 - c1
        u2 = c3 - c2
        return torch.atan2(
            torch.norm(torch.cross(u1, u2, dim=-1), dim=-1), -(u1 * u2).sum(dim=-1)
        )


class DihedralAngleCalculator(nn.Module):
    """Returns the dihedral angle in radians.
    Inputs:
    * c1: (batch, 3) or (3,)
    * c2: (batch, 3) or (3,)
    * c3: (batch, 3) or (3,)
    * c4: (batch, 3) or (3,)
    """

    def __init__(self):
        super(DihedralAngleCalculator, self).__init__()

    def forward(
        self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor
    ):
        """Calculate the dihedral angle between four atoms."""
        u1 = c2 - c1
        u2 = c3 - c2
        u3 = c4 - c3
        return torch.atan2(
            (
                (torch.norm(u2, dim=-1, keepdim=True) * u1)
                * torch.cross(u2, u3, dim=-1)
            ).sum(dim=-1),
            (torch.cross(u1, u2, dim=-1) * torch.cross(u2, u3, dim=-1)).sum(dim=-1),
        )


#  use three atoms to get the coordinates of the forth atom
class FourthAtomCalculator(nn.Module):
    def __init__(self, eps: float):
        super(FourthAtomCalculator, self).__init__()
        self.eps = eps

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        l: torch.Tensor,
        theta: torch.Tensor,
        chi: torch.Tensor,
    ):
        """Calculate the coordinates of the fourth atom given three atoms and the bond length, bond angle, and dihedral angle."""

        # if not ((-torch.pi <= theta) * (theta <= torch.pi)).all().item():
        #     raise ValueError(
        #         f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}"
        #     )
        ba = b - a + self.eps
        cb = c - b + self.eps

        n_plane = torch.cross(ba, cb, dim=-1)
        n_plane_ = torch.cross(n_plane, cb, dim=-1) + self.eps
        rotate = torch.stack([cb, n_plane_, n_plane], dim=-1)
        norm = torch.norm(rotate, dim=-2, keepdim=True) + self.eps
        rotate = rotate / norm

        d = torch.stack(
            [
                -torch.cos(theta),
                torch.sin(theta) * torch.cos(chi),
                torch.sin(theta) * torch.sin(chi),
            ],
            dim=-1,
        ).unsqueeze(-1)

        return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()


# Entity class is the parent class of all the classes
@dataclass
class Entity:
    _id: str | int
    _level: str
    _child_dict: OrderedDict
    _parent: List["Entity"] | None

    def __init__(self, id: str, parent: List["Entity"] = None):
        """Initialize the class. Levels are set by the subclass, not the parent class."""
        self._id = id
        self._child_dict = OrderedDict()
        self._parent = parent

    def __init_subclass__(cls, level: str, **kwargs):
        """Set the level for every subclass."""
        cls._level = level
        super().__init_subclass__(**kwargs)

    def __len__(self):
        return len(self._child_dict)

    def __getitem__(self, id):
        return self._child_dict[id]

    def __contains__(self, id):
        return id in self._child_dict

    def __iter__(self):
        yield from self._child_dict.values()

    def items(self):
        yield from self._child_dict.items()

    @property
    def id(self):
        """Return the read-only id."""
        return self._id

    @property
    def level(self):
        """Return the read-only level."""
        return self._level

    @property
    def child_dict(self):
        """Return the read-only child dictionary."""
        return self._child_dict

    @property
    def parent(self):
        """Return the read-only parent Entity object."""
        return self._parent


@dataclass
class Structure(Entity, level="St"):
    """The Structure class contains a collection of Chain instances."""

    def __init__(self, id: str):
        """Initialize the class."""
        super().__init__(id=id)

    def validate(self):
        """Validate: warning when this is an empty structure, duplicate chains."""
        if len(self) == 0:
            print(f"Warning: Empty structure {self}")
        for chain in self:
            chain.validate()

    @classmethod
    def from_CRAB(cls, C: np.ndarray, R: np.ndarray, A: np.ndarray):
        """Create a Structure instance from CRAB tensors."""
        # C tensor: (N_res), padded chain level mapping of residues.
        # R tensor: (N_res), padded sequence of amino acid residues.
        # A tensor: (N_res, N_atom, 3), padded coordinates of atoms, the last dimension means cartesian coordinates (x, y, z).

        assert (
            len(C.shape) == 1
        ), f"C should have a shape of (N_res), but got shape {C.shape}"
        assert (
            len(R.shape) == 1
        ), f"R should have a shape of (N_res), but got shape {R.shape}"
        assert (
            len(A.shape) == 3
        ), f"A should have a shape of (N_res, N_atom, 3), but got shape {A.shape}"
        assert A.shape[0] == R.shape[0] == C.shape[0]

        # create a new Structure instance
        obj = Structure(id="new_structure")
        for chain_idx, chain_id in enumerate(np.unique(C)):
            chain_bool = C == chain_id
            R_chain, A_chain = R[chain_bool], A[chain_bool]
            chain_new = Chain(
                id=chain_idx, parent=obj, chainname=CHAIN_ALPHABET[chain_id]
            )
            for res_idx, res_id in enumerate(R_chain):
                residue_new = Residue(id=res_idx, parent=chain_new, resname=res_id)
                for k, atom_coord in enumerate(A_chain[res_idx]):
                    atom_new = Atom(
                        id=ATOMS_BB[k],
                        parent=residue_new,
                        coord=np.array(atom_coord),
                        bfactor=10.00,
                    )
                    residue_new._child_dict[k] = atom_new
                chain_new._child_dict[residue_new.id] = residue_new
            obj._child_dict[chain_new.chainname] = chain_new
        return obj

    def to_CRAB(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CRAB tensors from Structure instance."""
        # C tensor: (N_res), padded chain level mapping of residues.
        # R tensor: (N_res), padded sequence of amino acid residues.
        # A tensor: (N_res, N_atom, 3), padded coordinates of atoms, the last dimension means cartesian coordinates (x, y, z).
        # TODO: add full-atom support, currently, only ATOMS_BB are considered.

        C = []
        for ch_idx, (ch_str, chain) in enumerate(self.items()):
            C.extend([ch_idx] * len(chain))

        R = [residue.resname for chain in self for residue in chain]
        A = []  # np.zeros([len(R), 4, 3])
        index_res = -1
        for chain in self:
            for residue in chain:
                index_res += 1

                bb_coord = []
                for i, atom_str in enumerate(ATOMS_BB):
                    atom = residue[atom_str]
                    bb_coord.append(atom.coord)
                A.append(bb_coord)

        C = np.array(C)
        R = np.array(R)
        A = np.array(A)

        assert (
            C.shape[0] == R.shape[0] == A.shape[0]
        ), f"Shape mismatch: C {C.shape}, R {R.shape}, A {A.shape}"

        return C, R, A

    @classmethod
    def from_file(cls, file_path: Path | str):
        """Create a Structure instance from a file."""
        # based on the file extension, use the appropriate parser
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        if file_path.suffix == ".pdb":
            parser = PDBParser()
        elif file_path.suffix == ".cif":
            parser = MMCIFParser()
        else:
            raise ValueError(
                f"Unsupported file format: {file_path}, only .pdb and .cif are supported."
            )

        structure = parser.get_structure(file_path.stem, file_path)
        obj = cls(id=file_path.stem)
        # load only the first model
        for chain in structure[0]:
            obj.add_chain(chain)
        obj.validate()
        return obj

    def to_file(self, file_path: str, format: str = "pdb"):
        """Save the Structure instance to a PDB file."""
        if format not in ["pdb", "cif"]:
            raise ValueError(
                f"Unsupported file format: {format}, only pdb and cif are supported."
            )
        if format == "pdb":
            pdb_io = PDBIO()
        elif format == "cif":
            pdb_io = MMCIFIO()

        bio_structure = Bio.PDB.Structure.Structure(self.id)
        bio_model = Bio.PDB.Model.Model(0)
        serial_num = 1
        for chain in self:
            bio_chain = Bio.PDB.Chain.Chain(chain.chainname)
            for residue in chain:
                bio_residue = Bio.PDB.Residue.Residue(
                    (" ", residue.id + 1, " "), residue.resname, " "
                )
                for atom in residue:
                    bio_atom = Bio.PDB.Atom.Atom(
                        atom.id,
                        atom.coord,
                        atom.bfactor,
                        1.0,
                        " ",
                        f" {atom.id} ",
                        serial_num,
                    )
                    bio_residue.add(bio_atom)
                    serial_num += 1
                bio_chain.add(bio_residue)
            bio_model.add(bio_chain)
        bio_structure.add(bio_model)
        pdb_io.set_structure(bio_structure)
        pdb_io.save(file_path)

    def add_chain(self, chain: Bio.PDB.Chain.Chain):
        """Add a Chain instance to the Structure instance."""
        # create a new Chain instance, and record chain to this instance
        chain_new = Chain(id=len(self._child_dict), parent=self, chainname=chain.id)
        for residue in chain:
            chain_new.add_residue(residue)
        self._child_dict[chain.id] = chain_new


@dataclass
class Chain(Entity, level="Ch"):
    _chainname: str
    """The Chain class contains a collection of Residue instances."""

    def __init__(self, id: int, parent: Structure, chainname: str = "A"):
        """Initialize the class."""
        super().__init__(id=id, parent=parent)
        self._chainname = chainname

    def add_residue(self, residue: Bio.PDB.Residue.Residue):
        """Add a Residue instance to the Chain instance."""
        # create a new Residue instance, and record residue to this instance
        # TODO: add support for hetero_flag and ins_code
        hetero_flag, seq_id, ins_code = residue.id
        if not hetero_flag == " ":
            print(
                f"Only standard amino acids are supported, not {seq_id} with resname {residue.get_resname()}, skipping."
            )
            return

        assert (
            ins_code == " "
        ), f"Insertion codes are not supported, not {ins_code} at {seq_id} with resname {residue.get_resname()}"
        residue_new = Residue(
            id=len(self._child_dict), parent=self, resname=residue.get_resname()
        )
        for atom in residue:
            residue_new.add_atom(atom)
        self._child_dict[residue_new.id] = residue_new

    def validate(self):
        """Validate: warning when this is an empty chain, missing residues."""
        if len(self) == 0:
            print(f"Warning: Empty chain {self} in structure {self.parent}")
        for i, j in zip(range(len(self)), range(1, len(self))):
            if self[j].id - self[i].id != 1:
                print(
                    f"Warning: Missing residue between {i} and {j} with id {self[j].id} and {self[i].id} in chain {self} in structure {self.parent}"
                )
        for residue in self:
            residue.validate()

    @property
    def chainname(self):
        """Return the read-only chainname."""
        return self._chainname

    def __repr__(self):
        """Return the chain identifier."""
        return f"<Chain id={self.id} chainname={self.chainname}>"


@dataclass
class Residue(Entity, level="Re"):
    _resname: str

    """The Residue class contains a collection of Atom instances."""

    def __init__(self, id: int, parent: Chain, resname: str = "UNK"):
        """Initialize the class."""
        super().__init__(id=id, parent=parent)
        self._resname = resname

    def add_atom(self, atom: Bio.PDB.Atom.Atom):
        """Add an Atom instance to the Residue instance."""
        # create a new Atom instance, and record atom to this instance
        # TODO: add support for alt_loc
        assert atom.get_altloc() in {
            " ",
            "A",
        }, f"Currently only the first occurance of an atom location is kept, got '{atom.get_altloc()}' at {atom.get_name()} with resname {self._resname}"
        atom_new = Atom(
            id=atom.get_name(),
            parent=self,
            coord=atom.get_coord(),
            bfactor=atom.get_bfactor(),
        )
        self._child_dict[atom.get_name()] = atom_new

    def validate(self):
        """Validate: warning when this is an empty residue, non-standard residue, missing backbone atoms."""
        if len(self) == 0:
            print(f"Warning: Empty residue {self} in chain {self.parent}")
        if self.resname not in AA20_3_TO_1:
            print(
                f"Warning: Non-standard residue {self.resname} in chain {self.parent}"
            )
        for atom_name in ["N", "CA", "C", "O"]:
            if atom_name not in self:
                print(
                    f"Warning: Missing backbone atom {atom_name} in residue {self} in chain {self.parent}"
                )
        for atom in self:
            atom.validate()

    @property
    def resname(self):
        """Return the read-only resname."""
        return self._resname

    def __repr__(self):
        """Return the residue identifier."""
        return f"<Residue id={self.id} resname={self.resname} >"


@dataclass
class Atom(Entity, level="At"):
    _coord: np.ndarray
    _bfactor: float
    _parent: str

    def __init__(self, id: str, parent: Residue, coord: np.ndarray, bfactor: float):
        """Initialize the class."""
        super().__init__(id=id, parent=parent)
        self._coord = coord
        self._bfactor = bfactor

    def validate(self):
        "Nothing to validate for Atom instances."
        pass

    @property
    def coord(self):
        """Return the read-only coordinate."""
        return self._coord

    @property
    def bfactor(self):
        """Return the read-only bfactor."""
        return self._bfactor

    @property
    def level(self):
        """Return the read-only level."""
        return self._level

    def __repr__(self):
        """Return the atom identifier."""
        return f"<Atom id={self.id} atom_type={self.atom_type} element={self.element}>"


# This class is for future use when we need to add bonds between atoms for non-protein ligands.
@dataclass
class Bond(Entity, level="Bo"):
    class BondType(Enum):
        SINGLE = 1
        DOUBLE = 2
        TRIPLE = 3
        AROMATIC = 1.5

    _bond_type: str

    def __init__(self, id: int, bond_type: BondType, atom1: Atom, atom2: Atom):
        """Initialize the class."""
        self._child_dict = OrderedDict()
        self._parent = [atom1, atom2]
        self._id = id
        self.bond_type = bond_type

    @property
    def bond_type(self):
        """Return the read-only bond_type."""
        return self._bond_type

    @property
    def bond_order(self):
        """Return the read-only bond-order."""
        return self._bond_type.value

    def __repr__(self):
        """Return the bond identifier."""
        return f"<Bond id={self._id} bond_type={self.bond_type} {self.parent[0]}-{self.parent[1]} >"


# API Marker: get the Structure instance from a file
# a = Structure.from_file('/home/ruohan/UnifyEverything/data/1WBK.cif')

if __name__ == "__main__":
    blc = BondLengthCalculator()
    bac = BondAngleCalculator()
    dac = DihedralAngleCalculator()
    fac = FourthAtomCalculator()

    # test in large dataset
    directory = Path(
        "/mnta/ruohan/cullpdb_pc50.0_res0.0-2.0_noBrks_len40-10000_R0.25_Xray_d2024_03_18_chains15946_20240422/"
    )
    n = 0
    rmsd_all = []
    # get all the files in the directory
    for filepath in directory.glob("*.cif"):
        n += 1
        a = Structure.from_file(filepath)
        C, R, A = a.to_CRAB()

        C = torch.from_numpy(C)
        R = torch.from_numpy(R)
        A = torch.from_numpy(A)

        # B is used to store the coordinates got from NeRF
        B = A.clone()

        ATOMS_BB = ["N", "CA", "C", "O"]

        # calculate the cartesian coordinates from internal coordinates for 1WBK Chain A
        for i in range(R.size(-2) - 1):
            # print(R[0][i].item())
            for j in range(3):
                # the situation of the first atom N in the residue
                if j == 0:
                    bond_length = blc(A[i][2], A[i + 1][0])
                    # bond_length = torch.tensor(1.3289373)
                    # bond_length = amino_acid_dict[R[0][i].item()][2]
                    bond_angle = bac(A[i][1], A[i][2], A[i + 1][0])
                    # bond_angle = torch.tensor(2.03)
                    # bond_angle = torch.deg2rad(amino_acid_dict_angle[R[0][i].item()][1])
                    dihedral_angle = dac(
                        A[0][i][0], A[0][i][1], A[0][i][2], A[0][i + 1][0]
                    )
                    B[0][i + 1][j] = fac(
                        B[0][i][0],
                        B[0][i][1],
                        B[0][i][2],
                        bond_length,
                        bond_angle,
                        dihedral_angle,
                    )

                # the situation of the second atom CA in the residue
                elif j == 1:
                    # bond_length = blc(A[0][i+1][0], A[0][i+1][j])
                    # bond_length = torch.tensor(1.4564931)
                    bond_length = amino_acid_dict[R[0][i + 1].item()][0]
                    bond_angle = bac(A[0][i][2], A[0][i + 1][0], A[0][i + 1][j])
                    # bond_angle = torch.tensor(2.08)
                    # bond_angle = torch.deg2rad(amino_acid_dict_angle[R[0][i+1].item()][2])
                    dihedral_angle = dac(
                        A[0][i][1], A[0][i][2], A[0][i + 1][0], A[0][i + 1][j]
                    )
                    B[0][i + 1][j] = fac(
                        B[0][i][1],
                        B[0][i][2],
                        B[0][i + 1][0],
                        bond_length,
                        bond_angle,
                        dihedral_angle,
                    )

                # the situation of the third atom C in the residue
                elif j == 2:
                    # bond_length = blc(A[0][i+1][1], A[0][i+1][j])
                    # bond_length = torch.tensor(1.524119)
                    bond_length = amino_acid_dict[R[0][i + 1].item()][1]
                    bond_angle = bac(A[0][i + 1][0], A[0][i + 1][1], A[0][i + 1][j])
                    # bond_angle = torch.tensor(1.9392343664169312)
                    # bond_angle = torch.deg2rad(amino_acid_dict_angle[R[0][i+1].item()][0])
                    dihedral_angle = dac(
                        A[0][i][2], A[0][i + 1][0], A[0][i + 1][1], A[0][i + 1][j]
                    )
                    B[0][i + 1][j] = fac(
                        B[0][i][2],
                        B[0][i + 1][0],
                        B[0][i + 1][1],
                        bond_length,
                        bond_angle,
                        dihedral_angle,
                    )

        rmsd = 0
        # calculate the RMSD
        for i in range(R.size(-2)):
            for j in range(3):
                # print(B[0][i][j])
                # print(A[0][i][j])
                for k in range(3):
                    rmsd = rmsd + (A[0][i][j][k] - B[0][i][j][k]) ** 2

        rmsd = np.sqrt(rmsd / (R.size(-2) * 3))

        print(rmsd)
        rmsd_all.append(rmsd)
        print(n)
        break

    # get the average RMSD
    rmsd_all.append(torch.tensor(float("nan")))
    print(rmsd_all)
    valid_values = [v for v in rmsd_all if torch.isfinite(v) and not torch.isnan(v)]
    print(np.mean(valid_values))

    # check the Structure instance
    # for chain in a:
    #     print(chain.id)
    #     for residue in chain:
    #         print(residue.id)
    #         for atom in residue:
    #             print(atom.id)
    #             print(atom.serial_number)
    #             print(atom.atom_type)
    #             print(atom.parent.id)
    #             print(atom.coord)
    #             print(atom.bfactor)
    #             print(atom.element)

    # check the Output functions
    # b.to_pdb('/home/ruohan/UnifyEverything/data/1WBK_test_fromCRA.cif')
