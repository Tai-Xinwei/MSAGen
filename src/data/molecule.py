from torch.utils.data import Data
from torch import LongTensor, FloatTensor
from typing import Union
import numpy as np
from rdkit import Chem
from pathlib import Path
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
import rdkit
from rdkit.Chem.rdmolops import RemoveHs
import torch
from rdkit.Chem.rdchem import RWMol

ATOM_FEAT_T = Union[LongTensor, np.array]
BOND_FEAT_T = Union[LongTensor, np.array]
BOND_INDEX_T = Union[LongTensor, np.array]
ATOM_POS_T = Union[FloatTensor, np.array]


# stores a mapping from bond order to rdkit bond type
bond_dict = {
    0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
    1: rdkit.Chem.rdchem.BondType.SINGLE,
    2: rdkit.Chem.rdchem.BondType.DOUBLE,
    3: rdkit.Chem.rdchem.BondType.TRIPLE,
    4: rdkit.Chem.rdchem.BondType.QUADRUPLE,
    5: rdkit.Chem.rdchem.BondType.QUINTUPLE,
    6: rdkit.Chem.rdchem.BondType.HEXTUPLE,
    7: rdkit.Chem.rdchem.BondType.ONEANDAHALF,
    8: rdkit.Chem.rdchem.BondType.TWOANDAHALF,
    9: rdkit.Chem.rdchem.BondType.THREEANDAHALF,
    10: rdkit.Chem.rdchem.BondType.FOURANDAHALF,
    11: rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
    12: rdkit.Chem.rdchem.BondType.AROMATIC,
    13: rdkit.Chem.rdchem.BondType.IONIC,
    14: rdkit.Chem.rdchem.BondType.HYDROGEN,
    15: rdkit.Chem.rdchem.BondType.THREECENTER,
    16: rdkit.Chem.rdchem.BondType.DATIVEONE,
    17: rdkit.Chem.rdchem.BondType.DATIVE,
    18: rdkit.Chem.rdchem.BondType.DATIVEL,
    19: rdkit.Chem.rdchem.BondType.DATIVER,
    20: rdkit.Chem.rdchem.BondType.OTHER,
    21: rdkit.Chem.rdchem.BondType.ZERO
}


def mol2graph(mol):
    """
    Modified from https://github.com/snap-stanford/ogb/blob/745531be13c5403a93c80e21a41848e38ea7637c/ogb/utils/mol.py#L12
    Converts an rdkit Mol string to graph Data object
    :input: rdkit Mol
    :return: graph object
    """

    # atoms
    atom_features_list = []
    pos = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(atom_to_feature_vector(atom))
        pos.append(list(mol.GetConformer().GetAtomPosition(i)))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['bond_index'] = edge_index
    graph['bond_feat'] = edge_attr
    graph['atom_feat'] = x
    graph['num_atoms'] = len(x)
    graph['atom_pos'] = np.array(pos)

    return graph


class Data():
    def __init__(self) -> None:
        pass


class Molecule(Data):
    """
    A representation of organic molecules.
    Stores the features extracted from ogb.utils.smiles2mol, and atom positions.
    Can be constructed from raw SMILES strings, or XYZ files with bond connection.
    """
    def __init__(
            self,
            atom_feat: ATOM_FEAT_T = None,
            bond_feat: BOND_FEAT_T = None,
            bond_index: BOND_INDEX_T = None,
            atom_pos: ATOM_POS_T = None,
            **kwargs) -> None:
        super().__init__()
        self.atom_feat = atom_feat
        self.bond_feat = bond_feat
        self.bond_index = bond_index
        self.atom_pos = atom_pos
        for key, value in kwargs:
            self.__dict__[key] = value

    @classmethod
    def from_smiles(cls, smiles: str, remove_hs: bool = True, keep_mol: bool = False, **kwargs) -> 'Molecule':
        mol = Chem.MolFromSmiles(smiles)
        if remove_hs:
            mol = RemoveHs(mol)
        graph = mol2graph(mol)
        molecule = cls(**graph, **kwargs)
        if keep_mol:
            molecule.mol = mol
        return molecule

    @classmethod
    def from_xyz_and_bond_index(cls, xyz_path: Path, bond_index: BOND_INDEX_T, bond_order: BOND_FEAT_T, remove_hs: bool, keep_mol: bool = False, **kwargs) -> 'Molecule':
        # check type and shape of bond_index
        if isinstance(bond_index, np.array):
            bond_index = torch.tensor(bond_index).long()
        bond_index_size = bond_index.size()
        assert len(bond_index_size) == 2 and bond_index_size[0] == 2, f"Bond index should have size [2, num_bonds], but {bond_index_size} found."
        bond_order_size = bond_order.size()
        assert len(bond_order_size) == 1
        assert bond_index_size[1] == bond_order_size[0], f"Numbers of bonds in bond_index and bond_order do not match."

        mol = rdkit.Chem.rdmolfiles.MolFromXYZFile(xyz_path)
        editable_mol = RWMol(mol)
        for connection, order in zip(bond_index.T, bond_order):
            # assuming that atom index starts from 0
            editable_mol.AddBond(int(connection[0]), int(connection[1]), bond_dict[order])
        if remove_hs:
            editable_mol = RemoveHs(editable_mol)
        graph = mol2graph(editable_mol)
        molecule = cls(**graph, **kwargs)
        if keep_mol:
            molecule.mol = editable_mol
        return molecule
