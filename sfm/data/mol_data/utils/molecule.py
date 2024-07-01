# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdDetermineBonds as rdDetermineBonds
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


def xyz2mol(
    atoms: List[Union[int, str]],
    coords: np.ndarray,
    charge: int,
    bond_orders: List[Tuple[int, int, int]] = None,
    check_charge: bool = True,
) -> Chem.Mol:
    """Create an RDKit molecule using 3D coordinates.

    Args:
        - atoms (List[Union[int, str]]): list of atomic numbers or symbols
        - coords (np.ndarray): 3D coordinates of atoms with shape (n_atoms, 3)
        - charge (int): charge of the molecule
        - bond_orders (List[Tuple[int, int, int]], optional): list of bond orders.
          Defaults to None. If None, rdDetermineBonds.DetermineBonds will be used to determine the bonds.
        - check_charge (bool, optional): check the charge of the molecule. Defaults to True.

    Returns:
        - Chem.Mol: RDKit molecule
    """
    assert len(atoms) == len(coords)
    with Chem.RWMol() as mw:
        conf = Chem.Conformer()
        for i, atm, (x, y, z) in zip(range(len(atoms)), atoms, coords):
            mw.AddAtom(Chem.Atom(atm))
            conf.SetAtomPosition(i, (x, y, z))
        mw.AddConformer(conf)

        if bond_orders:
            for i, j, order in bond_orders:
                if order == 1:
                    mw.AddBond(i, j, Chem.BondType.SINGLE)
                elif order == 2:
                    mw.AddBond(i, j, Chem.BondType.DOUBLE)
                elif order == 3:
                    mw.AddBond(i, j, Chem.BondType.TRIPLE)
                else:
                    mw.AddBond(i, j, Chem.BondType.SINGLE)
            Chem.SanitizeMol(mw)
        else:
            rdDetermineBonds.DetermineBonds(mw, charge=charge)

        if check_charge and Chem.GetFormalCharge(mw) != charge:
            raise ValueError(f"mol charge={Chem.GetFormalCharge(mw)} != {charge}")

        return mw


def mol2graph(mol: Chem.Mol, compress_graph_or_raise: bool = True) -> Dict[str, Any]:
    """Create graph features from a given RDKit molecule. The molecule is not allowed
    to have conformers, so we can ensure that the features do not depend on the coordinates of atoms.

    This function is modified from:
    https://github.com/snap-stanford/ogb/blob/745531be13c5403a93c80e21a41848e38ea7637c/ogb/utils/mol.py#L12

    Args:
        - mol (Chem.Mol): the RDkit molecule
        - compress_graph_or_raise (bool, optional): compress numpy arrays of features and
          raise exception if a graph is too big. Defaults to True.

    Raises:
        - ValueError: raise if there's any conformer for the given molecule
        - ValueError: raise if a graph is too big

    Returns:
        Dict[str, Any]: a dictionary of features
    """
    if len(mol.GetConformers()) > 0:
        raise ValueError(
            "No conformer is expected to generate graph features for a given molecule"
        )

    # atoms
    atom_features_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
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
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    g = dict()
    g["num_nodes"] = len(x)
    g["node_feat"] = x
    g["edge_index"] = edge_index
    g["edge_feat"] = edge_attr

    if compress_graph_or_raise:
        if (
            np.any(g["edge_index"] >= 65536)
            or np.any(g["edge_feat"] >= 256)
            or np.any(g["node_feat"] >= 256)
        ):
            raise ValueError("[GraphTooBigError]")
        g["edge_index"] = g["edge_index"].astype(np.int16)
        g["edge_feat"] = g["edge_feat"].astype(np.int8)
        g["node_feat"] = g["node_feat"].astype(np.int8)

    return g
