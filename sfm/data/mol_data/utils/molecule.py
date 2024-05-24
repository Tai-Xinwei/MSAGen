# -*- coding: utf-8 -*-
from typing import Any, Dict

import numpy as np
import rdkit.Chem as Chem
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


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
