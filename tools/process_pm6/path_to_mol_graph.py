# -*- coding: utf-8 -*-
import rdkit
from glob import glob
import numpy as np
from rdkit.Chem.rdchem import RWMol
import json

from rdkit import Chem

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector


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
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
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
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)
    graph['pos'] = np.array(pos)

    return graph


def compress(graph):
    assert np.all(graph['edge_index'] < 65536)
    graph['edge_index'] = np.array(graph['edge_index'], dtype=np.int16)
    assert np.all(graph['edge_feat'] < 256)
    graph['edge_feat'] = np.array(graph['edge_feat'], dtype=np.int8)
    assert np.all(graph['node_feat'] < 256)
    graph['node_feat'] = np.array(graph['node_feat'], dtype=np.int8)
    return graph


def path_to_mol_graph(dirname):
    try:
        json_fnames = glob(f"{dirname}/*.json")
        xyz_fnames = glob(f"{dirname}/*.xyz")
        assert len(json_fnames) == 1 and len(xyz_fnames) == 1, f"{dirname} has multiple xyz or json files"
        json_fname = json_fnames[0]
        xyz_fname = xyz_fnames[0]
        with open(json_fname, "r") as in_file:
            json_obj = json.load(in_file)
        connection_index = json_obj["pubchem"]["B3LYP@PM6"]["bonds"]["connections"]["index"]
        connection_order = json_obj["pubchem"]["B3LYP@PM6"]["bonds"]["order"]
        connection_index = np.array(connection_index).reshape(-1, 2)
        mol = rdkit.Chem.rdmolfiles.MolFromXYZFile(xyz_fname)
        editable_mol = RWMol(mol)
        for connection, order in zip(connection_index, connection_order):
            editable_mol.AddBond(int(connection[0]) - 1, int(connection[1]) - 1, bond_dict[order])
        # no_h_mol = RemoveHs(editable_mol)
        graph = mol2graph(editable_mol)
        graph['alpha_homo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['homo']
        graph['alpha_lumo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['lumo']
        graph['alpha_gap'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['gap']
        graph['beta_homo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['beta']['homo']
        graph['beta_lumo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['beta']['lumo']
        graph['beta_gap'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['beta']['gap']
        graph['total_energy'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['total']
        graph['smiles'] = json_obj['pubchem']['Isomeric SMILES']
        graph = compress(graph)
    except Exception as e:
        print(f"Failed {dirname}: {e}")
        return str(e)

    return graph
