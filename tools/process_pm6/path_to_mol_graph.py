# -*- coding: utf-8 -*-
import rdkit
from glob import glob
import numpy as np
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import RemoveHs
import json
from rdkit.Chem import rdDetermineBonds

from rdkit import Chem

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

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
    # pos = []
    # for i, atom in enumerate(mol.GetAtoms()):
    #     pos.append(list(mol.GetConformer().GetAtomPosition(i)))

    pos = mol.GetConformer().GetPositions()
    # assert type(pos) == np.ndarray, "pos is not np.ndarray"

    # remove conformation in mol
    mol.RemoveConformer(0)
    for i, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(atom_to_feature_vector(atom))

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
    # graph['pos'] = np.array(pos)
    graph['pos'] = pos

    return graph


def compress(graph):
    assert np.all(graph['edge_index'] < 65536)
    graph['edge_index'] = np.array(graph['edge_index'], dtype=np.int16)
    assert np.all(graph['edge_feat'] < 256)
    graph['edge_feat'] = np.array(graph['edge_feat'], dtype=np.int8)
    assert np.all(graph['node_feat'] < 256)
    graph['node_feat'] = np.array(graph['node_feat'], dtype=np.int8)
    return graph


def setAtomicCharge(mol, charge):
    mol = Chem.Mol(mol)
    molCharge = 0
    charges = []
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)

    for i, atom in enumerate(mol.GetAtoms()):
        # print(i, atom.GetTotalValence())
        atomCharge = atom.GetFormalCharge()
        molCharge += atomCharge
        if atom.GetAtomicNum() == 6:
            if atom.GetTotalValence() == 2 and atom.GetTotalDegree() == 2:
                molCharge += 1
                atomCharge = 0
            elif atom.GetTotalDegree() == 3 and (molCharge + 1 < charge):
                molCharge += 2
                atomCharge = 1
        charges.append(atomCharge)

    if molCharge != charge:
        raise ValueError(f'The total charge of the molecule ({molCharge}) does not match the given charge ({charge}).')

    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetFormalCharge(charges[i])

    return mol


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
        charge = int(json_obj["pubchem"]["B3LYP@PM6"]["properties"]["charge"])
        connection_index = np.array(connection_index).reshape(-1, 2)
        mol = rdkit.Chem.rdmolfiles.MolFromXYZFile(xyz_fname)

        editable_mol = RWMol(mol)
        # print(connection_index)
        for connection, order in zip(connection_index, connection_order):
            # if int(connection[0]) == 15 or int(connection[1]) == 15:
                # print(f"connection is {connection}, order is {order}")
            editable_mol.AddBond(int(connection[0]) - 1, int(connection[1]) - 1, bond_dict[order])

        # no_h_mol = RemoveHs(editable_mol)
        num_fragments = len(rdmolops.GetMolFrags(editable_mol))
        if num_fragments > 1:
            return f"Failed {dirname}: f'Number of fragments: {num_fragments}'"

        # editable_mol = setAtomicCharge(editable_mol, charge)
        Chem.SanitizeMol(editable_mol)
        no_h_mol = RemoveHs(editable_mol)
        mol_smile = Chem.MolToSmiles(no_h_mol, isomericSmiles=False)
        # Chem.SanitizeMol(editable_mol, sanitizeOps=Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES)

        graph = mol2graph(editable_mol)
        graph['alpha_homo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['homo']
        graph['alpha_lumo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['lumo']
        graph['alpha_gap'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['gap']
        graph['beta_homo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['beta']['homo']
        graph['beta_lumo'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['beta']['lumo']
        graph['beta_gap'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['beta']['gap']
        graph['total_energy'] = json_obj['pubchem']['B3LYP@PM6']['properties']['energy']['total']
        graph['charge'] = charge
        graph['smiles'] = json_obj['pubchem']['Isomeric SMILES']
        graph['mol_smile'] = mol_smile
        graph = compress(graph)
    except Exception as e:
        print(f"Failed {dirname}: {e}")
        # print(f"smile is {json_obj['pubchem']['Isomeric SMILES']}")
        # exit()
        return str(e)

    return graph


if __name__ == '__main__':
    path_to_mol_graph("/home/peiran/data/pm6_unzip/output/Compound_018125001_018150000/018147221")
