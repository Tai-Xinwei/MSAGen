#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import lmdb
import numpy as np
from absl import logging
from joblib import delayed
from joblib import Parallel
from ogb.utils.features import atom_to_feature_vector
from ogb.utils.features import bond_to_feature_vector
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr
from mmcif_processing import process_polymer_chain
from mmcif_processing import residue2resdict
from mmcif_processing import show_lmdb
from mmcif_processing import show_one_mmcif as show_one_complex
from pdb_parsing import parse_structure


logging.set_verbosity(logging.INFO)


def read_molecule(molecule_file: str,
                  sanitize: bool=False,
                  calc_charges: bool=False,
                  remove_hs: bool=False,
                  ) -> Chem.Mol:
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.
    Copied from https://github.com/HannesStark/EquiBind/blob/main/commons/process_mols.py#L1189

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                logging.warning('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol


def mol2graph(mol: Chem.Mol) -> dict:
    """Converts RDKit molecule to graph Data object.
    From https://github.com/snap-stanford/ogb/blob/master/ogb/utils/mol.py
    :input: molecule_path (str)
    :return: graph object
    """
    # atoms
    atom_coords_list = []
    atom_features_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_coords_list.append(list(mol.GetConformer().GetAtomPosition(i)))
        atom_features_list.append(atom_to_feature_vector(atom))
    node_coord = np.array(atom_coords_list, dtype = np.float32)
    node_feat = np.array(atom_features_list, dtype = np.int64)

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
    graph['node_coord'] = node_coord
    graph['node_feat'] = node_feat
    graph['num_nodes'] = len(node_feat)

    return graph


def process_one_target(target: str, rootdir: str) -> dict:
    try:
        pdb_path = Path(rootdir) / target / f'{target}_protein.pdb'
        assert pdb_path.exists(), f"PDBbind protein.pdb {pdb_path} must exist."
        sdf_path = Path(rootdir) / target / f'{target}_ligand.sdf'
        assert sdf_path.exists(), f"PDBbind ligand.sdf {sdf_path} must exist."
        mol2_path = Path(rootdir) / target / f'{target}_ligand.mol2'
        assert mol2_path.exists(), f"PDBbind ligand.mol2 {mol2_path} must exist."

        polymer_chains = {}
        with open(pdb_path, 'r') as fp:
            pdb_string = fp.read()
        assert pdb_string, f"Failed to read pdb string for {pdb_path}."
        protein = parse_structure(file_id=target, pdb_string=pdb_string)
        assert protein.pdb_object, f"The errors are {protein.errors}"
        for chain_id in sorted(protein.pdb_object.chain_to_seqres.keys()):
            seqres = protein.pdb_object.chain_to_seqres[chain_id]
            restype = protein.pdb_object.chain_to_restype[chain_id]
            struct = protein.pdb_object.seqres_to_structure[chain_id]
            # print('-'*80, f"{target}_{chain_id}", seqres, restype, sep='\n')
            current_chain = []
            # process residues one by one
            for sr, rt, (residue, atoms) in zip(seqres, restype, struct):
                if residue.position and chain_id != residue.position.chain_id:
                    raise ValueError(f"Chain '{chain_id}' has wrong {residue}")
                if sr != '?': # <=> rt != '*'
                    # Process polymer residues for protein, DNA and RNA.
                    # Must do this no matter it is standard residue or not.
                    resdict = residue2resdict(chain_id, sr, rt, residue, atoms)
                    # modify resdict for PDBbind
                    if resdict['restype'] == '*':
                        resdict['seqres'] = 'X'
                        resdict['restype'] = 'p'
                    current_chain.append(resdict)
            polymer = process_polymer_chain(current_chain)
            if sum(polymer['restype'] == 'p') < sum(polymer['restype'] == 'n'):
                raise ValueError(f"Have DNA or RNA in {target}_{chain_id}")
            if np.all(np.isnan(polymer['center_coord'])):
                logging.warning(f"All CA is 'nan' for {target}_{chain_id}")
                continue
            polymer_chains[chain_id] = polymer
        assert polymer_chains, f"Has no desirable chains for {target}."
        logging.debug(f"{pdb_path} processed successfully.")

        nonpoly_graphs = []
        RDLogger.DisableLog('rdApp.warning')
        mol = read_molecule(str(sdf_path), sanitize=True, remove_hs=False)
        if mol is None:
            mol = read_molecule(str(mol2_path), sanitize=True, remove_hs=False)
        assert mol, f"Failed to read molecule from {sdf_path} or {mol2_path}."
        graph = {
            'chain_id': ' ',
            'residue_number': 1,
            'name': 'LIG',
            'pdbx_formal_charge': Chem.GetFormalCharge(mol),
            'atomids': np.array([_.GetSymbol() for _ in mol.GetAtoms()]),
            'symbols': np.array([_.GetSymbol() for _ in mol.GetAtoms()]),
            'orders': np.array([
                (_.GetBeginAtomIdx(), _.GetEndAtomIdx(), _.GetBondType())
                for _ in mol.GetBonds()
                ]),
            "rdkitmol": mol,
        }
        graph.update(mol2graph(mol))
        if np.all(np.isnan(graph['node_coord'])):
            raise ValueError(f"All atom is 'nan' in nonpoly graph {target}")
        nonpoly_graphs.append(graph)
        logging.debug(f"{sdf_path} processed successfully.")

        data = {
            'pdbid': target,
            'structure_method': protein.pdb_object.header['structure_method'],
            'release_date': protein.pdb_object.header['release_date'],
            'resolution': protein.pdb_object.header['resolution'],
            'polymer_chains': polymer_chains,
            'nonpoly_graphs': nonpoly_graphs,
        }
        return data
    except Exception as e:
        logging.error(f"Process {target} failed, {e}")
        return {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--inpdir',
                        type=str,
                        required=True,
                        help="Input directory contains all PDBbind data.")
    parser.add_argument('--outlmdb',
                        type=str,
                        default="output.lmdb",
                        help="Output processed lmdb file.")
    args = parser.parse_args()

    inpdir = Path(args.inpdir).resolve()
    assert inpdir.exists(), f"Input directory {inpdir} not found."
    outlmdb = Path(args.outlmdb).resolve()
    assert not outlmdb.exists(), f"Output lmdb {outlmdb} already exists. Skip."

    # data = process_one_target('4mb9', inpdir)
    # show_one_complex(data)
    # exit('Debug')

    targets = [_.name for _ in Path(inpdir).glob("*/")]
    # URL = 'https://zenodo.org/record/6408497'
    # assert 19119 == len(targets), f"Wrong PDBbind {inpdir}, download from {URL}"
    URL = 'http://www.pdbbind.org.cn/download.php'
    assert 19443 == len(targets), f"Wrong PDBbind {inpdir}, download from {URL}"
    assert all(4==len(_) for _ in targets), "PDBID should be 4 characters long."

    metadata = {
        'keys': [],
        'num_polymers': [],
        'num_nonpolys': [],
        'structure_methods': [],
        'release_dates': [],
        'resolutions': [],
        'comment': (
            f'Created time: {datetime.now()}\n'
            f'Input PDBbind: {inpdir}\n'
            f'Output lmdb: {outlmdb}\n'
            f'Number of workers: 96\n'
            f'Comments: PDBbind downloaded from {URL}.\n'
        ),
    }

    results = Parallel(n_jobs=-1)(
        delayed(process_one_target)(p, inpdir) for p in tqdm(targets)
    )

    logging.info(f"Writing data to {outlmdb}")
    with lmdb.open(str(outlmdb), map_size=1024**4).begin(write=True) as txn: # 1TB size
        for data in tqdm(results):
            if not data:
                # skip empty data
                continue
            txn.put(data['pdbid'].encode(), obj2bstr(data))
            metadata['keys'].append(data['pdbid'])
            metadata['num_polymers'].append(len(data['polymer_chains']))
            metadata['num_nonpolys'].append(len(data['nonpoly_graphs']))
            metadata['structure_methods'].append(data['structure_method'])
            metadata['release_dates'].append(data['release_date'])
            metadata['resolutions'].append(data['resolution'])

        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    show_lmdb(outlmdb)
