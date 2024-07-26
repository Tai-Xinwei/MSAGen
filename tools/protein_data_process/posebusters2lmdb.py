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
from pdbbind2lmdb import read_molecule, mol2graph


logging.set_verbosity(logging.INFO)


def process_one_target(target: str, rootdir: str, removeHs=False) -> dict:
    try:
        assert target[4] == '_', f"Wrong PoseBusters target {target}"
        pdb_id, ccd_id = target.split('_')
        pdb_path = Path(rootdir) / target / f'{target}_protein.pdb'
        assert pdb_path.exists(), f"PoseBusters protein.pdb {pdb_path} missed."
        sdf_path = Path(rootdir) / target / f'{target}_ligand.sdf'
        assert sdf_path.exists(), f"PoseBusters ligand.sdf {sdf_path} missed."

        polymer_chains = {}
        with open(pdb_path, 'r') as fp:
            pdb_string = fp.read()
        assert pdb_string, f"Failed to read pdb string for {pdb_path}."
        protein = parse_structure(file_id=pdb_id, pdb_string=pdb_string)
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
        assert mol, f"Failed to read molecule from {sdf_path}."
        if removeHs:
            mol = Chem.RemoveHs(mol)
        else:
            mol = Chem.AddHs(mol)
        graph = {
            'chain_id': ' ',
            'residue_number': 1,
            'name': ccd_id,
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
            'pdbid': pdb_id,
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

    # data = process_one_target('5S8I_2LY', inpdir)
    # show_one_complex(data)
    # exit('Debug')

    targets = [_.name for _ in Path(inpdir).glob("*/")]
    URL = 'https://zenodo.org/records/8278563'
    assert 428 == len(targets), f"Wrong PoseBusters {inpdir}, download from {URL}"
    assert all(6<=len(_)<=8 for _ in targets), "Target is not PoseBusters."

    metadata = {
        'keys': [],
        'num_polymers': [],
        'num_nonpolys': [],
        'structure_methods': [],
        'release_dates': [],
        'resolutions': [],
        'comment': (
            f'Created time: {datetime.now()}\n'
            f'Input PoseBusters: {inpdir}\n'
            f'Output lmdb: {outlmdb}\n'
            f'Number of workers: 96\n'
            f'Comments: PoseBusters downloaded from {URL}.\n'
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
