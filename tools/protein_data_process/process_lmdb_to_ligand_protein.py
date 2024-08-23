#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from datetime import datetime
from pathlib import Path

import lmdb
import numpy as np
from absl import logging
from joblib import delayed
from joblib import Parallel
from rdkit import Chem
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from sfm.data.mol_data.utils.molecule import mol2graph


logging.set_verbosity(logging.INFO)


def remove_hydrogens_from_graph(graph):
    data = {
        'chain_id': graph['chain_id'],
        'residue_number': graph['residue_number'],
        'name': graph['name'],
        'pdbx_formal_charge': graph['pdbx_formal_charge'],
    }
    mask = graph['symbols'] != 'H'
    idx_old2new = {idx:i for i, idx in enumerate(np.where(mask)[0])}
    new_orders = [(idx_old2new[i], idx_old2new[j], _)
                  for i, j, _ in graph['orders'] if mask[i] and mask[j]]
    rdkitmol = Chem.RemoveHs(graph['rdkitmol'])
    data.update({
        'atomids': graph['atomids'][mask],
        'symbols': graph['symbols'][mask],
        'charges': graph['charges'][mask],
        'coords': graph['coords'][mask],
        'node_coord': graph['node_coord'][mask],
        'orders': np.array(new_orders),
        'rdkitmol': rdkitmol,
    })
    rdkitmol.RemoveAllConformers()
    data.update(mol2graph(rdkitmol))
    return data


def process_one_pdb(pdbid: str,
                    inplmdb: str,
                    remove_hydrogens: bool = True,
                    ) -> dict:
    try:
        with lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn:
            data = bstr2obj( inptxn.get(pdbid.encode()) )
        assert data, f"PDB {pdbid} not in {inplmdb}"

        assert 'pdbid' in data and data['pdbid'] == pdbid, (
            f"data['pdbid']={data['pdbid']} wrong with {pdbid} in {inplmdb}")

        assert 'polymer_chains' in data, f"'polymer_chains' not in {pdbid} data"
        polymer_chains = {}
        for chain_id, polymer in data['polymer_chains'].items():
            num_prot_res = sum(polymer['restype'] == 'p')
            if num_prot_res == 0:
                # Only process protein chain
                continue
            elif num_prot_res != len(polymer['restype']):
                raise ValueError(f"Chain {pdbid}_{chain_id} has wrong restype.")
            if np.all(np.isnan(polymer['center_coord'])):
                logging.warning(f"All CA is 'nan' for {pdbid}_{chain_id}")
                continue
            polymer_chains[chain_id] = polymer
        assert polymer_chains, f"No valid protein chain in {pdbid}"
        data['polymer_chains'] = polymer_chains

        assert 'nonpoly_graphs' in data, f"'nonpoly_graphs' not in {pdbid} data"
        nonpoly_graphs = []
        for graph in data['nonpoly_graphs']:
            if remove_hydrogens:
                graph = remove_hydrogens_from_graph(graph)
            if np.all(np.isnan(graph['node_coord'])):
                logging.warning(f"All atom is 'nan' in nonpoly graph {pdbid}")
                continue
            nonpoly_graphs.append(graph)
        data['nonpoly_graphs'] = nonpoly_graphs

        return data
    except Exception as e:
        logging.error(f"Failed to processing {pdbid}, {e}")
        return {}


def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input_lmdb> <output_lmdb> [max_release_date(=2020-04-30)]")
    inplmdb, outlmdb = sys.argv[1:3]
    datestr = sys.argv[3] if len(sys.argv) == 4 else '2020-04-30'

    assert not Path(outlmdb).exists(), f"{outlmdb} exists, please remove first."

    remove_hydrogens = False

    try:
        date_cutoff = datetime.strptime(datestr, '%Y-%m-%d')
    except ValueError as e:
        sys.exit(f"ERROR: {e}, max_release_date should like '2020-04-30'.")
    logging.info(f"Release date cutoff: release_date < {date_cutoff}")

    metadata = {
        'keys': [],
        'num_polymers': [],
        'num_nonpolys': [],
        'structure_methods': [],
        'release_dates': [],
        'resolutions': [],
        'comment' : (
            f'Postprocessed time: {datetime.now()}\n'
            f'Original lmdb: {inplmdb}\n'
            f'Postprocessed lmdb: {outlmdb}\n'
            f'Remove hydrogens: {remove_hydrogens}\n'
            f'PDB release date cutoff: {date_cutoff}\n'
            ),
        }

    with lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn:
        inpmeta = bstr2obj( inptxn.get('__metadata__'.encode()) )
    assert inpmeta, f"ERROR: {inplmdb} has no key '__metadata__'"

    logging.info(f"Processing original lmdb {inplmdb}")
    print(inpmeta['comment'], end='')
    for k, v in inpmeta.items():
        k != 'comment' and print(f"{k}: {len(v)}")

    metadata['comment'] = inpmeta['comment'] + metadata['comment']

    assert 'keys' in inpmeta, f"'keys' not in {inplmdb}"
    logging.info(f"Total original complexs: {len(inpmeta['keys'])}")

    assert 'release_dates' in inpmeta, f"'release_dates' not in {inplmdb}"
    filtered_keys = []
    for key, release_date in zip(inpmeta['keys'], inpmeta['release_dates']):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')
        if release_date > date_cutoff:
            logging.warning(f"PDB {key} release date {release_date.date()} > "
                           f"date cutoff {date_cutoff.date()}.")
        else:
            filtered_keys.append(key)
    logging.info(f"Filtered keys: {len(filtered_keys)}")

    results = Parallel(n_jobs=-1)(
        delayed(process_one_pdb)(p, inplmdb, remove_hydrogens)
        for p in tqdm(filtered_keys, desc='Processing PDBs...')
        )
    results = [_ for _ in results if _]

    with lmdb.open(outlmdb, map_size=1024**4).begin(write=True) as txn:
        for data in tqdm(results, desc='Writing to lmdb...'):
            if not data: continue
            pdbid = data['pdbid']
            txn.put(pdbid.encode(), obj2bstr(data))
            i = inpmeta['keys'].index(pdbid)
            metadata["keys"].append(inpmeta['keys'][i])
            metadata['num_polymers'].append(inpmeta['num_polymers'][i])
            metadata['num_nonpolys'].append(inpmeta['num_nonpolys'][i])
            metadata['structure_methods'].append(inpmeta['structure_methods'][i])
            metadata['release_dates'].append(inpmeta['release_dates'][i])
            metadata['resolutions'].append(inpmeta['resolutions'][i])
        max_release_date = max([datetime.strptime(_, '%Y-%m-%d')
                                for _ in metadata['release_dates']])
        metadata['comment'] += f"Current max_release_date: {max_release_date}\n"
        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    print(metadata['comment'], end='')
    for k, v in metadata.items():
        k != 'comment' and print(f"{k}: {len(v)}")
    logging.info(f"Total postprocessed complexs: {len(metadata['keys'])}")


if __name__ == "__main__":
    main()
