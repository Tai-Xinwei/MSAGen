#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
from datetime import datetime
from pathlib import Path

import lmdb
import numpy as np
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)
SEQUENCE_IDENTITY = [30, 40, 50, 70, 90, 95, 100]


def process_sequence_identity_clusters(seqid_path: str):
    clusters = {}
    with open(seqid_path, 'r') as fp:
        for line in fp:
            cols = line.split()
            assert len(cols) == 2, f"Wrong cluster line {line} in {seqid_path}"
            if cols[0] in clusters:
                clusters[cols[0]].append(cols[1])
            else:
                clusters[cols[0]] = [cols[1]]
    assert clusters and all(k == v[0] for k, v in clusters.items()), (
        f"Cluster center must come to the first in each cluster {seqid_path}")
    clusters = [sorted(v) for _, v in clusters.items()]
    return sorted(clusters, key=lambda x: len(x), reverse=True)


def process_one_pdb(pdbid: str,
                    inplmdb: str,
                    date_cutoff: datetime,
                    ) -> dict:
    try:
        with lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn:
            data = bstr2obj( inptxn.get(pdbid.encode()) )
        assert data, f"ERROR: {pdbid} not in {inplmdb}"
        assert 'polymer_chains' in data, f"'polymer_chains' not in {pdbid} data"
        chains = {}
        for chain_id, polymer in data['polymer_chains'].items():
            if sum(polymer['restype'] == 'p') < sum(polymer['restype'] == 'n'):
                # Only process protein chain
                continue
            chains[f'{pdbid}_{chain_id}'] = {
                'aa': polymer['seqres'],
                'pos': polymer['center_coord'],
                'size': len(polymer['seqres']),
                'structure_method': data['structure_method'],
                'release_date': data['release_date'],
                'resolution': data['resolution'],
            }
        return chains
    except Exception as e:
        logger.error(f"Failed to processing {pdbid}, {e}")
        return {}


def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        sys.exit(f"Usage: {sys.argv[0]} <input_lmdb> <clusters_directory> <output_lmdb> [max_release_date(=2020-04-30)]")
    inplmdb, cludir, outlmdb = sys.argv[1:4]
    datestr = sys.argv[4] if len(sys.argv) == 5 else '2020-04-30'

    assert not Path(outlmdb).exists(), f"{outlmdb} exists, please remove first."

    try:
        date_cutoff = datetime.strptime(datestr, '%Y-%m-%d')
    except ValueError as e:
        sys.exit(f"ERROR: {e}, max_release_date should like '2020-04-30'.")
    logger.info(f"Release date cutoff: release_date < {date_cutoff}")

    metadata = {
        'keys': [],
        'sizes': [],
        'resolutions': [],
        'release_dates': [],
        'structure_methods': [],
        'comment' : (
            f'Postprocessed time: {datetime.now()}\n'
            f'Original lmdb: {inplmdb}\n'
            f'Clusters directory: {cludir}\n'
            f'Postprocessed lmdb: {outlmdb}\n'
            f'PDB release date cutoff: {date_cutoff}\n'
            ),
        }

    for seqid in SEQUENCE_IDENTITY:
        seqid_path = Path(cludir) / f"pdb{seqid}_cluster.tsv"
        assert seqid_path.is_file(), f"File not found: {seqid_path}"
        logger.info(f"Processing sequence identity cluster file {seqid_path}")
        clusters = process_sequence_identity_clusters(seqid_path)
        metadata[f'pdb{seqid}'] = clusters
        metadata['comment'] += (f"Cluster 'pdb{seqid}': {len(clusters)} "
                                f"clusters in {seqid_path}\n")

    with lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn:
        inpmeta = bstr2obj( inptxn.get('__metadata__'.encode()) )
    assert inpmeta, f"ERROR: {inplmdb} has no key '__metadata__'"

    logger.info(f"Processing original lmdb {inplmdb}")
    print(inpmeta['comment'], end='')
    for k, v in inpmeta.items():
        k != 'comment' and print(f"{k}: {len(v)}")

    metadata['comment'] = inpmeta['comment'] + metadata['comment']

    assert 'keys' in inpmeta, f"'keys' not in {inplmdb}"
    logger.info(f"Total original complexs: {len(inpmeta['keys'])}")

    assert 'release_dates' in inpmeta, f"'release_dates' not in {inplmdb}"
    filtered_keys = []
    for key, release_date in zip(inpmeta['keys'], inpmeta['release_dates']):
        release_date = datetime.strptime(release_date, '%Y-%m-%d')
        if release_date > date_cutoff:
            logger.warning(f"PDB {key} release date {release_date.date()} > "
                           f"date cutoff {date_cutoff.date()}.")
        else:
            filtered_keys.append(key)
    logger.info(f"Filtered keys: {len(filtered_keys)}")

    results = Parallel(n_jobs=-1)(
        delayed(process_one_pdb)(p, inplmdb, date_cutoff)
        for p in tqdm(filtered_keys, desc='Processing PDBs...')
        )

    with lmdb.open(outlmdb, map_size=1024**4).begin(write=True) as txn:
        for res in tqdm(results, desc='Writing to lmdb...'):
            if not res: continue
            for name, data in res.items():
                simple_data = {'aa': data['aa'], 'pos': data['pos']}
                txn.put(name.encode(), obj2bstr(simple_data))
                metadata["keys"].append(name)
                metadata["sizes"].append(data['size'])
                metadata['structure_methods'].append(data['structure_method'])
                metadata['release_dates'].append(data['release_date'])
                metadata['resolutions'].append(data['resolution'])
        max_release_date = max([datetime.strptime(_, '%Y-%m-%d')
                                for _ in metadata['release_dates']])
        metadata['comment'] += f"Current max_release_date: {max_release_date}\n"
        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    logger.info(f"Total postprocessed chains: {len(metadata['keys'])}")

    print(metadata['comment'], end='')
    for k, v in metadata.items():
        if k == 'comment':
            continue
        print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main()
