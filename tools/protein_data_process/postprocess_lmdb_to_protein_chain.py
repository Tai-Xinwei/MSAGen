#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
from datetime import datetime
from pathlib import Path

import lmdb
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr


#logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SEQUENCE_IDENTITY = [30, 40, 50, 70, 90, 95, 100]


def complex2chain(pdb_id, complex_data):
    processed_data = {}
    for chain_id, chain_data in complex_data.items():
        if 'polymer' not in chain_data:
            raise SystemExit(f"polymer not found in {pdb_id}_{chain_id}")
        polymer = chain_data['polymer']
        if polymer['polymer_type'] == 'nucleotide': # != 'peptide':
            continue
        processed_data[f'{pdb_id}_{chain_id}'] = {
            'pos': polymer['full_coords'][:, :37, :],
            'aa': polymer['polyseq'],
        }
    return processed_data


def process_sequence_identity_clusters(seqid_path: str):
    clusters = []
    with open(seqid_path, 'r') as fp:
        for line in fp:
            clu = []
            for chain_name in line.split():
                if chain_name.startswith('AF_') or chain_name.startswith('MA_'):
                    # skip AFDB predicted protein chain
                    continue
                assert chain_name[4] == '_', f"Wrong chain name: {chain_name}"
                clu.append(chain_name[:4].lower() + chain_name[4:])
            if not clu:
                # skip group only AFDB predicted protein chain
                continue
            clusters.append(clu)
    return clusters


def main():
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        sys.exit(f"Usage: {sys.argv[0]} <input_lmdb> <output_lmdb> <clusters_directory> [max_release_date(=2020-04-30)]")
    inplmdb, outlmdb, cludir = sys.argv[1:4]
    datestr = sys.argv[4] if len(sys.argv) == 5 else '2020-04-30'

    assert not Path(outlmdb).exists(), f"{outlmdb} exists, please remove first."

    try:
        date_cutoff = datetime.strptime(datestr, '%Y-%m-%d')
    except ValueError as e:
        sys.exit(f"ERROR: {e}, max_release_date should like '2020-04-30'.")
    logger.info(f"Release date cutoff: pdb_release_date < {date_cutoff}")

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
        seqid_path = Path(cludir) / f"clusters-by-entity-{seqid}.txt"
        assert seqid_path.is_file(), f"File not found: {seqid_path}"
        clusters = process_sequence_identity_clusters(seqid_path)
        metadata[f'bc{seqid}'] = clusters
        logger.info(f"['bc{seqid}'] {len(clusters)} clusters in {seqid_path}")

    with lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn:
        inpmeta = bstr2obj( inptxn.get('__metadata__'.encode()) )
        assert inpmeta, f"ERROR: {inplmdb} has no key '__metadata__'"

        logger.info(f"Processing original lmdb {inplmdb}")
        print(inpmeta['comment'], end='')
        for k, v in inpmeta.items():
            k != 'comment' and print(f"{k}: {len(v)}")

        assert 'keys' in inpmeta, f"'keys' not in {inplmdb}"
        logger.info(f"Total original complexs: {len(inpmeta['keys'])}")
        for pdbid in tqdm(inpmeta['keys']):
            try:
                value = inptxn.get(pdbid.encode())
            except ValueError as e:
                sys.exit(f"ERROR: {e}, {pdbid} not in {inplmdb}")

            try:
                idx = inpmeta['keys'].index(pdbid)
                structure_method = inpmeta['structure_methods'][idx]
                release_date = inpmeta['release_dates'][idx]
                resolution = inpmeta['resolutions'][idx]
            except ValueError as e:
                sys.exit(f"ERROR: {e}, {pdbid} not in metadata['keys'] list")

            if datetime.strptime(release_date, '%Y-%m-%d') > date_cutoff:
                # skip if PDB released after date_cutoff
                continue

            with lmdb.open(outlmdb, map_size=1024**4).begin(write=True) as txn:
                processed_data = complex2chain(pdbid, bstr2obj(value))
                for chain_name, chain_data in processed_data.items():
                    length = len(chain_data['aa'])
                    assert chain_data['pos'].shape == (length, 37, 3)
                    assert chain_data['aa'].shape == (length,)
                    txn.put(chain_name.encode(), obj2bstr(chain_data))
                    metadata["keys"].append(chain_name)
                    metadata["sizes"].append(length)
                    metadata['structure_methods'].append(structure_method)
                    metadata['release_dates'].append(release_date)
                    metadata['resolutions'].append(resolution)
        metadata['comment'] = inpmeta['comment'] + metadata['comment']
        max_release_date = max([datetime.strptime(_, '%Y-%m-%d')
                                for _ in metadata['release_dates']])
        metadata['comment'] += f"Current max_release_date: {max_release_date}\n"
        logger.info(f"Total postprocessed chains: {len(metadata['keys'])}")

    with lmdb.open(outlmdb, map_size=1024**4).begin(write=True) as txn:
        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    print(metadata['comment'], end='')
    for k, v in metadata.items():
        if k == 'comment':
            continue
        print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main()
