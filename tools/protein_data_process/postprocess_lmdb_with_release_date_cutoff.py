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


def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input_lmdb> <output_lmdb> [max_release_date(=2020-04-30)]")
    inplmdb, outlmdb = sys.argv[1:3]
    datestr = sys.argv[3] if len(sys.argv) == 4 else '2020-04-30'

    assert not Path(outlmdb).exists(), f"{outlmdb} exists, please remove first."

    try:
        date_cutoff = datetime.strptime(datestr, '%Y-%m-%d')
    except ValueError as e:
        sys.exit(f"ERROR: {e}, max_release_date should like '2020-04-30'.")
    logger.info(f"Filtering cutoff: pdb_release_date < {date_cutoff}")

    metadata = {
        'keys': [],
        'resolutions': [],
        'release_dates': [],
        'structure_methods': [],
        'comment' : (
            f'Postprocessed time: {datetime.now()}\n'
            f'Original lmdb: {inplmdb}\n'
            f'Postprocessed lmdb: {outlmdb}\n'
            f'PDB release date cutoff: {date_cutoff}\n'
            ),
        }

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
                txn.put(pdbid.encode(), value)

            metadata['keys'].append(pdbid)
            metadata['structure_methods'].append(structure_method)
            metadata['release_dates'].append(release_date)
            metadata['resolutions'].append(resolution)
        metadata['comment'] = inpmeta['comment'] + metadata['comment']
        max_release_date = max([datetime.strptime(_, '%Y-%m-%d')
                                for _ in metadata['release_dates']])
        metadata['comment'] += f"Current max_release_date: {max_release_date}\n"
        logger.info(f"Total after filtering: {len(metadata['keys'])}")

    with lmdb.open(outlmdb, map_size=1024**4).begin(write=True) as txn:
        txn.put('__metadata__'.encode(), obj2bstr(metadata))

    print(metadata['comment'], end='')
    for k, v in metadata.items():
        if k == 'comment':
            continue
        print(f"{k}: {len(v)}")


if __name__ == "__main__":
    main()
