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


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
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
    logger.info(f"Release date cutoff: release_date < {date_cutoff}")

    with (lmdb.open(inplmdb, readonly=True).begin(write=False) as inptxn,
          lmdb.open(outlmdb, map_size=1024**4).begin(write=True) as txn):
        inpmeta = bstr2obj( inptxn.get('__metadata__'.encode()) )
        assert inpmeta, f"ERROR: {inplmdb} has no key '__metadata__'"
        logger.info(f"Processing original lmdb {inplmdb}")
        print(inpmeta['comment'], end='')
        for k, v in inpmeta.items():
            k != 'comment' and print(f"{k}: {len(v)}")

        assert 'keys' in inpmeta, f"'keys' not in {inplmdb}"
        logger.info(f"Total original complexs: {len(inpmeta['keys'])}")

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
                f'PDB release date cutoff: {date_cutoff}\n'
                ),
            }

        metadata['comment'] = inpmeta['comment'] + metadata['comment']

        for i in tqdm(range(len(inpmeta['keys']))):
            name = inpmeta['keys'][i]
            _date = datetime.strptime(inpmeta['release_dates'][i], '%Y-%m-%d')
            if _date > date_cutoff:
                logger.warning(f"PDB {name} release date {_date.date()} > "
                               f"date cutoff {date_cutoff.date()}.")
                continue
            key = name.encode()
            txn.put(key, inptxn.get(key))
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
    logger.info(f"Total postprocessed complexs: {len(metadata['keys'])}")


if __name__ == "__main__":
    main()
