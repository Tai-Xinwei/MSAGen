#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import lmdb

from commons import bstr2obj
from process_mmcif import show_lmdb


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(f'Usage: {sys.argv[0]} <lmdb_directory> <posebusters_list>')
    lmdbdir, listfile = sys.argv[1:3]

    assert os.path.exists(lmdbdir), f"{lmdbdir} not found."
    show_lmdb(lmdbdir)

    targets = []
    with open(listfile, 'r') as fp:
        for line in fp:
            cols = line.strip().split('_')
            assert len(cols) == 2 and len(cols[0]) == 4, f"Wrong format: {line}"
            targets.append(cols)

    with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
        metavalue = txn.get('__metadata__'.encode())
        assert metavalue, f"'__metadata__' not found in {lmdbdir}."

        metadata = bstr2obj(metavalue)

        for key, lig in targets:
            try:
                idx = metadata['keys'].index(key)
            except:
                print(f"ERROR: '{key}' not found in metadata['keys'].")

            # print('-'*80)
            # print(f">{key}")
            # print(metadata['structure_methods'][idx])
            # print(metadata['release_dates'][idx])
            # print("resolution", metadata['resolutions'][idx])

            value = txn.get(key.encode())
            if not value:
                print(f"ERROR: '{key}' not found in {lmdbdir}.")
                continue

            data = bstr2obj(value)
            assert 'nonpoly_graphs' in data, f"nonpoly_graphs not found in {key}."
            for graph in data['nonpoly_graphs']:
                if graph['name'] == lig:
                    break
            else:
                print(f"ERROR: {lig} not found in {key}.")
