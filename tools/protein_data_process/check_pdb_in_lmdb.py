#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import lmdb

from commons import bstr2obj
from process_mmcif import show_lmdb, show_one_structure


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(f'Usage: {sys.argv[0]} <lmdb_directory> <pdbid_prefix (e.g. 1ctf)>')
    lmdbdir, pdbid = sys.argv[1:3]

    assert os.path.exists(lmdbdir), f"{lmdbdir} not found."
    show_lmdb(lmdbdir)

    # assert 4 == len(pdbid), f"PDBID must be 4 characters, wrong id {pdbid}."
    print(f"Extracting {pdbid} from {lmdbdir}")

    with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
        metavalue = txn.get('__metadata__'.encode())
        assert metavalue, f"'__metadata__' not found in {lmdbdir}."
        metadata = bstr2obj(metavalue)

        selected_keys = [_ for _ in metadata['keys'] if _.startswith(pdbid)]
        assert selected_keys, f"{pdbid} no keys in __metadata__['keys']."

        print(f"Found {len(selected_keys)} keys for {pdbid}*: {selected_keys}.")
        for key in selected_keys:
            try:
                idx = metadata['keys'].index(key)
            except:
                print(f"ERROR: '{key}' not found in metadata['keys'].")

            print('-'*80)
            print(f">{key}")
            print("structure_methods:", metadata['structure_methods'][idx])
            print("release_dates:", metadata['release_dates'][idx])
            print("resolutions:", metadata['resolutions'][idx])

            value = txn.get(key.encode())
            if not value:
                print(f"ERROR: '{key}' not found in {lmdbdir}.")
                continue

            data = bstr2obj(value)
            show_one_structure(data)
