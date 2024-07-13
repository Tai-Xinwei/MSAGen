#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import lmdb

from commons import bstr2obj
from mmcif_processing import show_one_mmcif


def show_one_chain(data: dict):
    print(data.keys())
    print(''.join(data['aa']))
    for i, c in enumerate(['x', 'y', 'z']):
        arr = [f'{_:.2f}' for _ in data['pos'][:10, i]]
        print(f"pos[:10].{c}=[{', '.join(arr)}]")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(f'Usage: {sys.argv[0]} <lmdb_directory> <pdbid_prefix (e.g. 1ctf)>')
    lmdbdir, pdbid = sys.argv[1:3]

    assert 4 == len(pdbid), f"PDBID must be 4 characters, wrong id {pdbid}."
    print(f"Extracting {pdbid} from {lmdbdir}")

    with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
        metavalue = txn.get('__metadata__'.encode())
        assert metavalue, f"'__metadata__' not found in {lmdbdir}."

        metadata = bstr2obj(metavalue)

        assert 'keys' in metadata, (
            f"'keys' not in metadata for {lmdbdir}.")
        assert 'structure_methods' in metadata, (
            f"'structure_methods' not in metadata for {lmdbdir}.")
        assert 'release_dates' in metadata, (
            f"'release_dates' not in metadata for {lmdbdir}.")
        assert 'resolutions' in metadata, (
            f"'resolutions' not in metadata for {lmdbdir}.")
        assert 'comment' in metadata, (
            f"'comment' not in metadata for {lmdbdir}.")

        print('-'*80)
        print(metadata['comment'], end='')
        for k, v in metadata.items():
            k != 'comment' and print(k, len(v))
        print(f"{len(metadata['keys'])} samples in {lmdbdir}" )
        print(f"metadata['keys'][:10]={metadata['keys'][:10]}")

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
            print(metadata['structure_methods'][idx])
            print(metadata['release_dates'][idx])
            print("resolution", metadata['resolutions'][idx])

            value = txn.get(key.encode())
            if not value:
                print(f"ERROR: '{key}' not found in {lmdbdir}.")
                continue

            data = bstr2obj(value)
            if len(key) == 4:
                show_one_mmcif(data)
            else:
                show_one_chain(data)
