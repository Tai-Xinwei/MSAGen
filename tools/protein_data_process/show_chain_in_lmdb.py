#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import lmdb

from commons import bstr2obj


if len(sys.argv) != 3:
    sys.exit(f'Usage: {sys.argv[0]} <lmdb_directory> <chain_name, e.g. 1ctf_A>')
lmdbdir, pdbid = sys.argv[1:3]

assert len(pdbid) >= 6 and pdbid[4] == '_', (
    f"PDB Chain should like 1ctf_A, format {pdbid} is wrong.")
print(f"Searching {pdbid} from {lmdbdir}")

with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
    key = pdbid.encode()
    value = txn.get(key)
    if value:
        data = bstr2obj(value)
        print(f">{pdbid}")
        print(''.join(data['aa']))
        for i in range(10):
            arr = [f'{_:.3f}' for _ in data['pos'][:10, i, 0]]
            print(f"pos[:10,{i}].x=[{', '.join(arr)}]")
        print("...")
    else:
        print(f"ERROR: Key {pdbid} not found in {lmdbdir}.", file=sys.stderr)

    metakey = '__metadata__'.encode()
    metavalue = txn.get(metakey)
    if metavalue:
        metadata = bstr2obj(metavalue)

        assert 'keys' in metadata, (
            f"'keys' not in metadata for {lmdbdir}.")
        assert 'sizes' in metadata, (
            f"'sizes' not in metadata for {lmdbdir}.")
        assert 'structure_methods' in metadata, (
            f"'structure_methods' not in metadata for {lmdbdir}.")
        assert 'release_dates' in metadata, (
            f"'release_dates' not in metadata for {lmdbdir}.")
        assert 'resolutions' in metadata, (
            f"'resolutions' not in metadata for {lmdbdir}.")
        assert 'comment' in metadata, (
            f"'comment' not in metadata for {lmdbdir}.")

        assert pdbid in metadata['keys'], (
            f"'{pdbid}' not in metadata['keys'] for {lmdbdir}.")
        idx = metadata['keys'].index(pdbid)

        print('-'*80)
        print(metadata['comment'].strip())
        print(f"{len(metadata['keys'])} samples in {lmdbdir}" )
        print(f"metadata['keys'][:10]={metadata['keys'][:10]}")
        print('-'*80)
        print(pdbid)
        print(metadata['structure_methods'][idx])
        print(metadata['release_dates'][idx])
        print("resolution", metadata['resolutions'][idx])
    else:
        print(f"'__metadata__' not found in {lmdbdir}.", file=sys.stderr)
