#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import lmdb

from commons import bstr2obj
from process_pdb_complex import show_one_complex



if len(sys.argv) != 3:
    sys.exit(f'Usage: {sys.argv[0]} <lmdb_directory> <pdbid>')
lmdbdir, pdbid = sys.argv[1:3]

assert 4 == len(pdbid), f"PDBID must be 4 characters, wrong id {pdbid}."
print(f"Extracting {pdbid} from {lmdbdir}")

with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
    key = pdbid.encode()
    value = txn.get(key)
    if value:
        data = bstr2obj(value)
        show_one_complex(data)
    else:
        print(f"ERROR: Key {pdbid} not found in {lmdbdir}.", file=sys.stderr)

    metakey = '__metadata__'.encode()
    metavalue = txn.get(metakey)
    if metavalue:
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
