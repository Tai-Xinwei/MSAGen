#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import lmdb
import pickle
import sys
import zlib

from Bio.Data import PDBData
from process_pdb_complex import show_one_complex


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))


def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


if len(sys.argv) != 3:
    sys.exit(f'Usage: {sys.argv[0]} <lmdb_directory> <pdbid>')
lmdbdir, pdbid = sys.argv[1:3]

assert 4 == len(pdbid), f"ERROR: PDBID must be 4 characters, wrong id {pdbid}."
print(f"Extracting {pdbid} from {lmdbdir}")

with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
    metadata = bstr2obj( txn.get('__metadata__'.encode()) )
    print(f"{len(metadata['keys'])} keys in {lmdbdir}" )
    print(metadata['comment'].strip())

    key = pdbid.encode()
    value = txn.get(key)
    if not value:
        sys.exit(f"Key {pdbid} not found.")
    data = bstr2obj(value)
    show_one_complex(data)
