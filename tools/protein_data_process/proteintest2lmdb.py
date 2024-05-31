#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import sys
from pathlib import Path

import lmdb
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr


def parse_fastafile(fastafile):
    '''Parse fasta file.'''

    seqs = []
    try:
        with open(fastafile, 'r') as fin:
            header, seq = '', []
            for line in fin:
                if line[0] == '>':
                    seqs.append( (header, ''.join(seq)) )
                    header, seq = line.strip(), []
                else:
                    seq.append( line.strip() )
            seqs.append( (header, ''.join(seq)) )
            del seqs[0]

    except Exception as e:
        print('ERROR: wrong fasta file "%s"\n      ' % fastafile, e, file=sys.stderr)

    return seqs


def show_lmdb(lmdbdir):
    with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
        metadata = bstr2obj( txn.get('__metadata__'.encode()) )
        print(f"{len(metadata['keys'])} keys in {lmdbdir}" )
        print(f"{len(metadata['sizes'])} sizes in {lmdbdir}" )

        for key, length in zip(metadata['keys'], metadata['sizes']):
            value = txn.get(key.encode())
            assert value, f"Key {key} not found."
            data = bstr2obj(value)
            assert data.keys() == {'seq', 'pdb'}, f"Wrong keys {data.keys()}."
            print(f"name={key}, length={length}, lines={len(data['pdb'])}")
            print(data['seq'])
            print(data['pdb'][0], end='')
            print(data['pdb'][-1], end='')


def main():
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input_fasta> <pdb_directory> <output_lmdb>")
    inpfas, pdbdir, outlmdb = sys.argv[1:4]

    # parse fasta file
    seqs = parse_fastafile(inpfas)
    print(f"Parsed {len(seqs)} sequences from {inpfas}.")

    # open lmdb and write data
    env = lmdb.open(outlmdb, map_size=1536 ** 4)
    txn = env.begin(write=True)
    keys = []
    sizes = []
    for header, seq in tqdm(seqs):
        # parse target
        assert header[0] == '>' and 'length=' in header, (
            f"ERROR: wrong header {header} in fasta file {inpfas}.")
        cols = header[1:].split()
        target, length = cols[0], int(cols[1].split('=')[1])
        assert length == len(seq), f"ERROR: wrong sequence length for {target}"
        #print(f"Target information: name={target}, length={length}\n{seq}")

        # read in pdb file
        pdbfile = Path(pdbdir) / f'{target}.pdb'
        assert pdbfile.exists(), f"Native pdb dose not exist {pdbfile}."
        with open(pdbfile, 'r') as fp:
            atomlines = fp.readlines()
        assert atomlines, f"ERROR: empty pdb file {pdbfile}."

        # write to lmdb
        data = {'seq': seq, 'pdb': atomlines}
        txn.put(f'{target}'.encode(), obj2bstr(data))

        keys.append(target)
        sizes.append(length)
    metadata = {'keys': keys, 'sizes': sizes}
    txn.put("__metadata__".encode(), obj2bstr(metadata))
    txn.commit()

    show_lmdb(outlmdb)


if __name__ == "__main__":
    main()
