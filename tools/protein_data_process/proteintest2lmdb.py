#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import lmdb
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr
from metadata import metadata4target


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
        print(f"Failed to parse fasta {fastafile}, {e}", file=sys.stderr)
    return seqs


def main():
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input_fasta> <pdb_directory> <output_lmdb>")
    inpfas, pdbdir, outlmdb = sys.argv[1:4]

    # parse fasta file
    seqs = parse_fastafile(inpfas)
    print(f"Parsed {len(seqs)} sequences from {inpfas}.")

    # parse metainformation
    #for target, metadata in metadata4target.items():
    #    print(target, metadata)

    # open lmdb and write data
    env = lmdb.open(outlmdb, map_size=1536 ** 4)
    txn = env.begin(write=True)
    keys = []
    types = []
    sizes = []
    pdbs = []
    domains = []
    for header, seq in tqdm(seqs):
        # parse target
        assert header[0] == '>' and 'length=' in header, (
            f"ERROR: wrong header {header} in fasta file {inpfas}.")
        cols = header[1:].split()
        target, length = cols[0], int(cols[1].split('=')[1])
        assert target in metadata4target, f"ERROR: {target} metainfo not found."
        assert length == len(seq) == metadata4target[target]['size'], (
            f"ERROR: wrong sequence length for {target}")

        # read in pdb file
        pdbfile = Path(pdbdir) / f'{target}.pdb'
        assert pdbfile.exists(), f"Native pdb dose not exist {pdbfile}."
        with open(pdbfile, 'r') as fp:
            atomlines = fp.readlines()
        assert atomlines, f"ERROR: empty pdb file {pdbfile}."

        # write to lmdb
        txn.put(f'{target}'.encode(), obj2bstr(seq))
        keys.append(target)
        sizes.append(length)
        types.append(metadata4target[target]['type'])
        pdbs.append(atomlines)
        domains.append(metadata4target[target]['domain'])
    metadata = {
        'keys': keys,
        'sizes': sizes,
        'types': types,
        'pdbs': pdbs,
        'domains': domains,
        }
    txn.put("__metadata__".encode(), obj2bstr(metadata))
    txn.commit()

    with lmdb.open(outlmdb, readonly=True).begin(write=False) as txn:
        metadata = bstr2obj( txn.get('__metadata__'.encode()) )
        print(f"{len(metadata['keys'])} keys, sizes, ... in {outlmdb}" )

        for (k, s, t, p, d) in zip(metadata['keys'],
                                   metadata['sizes'],
                                   metadata['types'],
                                   metadata['pdbs'],
                                   metadata['domains']):
            value = txn.get(k.encode())
            assert value, f"Key {k} not found."
            seq = bstr2obj(value)
            print(f"name={k}, size={s}, type={t}, lines={len(p)}, domain={d}")
            print(seq)
            print(p[0], end='')
            print(p[-1], end='')


if __name__ == "__main__":
    main()
