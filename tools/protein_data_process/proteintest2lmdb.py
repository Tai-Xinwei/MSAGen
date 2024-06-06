#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import lmdb
import pandas as pd
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr


HERE = Path(__file__).resolve().parent
casp14_domain_csv = HERE / 'casp14_domain_definations_and_classifications.csv'
assert casp14_domain_csv.exists(), f"ERROR: {casp14_domain_csv} not found"
casp15_domain_csv = HERE / 'casp15_domain_definations_and_classifications.csv'
assert casp15_domain_csv.exists(), f"ERROR: {casp15_domain_csv} not found"
cameo_subset_csv = HERE / 'cameo_chain_definations_from_20220401_to_20220625.csv'
assert cameo_subset_csv.exists(), f"ERROR: {cameo_subset_csv} not found"


def parse_cameo_metadata(cameo_metadata_csv) -> dict:
    '''Parse metadata from cameo subset.'''
    tmpdict = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
    infodict = {}
    for _, row in pd.read_csv(cameo_metadata_csv).iterrows():
        target = f'{row["ref. PDB [Chain]"][:4]}_{row["ref. PDB [Chain]"][6]}'
        size = row['Sequence Length (residues)']
        type = 'CAMEO'
        domain = (target, size, tmpdict.get(row['Difficulty'], 'Hard'))
        infodict[target] = {
            'size': size,
            'type' : type,
            'domain': [domain],
        }
    return infodict


def parse_casp_metadata(casp_metadata_csv) -> dict:
    '''Parse metadata from casp domain definition.'''
    infodict = {}
    for _, row in pd.read_csv(casp_metadata_csv).iterrows():
        target = row['Target']
        size = row['Residues']
        type = 'CASP14' if int(target[1:5]) < 1104 else 'CASP15'
        domain = (row['Domains'], row['Residues in domain'], row['Classification'])
        if target in infodict:
            infodict[target]['domain'].append(domain)
        else:
            infodict[target] = {
                'size': size,
                'type' : type,
                'domain': [domain],
            }
    return infodict


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


def main():
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <input_fasta> <pdb_directory> <output_lmdb>")
    inpfas, pdbdir, outlmdb = sys.argv[1:4]

    # parse fasta file
    seqs = parse_fastafile(inpfas)
    print(f"Parsed {len(seqs)} sequences from {inpfas}.")

    # parse metainformation
    metainfo4target = {
        **parse_cameo_metadata(cameo_subset_csv),
        **parse_casp_metadata(casp14_domain_csv),
        **parse_casp_metadata(casp15_domain_csv),
        }
    #for target, metainfo in metainfo4target.items():
    #    print(target, metainfo)

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
        assert target in metainfo4target, f"ERROR: {target} metainfo not found."
        assert length == len(seq) == metainfo4target[target]['size'], (
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
        types.append(metainfo4target[target]['type'])
        pdbs.append(atomlines)
        domains.append(metainfo4target[target]['domain'])
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
