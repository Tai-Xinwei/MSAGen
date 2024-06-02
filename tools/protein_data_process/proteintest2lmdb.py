#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
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
        pdbchain = row['ref. PDB [Chain]']
        target = f'{pdbchain[:4]}_{pdbchain[6]}'
        infodict[target] = {
            'length': row['Sequence Length (residues)'],
            'type' : 'CAMEO',
            'group': tmpdict[row['Difficulty']],
        }
    return infodict


def parse_casp_metadata(casp_metadata_csv) -> dict:
    '''Parse metadata from casp domain definition.'''
    infodict = {}
    for _, row in pd.read_csv(casp_metadata_csv).iterrows():
        target = row['Target']
        domain = row['Domains'].split(':')[0]
        casp_round = 'CASP14' if int(target[1:5]) < 1104 else 'CASP15'
        infodict[target] = {
            'length': row['Residues'],
            'type' : casp_round,
            'group': row['Type'],
        }
        infodict[domain] = {
            'length': row['Residues in domain'],
            'type' : casp_round,
            'group': row['Classification'],
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


def show_lmdb(lmdbdir):
    with lmdb.open(lmdbdir, readonly=True).begin(write=False) as txn:
        metadata = bstr2obj( txn.get('__metadata__'.encode()) )
        print(f"{len(metadata['keys'])} keys in {lmdbdir}" )
        print(f"{len(metadata['sizes'])} sizes in {lmdbdir}" )
        print(f"{len(metadata['types'])} types in {lmdbdir}" )
        print(f"{len(metadata['groups'])} groups in {lmdbdir}" )
        print(f"{len(metadata['pdbs'])} pdbs in {lmdbdir}")

        for (k, l, t, g, p) in zip(metadata['keys'],
                                   metadata['sizes'],
                                   metadata['types'],
                                   metadata['groups'],
                                   metadata['pdbs']):
            value = txn.get(k.encode())
            assert value, f"Key {k} not found."
            seq = bstr2obj(value)
            print(f"name={k}, length={l}, type={t}, group={g}, lines={len(p)}")
            print(seq)
            print(p[0], end='')
            print(p[-1], end='')


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
    groups = []
    pdbs = []
    for header, seq in tqdm(seqs):
        # parse target
        assert header[0] == '>' and 'length=' in header, (
            f"ERROR: wrong header {header} in fasta file {inpfas}.")
        cols = header[1:].split()
        target, length = cols[0], int(cols[1].split('=')[1])
        assert target in metainfo4target, f"ERROR: {target} metainfo not found."
        assert length == len(seq) == metainfo4target[target]['length'], (
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
        groups.append(metainfo4target[target]['group'])
        pdbs.append(atomlines)
    metadata = {
        'keys': keys,
        'sizes': sizes,
        'types': types,
        'groups': groups,
        'pdbs': pdbs,
        }
    txn.put("__metadata__".encode(), obj2bstr(metadata))
    txn.commit()

    show_lmdb(outlmdb)


if __name__ == "__main__":
    main()
