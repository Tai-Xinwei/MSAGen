#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser
from pathlib import Path

import lmdb
import numpy as np
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


def parse_and_write_to_lmdb(inpfas: Path, pdbdir: Path, outlmdb: Path):
    # parse fasta file
    seqs = parse_fastafile(inpfas)
    print(f"Parsed {len(seqs)} sequences from {inpfas}.")

    # parse metainformation
    for target, metadata in metadata4target.items():
        print(target, metadata)

    env = lmdb.open(str(outlmdb), map_size=1024**4)
    txn = env.begin(write=True)

    metadata = {
        'keys': [],
        'sizes': [],
        'types': [],
        'pdbs': [],
        'domains': [],
        }
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
        metadata['keys'].append(target)
        metadata['sizes'].append(length)
        metadata['types'].append(metadata4target[target]['type'])
        metadata['pdbs'].append(atomlines)
        metadata['domains'].append(metadata4target[target]['domain'])
    txn.put("__metadata__".encode(), obj2bstr(metadata))

    txn.commit()
    env.close()


def show_lmdb(outlmdb: Path):
    env = lmdb.open(str(outlmdb), readonly=True)
    txn = env.begin(write=False)
    try:
        key = '__metadata__'.encode()
        value = txn.get(key)
        metadata = bstr2obj(value)
        for k, v in metadata.items():
            print(f"{k}: {len(v)}")

        assert 'keys' in metadata, f"'keys' not found in metadata {outlmdb}"
        for name in metadata['keys']:
            key = name.encode()
            value = txn.get(key)
            seq = bstr2obj(value)

            idx = metadata['keys'].index(name)
            size = metadata['sizes'][idx]
            type = metadata['types'][idx]
            pdb = metadata['pdbs'][idx]
            domain = metadata['domains'][idx]

            print(f">{name} "
                  f"size={size} "
                  f"type={type} "
                  f"lines={len(pdb)} "
                  f"domain={domain}")
            if isinstance(seq, str):
                print(seq)
            elif isinstance(seq, dict):
                print(''.join(seq['aa']))
            print(pdb[0], end='')
            print(pdb[1], end='')
            print(pdb[-1], end='')
    except ValueError as e:
        sys.exit(f"ERROR: Failed to read {outlmdb}, {e}")
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--inpfas',
                        type=str,
                        required=True,
                        help="Input fasta combined all test protein sequences.")
    parser.add_argument('--pdbdir',
                        type=str,
                        required=True,
                        help="Input directory containing all native pdb files.")
    parser.add_argument('--outdir',
                        type=str,
                        default="/tmp/proteintest/",
                        help="Output processed lmdb file.")
    args = parser.parse_args()

    inpfas = Path(args.inpfas).resolve()
    pdbdir = Path(args.pdbdir).resolve()
    outdir = Path(args.outdir).resolve()
    assert outdir.exists(), f"Output directory {outdir} does not exist."
    outlmdb = outdir / "cameo-subset-casp14-and-casp15-combined.lmdb"
    parse_and_write_to_lmdb(inpfas, pdbdir, outlmdb)
    show_lmdb(outlmdb)

    CATEGORY = {
        "CAMEO  Easy": ["Easy"],
        "CAMEO  Medi": ["Medium", "Hard"],
        "CASP14 Full": ["MultiDom"],
        "CASP15 Full": ["MultiDom"],
    }
    for category, groups in CATEGORY.items():
        print(f"Collecting data for {category}")
        _type, g = category.split()
        tmpdata = []
        tmpmeta = {'keys':[], 'sizes':[], 'types':[], 'pdbs':[],'domains':[]}

        try:
            env = lmdb.open(str(outlmdb), readonly=True)
            txn = env.begin(write=False)
            metadata = bstr2obj(txn.get('__metadata__'.encode()))
            for k, v in metadata.items():
                print(f"Original {k}: {len(v)}")
            for name in metadata['keys']:
                idx = metadata['keys'].index(name)
                type = metadata['types'][idx]
                domain = metadata['domains'][idx][0]
                assert domain[0].startswith(f'{name}-D0'), (
                    f"The first domain must be {name}-D0, {domain[0]}")

                if _type == type and domain[2] in groups:
                    seq = bstr2obj(txn.get(name.encode()))
                    tmpdata.append({
                        'aa': np.array([_ for _ in seq]),
                        'pos': np.tile(np.nan, [len(seq), 37, 3])
                        })
                    tmpmeta['keys'].append(name)
                    tmpmeta['sizes'].append(metadata['sizes'][idx])
                    tmpmeta['types'].append(type)
                    tmpmeta['pdbs'].append(metadata['pdbs'][idx])
                    tmpmeta['domains'].append(domain)
            for k, v in tmpmeta.items():
                print(f"Filtered {k}: {len(v)}")
            env.close()
        except Exception as e:
            sys.exit(f"ERROR: Failed to read information from {outlmdb}, {e}.")

        tmplmdb = Path(outdir) / f"proteintest-{_type.lower()}-{g.lower()}.lmdb"
        with lmdb.open(str(tmplmdb), map_size=1024**4).begin(write=True) as txn:
            for n, d in zip(tmpmeta['keys'], tmpdata):
                txn.put(n.encode(), obj2bstr(d))
            txn.put('__metadata__'.encode(), obj2bstr(tmpmeta))
        show_lmdb(tmplmdb)
