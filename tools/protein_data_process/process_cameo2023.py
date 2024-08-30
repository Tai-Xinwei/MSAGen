#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import dataclasses
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence, Tuple

import lmdb
import numpy as np
from absl import logging
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
from Bio.Data.PDBData import protein_letters_3to1_extended as aa3to1
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr
from metadata import cameo2023_metadata


logging.set_verbosity(logging.INFO)


@dataclasses.dataclass(frozen=True)
class Residue:
  name: str
  seqres: str
  is_missing: bool
  resid: str
  atoms: list


def parse_fastafile(fastafile: str) -> Sequence[Tuple[str, str]]:
    """Parse sequence file in fasta format."""
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
        logging.error(f"Failed to parse fasta {fastafile}, {e}")
    return seqs


def pdb2residues(pdbfile: str) -> Tuple[str, str, str, list]:
    protein = collections.defaultdict(dict)
    with open(pdbfile, 'r') as fp:
        for line in fp:
            if line.startswith('ENDMDL'):
                break
            if len(line) < 55:
                continue
#         1         2         3         4         5         6         7         8
#12345678901234567890123456789012345678901234567890123456789012345678901234567890
#ATOM     32  N  AARG A  -3      11.281  86.699  94.383  0.50 35.88           N
#ATOM     33  N  BARG A  -3      11.296  86.721  94.521  0.50 35.60           N
            record, altloc, resname = line[:6], line[16], line[17:20]
            if altloc not in (' ', 'A'):
                continue
            if record == 'ATOM  ' or (record == 'HETATM' and resname == 'MSE'):
                chainid, resnumb = line[21], int(line[22:26].strip())
                current = protein[chainid].get(resnumb, (resname, []))
                current[1].append(line)
                protein[chainid][resnumb] = current
    # fix missing residues
    for chainid, chaindata in protein.items():
        _min, _max = min(chaindata.keys()), max(chaindata.keys())
        for i in range(_min, _max + 1):
            if i not in chaindata:
                chaindata[i] = ('XAA', [])
    # convert to list
    residues = []
    for chainid, chaindata in protein.items():
        for resnumb in sorted(chaindata.keys()):
            resname, lines = chaindata[resnumb]
            residues.append(Residue(name=resname,
                                    seqres=aa3to1.get(resname, 'X'),
                                    is_missing=resname == 'XAA',
                                    resid=f'{chainid}{resnumb:>4d}',
                                    atoms=lines))
    return residues


def make_alignmets_by_biopython(seq: str, pdbseq: str) -> Any:
    alignments = PairwiseAligner(scoring='blastp').align(seq, pdbseq)
    if len(alignments) > 1:
        # parameters copy from hh-suite/scripts/renumberpdb.pl
        # https://github.com/soedinglab/hh-suite/blob/master/scripts/renumberpdb.pl
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -3
        aligner.target_open_gap_score = -200
        aligner.extend_gap_score = -0.1
        aligner.end_gap_score = -0.09
        aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
        alignments = aligner.align(seq, pdbseq)
    return alignments


def parse_and_write_to_lmdb(inpfas: str, pdbdir: str, outlmdb: str) -> None:
    # parse fasta file
    seqs = parse_fastafile(inpfas)
    logging.info(f"Parsed {len(seqs)} sequences from {inpfas}.")

    # parse metainformation
    # for target, metadata in cameo2023_metadata.items():
    #     print(target, metadata)

    env = lmdb.open(str(outlmdb), map_size=1024**4)
    txn = env.begin(write=True)

    logging.info(f"Writing {len(seqs)} sequences to {outlmdb}.")
    metadata = {
        'keys': [],
        'sizes': [],
        'types': [],
        'pdbs': [],
        'domains': [],
        }
    for header, seq in tqdm(seqs):
        # parse target name and length
        assert header[0] == '>' and 'length=' in header, (
            f"ERROR: wrong header {header} in fasta file {inpfas}.")
        cols = header[1:].split()
        target, length = cols[0], int(cols[1].split('=')[1])
        assert target in cameo2023_metadata, f"ERROR: {target} metainfo not found."
        # if target != '8bu0_A': continue
        assert length == len(seq) == cameo2023_metadata[target]['size'], (
            f"ERROR: wrong sequence length for {target}")
        # read in pdb file and parsing
        pdbfile = Path(pdbdir) / f'{target}.pdb'
        assert pdbfile.exists(), f"Native pdb dose not exist {pdbfile}."
        pdb_residues = pdb2residues(str(pdbfile))
        pdbseq = ''.join(_.seqres for _ in pdb_residues)
        alignments = make_alignmets_by_biopython(seq, pdbseq)
        if len(alignments) == 1:
            ali = alignments[0]
        elif target == '8bu0_A':
            ali = alignments[4]
        else:
            raise ValueError(f"{target} multiple alignments between seq and pdb.")
        # print(ali)
        seq_to_pdbseq_index_mapping = {}
        for i in range(ali.length):
            target_index = ali.indices[0][i]
            query_index = ali.indices[1][i]
            seqaa = ali.target[target_index] if target_index != -1 else '-'
            pdbaa = ali.query[query_index] if query_index != -1 else '-'
            assert ali[0][i] == seqaa and ali[1][i] == pdbaa, "Wrong alignment."
            if seqaa == '-':
                continue
            seq_to_pdbseq_index_mapping[target_index] = query_index
        # Append coordinates to the sequence
        aa, pos = ['']*len(seq), [[np.nan, np.nan, np.nan]]*len(seq)
        for i, c in enumerate(seq):
            aa[i] = c
            residx = seq_to_pdbseq_index_mapping[i]
            if residx == -1:
                continue
            for atomline in pdb_residues[residx].atoms:
                if atomline[12:16] == ' CA ':
                    x = float(atomline[30:38])
                    y = float(atomline[38:46])
                    z = float(atomline[46:54])
                    pos[i] = [x, y, z]
                    break
        data = {'aa': np.array(aa), 'pos': np.array(pos), 'size': len(aa),}
        # read in pdblines
        with open(pdbfile, 'r') as fp:
            atomlines = fp.readlines()
        assert atomlines, f"ERROR: empty pdb file {pdbfile}."
        # write to lmdb
        txn.put(target.encode(), obj2bstr(data))
        metadata['keys'].append(target)
        metadata['sizes'].append(length)
        metadata['types'].append(cameo2023_metadata[target]['type'])
        metadata['pdbs'].append(atomlines)
        metadata['domains'].append(cameo2023_metadata[target]['domain'])
    txn.put("__metadata__".encode(), obj2bstr(metadata))

    txn.commit()
    env.close()


def show_lmdb(outlmdb: str):
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
            data = bstr2obj(value)

            idx = metadata['keys'].index(name)
            size = metadata['sizes'][idx]
            type = metadata['types'][idx]
            pdb = metadata['pdbs'][idx]
            domain = metadata['domains'][idx]

            # print(f">{name} size={size} type={type} domain={domain}")
            # print(''.join(data['aa']))
            # for i, axis in enumerate('xyz'):
            #     arr = [f'{_:.3f}' for _ in data['pos'][:10, i]]
            #     print(f"data['pos'][:10].{axis}: [{', '.join(arr)}]")
            # print(pdb[0], end='')
            # print(pdb[1], end='')
            # print(pdb[2], end='')
    except ValueError as e:
        logging.error(f"ERROR: Failed to read {outlmdb}, {e}")
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
                        default="./ProteinTest/",
                        help="Output processed lmdb file.")
    args = parser.parse_args()

    inpfas = Path(args.inpfas).resolve()
    pdbdir = Path(args.pdbdir).resolve()
    outdir = Path(args.outdir).resolve()
    assert outdir.exists(), f"Output directory {outdir} does not exist."

    outlmdb = outdir / "cameo-subset-from-20230107-to-20231230.lmdb"
    print(f"Processing {inpfas} and {pdbdir}, save data into {outlmdb}.")
    parse_and_write_to_lmdb(str(inpfas), str(pdbdir), str(outlmdb))
    show_lmdb(str(outlmdb))
