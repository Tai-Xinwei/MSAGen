#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence
from typing import Tuple

import lmdb
import numpy as np
from absl import logging
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
from tqdm import tqdm

from commons import bstr2obj
from commons import obj2bstr
from metadata import metadata4target
from pdb_parsing import parse_structure


logging.set_verbosity(logging.INFO)


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


def make_alignmets_by_biopython(seq: str, pdbseq: str) -> None:
    alignments = PairwiseAligner(scoring='blastp').align(seq, pdbseq)
    if len(alignments) > 1:
        # parameters copy from hh-suite/scripts/renumberpdb.pl
        # https://github.com/soedinglab/hh-suite/blob/master/scripts/renumberpdb.pl
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -3
        aligner.target_open_gap_score = -20
        aligner.extend_gap_score = -0.1
        aligner.end_gap_score = -0.09
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        alignments = aligner.align(seq, pdbseq)
    return alignments


def parse_and_write_to_lmdb(inpfas: str, pdbdir: str, outlmdb: str) -> None:
    # parse fasta file
    seqs = parse_fastafile(inpfas)
    logging.info(f"Parsed {len(seqs)} sequences from {inpfas}.")

    # parse metainformation
    # for target, metadata in metadata4target.items():
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
        assert target in metadata4target, f"ERROR: {target} metainfo not found."
        assert length == len(seq) == metadata4target[target]['size'], (
            f"ERROR: wrong sequence length for {target}")
        # read in pdb file and parsing
        pdbfile = Path(pdbdir) / f'{target}.pdb'
        assert pdbfile.exists(), f"Native pdb dose not exist {pdbfile}."
        with open(pdbfile, 'r') as fin:
            pdb_string = fin.read()
        parsed_result = parse_structure(file_id=target, pdb_string=pdb_string)
        assert parsed_result.pdb_object, f"The errors are {parsed_result.errors}"
        chain_ids = list(parsed_result.pdb_object.chain_to_seqres.keys())
        assert len(chain_ids) == 1, f"Only allow one chain in {pdbfile}."
        chain_id = chain_ids[0]
        pdbseq = parsed_result.pdb_object.chain_to_seqres[chain_id]
        pdbstr = parsed_result.pdb_object.seqres_to_structure[chain_id]
        assert len(pdbseq) == len(pdbstr), f"Wrong parsing result for {pdbfile}"
        # align fasta sequence and pdb sequence and create mapping
        alignments = make_alignmets_by_biopython(seq, pdbseq)
        if len(alignments) > 1:
            logging.warning(f"{target} multiple alignments between seq and pdb.")
        ali = alignments[0]
        if target == 'T1119':
            ali = alignments[2]
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
        aa, pos = [], []
        for i, c in enumerate(seq):
            aa.append(c)
            _p = [float('nan'), float('nan'), float('nan')]
            residx = seq_to_pdbseq_index_mapping[i]
            if residx != -1:
                for atom in pdbstr[residx][1]: # (residue, atoms)
                    if atom.name == 'CA':
                        _p = [atom.x, atom.y, atom.z]
                        break
            pos.append(_p)
        data = {'aa': np.array(aa), 'pos': np.array(pos), 'size': len(aa),}
        with open(pdbfile, 'r') as fp:
            atomlines = fp.readlines()
        assert atomlines, f"ERROR: empty pdb file {pdbfile}."
        # write to lmdb
        txn.put(target.encode(), obj2bstr(data))
        metadata['keys'].append(target)
        metadata['sizes'].append(length)
        metadata['types'].append(metadata4target[target]['type'])
        metadata['pdbs'].append(atomlines)
        metadata['domains'].append(metadata4target[target]['domain'])
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

            print(f">{name} "
                  f"size={size} "
                  f"type={type} "
                  f"lines={len(pdb)} "
                  f"domain={domain}")
            print(''.join(data['aa']))
            for i, axis in enumerate('xyz'):
                arr = [f'{_:.3f}' for _ in data['pos'][:10, i]]
                print(f"data['pos'][:10].{axis}: [{', '.join(arr)}]")
            print(pdb[0], end='')
            print(pdb[1], end='')
            print(pdb[2], end='')
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
    outlmdb = outdir / "cameo-subset-casp14-and-casp15-combined.lmdb"
    parse_and_write_to_lmdb(str(inpfas), str(pdbdir), str(outlmdb))
    show_lmdb(str(outlmdb))

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
                    tmpdata.append(txn.get(name.encode()))
                    tmpmeta['keys'].append(name)
                    tmpmeta['sizes'].append(metadata['sizes'][idx])
                    tmpmeta['types'].append(type)
                    tmpmeta['pdbs'].append(metadata['pdbs'][idx])
                    tmpmeta['domains'].append(domain)
            for k, v in tmpmeta.items():
                print(f"Filtered {k}: {len(v)}")
            env.close()
        except Exception as e:
            logging.error(f"Failed to read information from {outlmdb}, {e}.")

        tmplmdb = Path(outdir) / f"proteintest-{_type.lower()}-{g.lower()}.lmdb"
        with lmdb.open(str(tmplmdb), map_size=1024**4).begin(write=True) as txn:
            for name, data in zip(tmpmeta['keys'], tmpdata):
                txn.put(name.encode(), data)
            txn.put('__metadata__'.encode(), obj2bstr(tmpmeta))
        show_lmdb(tmplmdb)
