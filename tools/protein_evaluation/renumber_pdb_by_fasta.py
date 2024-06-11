#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys


def parse_fastafile(fastafile: str):
    import logging
    seqs = []
    try:
        with open(fastafile, "r") as fp:
            header, seq = "", []
            for line in fp:
                if line[0] == ">":
                    seqs.append( (header, "".join(seq)) )
                    header, seq = line.strip(), []
                else:
                    seq.append( line.strip() )
            seqs.append( (header, "".join(seq)) )
            del seqs[0]
    except Exception as e:
        logging.error(f"Failed to read fasta file {fastafile}, {e}")
    return seqs


def pdb2fasta(pdbfile: str):
    """
    Converting PDB format protein structure to FASTA format amino acid sequence.
    Copy from https://github.com/kad-ecoli/pdb2fasta/blob/master/pdb2fasta.py.
    These programs have consistent behavior over the following scenarios:
      - If atoms have alternative locations (e.g. 3b2c), only those atoms with
        alternative location identifier ' ' or 'A' will be considered.
      - If a protein contain non-standard amino acids (e.g. 1a62), only the
        "MSE" residue will be converted to "MET", while other non-standard
        amino acids are ignored.
      - If a residue have the insetion code (e.g. 2p83), it will still be
        considered.
      - If PDB file contains multi-models (e.g. 2m9l), only the first model
        will be considered.
    """
    import re
    aa3to1 = {
        "ALA": "A",
        "ASX": "B", # D, N
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "XLE": "J", # I, L
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "MSE": "M", # MSE ~ MET
        "ASN": "N",
        "PYL": "O",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "SEC": "U",
        "VAL": "V",
        "TRP": "W",
        "XAA": "X", # All
        "TYR": "Y",
        "GLX": "Z", # E, Q
        }
    ca_pattern = re.compile(
        "^ATOM\s{2,6}\d{1,5}\s{2}CA\s[\sA]([A-Z]{3})\s([\s\w])|"
        "^HETATM\s{0,4}\d{1,5}\s{2}CA\s[\sA](MSE)\s([\s\w])"
        )
    chain_list, chain_dict = [], dict()
    with open(pdbfile, "r") as fp:
        for line in fp.read().splitlines():
            if line.startswith("ENDMDL"):
                break
            match_list = ca_pattern.findall(line)
            if match_list:
                resn = match_list[0][0] + match_list[0][2]
                chain = match_list[0][1] + match_list[0][3]
                if chain in chain_dict:
                    chain_dict[chain] += aa3to1[resn]
                else:
                    chain_dict[chain] = aa3to1[resn]
                    chain_list.append(chain)
    return chain_list, chain_dict


def pdb2residues(pdbfile: str):
    aa3to1 = {
        "ALA": "A",
        "ASX": "B", # D, N
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "XLE": "J", # I, L
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "MSE": "M", # MSE ~ MET
        "ASN": "N",
        "PYL": "O",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "SEC": "U",
        "VAL": "V",
        "TRP": "W",
        "XAA": "X", # All
        "TYR": "Y",
        "GLX": "Z", # E, Q
        }
    residues, resid = [], ""
    with open(pdbfile, 'r') as fp:
        for line in fp:
            if line.startswith("ENDMDL"):
                break
            if len(line) < 55:
                continue
#         1         2         3         4         5         6         7         8
#12345678901234567890123456789012345678901234567890123456789012345678901234567890
#ATOM     32  N  AARG A  -3      11.281  86.699  94.383  0.50 35.88           N
#ATOM     33  N  BARG A  -3      11.296  86.721  94.521  0.50 35.60           N
            type, altloc, resname = line[:6], line[16], line[17:20]
            if type == "ATOM  " or (type == "HETATM" and resname == "MSE"):
                if altloc != " " and altloc != "A":
                    continue
                if line[21:26] == resid:
                    residues[-1][-1].append(line)
                else:
                    resid = line[21:26]
                    residues.append( (resname, aa3to1[resname], resid, [line]) )
    return residues


def renumberpdb(seq, pdbseq, residues):
    import logging
    from Bio.Align import PairwiseAligner
    alignments = PairwiseAligner(scoring='blastp').align(seq, pdbseq)
    if len(alignments) > 1:
        logging.error("Multiple alignments between fasta and pdb")
    ali = alignments[0]
    assert seq == ali[0, :], f"The pdbseq must be subseq of fasta\n{ali}"
    gap_ratio = 1.0 * sum(x=='-' for x in ali[1, :]) / ali.length
    if gap_ratio > 0.2:
        logging.error(f"Too many gaps and the pdbseq is too short\n{ali}")
    assert len(pdbseq) == len(residues), "Different pdbseq and residues"
    lines, idx = [], 0
    for resnum, (x, y) in enumerate(zip(seq, ali[1, :]), start=1):
        if y == '-': continue
        lines.extend([_[:22]+f'{resnum:4d}'+_[26:] for _ in residues[idx][-1]])
        idx += 1
    # assert idx == len(residues), "Different number of residues"
    return lines


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('Usage: %s <raw_structure_pdb> <sequence_fasta> <new_structure_pdb>' % sys.argv[0])
    rawpdb, seqfas, newpdb = sys.argv[1:4]
    print(rawpdb, seqfas, newpdb, sep='\n')

    seqs = parse_fastafile(seqfas)
    assert 1 == len(seqs), f"More than one sequence in {seqfas}"
    assert 2 == len(seqs[0]), f"Wrong fasta format {seqfas}"
    header, seq = seqs[0]
    if 'length=' in header:
        length = int(header.split('length=')[1].split()[0])
        assert len(seq) == length, f"Wrong sequence length {seqfas}"
    #print(header)
    #print(seq)

    chain_list, chain_dict = pdb2fasta(rawpdb)
    assert 1 == len(chain_list), f"More than one chain in {rawpdb}"
    pdbseq = chain_dict[ chain_list[0] ]
    #print(pdbseq)

    residues = pdb2residues(rawpdb)
    assert len(pdbseq) == len(residues), f"Failed to get residues {rawpdb}"
    for i, aa in enumerate(pdbseq):
        assert aa == residues[i][1], f"Wrong residue {aa} in {rawpdb}"
    #for res in residues:
    #    print(res[:3], f"{len(res[3]):2}", "atoms")

    lines = renumberpdb(seq, pdbseq, residues)
    with open(newpdb, 'w') as fp:
        fp.writelines(lines)
        print('TER', 'END', sep='\n', file=fp)
