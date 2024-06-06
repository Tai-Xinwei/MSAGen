#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd


casp14_domain_definition = Path('/data/database/casp/casp14/casp14_domain_definations_and_classifications.csv')
if not casp14_domain_definition.exists():
    raise FileNotFoundError(f"ERROR: {casp14_domain_definition} not found")

casp15_domain_definition = Path('/data/database/casp/casp15/casp15_domain_definations_and_classifications.csv')
if not casp15_domain_definition.exists():
    raise FileNotFoundError(f"ERROR: {casp15_domain_definition} not found")


def select_residues_by_index(atomlines: list, residueindex: set) -> list:
    # parse pdbfile lines by residue index
    lines = []
    for line in atomlines:
        if line.startswith('ATOM'):
            resnum = int( line[22:26].strip() )
            if resnum in residueindex:
                lines.append(line)
        elif line.startswith('TER') or line.startswith('END'):
            continue
        else:
            lines.append(line)
    lines.append('TER\n')
    lines.append('END\n')
    return lines


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <output_root_directory> <server_id>")
    rootdir, server_id = sys.argv[1:3]

    # check directory
    inptargetdir = Path(rootdir) / 'casp-official-targets.prediction'
    outdomaindir = Path(rootdir) / 'casp-official-trimmed-to-domains.prediction'
    if not inptargetdir.exists() or not outdomaindir.exists():
        raise FileNotFoundError(f"{inptargetdir} or {outdomaindir} not found")

    # read in domain definition
    df = pd.concat([
        pd.read_csv(casp14_domain_definition),
        pd.read_csv(casp15_domain_definition)
        ]).reset_index(drop=True)
    print(df)

    for idx, row in df.iterrows():
        # parse domain and residue index
        domainstring = row['Domains']
        cols = domainstring.split(':')
        assert 2 == len(cols), f"ERROR: wrong domain format: {domainstring}"
        domname, domseg = cols[0], cols[1]
        residueindex = set()
        for seg in domseg.split(','):
            arr = seg.split('-')
            assert len(arr) == 2, f"ERROR: wrong segment: {domainstring} {seg}"
            start, finish = int(arr[0]), int(arr[1])
            residueindex.update( range(start, finish+1) )
        assert row['Residues in domain'] == len(residueindex), (
            f"ERROR: wrong domain definition: {domainstring}")

        # check residue index
        domstr = f'{domname}: '
        for i in sorted(residueindex):
            if i-1 in residueindex and i+1 in residueindex:
                continue
            elif i-1 in residueindex:
                domstr += f'{i},'
            elif i+1 in residueindex:
                domstr += f'{i}-'
            else:
                raise ValueError(f"ERROR: wrong domain index: {domainstring}")
        domstr = domstr.rstrip(',')
        assert domainstring == domstr, f"ERROR: wrong: {domainstring} {domstr}"

        # check domain is D0 or not
        if domname.endswith('D0'):
            continue

        # check native pdb exists or not
        tarname = row['Target']
        pdbfile = inptargetdir / f'{tarname}.pdb'
        preddir = inptargetdir / tarname
        if not pdbfile.is_file() or not preddir.exists():
            continue
        print(f"{domname:10} {len(residueindex):4} {domseg}")

        # parse pdbfile lines by residue index
        domlines = []
        with open(pdbfile, 'r') as fp:
            domlines = select_residues_by_index(fp.readlines(), residueindex)

        # write out domain pdb file
        if (outdomaindir / domname).exists():
            dompdbfile = outdomaindir / f'{domname}.pdb'
            print(dompdbfile)
            with open(dompdbfile, 'w') as fp:
                fp.writelines(domlines)

        # parse prediction file
        for model in sorted( preddir.glob(f'{tarname}TS{server_id:3}_*') ):
            print(model)
            # read lines from prediction file
            modlines = []
            with open(model, 'r') as fp:
                modlines = select_residues_by_index(fp.readlines(), residueindex)
            # write out domain prediction file
            if (outdomaindir / domname).exists():
                dompredfile = outdomaindir / domname / f'{model.name}{domname[-3:]}'
                with open(dompredfile, 'w') as fp:
                    fp.writelines(modlines)
