# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd


def parse_cameo_metadata(cameo_metadata_csv: str) -> dict:
    '''Parse metadata from cameo subset.'''
    diffdict = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
    infodict = {}
    for _, row in pd.read_csv(cameo_metadata_csv).iterrows():
        target = f'{row["ref. PDB [Chain]"][:4]}_{row["ref. PDB [Chain]"][6]}'
        size = row['Sequence Length (residues)']
        type = 'CAMEO'
        domain0 = (f'{target}-D0: 1-{size}',
                  size,
                  diffdict.get(row['Difficulty'], 'Hard'))
        infodict[target] = {
            'size': size,
            'type' : type,
            'domain': [domain0],
        }
    return infodict


def parse_casp_metadata(casp_metadata_csv: str) -> dict:
    '''Parse metadata from casp domain definition.'''
    infodict = {}
    for _, row in pd.read_csv(casp_metadata_csv).iterrows():
        if '-D0' in row['Domains']:
            # skip pre-defined -D0 domain and re-define -D0 with MultiDom flag
            continue
        target = row['Target']
        size = row['Residues']
        type = 'CASP14' if int(target[1:5]) < 1104 else 'CASP15'
        domain = (row['Domains'],
                  row['Residues in domain'],
                  row['Classification'])
        if target in infodict:
            infodict[target]['domain'].append(domain)
        else:
            domain0 = (f'{target}-D0: 1-{size}', size, 'MultiDom')
            infodict[target] = {
                'size': size,
                'type' : type,
                'domain': [domain0, domain],
            }
    return infodict


HERE = Path(__file__).resolve().parent

cameo_subset_csv = HERE / 'cameo_chain_definations_from_20220401_to_20220625.csv'
assert cameo_subset_csv.exists(), f"Could not find {cameo_subset_csv}"
cameo_metadata = parse_cameo_metadata(cameo_subset_csv)

casp14_domain_csv = HERE / 'casp14_domain_definations_and_classifications.csv'
assert casp14_domain_csv.exists(), f"Could not find {casp14_domain_csv}"
casp14_metadata = parse_casp_metadata(casp14_domain_csv)

casp15_domain_csv = HERE / 'casp15_domain_definations_and_classifications.csv'
assert casp15_domain_csv.exists(), f"Could not find {casp15_domain_csv}"
casp15_metadata = parse_casp_metadata(casp15_domain_csv)

metadata4target = {**cameo_metadata, **casp14_metadata, **casp15_metadata}
