# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd


def parse_cameo_metadata(cameo_metadata_csv: str) -> dict:
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


def parse_casp_metadata(casp_metadata_csv: str) -> dict:
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


HERE = Path(__file__).resolve().parent.parent / 'protein_data_process'

casp14_domain_csv = HERE / 'casp14_domain_definations_and_classifications.csv'
assert casp14_domain_csv.exists(), f"Could not find {casp14_domain_csv}"
casp14_metadata = parse_casp_metadata(casp14_domain_csv)

casp15_domain_csv = HERE / 'casp15_domain_definations_and_classifications.csv'
assert casp15_domain_csv.exists(), f"Could not find {casp15_domain_csv}"
casp15_metadata = parse_casp_metadata(casp15_domain_csv)

cameo_subset_csv = HERE / 'cameo_chain_definations_from_20220401_to_20220625.csv'
assert cameo_subset_csv.exists(), f"Could not find {cameo_subset_csv}"
cameo_metadata = parse_cameo_metadata(cameo_subset_csv)

metadata4target = {**casp14_metadata, **casp15_metadata, **cameo_metadata}
