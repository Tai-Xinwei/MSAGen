# -*- coding: utf-8 -*-
import gzip

def sdf2json(path):
    with gzip.open(path, 'rt') as sdf_file:
        i = 0
        data = []
        item = {}
        flag = 0
        for line in sdf_file:
            # print(line.strip())
            if line.strip() == '> <PUBCHEM_COMPOUND_CID>':
                flag = 1
            elif line.strip() == '> <PUBCHEM_IUPAC_OPENEYE_NAME>':
                flag = 2
            elif line.strip() == '> <PUBCHEM_IUPAC_CAS_NAME>':
                flag = 3
            elif line.strip() == '> <PUBCHEM_IUPAC_NAME>':
                flag = 4
            elif line.strip() == '> <PUBCHEM_IUPAC_SYSTEMATIC_NAME>':
                flag = 5
            elif line.strip() == '> <PUBCHEM_IUPAC_TRADITIONAL_NAME>':
                flag = 6
            elif line.strip() == '> <PUBCHEM_IUPAC_INCHI>':
                flag = 7
            elif line.strip() == '> <PUBCHEM_IUPAC_INCHIKEY>':
                flag = 8
            elif line.strip() == '> <PUBCHEM_XLOGP3_AA>':
                flag = 9
            elif line.strip() == '> <PUBCHEM_EXACT_MASS>':
                flag = 10
            elif line.strip() == '> <PUBCHEM_MOLECULAR_FORMULA>':
                flag = 11
            elif line.strip() == '> <PUBCHEM_MOLECULAR_WEIGHT>':
                flag = 12
            elif line.strip() == '> <PUBCHEM_OPENEYE_CAN_SMILES>':
                flag = 13
            elif line.strip() == '> <PUBCHEM_OPENEYE_ISO_SMILES>':
                flag = 14
            elif line.strip() == '> <PUBCHEM_CACTVS_TPSA>':
                flag = 15
            elif line.strip() == '> <PUBCHEM_MONOISOTOPIC_WEIGHT>':
                flag = 16
            elif line.strip() == '> <PUBCHEM_TOTAL_CHARGE>':
                flag = 17
            elif line.strip() == '> <PUBCHEM_BONDANNOTATIONS>':
                flag = 100

            else:
                if flag == 100:
                    data.append(item)
                    item = {}
                    flag = 0
                elif flag == 1:
                    item['cid'] = line.strip()
                    flag = 0
                elif flag == 2:
                    item['openeye_name'] = line.strip()
                    flag = 0
                elif flag == 3:
                    item['cas_name'] = line.strip()
                    flag = 0
                elif flag == 4:
                    item['name'] = line.strip()
                    flag = 0
                elif flag == 5:
                    item['systematic_name'] = line.strip()
                    flag = 0
                elif flag == 6:
                    item['traditional_name'] = line.strip()
                    flag = 0
                elif flag == 7:
                    item['inchi'] = line.strip()
                    flag = 0
                elif flag == 8:
                    item['inchikey'] = line.strip()
                    flag = 0
                elif flag == 9:
                    item['xlogp3_aa'] = float(line.strip())
                    flag = 0
                elif flag == 10:
                    item['exact_mass'] = float(line.strip())
                    flag = 0
                elif flag == 11:
                    item['molecular_formula'] = line.strip()
                    flag = 0
                elif flag == 12:
                    item['molecular_weight'] = float(line.strip())
                    flag = 0
                elif flag == 13:
                    item['openeye_can_smiles'] = line.strip()
                    flag = 0
                elif flag == 14:
                    item['openeye_iso_smiles'] = line.strip()
                    flag = 0
                elif flag == 15:
                    item['cactvs_tpsa'] = float(line.strip())
                    flag = 0
                elif flag == 16:
                    item['monoisotopic_weight'] = float(line.strip())
                    flag = 0
                elif flag == 17:
                    item['total_charge'] = int(line.strip())
                    flag = 0
                else:
                    pass

        return data

import os
import json
import multiprocessing

def savejson(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)

def process_file(file):
    path = os.path.join('/mnt/pubchem/sdf/', file)
    print(f"Processing {path}")
    data = sdf2json(path)
    savepath = os.path.join('/mnt/pubchem/json/', file.replace('.sdf.gz', '.json'))
    print(f"Saving to {savepath}")
    savejson(data, savepath)


# Get the list of files
files = []
for file in os.listdir('/mnt/pubchem/sdf/'):
    if file.endswith('.sdf.gz'):
        files.append(file)
print(files)

# Create a process pool and start processing the files
with multiprocessing.Pool(12) as pool:
    pool.map(process_file, files)
