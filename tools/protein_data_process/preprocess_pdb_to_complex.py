# -*- coding: utf-8 -*-

import torch
import lmdb
import pickle as pkl
import zlib

from joblib import delayed
from joblib import Parallel
from tqdm import tqdm
from queue import Queue

SEQUENCE_LENGTH_LIMIT = 512
EXCEPTIONS = {'144', '15P', '1PE', '2F2', '2JC', '3HR', '3SY', '7N5', '7PE', '9JE', 'AAE', 'ABA', 'ACE', 'ACN', 'ACT', 'ACY', 'AZI', 'BAM', 'BCN', 'BCT', 'BDN', 'BEN', 'BME', 'BO3', 'BTB', 'BTC', 'BU1', 'C8E', 'CAD', 'CAQ', 'CBM', 'CCN', 'CIT', 'CL', 'CLR', 'CM', 'CMO', 'CO3', 'CPT', 'CXS', 'D10', 'DEP', 'DIO', 'DMS', 'DN', 'DOD', 'DOX', 'EDO', 'EEE', 'EGL', 'EOH', 'EOX', 'EPE', 'ETF', 'FCY', 'FJO', 'FLC', 'FMT', 'FW5', 'GOL', 'GSH', 'GTT', 'GYF', 'HED', 'IHP', 'IHS', 'IMD', 'IOD', 'IPA', 'IPH', 'LDA', 'MB3', 'MEG', 'MES', 'MLA', 'MLI', 'MOH', 'MPD', 'MRD', 'MSE', 'MYR', 'N', 'NA', 'NH2', 'NH4', 'NHE', 'NO3', 'O4B', 'OHE', 'OLA', 'OLC', 'OMB', 'OME', 'OXA', 'P6G', 'PE3', 'PE4', 'PEG', 'PEO', 'PEP', 'PG0', 'PG4', 'PGE', 'PGR', 'PLM', 'PO4', 'POL', 'POP', 'PVO', 'SAR', 'SCN', 'SEO', 'SEP', 'SIN', 'SO4', 'SPD', 'SPM', 'SR', 'STE', 'STO', 'STU', 'TAR', 'TBU', 'TME', 'TPO', 'TRS', 'UNK', 'UNL', 'UNX', 'UPL', 'URE'}
EXCEPTIONS |= {'HOH', 'DOD'} # Added by Jianwei Zhu

VOCAB = {
    # "<pad>": 0,  # padding
    # "1"-"127": 1-127, # atom type
    # "<cell_corner>": 128, use for pbc material
    "L": 130,
    "A": 131,
    "G": 132,
    "V": 133,
    "S": 134,
    "E": 135,
    "R": 136,
    "T": 137,
    "I": 138,
    "D": 139,
    "P": 140,
    "K": 141,
    "Q": 142,
    "N": 143,
    "F": 144,
    "Y": 145,
    "M": 146,
    "H": 147,
    "W": 148,
    "C": 149,
    "X": 150,
    "B": 151,
    "U": 152,
    "Z": 153,
    "O": 154,
    "-": 155,
    ".": 156,
    "<mask>": 157,
    "<cls>": 158,
    "<eos>": 159,
    # "<unk>": 160,
}
AA1TO3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR"
}

AA3TO1 = {v: k for k, v in AA1TO3.items()}

def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def generate_2dgraphfeat(data, ligand, protein):
    N = data["token_type"].shape[0]
    adj = torch.zeros([N, N], dtype=torch.bool)
    indgree = adj.long().sum(dim=1).view(-1)

    data["edge_index"] = torch.add(data['protein_len'], torch.tensor(ligand["edge_index"], dtype=torch.long))
    data["edge_attr"] = torch.tensor(ligand["edge_feat"], dtype=torch.long)

    attn_edge_type = torch.zeros([N, N, data["edge_attr"].size(-1)], dtype=torch.long)
    attn_edge_type[data["edge_index"][0, :], data["edge_index"][1, :]] = convert_to_single_emb(
        data["edge_attr"]
    )
    adj[data["edge_index"][0, :], data["edge_index"][1, :]] = True
    indgree = adj.long().sum(dim=1).view(-1)
    protein_attr = torch.cat(
        [
            data["token_type"].unsqueeze(-1),
            torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
        ],
        dim=-1,
    )[:data['protein_len']]
    data["node_attr"] = torch.cat([protein_attr, torch.tensor(ligand["node_feat"])], dim=0)

    data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
    data["in_degree"] = indgree

    if False: #args.preprocess_2d_bond_features_with_cuda:
        attn_edge_type = torch.zeros([N, N, data["edge_attr"].size(-1)], dtype=torch.long)
        data["adj"] = adj
        data["attn_edge_type"] = attn_edge_type
    else:
        shortest_path_result = (
            torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
        )
        edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        data["edge_input"] = edge_input
        data["spatial_pos"] = spatial_pos

    return data

def process_protein(record, handle):
    # print('Processing', key)
    protein, key = record
    protein = pkl.loads(zlib.decompress(protein))
    chains_keys = protein.keys()
    record = []
    for chain_key in chains_keys:
        chain = protein[chain_key]
        if 'polymer' not in chain:
            continue
        polymer = chain['polymer']
        if polymer['polymer_type'] != 'peptide':
            continue
        ligands_keys = chain.keys() - {'polymer'}
        valid = True
        for token in polymer['types']:
            if token not in AA3TO1:
                valid = False
                # print('Ignore invalid protein type', token)
                break
        if not valid:
            continue
        protein_type = torch.tensor([VOCAB[AA3TO1[aa]] for aa in polymer['types']], dtype=torch.long)
        for ligand_key in ligands_keys:
            # create a new record for each ligand
            ligand = chain[ligand_key]
            # combine the protein and ligand
            data = {}
            data['coords'] = torch.cat([torch.Tensor(polymer['coords']), torch.Tensor(ligand['coords'])], dim=0)
            data['protein_len'] = polymer['num_residues']
            data['num_atoms'] = polymer['num_residues'] + ligand['num_nodes']

            ligand_type = torch.tensor(ligand['node_feat'][:, 0])
            data['token_type'] = torch.cat([protein_type, ligand_type], dim=0)
            data = generate_2dgraphfeat(data, ligand, protein)

            #place holders
            data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
            data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
            data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=data['token_type'].device)
            data["forces"] = torch.zeros(
                (data['token_type'].size()[0], 3), dtype=torch.float64, device=data['token_type'].device
            )
            data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=data['token_type'].device)
            data["energy_per_atom"] = torch.tensor(
                [0.0], dtype=torch.float64, device=data['token_type'].device
            )
            record.append(((key + '_' + chain_key + '-' + ligand_key), zlib.compress(pkl.dumps(data)), data['num_atoms']))

    result = []
    for name, raw, nums in record:
        if name.split('_')[-1] in EXCEPTIONS:
            raise ValueError(name)
        if nums > SEQUENCE_LENGTH_LIMIT:
            continue
        result.append((name, raw))
    with handle.begin(write=True) as txn:
       for name, raw in result:
           txn.put(name.encode(), raw)

pdb_handle = lmdb.open('/home/v-zhezhan/data/')
pdb_txn = pdb_handle.begin(write=False)
keys = []
for key, _ in pdb_txn.cursor():
    keys.append(key.decode())

result_handle = lmdb.open('/home/v-zhezhan/preprocessed', map_size=1024**4)
data = []
for key in tqdm(keys):
    data.append((pdb_txn.get(key.encode()), key))

for record in tqdm(data):
    process_protein(record, result_handle)

pdb_handle.close()
result_handle.close()
