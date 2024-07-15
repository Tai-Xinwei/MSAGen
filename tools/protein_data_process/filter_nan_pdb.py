# -*- coding: utf-8 -*-
import numpy as np
import torch
import pickle
import zlib
import lmdb
import copy
def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))

def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

lmdb_path = '/fastdata/peiran/psm/20240630_PDB_Training_Data/20240630_snapshot.20240711_dd3e1b69.subset_release_date_before_20200430.ligand_protein.lmdb'
lmdb_path2 = '/fastdata/peiran/psm/20240630_PDB_Training_Data/20240630_snapshot.20240711_dd3e1b69.subset_release_date_before_20200430.ligand_protein_filteredNan.lmdb'

env = lmdb.open(
    str(lmdb_path),
    subdir=True,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)
txn = env.begin(write=False)
metadata = bstr2obj(txn.get("__metadata__".encode()))
keys = metadata["keys"]

env2 = lmdb.open(lmdb_path2, map_size=1024 ** 4)
txn2 = env2.begin(write=True)

keys2 = []
for idx in range(len(keys)):
    key = keys[idx]
    value = txn.get(key.encode())
    if value is None:
        raise IndexError(f"Name {key} has no data in the dataset")
    ori_data = copy.deepcopy(bstr2obj(value))
    polymer_chains = ori_data["polymer_chains"]
    non_polymers = ori_data["nonpoly_graphs"]

    filtered_ligand = []
    for ligand in non_polymers:
        ligand_pos = ligand["node_coord"]
        # pick random atom from ligand as crop center, there is nan in the ligand node_coord, avoid it
        if np.all(np.isnan(ligand_pos)):
            continue
        else:
            filtered_ligand.append(ligand)

    filtered_polymer_chains = {}
    num_polymer = 0
    for key in polymer_chains.keys():
        # # TODO: filter DNA/RNA chains, needs to be considered in the future
        if np.any(polymer_chains[key]["restype"] == "N"):
            continue

        # some croped polymer has all Nan coords, so we need to avoid it
        if np.all(np.isnan(polymer_chains[key]["center_coord"])):
            continue
        else:
            filtered_polymer_chains[key] = polymer_chains[key]
            num_polymer += 1

    if num_polymer > 0 or len(filtered_ligand) > 0:
        ori_data["polymer_chains"] = filtered_polymer_chains
        ori_data["nonpoly_graphs"] = filtered_ligand
        txn2.put(key.encode(), obj2bstr(ori_data))
        keys2.append(key)


metadata2 = copy.deepcopy(metadata)
metadata2["keys"] = keys2
txn2.put("__metadata__".encode(), obj2bstr(metadata2))

txn2.commit()
env2.sync()
env2.close()
