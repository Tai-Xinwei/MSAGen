# -*- coding: utf-8 -*-
import argparse

import lmdb
from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from data.utils.lmdb import obj2bstr


def process_dataset(data_dir, db_path):
    dataset = PCQM4Mv2Dataset(root=data_dir, only_smiles=True)
    indices = dataset.get_idx_split()["train"]

    graphs = PCQM4Mv2Dataset(root=data_dir, smiles2graph=smiles2graph)
    suppl = Chem.SDMolSupplier(f"{data_dir}/pcqm4m-v2/pcqm4m-v2-train.sdf")

    assert len(indices) == len(suppl), "Length mismatch for dataset, sdf"

    labels = ["smiles", "homo_lumo_gap"]

    env = lmdb.open(db_path, map_size=1024**4)
    with env.begin(write=True) as txn:
        keys = []
        size = []

        for idx in tqdm(indices):
            data = dict(zip(labels, dataset[idx]))
            data.update(graphs[idx][0])

            mol = suppl[int(idx)]
            data.update({"pos": mol.GetConformer().GetPositions()})
            data.update({"mol": mol})

            if data["pos"].shape[0] != data["num_nodes"]:
                continue

            txn.put(str(idx).encode(), obj2bstr(data))

            keys.append(str(idx))
            size.append(data["num_nodes"])

        metadata = {"keys": keys, "size": size}
        txn.put("metadata".encode(), obj2bstr(metadata))

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--db-path", type=str)

    args = parser.parse_args()

    process_dataset(args.data_dir, args.db_path)
