# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from typing import Any, Dict

import lmdb
from ogb.lsc import PCQM4Mv2Dataset
from rdkit import Chem
from tqdm import tqdm

from sfm.data.mol_data.utils.lmdb import obj2bstr
from sfm.data.mol_data.utils.molecule import mol2graph
from sfm.logging import logger


def smiles2graph(smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    return mol2graph(mol, compress_graph_or_raise=True)


def process_dataset(data_dir, output_dir, split):
    assert split in ["train", "valid", "test-dev", "test-challenge"]

    dataset = PCQM4Mv2Dataset(root=data_dir, only_smiles=True)
    indices = dataset.get_idx_split()[split]

    graphs = PCQM4Mv2Dataset(root=data_dir, smiles2graph=smiles2graph)
    suppl = None
    if split == "train":
        sdf_path = os.path.join(data_dir, "pcqm4m-v2", "pcqm4m-v2-train.sdf")
        if not os.path.exists(sdf_path):
            logger.error(f"3D file [{sdf_path}] not existing, download it first")
            exit(1)
        suppl = Chem.SDMolSupplier(sdf_path)
        assert len(indices) == len(suppl), "Length mismatch for dataset, sdf"

    labels = ["smiles", "homo_lumo_gap"]
    errors = {"num_atoms_mismatch": 0}

    db_path = os.path.join(output_dir, split)
    env = lmdb.open(db_path, map_size=1024**4)
    with env.begin(write=True) as txn:
        keys = []
        size = []

        for idx in tqdm(indices):
            data = dict(zip(labels, dataset[idx]))
            data.update(graphs[idx][0])

            if suppl:
                mol = suppl[int(idx)]
                data["pos"] = mol.GetConformer().GetPositions()
                if data["pos"].shape[0] != data["num_nodes"]:
                    errors["num_atoms_mismatch"] += 1
                    continue

            txn.put(str(idx).encode(), pickle.dumps(data))

            keys.append(str(idx))
            size.append(data["num_nodes"])

        metadata = {"keys": keys, "size": size}
        txn.put("metadata".encode(), obj2bstr(metadata))

    env.close()

    logger.info(f"PCQM4Mv2Dataset[{split}] was processed and saved in {db_path}")
    total, valid = len(indices), len(keys)
    logger.info(
        f"{valid}/{total} ({valid*100/total:.2f}%) molecules were extracted. errors={errors}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    process_dataset(args.data_dir, args.output_dir, split="train")
    process_dataset(args.data_dir, args.output_dir, split="valid")
    process_dataset(args.data_dir, args.output_dir, split="test-dev")
    process_dataset(args.data_dir, args.output_dir, split="test-challenge")
