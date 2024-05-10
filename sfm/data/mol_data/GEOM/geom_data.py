# -*- coding: utf-8 -*-
import argparse
import json
import pickle
from pathlib import Path

import lmdb
from rdkit import Chem
from tqdm import tqdm

from sfm.data.mol_data.utils.lmdb import obj2bstr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GEOM raw data to LMDB database."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        help="Path to the summary file from GEOM dataset, rdkit_folder sub-directory",
    )
    parser.add_argument(
        "--outlmdb_path",
        type=str,
        help="Path to the output LMDB database.",
    )
    args = parser.parse_args()
    summary = json.load(open(args.summary_path, "r"))
    """
    summary is in this format
    "C": {
        "charge": 0,
        "ensembleenergy": 0.0,
        "ensembleentropy": 0.0,
        "ensemblefreeenergy": 0.0,
        "lowestenergy": -4.17522,
        "pickle_path": "qm9/C.pickle",
        "poplowestpct": 100.0,
        "temperature": 298.15,
        "totalconfs": 1,
        "uniqueconfs": 1
    }
    """
    print(f"{len(summary)} entries in {args.summary_path}")

    env = lmdb.open(args.outlmdb_path, map_size=8 * 1024**4)
    txn = env.begin(write=True)

    for idx, (smiles, value) in tqdm(enumerate(summary.items()), ncols=80):
        # print(f"{smiles}: {value['totalconfs']} conformers")
        try:
            data_dict = {
                "smiles": smiles,
                "charge": value["charge"],
                "ensembleenergy": value["ensembleenergy"],
                "ensembleentropy": value["ensembleentropy"],
                "ensemblefreeenergy": value["ensemblefreeenergy"],
                "lowestenergy": value["lowestenergy"],
                "mol": pickle.load(
                    open(Path(args.summary_path).parent / value["pickle_path"], "rb")
                ),
                "poplowestpct": value["poplowestpct"],
                "temperature": value["temperature"],
                "totalconfs": value["totalconfs"],
                "uniqueconfs": value["uniqueconfs"],
            }
        except Exception as e:
            print(f"Error loading {smiles} - {idx}: {e}")
            print(value)
            continue

        txn.put(f"{idx}".encode(), obj2bstr(data_dict))

    txn.commit()
    env.close()
