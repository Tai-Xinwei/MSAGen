# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from glob import glob
import numpy as np
import os
from .path_to_mol_graph import path_to_mol_graph
import lmdb
import pickle as pkl
from tqdm import tqdm
import random
from rdkit import Chem
from sfm.data.prot_data.util import bstr2obj, obj2bstr

def process_files():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default="/home/peiran/data/pm6_unzip/Compounds")
    # arg_parser.add_argument('start_index', type=int, default=1)
    # arg_parser.add_argument('end_index', type=int)
    # arg_parser.add_argument('num_processes', type=int)
    arg_parser.add_argument('--output_dir', type=str, default="/home/peiran/data/pm6_unzip/output")
    arg_parser.add_argument('--lmdb_dir', type=str, default="/home/peiran/data/pm6_unzip/pm6_8M.lmdb")

    args = arg_parser.parse_args()

    os.system(f"mkdir {args.output_dir}")

    write_env = lmdb.open(f"{args.lmdb_dir}", map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    # xz_names = np.sort(glob(f"{args.data_path}/Compound_*.tar.xz"))[args.start_index: args.end_index]
    xz_names = os.listdir(args.data_path)
    # print(xz_names)

    keys = []
    mol_sizes = []

    i = 0
    filter_i = 0
    for xz_name in xz_names:
        xz_name = os.path.join(args.data_path, xz_name)
        print(xz_name)
        os.system(f"tar -xf {xz_name} -C {args.output_dir}")

        # unzip_dir_names = [xz_name.split("/")[-1].split(".")[0] for xz_name in xz_names]
        unzip_dir_name = xz_name.split("/")[-1].split(".")[0]
        unzip_dir_name = os.path.join(args.output_dir, unzip_dir_name)
        for mol_file in os.listdir(unzip_dir_name):
            if random.random() > 0.12:
                continue

            mol_path = os.path.join(unzip_dir_name, mol_file)

            graph = path_to_mol_graph(mol_path)

            # for graph in tqdm(graphs):
            if "smiles" not in graph or graph["smiles"] is None:
                filter_i += 1
                continue

            try:
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(graph["smiles"]))
            except:
                filter_i += 1
                continue

            write_txn.put(smiles.encode(), pkl.dumps(graph))
            keys.append(smiles)
            mol_sizes.append(graph["num_nodes"])
            i += 1

        print(f"Processed {i} graphs, {filter_i} filtered")

        write_txn.commit()
        write_txn = write_env.begin(write=True)

        #remvoe the folder created by os.system
        os.system(f"rm -rf {unzip_dir_name}")

    metadata = {}
    metadata['keys'] = keys
    metadata['size'] = mol_sizes
    write_txn.put("metadata".encode(), obj2bstr(metadata))

    write_txn.commit()
    write_env.close()

if __name__ == '__main__':
    process_files()
