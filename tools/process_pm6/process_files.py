# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from glob import glob
import numpy as np
import os
from path_to_mol_graph import path_to_mol_graph
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

    args = arg_parser.parse_args()

    os.system(f"mkdir {args.output_dir}")

    write_env = lmdb.open(f"{args.output_dir}", map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    # xz_names = np.sort(glob(f"{args.data_path}/Compound_*.tar.xz"))[args.start_index: args.end_index]
    xz_names = os.listdir(args.data_path)
    print(xz_names)

    for xz_name in xz_names:
        xz_name = os.path.join(args.data_path, xz_name)
        print(xz_name)
    # for xz_name in xz_names:
        os.system(f"tar -xf {xz_name} -C {args.output_dir}")

        # unzip_dir_names = [xz_name.split("/")[-1].split(".")[0] for xz_name in xz_names]
        unzip_dir_name = xz_name.split("/")[-1].split(".")[0]

        # dir_names = []
        # for unzip_dir_name in unzip_dir_names:
        #     dir_names.extend(glob(f"{unzip_dir_name}/*"))

        # pool = Pool(args.num_processes)
        # graphs = pool.map(path_to_mol_graph, dir_names)
        # ids = [dir_name.split("/")[-1] for dir_name in dir_names]

        graphs = path_to_mol_graph(unzip_dir_name)

        i = 0
        for graph in tqdm(graphs):
            if "smiles" not in graph:
                continue

            if random.random() > 0.1:
                continue

            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(graph["smiles"]))
            write_txn.put(smiles.encode(), pkl.dumps(graph))
            i += 1

        write_txn.commit()
        write_txn = write_env.begin(write=True)

    metadata = {}
    metadata['keys'] = keys
    metadata['size'] = mol_sizes
    write_txn.put("metadata".encode(), obj2bstr(metadata))

    write_txn.commit()
    write_env.close()

if __name__ == '__main__':
    process_files()
