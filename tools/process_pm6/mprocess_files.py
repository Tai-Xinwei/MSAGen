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
import multiprocessing
from multiprocessing import Pool
from functools import partial

def process_file(xz_file, data_path, output_dir):
    xz_file = os.path.join(data_path, xz_file)
    os.system(f"tar -xf {xz_file} -C {output_dir}")

    filter_count = 0
    molecule_keys = []
    molecule_sizes = []
    molecule_graphs = []

    unzipped_dir_name = xz_file.split("/")[-1].split(".")[0]
    unzipped_dir_name = os.path.join(output_dir, unzipped_dir_name)
    for molecule_file in os.listdir(unzipped_dir_name):
        if random.random() > 0.12:
            continue

        molecule_path = os.path.join(unzipped_dir_name, molecule_file)

        graph = path_to_mol_graph(molecule_path)

        if "smiles" not in graph or graph["smiles"] is None:
            filter_count += 1
            continue

        try:
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(graph["smiles"]))
        except:
            filter_count += 1
            continue

        if smiles in molecule_keys:
            filter_count += 1
            continue

        cononical_smile = Chem.MolToSmiles(Chem.MolFromSmiles(graph["smiles"]), isomericSmiles=False)
        if cononical_smile != graph["mol_smile"]:
            print(f"smiles is {cononical_smile}, mol_smile is {graph['mol_smile']}")
            filter_count += 1
            continue

        # write_txn.put(smiles.encode(), pkl.dumps(graph))
        # molecule_keys.add(smiles)
        # molecule_sizes.append(graph["num_nodes"])
        # count += 1

        molecule_keys.append(smiles)
        molecule_sizes.append(graph["num_nodes"])
        molecule_graphs.append(graph)


    # print(f"Processed {count} graphs, {filter_count} filtered")
    os.system(f"rm -rf {unzipped_dir_name}")
    # write_txn.commit()
    # write_txn = write_env.begin(write=True)

    return molecule_keys, molecule_sizes, filter_count, molecule_graphs


def process_files():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default="/home/peiran/data/pm6_unzip/Compounds")
    # arg_parser.add_argument('start_index', type=int, default=1)
    # arg_parser.add_argument('end_index', type=int)
    # arg_parser.add_argument('num_processes', type=int)
    arg_parser.add_argument('--output_dir', type=str, default="/home/peiran/data/pm6_unzip/output")
    arg_parser.add_argument('--lmdb_dir', type=str, default="/home/peiran/data/pm6_unzip/pm6_10M_refined4.lmdb")

    args = arg_parser.parse_args()

    os.system(f"mkdir {args.output_dir}")

    write_env = lmdb.open(f"{args.lmdb_dir}", map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    # xz_names = np.sort(glob(f"{args.data_path}/Compound_*.tar.xz"))[args.start_index: args.end_index]
    xz_names = os.listdir(args.data_path)
    # print(xz_names)

    mol_keys = set()
    mol_sizes = []
    i = 0
    filter_i = 0

    pool = Pool(24)
    partial_process_file = partial(process_file, data_path=args.data_path, output_dir=args.output_dir)
    output_tuples = pool.map(partial_process_file, xz_names)

    for index, output_tuple in tqdm(enumerate(output_tuples)):
        keys, sizes, filter_count, graphs = output_tuple

        for key, size, graph in zip(keys, sizes, graphs):
            if key in mol_keys:
                filter_i += 1
                continue

            write_txn.put(key.encode(), pkl.dumps(graph))
            mol_keys.add(key)
            mol_sizes.append(size)
            i += 1

            if i % 10000 == 0:
                write_txn.commit()
                write_txn = write_env.begin(write=True)

        print(f"Processed {i} graphs, {filter_i} filtered")
        write_txn.commit()
        write_txn = write_env.begin(write=True)


    mol_keys = list(mol_keys)
    metadata = {}
    metadata['keys'] = mol_keys
    metadata['size'] = mol_sizes
    write_txn.put("metadata".encode(), obj2bstr(metadata))

    write_txn.commit()
    write_env.close()

if __name__ == '__main__':
    process_files()
