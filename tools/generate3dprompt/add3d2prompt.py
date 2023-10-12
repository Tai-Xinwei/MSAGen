# -*- coding: utf-8 -*-
import lmdb
import json
from tqdm import tqdm
import pickle as pkl
import re
import transformers
import torch
import os
import numpy as np
from rdkit import Chem
import functools
from multiprocessing import Pool
import multiprocessing
import pickle

def read3dconf(path):
    pos_file = os.path.join(path, "S0")
    smile_file = os.path.join(path, "S0-smiles")
    smile2pos = {}
    read_env = lmdb.open(pos_file, map_size=1024 ** 4)
    smile_env = lmdb.open(smile_file, map_size=1024 ** 4)

    with read_env.begin() as txn, smile_env.begin() as txnsmile:
        cursor = txn.cursor()
        for key, value in cursor:
            ori_data = pickle.loads(value)
            smile = pickle.loads(txnsmile.get(key))
            print(key)
            print(smile)
            # print(ori_data["atom_coords"])
            exit()

    read_env.close()
    smile_env.close()

    return smile2pos

def add3d2prompt(path, smile2pos):
    print("add 3d to prompt")

    output_path = path + "-3d"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    read_env = lmdb.open(path, map_size=1024 ** 4)

    write_env = lmdb.open(output_path, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    index = 1
    with read_env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            intput_ids, len, smile_list, Ntoken = pickle.loads(value)
            flag = 0
            poses = []
            for smile in smile_list:
                if smile not in smile2pos:
                    flag = 1
                    break
                poses.append(smile2pos[smile])

            if flag == 0:
                print(pickle.loads(value))
                smile3d = [smile_list, poses]
                write_txn.put(f"{key}".encode(), pkl.dumps((intput_ids, len, smile3d, Ntoken)))
                if (index + 1) % 10000 == 0:
                    write_txn.commit()
                    write_txn = write_env.begin(write=True)

                index += 1

    if (index + 1) % 10000 != 0:
        write_txn.commit()

    write_env.close()
    read_env.close()



if __name__ == "__main__":
    # smile2pos = read3dconf("/mnt/data/pm6-86m-3d-filter/merged/")
    smile2pos = {}
    add3d2prompt("/mnt/chemical-copilot-special-token/mol-instruction-mol-desc/all", smile2pos)
