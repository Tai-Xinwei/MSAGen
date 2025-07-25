# -*- coding: utf-8 -*-
import multiprocessing
from multiprocessing import Pool
import tqdm
import argparse
import struct
import random
import numpy as np
import os, sys
import lmdb
import pickle as pkl
from multiprocessing import Pool

from .commons import bstr2obj, obj2bstr

vocab = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<mask>': 31}


def main():
    write_file = '/mnt/protein/48organism1m.lmdb/'
    write_env = lmdb.open(write_file, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    lmdb_path = '/mnt/protein/48organism.lmdb/'

    env = lmdb.open(
        str(lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
    )
    txn = env.begin(write=False)

    metadata = bstr2obj(txn.get("metadata".encode()))
    lengths, keys = metadata["sizes"], metadata["names"]
    comment = metadata["comment"]

    buffer_len = 0
    sequence_length = 1024
    buffer = []
    last_tokens = None
    metadata = {}
    names = []
    sizes = []

    idx = 0

    for i in tqdm.tqdm(keys):
        value = txn.get(i.encode())
        write_txn.put(f"{i}".encode(), value)


    metadata['keys'] = keys
    metadata['sizes'] = lengths
    metadata["comment"] = comment

    write_txn.put("__metadata__".encode(), obj2bstr(metadata))
    write_txn.commit()
    print(f"Finish processing {write_file}")

    env.close()
    write_env.close()


if __name__ == "__main__":
    main()
