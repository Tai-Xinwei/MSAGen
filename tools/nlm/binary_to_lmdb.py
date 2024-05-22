# -*- coding: utf-8 -*-
import os
import concurrent.futures
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import lmdb
import numpy as np
import tqdm

from sfm.data.prot_data.util import obj2bstr
from sfm.logging import logger


def process_file(args, file):
    file_name_with_extension = os.path.basename(file)
    file_name, _ = os.path.splitext(file_name_with_extension)
    save_path = os.path.join(args.output, file_name + ".lmdb")
    file_size = int(os.path.getsize(file) / (1024 * 1024 * 1024))
    env = lmdb.open(
        str(save_path),
        subdir=True,
        readonly=False,
        lock=False,
        readahead=False,
        map_size=(file_size + 1) * 1024**3,
    )
    keys = []
    data = np.load(file, mmap_mode="r")
    for i in range(data.shape[0]):
        with env.begin(write=True) as txn:
            txn.put(str(i).encode(), data[i])
            keys.append(i)
    metadata = {
        "dtype": str(data.dtype),
        "keys": keys,
        "size": len(keys),
        "processed_seq_len": args.seq_len,
    }
    with env.begin(write=True) as txn:
        txn.put("metadata".encode(), obj2bstr(metadata))


def main():
    parser = ArgumentParser()
    parser.add_argument("input_files", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--pad_idx", type=int, default=32000)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    files = [e.strip() for e in args.input_files.split(",")]
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(partial(process_file, args), files)


if __name__ == "__main__":
    main()
