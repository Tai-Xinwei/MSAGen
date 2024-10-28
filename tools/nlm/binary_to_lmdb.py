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
    print("begin: " + file)
    file_name_with_extension = os.path.basename(file)
    file_name, _ = os.path.splitext(file_name_with_extension)
    save_path = os.path.join(
        args.output, file.split("/")[-2] + "_" + file_name + ".lmdb"
    )
    file_size = int(os.path.getsize(file) / (1024 * 1024 * 1024))
    print(file_size)
    env = lmdb.open(
        str(save_path),
        subdir=True,
        readonly=False,
        lock=False,
        readahead=False,
        map_size=(file_size + 100) * 1024**3,
    )
    keys = []
    with open(file, "rb") as fbin:
        data = np.frombuffer(fbin.read(), dtype="<H").astype(np.uint16)
    token_count = data.shape[0]
    logger.info("number of tokens: {}".format(token_count))
    logger.info("first token loaded: {}, type {}".format(data[:10], data.dtype))
    if token_count % args.seq_len != 0:
        # pad to multiple of seq_len
        data = np.append(
            data,
            np.ones(args.seq_len - token_count % args.seq_len, dtype=np.uint16)
            * args.pad_idx,
        )
    data = data.reshape(-1, args.seq_len)
    print(data.shape)
    # data = np.load(file, mmap_mode="r")
    with env.begin(write=True) as txn:
        for i in range(data.shape[0]):
            txn.put(str(i).encode(), data[i])
            keys.append(i)
        metadata = {
            "dtype": str(data.dtype),
            "keys": keys,
            "size": len(keys),
            "processed_seq_len": args.seq_len,
        }
        txn.put("metadata".encode(), obj2bstr(metadata))
    print("done: " + file)


def main():
    parser = ArgumentParser()
    parser.add_argument("input_files", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--pad_idx", type=int, default=32000)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    files = [e.strip() for e in args.input_files.split(",")]
    for file in files:
        process_file(args, file)
    # with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    # executor.map(partial(process_file, args), files)


if __name__ == "__main__":
    main()
