# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import tqdm

import mmap
import struct
import lmdb
import os
import multiprocessing as mp

from sfm.data.sci_data.NlmTokenizer import NlmTokenizer
from sfm.logging import logger
from sfm.data.prot_data.util import obj2bstr


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = NlmTokenizer.from_pretrained(tokenizer_path)


def tokenize(line):
    global tokenizer
    try:
        line = line.replace("\\n", "\n")
        tokens = tokenizer.tokenize(line)
        tokens = (
            [tokenizer.bos_token_id]
            + tokenizer.convert_tokens_to_ids(tokens)
            + [tokenizer.eos_token_id]
        )
        return tokens
    except:
        # some lines have weird tags that can't be tokenized
        return []


def read_lines_mmap(path):
    with open(path, "r") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b""):
                yield line.decode("utf8").strip()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", type=bool, default=True)
    parser.add_argument("--pad_idx", type=int, default=32000)
    parser.add_argument("--dtype", type=str, default="H")
    args = parser.parse_args()

    b_dtype = args.dtype
    if args.dtype == "I":
        np_dtype = np.uint32
    elif args.dtype == "H":
        np_dtype = np.uint16
    else:
        raise ValueError(f"dtype {args.dtype} error")

    files = []
    # ensure all files exist and not empty
    for file in args.input.split(","):
        file = file.strip()
        if file:
            assert os.path.isfile(file), "file {} not exist".format(file.strip())
            assert os.path.getsize(file) > 0, "file {} is empty".format(file.strip())
            files.append(file)

    n_lines = 0
    n_tokens = 0
    n_unks = 0
    for file in files:
        file_name = os.path.basename(file.strip())
        save_path = os.path.join(args.output, file_name + ".lmdb")
        if not args.overwrite:
            if os.path.exists(save_path):
                logger.info(
                    f"overwrite is set to false, {save_path} already exists, skip"
                )
                continue
        packed_data = b""
        with mp.Pool(
            args.num_workers,
            initializer=init_tokenizer,
            initargs=(args.tokenizer_path,),
        ) as pool:
            line_iter = read_lines_mmap(file)
            n_new_lines = 0
            n_new_tokens = 0
            n_new_unks = 0

            for tokens in tqdm.tqdm(
                pool.imap(tokenize, line_iter, chunksize=1000),
                desc=file_name,
                miniters=100000,
            ):
                if len(tokens) > 0:
                    packed_data += struct.pack("<" + b_dtype * len(tokens), *tokens)
                    n_new_lines += 1
                    n_new_tokens += len(tokens)
                    n_new_unks += tokens.count(0)  # unk token id is 0
            unk_rate = n_new_unks / n_new_tokens

            logger.info(
                "processed {:10,} lines, {:13,} tokens, {:,} unks ({:.4g}), from {}".format(
                    n_new_lines, n_new_tokens, n_new_unks, unk_rate, file.strip()
                )
            )
            data = np.frombuffer(packed_data, dtype="<" + b_dtype).astype(np_dtype)
            token_count = data.shape[0]
            logger.info("number of tokens: {}".format(token_count))
            logger.info("first token loaded: {}, type {}".format(data[:10], data.dtype))
            if token_count % args.seq_len != 0:
                # pad to multiple of seq_len
                data = np.append(
                    data,
                    np.ones(args.seq_len - token_count % args.seq_len, dtype=np_dtype)
                    * args.pad_idx,
                )
            data = data.reshape(-1, args.seq_len)
            logger.info("total tokens: {:,}".format(data.shape[0] * args.seq_len))
            logger.info("shape {}".format(data.shape))
            logger.info("shuffle the data")
            np.random.seed(42)
            np.random.shuffle(data)

            file_size = int(data.nbytes / (1024 * 1024 * 1024))
            env = lmdb.open(
                str(save_path),
                subdir=True,
                readonly=False,
                lock=False,
                readahead=False,
                map_size=(file_size + 1) * 1024**3,
            )
            keys = []
            # data = np.load(file, mmap_mode="r")
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

            n_lines += n_new_lines
            n_tokens += n_new_tokens
            n_unks += n_new_unks
            logger.info(
                "saved to {}, dtype :{}, size :{}, processed_seq_len :{}".format(
                    save_path, str(data.dtype), len(keys), args.seq_len
                )
            )

    logger.info(
        "total {:,} lines, {:,} tokens, {:,} unks".format(n_lines, n_tokens, n_unks)
    )


if __name__ == "__main__":
    main()
