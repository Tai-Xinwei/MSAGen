# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import tqdm

import mmap

import os
import multiprocessing as mp

from sfm.data.sci_data.NlmTokenizer import NlmTokenizer
from sfm.logging import logger

import struct


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = NlmTokenizer.from_pretrained(
        tokenizer_path
    )

def tokenize(line):
    global tokenizer
    try:
        line = line.replace('\\n', '\n')
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
    with open(path, 'r') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for line in iter(mm.readline, b''):
                yield line.decode('utf8').strip()

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    files = []
    # ensure all files exist and not empty
    for file in args.input.split(','):
        file = file.strip()
        if file:
            assert os.path.isfile(file), "file {} not exist".format(file.strip())
            assert os.path.getsize(file) > 0, "file {} is empty".format(file.strip())
            files.append(file)

    n_lines = 0
    n_tokens = 0
    n_unks = 0

    with open(args.output+'.bin', 'wb') as fbin:
        with mp.Pool(args.num_workers, initializer=init_tokenizer, initargs=(args.tokenizer_path,)) as pool:
            for file in files:
                line_iter = read_lines_mmap(file)
                n_new_lines = 0
                n_new_tokens = 0
                n_new_unks = 0

                file_name = os.path.basename(file.strip())

                for tokens in tqdm.tqdm(pool.imap(tokenize, line_iter, chunksize=1000), desc=file_name, miniters=100000):
                    if len(tokens) > 0:
                        fbin.write(struct.pack('<'+'H'*len(tokens), *tokens))
                        n_new_lines += 1
                        n_new_tokens += len(tokens)
                        n_new_unks += tokens.count(0) # unk token id is 0
                unk_rate = n_new_unks / n_new_tokens
                logger.info("processed {:10,} lines, {:13,} tokens, {:,} unks ({:.4g}), from {}".format(n_new_lines, n_new_tokens, n_new_unks, unk_rate, file.strip()))

                n_lines += n_new_lines
                n_tokens += n_new_tokens
                n_unks += n_new_unks


    logger.info("total {:,} lines, {:,} tokens, {:,} unks".format(n_lines, n_tokens, n_unks))
    logger.info("saved to {}".format(args.output+'.bin'))

if __name__ == "__main__":
    main()
