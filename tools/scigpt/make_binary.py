# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import tqdm

import os
import multiprocessing as mp

from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger

import struct

import random

def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = SFMDecTokenizer.from_pretrained(
        tokenizer_path,
        prot_spm_path='/blob/shufxi/data/scigpt/ur50bpe/bpe',
        dna_spm_path='/blob/shufxi/data/scigpt/dnabpe/bpe',
        rna_spm_path='/blob/shufxi/data/scigpt/rnabpe/bpe'
    )

def tokenize(line):
    global tokenizer
    try:
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

    # lines = []
    # for file in files:
    #     with open(file.strip(), 'r') as f:
    #         lines.extend(f.readlines())
    #         logger.info("Total {:,} lines after reading {}".format(len(lines), file.strip()))

    # logger.info("total lines: {:,}".format(len(lines)))

    n_lines = 0
    n_tokens = 0
    n_unks = 0

    with open(args.output+'.bin', 'wb') as fbin:
        with mp.Pool(args.num_workers, initializer=init_tokenizer, initargs=(args.tokenizer_path,)) as pool:
            for file in files:
                with open(file.strip(), 'r') as f:
                    lines = f.readlines()
                    n_new_lines = len(lines)
                    n_new_tokens = 0
                    n_new_unks = 0

                    file_name = os.path.basename(file.strip())

                    for tokens in tqdm.tqdm(pool.imap(tokenize, lines), total=len(lines), desc=file_name):
                        if len(tokens) > 0:
                            fbin.write(struct.pack('<'+'H'*len(tokens), *tokens))
                            n_new_tokens += len(tokens)
                            n_new_unks += tokens.count(0) # unk token id is 0
                    unk_rate = n_new_unks / n_new_tokens
                    logger.info("processed {:10,} lines, {:13,} tokens, {:,} unks ({:.4g}), from {}".format(n_new_lines, n_new_tokens, n_new_unks, unk_rate, file.strip()))
                    n_lines += n_new_lines
                    n_tokens += n_new_tokens
                    n_unks += n_new_unks
    logger.info("total {:,} lines, {:,} tokens, {:,} unks".format(n_lines, n_tokens, n_unks))
    logger.info("saved to {}".format(args.output+'.bin'))

    del lines

    with open(args.output+'.bin', 'rb') as fbin:
        data = np.frombuffer(fbin.read(), dtype='<H').astype(np.uint16)
        logger.info("first token loaded: {}, type {}".format(data[:10], data.dtype))
        token_count = data.shape[0]
        if token_count % args.seq_len != 0:
            # pad to multiple of seq_len
            data = np.append(data, np.ones(args.seq_len - token_count % args.seq_len, dtype=np.uint16)*32000)
        data = data.reshape(-1, args.seq_len)
        logger.info("total tokens: {:,}".format(data.shape[0] * args.seq_len))
        logger.info("shape {}".format(data.shape))
        # shuffle the data by rows
        logger.info("shuffle the data")
        np.random.seed(42)
        np.random.shuffle(data)
        logger.info("save to {}".format(args.output))
        np.save(args.output, data)
    # delete binary file
    os.remove(args.output+'.bin')



if __name__ == "__main__":
    main()
