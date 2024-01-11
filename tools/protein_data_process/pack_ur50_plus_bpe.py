# -*- coding: utf-8 -*-
import multiprocessing
import sentencepiece as spm
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import struct
import random
import numpy as np
import os, sys
import lmdb
import pickle as pkl
from multiprocessing import Pool
import functools

from .commons import bstr2obj, obj2bstr

vocab = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<mask>': 31}

def tokenize(line):
    if getattr(tokenize, "sp", None) is None:
        setattr(tokenize, "sp", spm.SentencePieceProcessor(model_file="/home/peiran/FMproj/data/bfm/ur50bpe.model"))

    sp = getattr(tokenize, "sp")
    tokens = sp.encode(line, out_type=int)
    return tokens # [bos] + tokens + [eos]

def data_process(data):

    tokens = [0] + [vocab[tok] for tok in data] + [2]
    bpe_token = tokenize(data)

    return tokens, bpe_token


def main():

    with open("/home/peiran/FMproj/data/bfm/ur50bpe.vocab") as file:
        vocab_lines = file.readlines()
        vocab_lines = [word.strip() for word in vocab_lines]

    bpe_vocab = {}
    bpe_idx2word = {}
    for idx, word in enumerate(vocab_lines):
        bpe_idx2word[idx] = word
        bpe_vocab[word] = idx

    write_file = '/home/peiran/protein/uniref50_bpepack1024_train_2.lmdb'
    write_env = lmdb.open(write_file, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    lmdb_path = '/mnt/protein/uniref50_train.lmdb'

    env = lmdb.open(
        str(lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
    )
    txn = env.begin(write=False)

    metadata = bstr2obj(txn.get("metadata".encode()))
    lengths, keys = metadata["lengths"], metadata["prot_accessions"]

    # raw_data = []
    # for key in tqdm(keys):
    #     value = txn.get(key.encode())
    #     data = list(bstr2obj(value))
    #     if len(data) > 1022:
    #         continue
    #     else:
    #         raw_data.append(data)

    # pkl.dump(raw_data, open("/home/peiran/FMproj/data/bfm/ur50_raw_data.pkl", "wb"))

    env.close()

    raw_data = pkl.load(open("/home/peiran/FMproj/data/bfm/ur50_raw_data.pkl", "rb"))
    print("Finish loading raw data")

    buffer_len = 0
    sequence_length = 1024
    buffer = []
    bpe_buffer = []
    last_tokens = None
    metadata = {}
    names = []
    sizes = []
    token_list = []
    bpe_token_list = []

    idx = 0
    with Pool(processes=12) as pool:
        length = len(raw_data)
        iter = pool.imap(data_process, raw_data)
        for idx, results in tqdm(enumerate(iter), total=length):
            tokens, bpe_token = results
            if len(tokens) > 1024:
                continue

            token_list.append(tokens)
            bpe_token_list.append(bpe_token)

    del raw_data

    for tokens, bpe_token in tqdm(zip(token_list, bpe_token_list), total=len(token_list)):
        # pack 1024
        buffer.extend(tokens)

        bpe_token_long = []
        for token in bpe_token:
            token_len = len(bpe_idx2word[token[0]])
            bpe_token_long.extend([token[0]] * token_len)

        bpe_token_long = [0] + bpe_token_long + [2]

        bpe_buffer.extend(bpe_token_long)

        buffer_len += len(tokens)
        if buffer_len >= sequence_length:

            if buffer_len == sequence_length:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer
                dict_buffer['bpe_seq'] = bpe_buffer
                write_txn.put(f"{idx}".encode(), pkl.dumps(dict_buffer))
                buffer_len = 0
                buffer = []
                bpe_buffer = []

                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
            else:
                dict_buffer = {}
                dict_buffer['aa_seq'] = buffer[:sequence_length]
                dict_buffer['bpe_seq'] = bpe_buffer[:sequence_length]
                write_txn.put(f"{idx}".encode(), pkl.dumps(dict_buffer))
                names.append(idx)
                sizes.append(sequence_length)
                idx += 1
                buffer_len = len(tokens)
                buffer = tokens
                bpe_buffer = bpe_token_long

    metadata['prot_accessions'] = names
    metadata['lengths'] = sizes

    write_txn.put("metadata".encode(), obj2bstr(metadata))
    write_txn.commit()
    print(f"Finish processing {write_file}")

    write_env.close()


if __name__ == "__main__":
    main()
