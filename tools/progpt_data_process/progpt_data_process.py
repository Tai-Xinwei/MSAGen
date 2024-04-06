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
import pickle

from copy import deepcopy
from joblib import Parallel, delayed
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS
from transformers import AutoTokenizer
from sfm.data.prot_data.util import bstr2obj, obj2bstr

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def tokenize():
    tokenizer = AutoTokenizer.from_pretrained(
        "/fastdata/peiran/llama-2-7b",
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    special_tokens_dict["additional_special_tokens"] = SCIENCE_TAG_TOKENS
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer

vocab = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<mask>': 31}
# revert vocab to get id to word
idx2word = {v: k for k, v in vocab.items()}

def protein_process(protein):
    protein_id = [vocab[tok] for tok in protein]

    return protein_id

def process(text):
    # find the part of protein seq that surrounded by <protein> and </protein> in text
    protein = []
    res = []
    text1 = text.split("<protein>")
    res.append(text1[0])
    for i in range(1, len(text1)):
        text2 = text1[i].split("</protein>")
        protein.append(text2[0])
        res.append(text2[1])

    return res, protein

def main():
    path = "/fastdata/peiran/nlm/data/"

    tokenizer = tokenize()
    write_file = "/fastdata/peiran/nlm/progpt_train.lmdb"
    write_file2 = "/fastdata/peiran/nlm/progpt_valid.lmdb"

    write_env = lmdb.open(write_file, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    write_env2 = lmdb.open(write_file2, map_size=1024 ** 4)
    write_txn2 = write_env2.begin(write=True)

    idx = 0
    keys_train = []
    keys_valid = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as file:
            vocab_lines = file.readlines()

        flag = 0
        for line in vocab_lines:

            text, protein = process(line)
            protein_id_list = []
            for p in protein:
                try:
                    if len(p) == 0:
                        print(f"Empty protein sequence in {line}")
                        flag = 1
                        break
                    protein_id = protein_process(p)
                    protein_id_list.append(protein_id)
                except:
                    print(f"Error in processing {p}")
                    flag = 1
                    break

            if flag == 1:
                flag = 0
                continue

            input_ids = []
            for i in range(len(text)):
                if i == 0:
                    input_ids.extend(tokenizer.encode(text[i] + " <protein>"))
                elif i != len(text) - 1:
                    input_ids.append(-1)
                    input_ids.extend(tokenizer.encode("</protein> " + text[i] + " <protein>")[1:])
                else:
                    input_ids.append(-1)
                    input_ids.extend(tokenizer.encode("</protein> " + text[i])[1:])

            if random.random() < 0.9:
                write_txn.put(str(idx).encode(), pkl.dumps((input_ids, protein_id_list)))
                keys_train.append(idx)
            else:
                write_txn2.put(str(idx).encode(), pkl.dumps((input_ids, protein_id_list)))
                keys_valid.append(idx)
            idx += 1

    metadata = {}
    metadata['keys'] = keys_train
    metadata['size'] = len(keys_train)
    write_txn.put("metadata".encode(), obj2bstr(metadata))

    write_txn.commit()
    print(f"Finish processing {write_file}")

    metadata['keys'] = keys_valid
    metadata['size'] = len(keys_valid)
    write_txn2.put("metadata".encode(), obj2bstr(metadata))
    write_txn2.commit()
    print(f"Finish processing {write_file2}")

    write_env.close()
    write_env2.close()


if __name__ == "__main__":
    main()
