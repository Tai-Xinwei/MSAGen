# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import numpy as np
import tqdm
import re
import csv
import mmap
import struct
import lmdb
import os
import multiprocessing as mp
from transformers import AutoTokenizer
import pickle as pkl

from sfm.data.sci_data.NlmTokenizer import NlmLlama3Tokenizer
from sfm.logging import logger
from sfm.data.prot_data.util import obj2bstr
from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS, SCIENCE_TOKENS



csv.field_size_limit(10000000)

IGNORE_INDEX = -100


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/home/v-zekunguo/nlm/llama/Meta-Llama-3-8B",
    )
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--overwrite", type=bool, default=True)
    parser.add_argument("--pad_idx", type=int, default=128256)
    parser.add_argument("--is_hf", type=bool, default=False)

    args = parser.parse_args()
    result = []
    # tokenizer = NlmLlama3Tokenizer.from_pretrained(args.tokenizer_path,is_dna_six=True)
    tokenizer = NlmLlama3Tokenizer.from_pretrained(args.tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    response_result = []
    hf_dic_keys={}
    with open(args.input, "r") as f:
        reader = csv.reader(f,delimiter='\t')
        # reader = csv.reader(f)
        i=-1
        for row in reader:
            i+=1
            prompt = f"Instruction: {row[0].strip()}\n\n\nResponse: "
            prompt=prompt.replace('\\n','\n')
            # prompt = row[0].strip()
            result.append(prompt)
            response_result.append(row[1].strip().replace('\\n','\n'))
            if args.is_hf:
                if row[1].strip() in hf_dic_keys:
                    hf_dic_keys[row[1].strip()].append(i)
                else:
                    hf_dic_keys[row[1].strip()]=[i]
    print(len(response_result))
    env = lmdb.open(
        str(args.output),
        subdir=True,
        readonly=False,
        lock=False,
        readahead=False,
        map_size=(200 + 1) * 1024**3,
    )
    keys = []
    with env.begin(write=True) as txn:
        for i in range(len(result)):
            inputs_id = tokenizer.encode(result[i],prepend_bos=False)
            # inputs_id = tokenizer.encode(result[i],add_special_tokens=False)
            # res_out = tokenizer.encode(response_result[i],add_special_tokens=False)

            res_out = tokenizer.encode(response_result[i],prepend_bos=False)
            origin_labels = (
                (len(inputs_id) + 1) * [IGNORE_INDEX]
                + res_out
                + [tokenizer.eos_token_id]
            )
            inputs_id = (
                [tokenizer.bos_token_id]
                + inputs_id
                + res_out
                + [tokenizer.eos_token_id]
            )
            if len(inputs_id) > args.seq_len:
                continue
            data = pkl.dumps((inputs_id, origin_labels))

            txn.put(str(i).encode(), data)
            keys.append(i)
    if args.is_hf:
        dic_keys=[]
        for value in hf_dic_keys.values():
            dic_keys.append(value)
        metadata = {
            "keys": keys,
            "size": len(keys),
            "dict_keys":dic_keys,
        }
        # print(dic_keys.keys())
    else:
        metadata = {
            "keys": keys,
            "size": len(keys),
        }
    with env.begin(write=True) as txn:
        txn.put("metadata".encode(), obj2bstr(metadata))

    print(inputs_id, origin_labels)


if __name__ == "__main__":
    main()
