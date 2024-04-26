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
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
mount_dir = "/home/v-wukehan/blob1.v2"

def tokenize():
    llm_model_name_or_path = mount_dir+"/v-kehanwu/SFM/scigpt/stageB.prot/global_step224655"
    tokenizer_path = mount_dir+"/shufxi/data/scigpt"
    tokenizer = SFMDecTokenizer.from_pretrained(
        llm_model_name_or_path,
        prot_spm_path=os.path.join(tokenizer_path, "ur50bpe/bpe"),
        dna_spm_path=os.path.join(tokenizer_path, "dnabpe/bpe"),
        rna_spm_path=os.path.join(tokenizer_path, "rnabpe/bpe"),
    )
    return tokenizer

vocab = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<mask>': 31}
# revert vocab to get id to word
idx2word = {v: k for k, v in vocab.items()}

scigpt_vacab = {'L': 33874, 'A': 33875, 'G': 33878, 'V': 33877, 'S': 33876, 'E': 33879, 'R': 33880, 'T': 33881, 'I': 33882, 'D': 33884, 'P': 33886, 'K': 33883, 'Q': 33885, 'N': 33887, 'F': 33888, 'Y': 33890, 'M': 33873, 'H': 33889, 'W': 33891, 'C': 33892, 'X': 34276, 'B': 37965, 'U': 37967, 'Z': 37966, 'O': 0}

def protein_process(protein):
    protein_id = [vocab[tok] for tok in protein]
    protein_bpe_id = [scigpt_vacab[tok] for tok in protein]

    return protein_id, protein_bpe_id

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

def process_line(line, tokenizer):
    text, protein = process(line)
    protein_id_list = []
    protein_bpe_id_list = []
    for p in protein:
        try:
            if len(p) == 0:
                print(f"Empty protein sequence in {line}")
                return None
            protein_id, protein_bpe_id = protein_process(p)
            protein_id_list.append(protein_id)
            protein_bpe_id_list.append(protein_bpe_id)
        except:
            print(f"Error in processing {p}")
            return None

    input_ids = []
    for i in range(len(text)):
        if i == 0:
            input_ids.extend(tokenizer.encode(text[i] + " <protein>"))
        elif i != len(text) - 1:
            input_ids.append(-1)
            # input_ids.extend(protein_bpe_id_list[i-1])
            input_ids.extend(tokenizer.encode("</protein> " + text[i] + " <protein>")[1:])
        else:
            input_ids.append(-1)
            # input_ids.extend(protein_bpe_id_list[i-1])
            input_ids.extend(tokenizer.encode("</protein> " + text[i])[1:])
    return input_ids, protein_id_list, protein_bpe_id_list

def main():
    data_path = "/home/v-wukehan/hai1/kaiyuan/scigpt/molinst/data/scigpt_molinst_pro_v3"
    save_path = "/home/v-wukehan/blob1.v2/v-kehanwu/data/progpt_data"
    # save_path = "tmp/"
    tokenizer = tokenize()
    for split in ["train", "test"]:
        with open(os.path.join(data_path, f"{split}.tsv"), "r") as f:
            lines = f.readlines()
        write_file = os.path.join(save_path, f"progpt_instructions_{split}_bpe.lmdb")
        write_env = lmdb.open(write_file, map_size=1024 ** 4)
        write_txn = write_env.begin(write=True)

        for idx, line in enumerate(tqdm(lines)):
            prompt, target = line.strip().split("\t")
            # line = f"Instruction: {instruction}\n\n\n Response: {response}"
            prompt = f"Instruction: {prompt}\n\n\n Response:"

            processed_prompt = process_line(prompt, tokenizer)
            if processed_prompt is None:
                continue
            prompt_input_ids, prompt_protein_id_list, prompt_protein_bpe_id_list = processed_prompt
            processed_target = process_line(target, tokenizer)
            if processed_target is None:
                continue
            target_input_ids, target_protein_id_list, target_protein_bpe_id_list = processed_target

            input_ids = prompt_input_ids + target_input_ids[1:] # remove bos token
            protein_id_list = prompt_protein_id_list + target_protein_id_list
            protein_bpe_id_list = prompt_protein_bpe_id_list + target_protein_bpe_id_list
            labels = [-100] * len(prompt_input_ids) + target_input_ids[1:]
            assert len(input_ids) == len(labels)
            # print("prompt_input_ids", prompt_input_ids)
            # print("target_input_ids", target_input_ids)
            # print("input_ids", input_ids)
            # print("labels", labels)
            write_txn.put(str(idx).encode(), pkl.dumps((input_ids, protein_id_list, protein_bpe_id_list, labels)))

        metadata = {}
        metadata['keys'] = list(range(len(lines)))
        metadata['size'] = len(lines)
        write_txn.put("metadata".encode(), obj2bstr(metadata))
        write_txn.commit()
        print(f"Finish processing {write_file}")
        write_env.close()

if __name__ == "__main__":
    main()
