# -*- coding: utf-8 -*-
import lmdb
import json
from tqdm import tqdm
import pickle as pkl
import numpy as np
from rdkit import Chem
import functools
import pickle
import transformers
import torch
import csv
import re

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/home/peiran/FMproj/llama2/llama-2-7b/",
        model_max_length=1000000,
        padding_side="right",
        use_fast=False,
    )
special_tokens_dict = {}
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
special_tokens_dict["additional_special_tokens"] = ["<mol>", "</mol>", "<num>", "</num>"]
tokenizer.add_special_tokens(special_tokens_dict)

Prompt_list = [
    "Here is the conformation of a pocket and ligand <<|mol0|>>, its free energy is <<|num0|>>.",
    "The conformation of pocket and ligand <<|mol0|>> has an free energy of <<|num0|>>.",
    "For the pocket and ligand conformation <<|mol0|>>, the free energy is <<|num0|>>.",
    "In the conformation of pocket and ligand <<|mol0|>>, the free energy measures <<|num0|>>.",
    "The pocket and ligand conformation <<|mol0|>> features an free energy of <<|num0|>>.",
    "The free energy of the pocket and ligand conformation, <<|mol0|>>, is <<|num0|>>.",
    "The <<|mol0|>> conformation of pocket and ligand has an free energy of <<|num0|>>.",
    "The conformation for pocket and ligand <<|mol0|>> possesses an free energy of <<|num0|>>.",
    "The pocket and ligand conformation, represented by <<|mol0|>>, displays an free energy of <<|num0|>>.",
]


def replace_num(data):
    nums = []
    cur_num_ptr = 0
    while True:
        num_key = f"<<|num{cur_num_ptr}|>>"
        if num_key in data["entity"]:
            num = data["entity"][num_key]
            nums.append(num)
        else:
            break
        cur_num_ptr += 1
    if cur_num_ptr == 0:
        return data
    for i, num in enumerate(nums):
        data["text"] = data["text"].replace(f"<<|num{i}|>>", f"<num> {num} </num>")
    return data

def read3dconf(path, id2energy, add_mol_token=True):
    print("read 3d conf")
    read_env = lmdb.open(path, map_size=1024 ** 4)

    write_file = "/mnt/chemical-copilot-special-token/pdbbind/train"
    write_env = lmdb.open(write_file, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)

    with read_env.begin() as txn:
        cursor = txn.cursor()
        i = 0
        for key, value in cursor:
            ori_data = pickle.loads(value)
            pdbid = ori_data.pdbid

            if pdbid not in id2energy:
                continue

            freeenergy = id2energy[pdbid]

            prommp_text = Prompt_list[i%len(Prompt_list)]
            text = f"{prommp_text}{tokenizer.eos_token}"
            if add_mol_token:
                text = re.sub("<<\|mol[0-9]+\|>>", "<mol> <unk> </mol>", text)
            else:
                text = re.sub("<<\|mol[0-9]+\|>>", "<unk>", text)

            tokenized = tokenizer(text, return_tensors="pt")
            input_ids = tokenized.input_ids[0]
            input_ids[input_ids == 0] = -1

            mol_sizes = [ori_data.x.shape[0]]

            input_ids_len = 0
            write_txn.put(f"{i}".encode(), pkl.dumps((input_ids, input_ids_len, [ori_data], mol_sizes, [freeenergy])))
            i += 1

    write_txn.commit()
    print(f"Finish processing {write_file}")

    read_env.close()
    write_env.close()

def getfreeenergy():
    id2energy = {}
    with open('/mnt/data/pdbbind/pdbbind_engergy.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        # Read the header (first row) and store it in a variable
        header = next(csv_reader)
        print("Header:", header)
        for row in csv_reader:
            id2energy[row[9]] = float(row[11])

    return id2energy

if __name__ == "__main__":
    id2energy = getfreeenergy()
    read3dconf("/mnt/data/pdbbind/pdbbind_lmdb_remove_512/train", id2energy)
    # smile2pos = {}
    # add3d2prompt("/mnt/chemical-copilot-special-token/mol-instruction-mol-desc/all", smile2pos)
