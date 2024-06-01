# -*- coding: utf-8 -*-
import lmdb
import json
from tqdm import tqdm
import pickle as pkl
import re
import transformers
import torch
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
import functools
from multiprocessing import Pool
import multiprocessing
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sfm.utils.science_tokens import SCIENCE_TAG_TOKENS
from sfm.models.progpt.progpt import ProGPTModel
from sfm.models.progpt.progpt_config import ProGPTConfig
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.utils import arg_utils
from sfm.data.prot_data.util import bstr2obj, obj2bstr

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
scigpt_vacab = {
    "L": 33874,
    "A": 33875,
    "G": 33878,
    "V": 33877,
    "S": 33876,
    "E": 33879,
    "R": 33880,
    "T": 33881,
    "I": 33882,
    "D": 33884,
    "P": 33886,
    "K": 33883,
    "Q": 33885,
    "N": 33887,
    "F": 33888,
    "Y": 33890,
    "M": 33873,
    "H": 33889,
    "W": 33891,
    "C": 33892,
    "X": 34276,
    "B": 37965,
    "U": 37967,
    "Z": 37966,
    "O": 0,
}

vocab = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 24,
    "B": 25,
    "U": 26,
    "Z": 27,
    "O": 28,
    ".": 29,
    "-": 30,
    "<mask>": 31,
}


def init_tokenizer(use_llama=False):
    mount_dir = "/home/v-zekunguo/data"
    if not use_llama:
        llm_model_name_or_path = (
            mount_dir + "/v-kehanwu/SFM/scigpt/stageB.prot/global_step224655"
        )
        tokenizer_path = mount_dir + "/shufxi/data/scigpt"
        tokenizer = SFMDecTokenizer.from_pretrained(
            llm_model_name_or_path,
            prot_spm_path=os.path.join(tokenizer_path, "ur50bpe/bpe"),
            dna_spm_path=os.path.join(tokenizer_path, "dnabpe/bpe"),
            rna_spm_path=os.path.join(tokenizer_path, "rnabpe/bpe"),
        )
    else:
        llm_model_name_or_path = (
            mount_dir + "/v-kehanwu/SFM/scigpt/stageB.prot/global_step224655"
        )
        tokenizer_path = mount_dir + "/shufxi/data/scigpt"
        model_max_length = 2048
        tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name_or_path,
            model_max_length=model_max_length,
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


tokenizer = init_tokenizer(use_llama=False)


# def tokenize_sample(in_fname, out_dir_name, is_instruction=False, add_mol_token=False):
#     os.system(f"mkdir -p {out_dir_name}")
#     write_env = lmdb.open(out_dir_name, map_size=1024**4)
#     write_txn = write_env.begin(write=True)
#     with open(in_fname, "r") as in_file:
#         for index, line in tqdm(enumerate(in_file)):
#             datas = json.loads(line.strip())
#             for data in datas:
#                 text = f"{data['text']}{tokenizer.eos_token}"
#                 unique_mol_cnt = 0
#                 remap_mol = {}
#                 smiless = []
#                 mol_sizes = []
#                 for key in data["entity"]:
#                     if key.startswith("<<|mol"):
#                         mol_idx = int(key.split("<<|mol")[-1].split("|>>")[0])
#                         smiles = data["entity"][key]["smiles"]
#                         smiless.append(smiles)
#                         remap_mol[mol_idx] = unique_mol_cnt
#                         unique_mol_cnt += 1
#                 mol_occurs = [int(t.split("|>>")[0]) for t in text.split("<<|mol")[1:]]
#                 remap_mol_idxs = torch.tensor(
#                     [remap_mol[mol_occur] for mol_occur in mol_occurs], dtype=torch.long
#                 )
#                 if add_mol_token:
#                     text = re.sub("<<\|mol[0-9]+\|>>", "<mol> <unk> </mol>", text)
#                 else:
#                     text = re.sub("<<\|mol[0-9]+\|>>", "<unk>", text)
#                 tokenized = tokenizer(text, return_tensors="pt")
#                 input_ids = tokenized.input_ids[0]
#                 assert len(input_ids[input_ids == 0]) == len(remap_mol_idxs)
#                 input_ids[input_ids == 0] = -remap_mol_idxs - 1
#                 if is_instruction:
#                     soruce = text.split("Response:\n")[0] + "Response:\n"
#                     soruce_tokenized = tokenizer(soruce, return_tensors="pt")
#                     soruce_input_ids = soruce_tokenized.input_ids[0]
#                     input_ids_len = len(soruce_input_ids)
#                 else:
#                     input_ids_len = 0
#                 write_txn.put(
#                     f"{index}".encode(),
#                     pkl.dumps((input_ids, input_ids_len, smiless, mol_sizes)),
#                 )
#                 if (index + 1) % 10000 == 0:
#                     write_txn.commit()
#                     write_txn = write_env.begin(write=True)
#     if (index + 1) % 10000 != 0:
#         write_txn.commit()
#     write_env.close()


def process(
    files, lmdb_path, process_index=0, is_instruction=False, add_mol_token=True
):

    write_env = lmdb.open(lmdb_path, map_size=1024**4)
    write_txn = write_env.begin(write=True)
    index = 0
    keys = []
    for file in files:
        print(f"Processing {file}")

        datas = json.load(open(file, "r"))
        # print(len(datas))
        for data in tqdm(datas):
            text = f"{data['text']}{tokenizer.eos_token}"
            for key in data["entity"]:
                if key == "<<|protein|>>":
                    bpe_protein_ids = []
                    protein_ids = []
                    for protein in data["entity"][key]:
                        tmp_bpe_protein_id = []
                        tmp_protein_id = []
                        for aa in protein:
                            tmp_bpe_protein_id.append(scigpt_vacab[aa])
                            tmp_protein_id.append(vocab[aa])
                        bpe_protein_ids.append(tmp_bpe_protein_id)
                        protein_ids.append(tmp_protein_id)
            if add_mol_token:
                text = re.sub("<<\|protein[0-9]+\|>>", "<protein><unk></protein>", text)
            else:
                text = re.sub("<<\|protein[0-9]+\|>>", "<unk>", text)
            # list Keep the same type as before
            tokenized = tokenizer(text)
            input_ids = tokenized.input_ids
            count0 = 0
            for i in range(len(input_ids)):
                if input_ids[i] == 0:
                    count0 += 1
                    input_ids[i] = -1
            assert count0 == len(protein_ids)
            keys.append(index)
            write_txn.put(
                f"{index}".encode(),
                pkl.dumps((input_ids, protein_ids, bpe_protein_ids)),
            )

            if (index + 1) % 10000 == 0:
                write_txn.put(
                    "metadata".encode(), obj2bstr({"keys": keys, "size": len(keys)})
                )
                write_txn.commit()
                write_txn = write_env.begin(write=True)
            index += 1

        if (index + 1) % 10000 != 0:
            write_txn.put(
                "metadata".encode(), obj2bstr({"keys": keys, "size": len(keys)})
            )
            write_txn.commit()
            write_txn = write_env.begin(write=True)
        print(f"Finish processing {file}")

    write_env.close()


# def tokenize_sample_parallel(
#     in_fname, out_dir_name, num_processes, is_instruction=False, add_mol_token=False
# ):
#     with open(in_fname, "r") as in_file:
#         lines = in_file.readlines()

#     chunk_size = (len(lines) + num_processes - 1) // num_processes
#     chunk_starts = [chunk_size * i for i in range(num_processes)]
#     chunk_ends = [
#         min(len(lines), chunk_start + chunk_size) for chunk_start in chunk_starts
#     ]
#     chunks = [
#         lines[chunk_start:chunk_end]
#         for chunk_start, chunk_end in zip(chunk_starts, chunk_ends)
#     ]

#     pool = Pool(num_processes)
#     pool.starmap(
#         process,
#         zip(
#             chunks,
#             range(num_processes),
#             [out_dir_name for _ in range(num_processes)],
#             [is_instruction for _ in range(num_processes)],
#             [add_mol_token for _ in range(num_processes)],
#         ),
#     )


def split_files(files, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for i, file in enumerate(files):
        chunks[i % num_chunks].append(file)
    return chunks


if __name__ == "__main__":

    base_prompt_path = "/home/v-zekunguo/zekun_data/protein/prompt"
    files = []
    for file in os.listdir(base_prompt_path):
        if file.endswith(".json"):
            files.append(os.path.join(base_prompt_path, file))
    length = len(files)
    print(files)

    outpth = "/home/v-zekunguo/zekun_data/protein/prompt_tokenized/"

    num_lmdbs = 2
    lmdb_paths = [f"{outpth}data{i}" for i in range(num_lmdbs)]
    file_chunks = split_files(files, num_lmdbs)

    process_with_token = functools.partial(process, add_mol_token=True)

    with multiprocessing.Pool(num_lmdbs) as pool:
        pool.starmap(process_with_token, zip(file_chunks, lmdb_paths))
