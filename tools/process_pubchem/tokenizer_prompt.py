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

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = RemoveHs(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = torch.tensor(edge_index)
    graph['edge_attr'] = torch.tensor(edge_attr)
    graph['x'] = torch.tensor(x)
    graph['num_nodes'] = len(x)

    return graph

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

def tokenize_sample(in_fname, out_dir_name, is_instruction=False, add_mol_token=False):
    os.system(f"mkdir -p {out_dir_name}")
    write_env = lmdb.open(out_dir_name, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)
    with open(in_fname, "r") as in_file:
        for index, line in tqdm(enumerate(in_file)):
            datas = json.loads(line.strip())
            for data in datas:
                data = replace_num(data)
                text = f"{data['text']}{tokenizer.eos_token}"
                unique_mol_cnt = 0
                remap_mol = {}
                smiless = []
                mol_sizes = []
                for key in data["entity"]:
                    if key.startswith("<<|mol"):
                        mol_idx = int(key.split("<<|mol")[-1].split("|>>")[0])
                        smiles = data["entity"][key]["smiles"]
                        smiless.append(smiles)
                        remap_mol[mol_idx] = unique_mol_cnt
                        unique_mol_cnt += 1
                        try:
                            graph = smiles2graph(smiles)
                            mol_sizes.append(graph['num_nodes'])
                        except Exception as e:
                            print(f"{e}")
                            mol_sizes.append(np.nan)
                mol_occurs = [int(t.split("|>>")[0]) for t in text.split("<<|mol")[1:]]
                remap_mol_idxs = torch.tensor([remap_mol[mol_occur] for mol_occur in mol_occurs], dtype=torch.long)
                if add_mol_token:
                    text = re.sub("<<\|mol[0-9]+\|>>", "<mol> <unk> </mol>", text)
                else:
                    text = re.sub("<<\|mol[0-9]+\|>>", "<unk>", text)
                tokenized = tokenizer(text, return_tensors="pt")
                input_ids = tokenized.input_ids[0]
                assert len(input_ids[input_ids == 0]) == len(remap_mol_idxs)
                input_ids[input_ids == 0] = -remap_mol_idxs - 1
                if is_instruction:
                    soruce = text.split("Response:\n")[0] + "Response:\n"
                    soruce_tokenized = tokenizer(soruce, return_tensors="pt")
                    soruce_input_ids = soruce_tokenized.input_ids[0]
                    input_ids_len = len(soruce_input_ids)
                else:
                    input_ids_len = 0
                write_txn.put(f"{index}".encode(), pkl.dumps((input_ids, input_ids_len, smiless, mol_sizes)))
                if (index + 1) % 10000 == 0:
                    write_txn.commit()
                    write_txn = write_env.begin(write=True)
    if (index + 1) % 10000 != 0:
        write_txn.commit()
    write_env.close()


def process(files, lmdb_path, process_index=0, is_instruction=False, add_mol_token=False):

    write_env = lmdb.open(lmdb_path, map_size=1024 ** 4)
    write_txn = write_env.begin(write=True)
    index = 0

    for file in files:
        print(f"Processing {file}")


        datas = json.load(open(file, 'r'))
        for data in tqdm(datas):
            data = replace_num(data)
            text = f"{data['text']}{tokenizer.eos_token}"
            unique_mol_cnt = 0
            remap_mol = {}
            smiless = []
            mol_sizes = []
            for key in data["entity"]:
                if key.startswith("<<|mol"):
                    mol_idx = int(key.split("<<|mol")[-1].split("|>>")[0])
                    smiles = data["entity"][key]["smiles"]
                    smiless.append(smiles)
                    remap_mol[mol_idx] = unique_mol_cnt
                    unique_mol_cnt += 1
                    try:
                        graph = smiles2graph(smiles)
                        mol_sizes.append(graph['num_nodes'])
                    except Exception as e:
                        print(f"{e}")
                        mol_sizes.append(np.nan)
            mol_occurs = [int(t.split("|>>")[0]) for t in text.split("<<|mol")[1:]]
            remap_mol_idxs = torch.tensor([remap_mol[mol_occur] for mol_occur in mol_occurs], dtype=torch.long)
            if add_mol_token:
                text = re.sub("<<\|mol[0-9]+\|>>", "<mol> <unk> </mol>", text)
            else:
                text = re.sub("<<\|mol[0-9]+\|>>", "<unk>", text)
            tokenized = tokenizer(text, return_tensors="pt")
            input_ids = tokenized.input_ids[0]
            assert len(input_ids[input_ids == 0]) == len(remap_mol_idxs)
            input_ids[input_ids == 0] = -remap_mol_idxs - 1
            if is_instruction:
                soruce = text.split("Response:\n")[0] + "Response:\n"
                soruce_tokenized = tokenizer(soruce, return_tensors="pt")
                soruce_input_ids = soruce_tokenized.input_ids[0]
                input_ids_len = len(soruce_input_ids)
            else:
                input_ids_len = 0
            write_txn.put(f"{index}".encode(), pkl.dumps((input_ids, input_ids_len, smiless, mol_sizes)))

            if (index + 1) % 10000 == 0:
                write_txn.commit()
                write_txn = write_env.begin(write=True)
            index += 1

        if (index + 1) % 10000 != 0:
            write_txn.commit()
            write_txn = write_env.begin(write=True)
        print(f"Finish processing {file}")

    write_env.close()


def tokenize_sample_parallel(in_fname, out_dir_name, num_processes, is_instruction=False, add_mol_token=False):
    with open(in_fname, "r") as in_file:
        lines = in_file.readlines()

    chunk_size = (len(lines) + num_processes - 1) // num_processes
    chunk_starts = [chunk_size * i for i in range(num_processes)]
    chunk_ends = [min(len(lines), chunk_start + chunk_size) for chunk_start in chunk_starts]
    chunks = [lines[chunk_start: chunk_end] for chunk_start, chunk_end in zip(chunk_starts, chunk_ends)]

    pool = Pool(num_processes)
    pool.starmap(process, zip(chunks, range(num_processes), [out_dir_name for _ in range(num_processes)], [is_instruction for _ in range(num_processes)], [add_mol_token for _ in range(num_processes)]))


def split_files(files, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for i, file in enumerate(files):
        chunks[i % num_chunks].append(file)
    return chunks


if __name__ == "__main__":
    files = []
    for file in os.listdir('/mnt/pubchem/prompt/'):
        if file.endswith('.json'):
            files.append(os.path.join('/mnt/pubchem/prompt/', file))
    print(files)

    outpth = '/mnt/pubchem/prompt_tokenized/'

    num_lmdbs = 64
    lmdb_paths = [f'{outpth}data{i}' for i in range(num_lmdbs)]
    file_chunks = split_files(files, num_lmdbs)

    process_with_token = functools.partial(process, add_mol_token=True)

    with multiprocessing.Pool(num_lmdbs) as pool:
        pool.starmap(process_with_token, zip(file_chunks, lmdb_paths))
