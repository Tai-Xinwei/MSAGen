# -*- coding: utf-8 -*-
import json
import os
import pickle as pkl
import re
from multiprocessing import Pool

import lmdb
import numpy as np
import torch
import transformers
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from tqdm import tqdm

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "/home/v-peiqizhi/models/converted/llama-2-7b/",
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
special_tokens_dict["additional_special_tokens"] = [
    "<mol>",
    "</mol>",
    "<num>",
    "</num>",
]
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
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
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
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = torch.tensor(edge_index)
    graph["edge_attr"] = torch.tensor(edge_attr)
    graph["x"] = torch.tensor(x)
    graph["num_nodes"] = len(x)

    return graph


def replace_num(data):
    nums = []
    cur_num_ptr = 0
    while True:
        num_key = f"<<|num{cur_num_ptr}|>>"
        if num_key in data["entities"]:
            num = data["entities"][num_key]
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
    write_env = lmdb.open(out_dir_name, map_size=1024**4)
    write_txn = write_env.begin(write=True)
    with open(in_fname, "r") as in_file:
        for index, line in tqdm(enumerate(in_file)):
            data = json.loads(line.strip())
            data = replace_num(data)
            text = f"{data['text']}{tokenizer.eos_token}"
            unique_mol_cnt = 0
            remap_mol = {}
            smiless = []
            mol_sizes = []
            nums = []
            for key in data["entities"]:
                if key.startswith("<<|mol"):
                    mol_idx = int(key.split("<<|mol")[-1].split("|>>")[0])
                    smiles = data["entities"][key]["smiles"]
                    smiless.append(smiles)
                    remap_mol[mol_idx] = unique_mol_cnt
                    unique_mol_cnt += 1
                    try:
                        graph = smiles2graph(smiles)
                        mol_sizes.append(graph["num_nodes"])
                    except Exception as e:
                        print(f"{e}")
                        mol_sizes.append(np.nan)
                if key.startswith("<<|num"):
                    num = data["entities"][key]
                    nums.append(num)
            mol_occurs = [int(t.split("|>>")[0]) for t in text.split("<<|mol")[1:]]
            remap_mol_idxs = torch.tensor(
                [remap_mol[mol_occur] for mol_occur in mol_occurs], dtype=torch.long
            )
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
            write_txn.put(
                f"{index}".encode(),
                pkl.dumps((input_ids, input_ids_len, smiless, mol_sizes, nums)),
            )
            if (index + 1) % 10000 == 0:
                write_txn.commit()
                write_txn = write_env.begin(write=True)
    if (index + 1) % 10000 != 0:
        write_txn.commit()
    write_env.close()


def process(lines, process_index, out_dir_name, is_instruction):
    os.system(f"mkdir -p {out_dir_name}_{process_index}")
    write_env = lmdb.open(f"{out_dir_name}_{process_index}", map_size=1024**4)
    write_txn = write_env.begin(write=True)
    for index, line in tqdm(enumerate(lines)):
        data = json.loads(line.strip())
        data = replace_num(data)
        text = f"{data['text']}{tokenizer.eos_token}"
        unique_mol_cnt = 0
        remap_mol = {}
        smiless = []
        mol_sizes = []
        for key in data["entities"]:
            if key.startswith("<<|mol"):
                mol_idx = int(key.split("<<|mol")[-1].split("|>>")[0])
                smiles = data["entities"][key]["smiles"]
                smiless.append(smiles)
                remap_mol[mol_idx] = unique_mol_cnt
                unique_mol_cnt += 1
                try:
                    graph = smiles2graph(smiles)
                    mol_sizes.append(graph["num_nodes"])
                except Exception as e:
                    print(f"{e}")
                    mol_sizes.append(np.nan)
        mol_occurs = [int(t.split("|>>")[0]) for t in text.split("<<|mol")[1:]]
        if is_instruction:
            response_splits = text.split("Response:")
            assert len(response_splits) == 2
            source = response_splits[0] + "Response:"
            response = response_splits[1]
            assert len(re.split("<<\|mol[0-9]+\|>>", response)) == 1

            source_parts = re.split("<<\|mol[0-9]+\|>>", source)
            source_ids = []
            assert (
                len(mol_occurs) == len(source_parts) - 1
            ), f"{text}, {source_parts}, {mol_occurs}"
            for i, source_part in enumerate(source_parts):
                if len(source_parts) > 1:
                    if i == 0:
                        source_part += "<mol>"
                    elif i > 0 and i < len(source_parts) - 1:
                        source_part = "</mol>" + source_part + "<mol>"
                    elif i == len(source_parts) - 1:
                        source_part = "</mol>" + source_part
                tokenized = tokenizer(source_part, return_tensors="pt")
                if i > 0:
                    source_ids.extend(tokenized.input_ids[0][1:])
                else:
                    source_ids.extend(tokenized.input_ids[0])
                if i != len(source_parts) - 1:
                    source_ids.append(-remap_mol[mol_occurs[i]] - 1)
            source_ids = torch.tensor(source_ids, dtype=torch.int64)

        text_parts = re.split("<<\|mol[0-9]+\|>>", text)
        input_ids = []
        assert (
            len(mol_occurs) == len(text_parts) - 1
        ), f"{text}, {text_parts}, {mol_occurs}"
        for i, text_part in enumerate(text_parts):
            if len(text_parts) > 1:
                if i == 0:
                    text_part += "<mol>"
                elif i > 0 and i < len(text_parts) - 1:
                    text_part = "</mol>" + text_part + "<mol>"
                elif i == len(text_parts) - 1:
                    text_part = "</mol>" + text_part
            tokenized = tokenizer(text_part, return_tensors="pt")
            if i > 0:
                input_ids.extend(tokenized.input_ids[0][1:])
            else:
                input_ids.extend(tokenized.input_ids[0])
            if i != len(text_parts) - 1:
                input_ids.append(-remap_mol[mol_occurs[i]] - 1)
                assert remap_mol[mol_occurs[i]] < len(smiless), f"{line}"
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        if is_instruction:
            assert torch.all(
                input_ids[: len(source_ids)][:-1] == source_ids[:-1]
            ), f"{input_ids[:len(source_ids)]} vs. {source_ids}"
            input_ids_len = len(source_ids)
        else:
            input_ids_len = 0
        write_txn.put(
            f"{process_index}_{index}".encode(),
            pkl.dumps((input_ids, input_ids_len, smiless, mol_sizes)),
        )
    write_txn.commit()
    write_env.close()


def tokenize_sample_parallel(
    in_fname, out_dir_name, num_processes, is_instruction=False
):
    with open(in_fname, "r") as in_file:
        lines = in_file.readlines()

    chunk_size = (len(lines) + num_processes - 1) // num_processes
    chunk_starts = [chunk_size * i for i in range(num_processes)]
    chunk_ends = [
        min(len(lines), chunk_start + chunk_size) for chunk_start in chunk_starts
    ]
    chunks = [
        lines[chunk_start:chunk_end]
        for chunk_start, chunk_end in zip(chunk_starts, chunk_ends)
    ]

    pool = Pool(num_processes)
    pool.starmap(
        process,
        zip(
            chunks,
            range(num_processes),
            [out_dir_name for _ in range(num_processes)],
            [is_instruction for _ in range(num_processes)],
        ),
    )


# tokenize_sample("chembi.json", "../tmp_data/chembi/mol-instruction-mol-desc/clean", is_instruction=True, add_mol_token=True)
# tokenize_sample(
#     "../tmp_data/json/ESOL_train.json",
#     "../tmp_data/ESOL/mol-instruction-mol-desc/train",
#     is_instruction=True,
#     add_mol_token=True,
# )
# tokenize_sample(
#     "../tmp_data/json/ESOL_valid.json",
#     "../tmp_data/ESOL/mol-instruction-mol-desc/valid",
#     is_instruction=True,
#     add_mol_token=True,
# )
# tokenize_sample(
#     "../tmp_data/json/ESOL_test.json",
#     "../tmp_data/ESOL/mol-instruction-mol-desc/test",
#     is_instruction=True,
#     add_mol_token=True,
# )

result_path = "/sfm/ds_dataset/qizhi_numerical"
# dataset_name = "freesolv"
# dataset_name = "ESOL"
# dataset_name = "bbbp"
# dataset_name = "bace"
# dataset_name = "hiv"
# dataset_name = "lipo"
# dataset_name = 'bbbp_mlp'
# dataset_name = 'molnet_3_reg'
dataset_name = 'molnet_3_reg_up'
for split in ["train", "valid", "test"]:
    tokenize_sample(
        f"{result_path}/json/{dataset_name}_{split}.json",
        f"{result_path}/{dataset_name}/mol-instruction-mol-desc/{split}",
        is_instruction=True,
        add_mol_token=True,
    )
    print(f"Done for {split} set")
