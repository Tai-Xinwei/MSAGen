# -*- coding: utf-8 -*-
import logging

import torch

logging.getLogger().setLevel(logging.ERROR)

import copy
import json
import os
import pickle as pkl
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Dict, Optional, Sequence

import lmdb
import numpy as np
import transformers
from data.mol_data.collator import collator_copilot, collator_copilot_multi_mol
from data.mol_data.wrapper import smiles2graph
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from utils.jload import jload

from . import algos

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess(input_ids, label, llm_mask, data, idx, mask_ratio=0.0):
    # graph = smiles2graph(smile)
    # data = Data()

    # data.__num_nodes__ = int(graph['num_nodes'])
    # data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    # data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
    # data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
    data.pos = None
    data.y = None
    data.idx = idx
    # data.smile = graph['smile']

    # edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x

    N = data.x.size(0)
    data.x = convert_to_single_emb(data.x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[data.edge_index[0, :], data.edge_index[1, :]] = True

    # edge feature here
    if len(data.edge_attr.size()) == 1:
        data.edge_attr = data.edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, data.edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[data.edge_index[0, :], data.edge_index[1, :]] = (
        convert_to_single_emb(data.edge_attr) + 1
    )
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    mask_N = int(N * mask_ratio)
    mask_idx = torch.from_numpy(np.random.choice(N, mask_N, replace=False))

    node_mask = torch.zeros(N).float()
    node_mask[mask_idx] = 1.0

    data.attn_bias = attn_bias
    data.attn_edge_type = attn_edge_type
    data.spatial_pos = spatial_pos
    data.in_degree = adj.long().sum(dim=1).view(-1)
    data.out_degree = data.in_degree  # for undirected graph
    data.edge_input = torch.from_numpy(edge_input).long()

    # data.node_mask = None
    data.node_mask = node_mask.unsqueeze(-1)

    data.input_ids = input_ids
    data.target_ids = label
    data.llm_mask = llm_mask

    return data


# TODO (Roger)
def _tokenize_fn_moleculenet(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    smiles: Sequence[str],
    smiles_dict: Dict,
    nnodes: Sequence[int],
    pool_mode: str = "cls",
    embedding_length: int = 1,
) -> Dict:
    """Tokenize a list of strings."""

    tokenized_list = []
    for idx, text in enumerate(strings):
        text = " ".join(text)
        split_text = text.split("<<|mol0|>>")
        split_tokenized_list = [
            tokenizer(
                tt,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for tt in split_text
        ]
        smiles_idx = -smiles_dict.get(smiles[idx], 1) - 1
        # graph = smiles2graph(smiles[idx])
        # nnode = int(graph['num_nodes'])
        if pool_mode == "full":
            to_replace_list = [1] + [smiles_idx for i in range(nnodes[idx])]
            to_replace_list = torch.tensor(to_replace_list).to(torch.long)
            input_id = torch.cat(
                [
                    split_tokenized_list[0].input_ids[0],
                    to_replace_list,
                    split_tokenized_list[-1].input_ids[0],
                ]
            )
        elif pool_mode == "qformer" or pool_mode == "multimol":
            to_replace_list = [1] + [smiles_idx for i in range(embedding_length)]
            to_replace_list = torch.tensor(to_replace_list).to(torch.long)
            input_id = torch.cat(
                [
                    split_tokenized_list[0].input_ids[0],
                    to_replace_list,
                    split_tokenized_list[-1].input_ids[0],
                ]
            )
        else:
            to_replace_id = torch.tensor([smiles_idx]).to(torch.long)
            input_id = torch.cat(
                [
                    split_tokenized_list[0].input_ids[0],
                    to_replace_id,
                    split_tokenized_list[-1].input_ids[0],
                ]
            )

        tokenized_list.append(input_id)

    # tokenized_list = [
    #     tokenizer(
    #         text,
    #         return_tensors="pt",
    #         padding="longest",
    #         max_length=tokenizer.model_max_length,
    #         truncation=True,
    #     )
    #     for text in strings
    # ]
    input_ids = labels = tokenized_list
    input_ids_lens = labels_lens = [
        tokenized.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


def preprocess_mask(
    tokenizer: transformers.PreTrainedTokenizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    return (input_ids, labels, input_ids.ne(tokenizer.pad_token_id))


# TODO (Roger)
def preprocess_moleculenet(
    sources: Sequence[str],
    targets: Sequence[str],
    smiles: Sequence[str],
    smiles_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    nnodes: Sequence[int],
    pool_mode: str = "cls",
    embedding_length=1,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + [t] for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn_moleculenet(
            strings,
            tokenizer,
            smiles,
            smiles_dict,
            nnodes,
            pool_mode=pool_mode,
            embedding_length=embedding_length,
        )
        for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    # labels = examples_tokenized["labels"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


# TODO (Roger)
class SupervisedMoleculeNetDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        dataset_names: str,
        smiles_dict_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        mode="train",
        pool_mode="cls",
        embedding_length=1,
    ):
        # data_path: ./data/
        # dataset_name: hiv
        # smiles_dict_path: ./data/mol2idx_dict.jsonl

        super(SupervisedMoleculeNetDataset, self).__init__()
        dataset_names = dataset_names.split(",")
        list_data_dict = []
        for dataset_name in dataset_names:
            assert dataset_name in ["hiv", "clintox", "sider", "tox21", "bbbp", "bace"]
            logging.warning(f"Loading data {dataset_name} ...")
            dir_name = os.path.join(data_path, dataset_name)
            for fname in os.listdir(dir_name):
                # if fname.endswith("-train.jsonl"):
                #     with open(os.path.join(os.path.join(data_path, dataset_name), fname), 'r') as f:
                #         content = f.readlines()
                #         list_data_dict.extend([json.loads(item) for item in content])

                if mode == "train" and fname.endswith("-train.jsonl"):
                    with open(
                        os.path.join(os.path.join(data_path, dataset_name), fname), "r"
                    ) as f:
                        content = f.readlines()
                        list_data_dict.extend([json.loads(item) for item in content])
                elif mode == "valid" and fname.endswith("-dev.jsonl"):
                    with open(
                        os.path.join(os.path.join(data_path, dataset_name), fname), "r"
                    ) as f:
                        content = f.readlines()
                        list_data_dict.extend([json.loads(item) for item in content])
                elif mode == "test" and fname.endswith("-test.jsonl"):
                    with open(
                        os.path.join(os.path.join(data_path, dataset_name), fname), "r"
                    ) as f:
                        content = f.readlines()
                        list_data_dict.extend([json.loads(item) for item in content])
                # else:
                # raise NotImplementedError

        # format: {'text': 'xxx', 'entities': {'<<|mol0|>>': {'smiles': 'xxx'}}}
        print(smiles_dict_path)
        smiles_dict = jload(smiles_dict_path)

        # TODO: formating & preprocessing
        logging.warning("Formatting inputs...")
        # prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # sources = [
        #     prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #     for example in list_data_dict
        # ]

        sources = []
        targets = []
        entity_replace_smiles = []
        for example in list_data_dict:
            cur_text = example.get("text", "")
            cur_source = cur_text.split()[:-2]
            cur_target = f"{' '.join(cur_text.split()[-2:])}{tokenizer.eos_token}"
            cur_smiles = (
                example.get("entities", "").get("<<|mol0|>>", "").get("smiles", "")
            )

            sources.append(cur_source)
            targets.append(cur_target)
            entity_replace_smiles.append(cur_smiles)

        self.smiles = entity_replace_smiles
        self.data = []
        nnodes = []
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, entity_replace_smiles)

            for i, graph in enumerate(iter):
                data = Data()
                data.__num_nodes__ = int(graph["num_nodes"])
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

                self.data.append(data)
                nnodes.append(data.__num_nodes__)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess_moleculenet(
            sources,
            targets,
            entity_replace_smiles,
            smiles_dict,
            tokenizer,
            nnodes,
            pool_mode=pool_mode,
            embedding_length=embedding_length,
        )

        input_ids = data_dict["input_ids"]
        labels = data_dict["labels"]
        self.input_ids, self.labels, self.llm_mask = preprocess_mask(
            tokenizer, input_ids, labels
        )

        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = preprocess(
            self.input_ids[i], self.labels[i], self.llm_mask[i], self.data[i], i
        )
        return item

    def collater(self, samples):
        return collator_copilot(
            samples,
            max_node=1024,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


class SupervisedProcessedData(Dataset):
    def __init__(
        self,
        data_path: str,
        mol_size_path: str,
        mol2idx_dict_path: str,
        dataset_names: str,
        dataset_splits: str,
        in_memory: bool,
        pad_token_id: int,
        pool_mode: str,
        embedding_length: int = 1,
        dataset_ratios: str = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.dataset_names = dataset_names.split(",")
        self.dataset_ratios = (
            None
            if dataset_ratios is None
            else [float(ratio) for ratio in dataset_ratios.split(",")]
        )
        self.dataset_splits = [
            dataset_split.split("+") for dataset_split in dataset_splits.split(",")
        ]
        assert len(self.dataset_splits) == len(
            self.dataset_names
        ), "Lengths of dataset_splits and dataset_names do not match."
        assert self.dataset_ratios is None or len(self.dataset_ratios) == len(
            self.dataset_names
        ), "Lengths of dataset_ratios and dataset_names do not match."
        self.in_memory = in_memory
        self.pad_token_id = pad_token_id
        self.pool_mode = pool_mode
        self.embedding_length = embedding_length

        self.len = 0
        self.index_to_key_map = []
        self.in_memory_data = {}
        self.read_txns = {}
        self.read_envs = {}
        self.weight_dict = {}
        self.len_read_txns = {}
        self.len_read_envs = {}
        self.dataset_count = {}
        self.dataset_filtered = {}
        self.molidx_to_smiles = {}
        self.mol_data = []
        self.mol_data_offset = []
        smile_list = []

        self.multi_hop_max_dist = 5
        self.spatial_pos_max = 1024
        self.max_node = 256
        max_mol_per_sample = 1

        with open(mol2idx_dict_path, "rb") as in_file:
            mol2idx_dict = jload(mol2idx_dict_path)
            for smiles in mol2idx_dict:
                self.molidx_to_smiles[int(mol2idx_dict[smiles])] = smiles
                smile_list.append(smiles)

        smile2data = {}
        with Pool(processes=120) as pool:
            iter = pool.imap(smiles2graph, smile_list)

            for i, graph in enumerate(iter):
                try:
                    data = Data()
                    data.__num_nodes__ = int(graph["num_nodes"])
                    data.edge_index = torch.from_numpy(graph["edge_index"]).to(
                        torch.int64
                    )
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(
                        torch.int64
                    )
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

                    smile2data[graph["smile"]] = data
                except:
                    continue
                # self.data.append(data)
                # nnodes.append(data.__num_nodes__)

        with open(mol_size_path, "rb") as in_file:
            self.mol_size_dict = pkl.load(in_file)

        num_mols = 0
        max_mol = 0
        max_node = 0
        if not self.in_memory:
            for i, (dataset_name, dataset_splits) in enumerate(
                zip(self.dataset_names, self.dataset_splits)
            ):
                self.read_txns[dataset_name] = {}
                self.read_envs[dataset_name] = {}
                self.len_read_envs[dataset_name] = {}
                self.len_read_txns[dataset_name] = {}
                start_index = self.len
                self.dataset_count[dataset_name] = {}
                self.dataset_filtered[dataset_name] = {}
                for dataset_split in dataset_splits:
                    logging.warning(
                        f"Loading dataset {dataset_name} split {dataset_split}"
                    )
                    read_env = lmdb.open(
                        f"{self.data_path}/{dataset_name}/{dataset_split}/"
                    )
                    read_txn = read_env.begin(write=False)
                    self.read_txns[dataset_name][dataset_split] = read_txn
                    self.read_envs[dataset_name][dataset_split] = read_env
                    len_read_env = lmdb.open(
                        f"{self.data_path}/{dataset_name}/{dataset_split}-len/"
                    )
                    len_read_txn = len_read_env.begin(write=False)
                    self.len_read_envs[dataset_name][dataset_split] = len_read_env
                    self.len_read_txns[dataset_name][dataset_split] = len_read_txn
                    self.dataset_count[dataset_name][dataset_split] = 0
                    self.dataset_filtered[dataset_name][dataset_split] = 0
                    for key, val in read_txn.cursor():
                        val = pkl.loads(val)[0]
                        length_and_mol_idxs = pkl.loads(len_read_txn.get(key))
                        mol_idxs = length_and_mol_idxs[1]
                        to_skip = (
                            len(mol_idxs) == 0 or len(mol_idxs) > max_mol_per_sample
                        )
                        max_mol = max(max_mol, len(mol_idxs))
                        for mol_idx in mol_idxs:
                            mol_idx = int(mol_idx)
                            if mol_idx not in self.mol_size_dict:
                                to_skip = True
                                break
                        if to_skip:
                            continue
                        for mol_idx in mol_idxs:
                            mol_idx = int(mol_idx)
                            smiles = self.molidx_to_smiles[mol_idx]
                            # mol_data = self._process_smiles(smiles)
                            mol_data = smile2data[smiles]
                            max_node = max(max_node, mol_data.__num_nodes__)

                            if mol_data.__num_nodes__ > self.max_node:
                                to_skip = True
                                break
                            self.mol_data.append(mol_data)
                        if to_skip:
                            continue
                        self.mol_data_offset.append(num_mols)
                        num_mols += len(mol_idxs)
                        assert len(mol_idxs) == len(
                            val[val < 0]
                        ), f"{mol_idxs} vs. {val}"
                        self.index_to_key_map.append(
                            (dataset_name, dataset_split, key.decode())
                        )
                        self.dataset_count[dataset_name][dataset_split] += 1
                        self.len += 1
                if self.dataset_ratios is not None:
                    self.weight_dict[(start_index, self.len)] = self.dataset_ratios[i]
            self.mol_data_offset.append(num_mols)
        else:
            for i, (dataset_name, dataset_splits) in enumerate(
                zip(self.dataset_names, self.dataset_splits)
            ):
                self.in_memory_data[dataset_name] = {}
                start_index = self.len
                self.dataset_count[dataset_name] = {}
                self.dataset_filtered[dataset_name] = {}
                for dataset_split in dataset_splits:
                    logging.warning(
                        f"Loading dataset {dataset_name} split {dataset_split}"
                    )
                    read_env = lmdb.open(
                        f"{self.data_path}/{dataset_name}/{dataset_split}/"
                    )
                    read_txn = read_env.begin(write=False)
                    len_read_env = lmdb.open(
                        f"{self.data_path}/{dataset_name}/{dataset_split}-len/"
                    )
                    len_read_txn = len_read_env.begin(write=False)
                    self.in_memory_data[dataset_name][dataset_split] = {}
                    self.dataset_count[dataset_name][dataset_split] = 0
                    self.dataset_filtered[dataset_name][dataset_split] = 0
                    for key, val in read_txn.cursor():
                        val = pkl.loads(val)
                        length_and_mol_idxs = pkl.loads(len_read_txn.get(key))
                        mol_idxs = length_and_mol_idxs[1]
                        to_skip = (
                            len(mol_idxs) == 0 or len(mol_idxs) > max_mol_per_sample
                        )
                        max_mol = max(max_mol, len(mol_idxs))
                        for mol_idx in mol_idxs:
                            mol_idx = int(mol_idx)
                            if mol_idx not in self.mol_size_dict:
                                to_skip = True
                                break
                        if to_skip:
                            continue
                        for mol_idx in mol_idxs:
                            mol_idx = int(mol_idx)
                            smiles = self.molidx_to_smiles[mol_idx]
                            # mol_data = self._process_smiles(smiles)
                            mol_data = smile2data[smiles]
                            max_node = max(max_node, mol_data.__num_nodes__)

                            if mol_data.__num_nodes__ > self.max_node:
                                to_skip = True
                                break
                            self.mol_data.append(mol_data)
                        if to_skip:
                            continue
                        self.mol_data_offset.append(num_mols)
                        num_mols += len(mol_idxs)
                        key = key.decode()
                        self.index_to_key_map.append((dataset_name, dataset_split, key))
                        self.dataset_count[dataset_name][dataset_split] += 1
                        self.in_memory_data[dataset_name][dataset_split][key] = val
                        self.len += 1
                    read_env.close()
                if self.dataset_ratios is not None:
                    self.weight_dict[(start_index, self.len)] = self.dataset_ratios[i]
            self.mol_data_offset.append(num_mols)
        logging.warning(f"{self.len} sentences loaded.")
        for dataset_name in self.dataset_count:
            for dataset_split in self.dataset_count[dataset_name]:
                logging.warning(
                    f"Dataset {dataset_name} split {dataset_split}: {self.dataset_count[dataset_name][dataset_split]} loaded, {self.dataset_filtered[dataset_name][dataset_split]} filtered."
                )

        print(f"Max mol: {max_mol}, max node: {max_node}")
        if len(self.weight_dict) == 0:
            self.weight_dict = None

        if self.weight_dict is not None:
            equal_ratio = True
            for begin, end in self.weight_dict:
                if int(self.weight_dict[(begin, end)]) != 1:
                    equal_ratio = False
            if equal_ratio:
                self.weight_dict = None

    def _process_smiles(self, smiles):
        mol_data = smiles2graph(smiles)
        data = Data()
        data.__num_nodes__ = int(mol_data["num_nodes"])
        data.edge_index = torch.from_numpy(mol_data["edge_index"]).to(torch.int64)
        data.edge_attr = torch.from_numpy(mol_data["edge_feat"]).to(torch.int64)
        data.x = torch.from_numpy(mol_data["node_feat"]).to(torch.int64)
        return data

    def __del__(self):
        if not self.in_memory:
            for dataset_name in self.read_envs:
                for dataset_split in self.read_envs[dataset_name]:
                    self.read_envs[dataset_name][dataset_split].close()

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        dataset_name, dataset_split, key = self.index_to_key_map[index]
        if not self.in_memory:
            input_ids, input_ids_len = pkl.loads(
                self.read_txns[dataset_name][dataset_split].get(key.encode())
            )
        else:
            input_ids, input_ids_len = self.in_memory_data[dataset_name][dataset_split][
                key
            ]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
        input_ids = input_ids.to(dtype=torch.int64)
        input_ids_len = int(input_ids_len)

        mol_idx_pos = torch.nonzero(input_ids < 0).squeeze()
        if len(mol_idx_pos.size()) == 0:
            mol_idx_pos = mol_idx_pos.unsqueeze(-1)
        mol_idxs = input_ids[mol_idx_pos]

        mol_idx_pos = torch.cat(
            [torch.tensor([-1]), mol_idx_pos, torch.tensor([len(input_ids)])]
        )

        assert self.mol_data_offset[index + 1] - self.mol_data_offset[index] == len(
            mol_idxs
        ), f"{input_ids} vs. {key} vs. {self.mol_data_offset[index + 1] - self.mol_data_offset[index]} vs. {len(mol_idxs)}, {mol_idxs}"

        if (
            self.pool_mode == "full"
            or self.pool_mode == "qformer"
            or self.pool_mode == "multimol"
        ):
            input_ids_expanded = []
            for i in range(len(mol_idx_pos) - 1):
                cur_pos = mol_idx_pos[i]
                if cur_pos >= 0 and cur_pos < len(input_ids):
                    mol_idx = int(input_ids[cur_pos])
                    assert mol_idx < 0
                    mol_size = self.mol_size_dict[-mol_idx - 1] - 1
                    if self.pool_mode == "full":
                        input_ids_expanded.append(
                            torch.tensor([mol_idx for _ in range(mol_size)])
                        )
                    elif self.pool_mode == "qformer" or self.pool_mode == "multimol":
                        input_ids_expanded.append(torch.tensor([32001]))
                        input_ids_expanded.append(
                            torch.tensor(
                                [mol_idx for _ in range(self.embedding_length)]
                            )
                        )
                        input_ids_expanded.append(torch.tensor([32002]))
                input_ids_expanded.append(input_ids[cur_pos + 1 : mol_idx_pos[i + 1]])

            input_ids = torch.cat(input_ids_expanded)
        else:
            raise Exception(
                f"Unknown pool mode {self.pool_mode}, should be full, qformer or multimol"
            )

        processed_mol_data = []
        for j, mol_data_idx in enumerate(
            range(self.mol_data_offset[index], self.mol_data_offset[index + 1])
        ):
            mol_data = self.mol_data[mol_data_idx].clone()
            processed_mol_data.append(preprocess(None, None, None, mol_data, i, 0.0))

        labels = input_ids.clone()
        labels[:input_ids_len] = IGNORE_INDEX
        labels[labels < 0] = IGNORE_INDEX
        return dict(
            input_ids=input_ids,
            labels=labels,
            llm_mask=input_ids.ne(self.pad_token_id),
            processed_mol_data=processed_mol_data,
        )

    def collater(self, samples):
        return collator_copilot_multi_mol(
            samples,
            max_node=1024,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )


if __name__ == "__main__":
    pass
