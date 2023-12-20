# -*- coding: utf-8 -*-
import logging

import torch

logging.getLogger().setLevel(logging.ERROR)

import copy
import gc
import json
import os
import pickle as pkl
from argparse import Namespace
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lmdb
import numpy as np
import transformers
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveHs
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from sfm.logging.loggers import logger
from sfm.utils.jload import jload

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


@torch.jit.script
def convert_to_single_emb_last(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


@torch.jit.script
def convert_to_single_emb_light(x, num_feature_values: List[int]):
    num_feature_values = torch.tensor(
        num_feature_values, dtype=torch.long, device=x.device
    )
    offsets = torch.cumsum(num_feature_values, dim=-1)
    offsets = torch.cat(
        [torch.tensor([0], dtype=offsets.dtype, device=offsets.device), offsets], dim=-1
    )[:-1]
    assert len(x.size()) > 1
    x = x + offsets + 1  # 0 is for padding
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


def _calc_input_len_static(
    model_max_length,
    molecule_max_size,
    input_ids,
    source_len,
    smiless,
    mol_sizes,
    nums=None,
):
    try:
        mol_sizes = torch.tensor(mol_sizes)
        if torch.any(torch.isnan(mol_sizes)) or torch.any(
            mol_sizes > molecule_max_size
        ):
            return model_max_length + 1
        mol_idxs = -input_ids[input_ids < 0] - 1
        return int(len(input_ids) + torch.sum(mol_sizes[mol_idxs]) - len(mol_idxs))
    except Exception as e:
        print(f"Failed to convert smiles to graph: {e} {input_ids} {smiless}")
        return model_max_length + 1


def _load(
    data_path,
    model_max_length,
    molecule_max_size,
    process_index,
    num_processes,
    dataset_name,
    dataset_split,
    threshold_maxmol,
):
    read_env = lmdb.open(f"{data_path}/{dataset_name}/{dataset_split}/", readonly=True)
    read_txn = read_env.begin(write=False)
    dataset_count = 0
    dataset_filtered = 0
    index_to_key_map = []

    for i, (key, val) in tqdm(
        enumerate(read_txn.cursor()),
        desc=f"Process {process_index}. Loading for {dataset_name} {dataset_split}",
        miniters=10000,
    ):
        if i % num_processes == process_index:
            val = pkl.loads(val)
            if (
                _calc_input_len_static(model_max_length, molecule_max_size, *val)
                > model_max_length
                or len(val[3]) > threshold_maxmol
            ):
                dataset_filtered += 1
                continue
            index_to_key_map.append((dataset_name, dataset_split, key.decode()))
            dataset_count += 1
        val = None
        if (i + 1) % 100000 == 0:
            gc.collect()
    gc.collect()
    read_env.close()
    return dataset_count, dataset_filtered, index_to_key_map


class SupervisedProcessedDataWithSmiles(Dataset):
    def __init__(
        self,
        data_path: str,
        dataset_names: str,
        dataset_splits: str,
        in_memory: bool,
        model_max_length: int,
        mol_embed_type: str,
        molecule_max_size: int,
        pad_token_id: int,
        dataset_ratios: Optional[str] = None,
        pool_mode: Optional[str] = "full",
        embedding_length: int = 20,
        num_token_id: int = 32003,
        use_pp: bool = True,
        use_pbc: bool = False,
        local_rank: int = 0,
        num_data_loading_workers: int = 16,
        skip_num_datasets: str = "",
        temp_dir: str = "./generalist-data-temp",
        max_num_mol_per_sample: int = 8,
        use_global_padding: bool = False,
        multi_hop_max_dist: int = 20,
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
        self.model_max_length = model_max_length
        self.mol_embed_type = mol_embed_type
        self.molecule_max_size = molecule_max_size
        self.pool_mode = pool_mode
        self.embedding_length = embedding_length
        self.num_token_id = num_token_id
        self.local_rank = local_rank
        self.skip_num_datasets = set(
            list(filter(lambda x: x != "", skip_num_datasets.split(",")))
        )
        self.num_data_loading_workers = num_data_loading_workers
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

        self.use_global_padding = use_global_padding
        self.multi_hop_max_dist = multi_hop_max_dist

        self.max_num_mol_per_sample = max_num_mol_per_sample

        self.len = 0
        self.index_to_key_map = []
        self.in_memory_data = {}
        self.read_txns = {}
        self.read_envs = {}
        self.weight_dict = {}
        self.dataset_count = {}
        self.dataset_filtered = {}
        self.data_collater = DataCollatorForSupervisedDataset(
            pad_token_id=pad_token_id,
            use_pp=use_pp,
            add_mfm=True,
            use_pbc=use_pbc,
            use_global_padding=self.use_global_padding,
            model_max_length=self.model_max_length,
            molecule_max_size=self.molecule_max_size,
            max_num_mol_per_sample=self.max_num_mol_per_sample,
            multi_hop_max_dist=self.multi_hop_max_dist,
        )

        if not self.in_memory:
            for i, (dataset_name, dataset_splits) in enumerate(
                zip(self.dataset_names, self.dataset_splits)
            ):
                self.read_txns[dataset_name] = {}
                self.read_envs[dataset_name] = {}
                start_index = self.len
                self.dataset_count[dataset_name] = {}
                self.dataset_filtered[dataset_name] = {}
                for dataset_split in dataset_splits:
                    logger.warning(
                        f"Loading dataset {dataset_name} split {dataset_split}"
                    )
                    read_env = lmdb.open(
                        f"{self.data_path}/{dataset_name}/{dataset_split}/"
                    )
                    read_txn = read_env.begin(write=False)
                    self.read_txns[dataset_name][dataset_split] = read_txn
                    self.read_envs[dataset_name][dataset_split] = read_env
                    if self.local_rank <= 0:
                        self._load_multiprocessing(
                            dataset_name,
                            dataset_split,
                            self.max_num_mol_per_sample,
                            self.num_data_loading_workers,
                        )
                if self.dataset_ratios is not None:
                    self.weight_dict[(start_index, self.len)] = self.dataset_ratios[i]
            # clear from previous runs
            if os.path.exists(f"{self.temp_dir / 'data_meta.pkl'}"):
                os.system(f"rm {self.temp_dir / 'data_meta.pkl'}")
            if os.path.exists(f"{self.temp_dir / 'DATA_READY'}"):
                os.system(f"rm {self.temp_dir / 'DATA_READY'}")
            if self.local_rank <= 0:
                with open(f"{self.temp_dir / 'data_meta.pkl'}", "wb") as out_file:
                    pkl.dump(self.dataset_count, out_file)
                    pkl.dump(self.dataset_filtered, out_file)
                    pkl.dump(self.index_to_key_map, out_file)
                    pkl.dump(self.len, out_file)
                    pkl.dump(self.weight_dict, out_file)
                os.system(f"touch {self.temp_dir / 'DATA_READY'}")
            else:
                while not os.path.exists(f"{self.temp_dir / 'DATA_READY'}"):
                    pass
                with open(f"{self.temp_dir / 'data_meta.pkl'}", "rb") as in_file:
                    self.dataset_count = pkl.load(in_file)
                    self.dataset_filtered = pkl.load(in_file)
                    self.index_to_key_map = pkl.load(in_file)
                    self.len = pkl.load(in_file)
                    self.weight_dict = pkl.load(in_file)
        else:
            for i, (dataset_name, dataset_splits) in enumerate(
                zip(self.dataset_names, self.dataset_splits)
            ):
                self.in_memory_data[dataset_name] = {}
                start_index = self.len
                self.dataset_count[dataset_name] = {}
                self.dataset_filtered[dataset_name] = {}
                for dataset_split in dataset_splits:
                    logger.warning(
                        f"Loading dataset {dataset_name} split {dataset_split}"
                    )
                    read_env = lmdb.open(
                        f"{self.data_path}/{dataset_name}/{dataset_split}/"
                    )
                    read_txn = read_env.begin(write=False)
                    self.in_memory_data[dataset_name][dataset_split] = {}
                    self.dataset_count[dataset_name][dataset_split] = 0
                    self.dataset_filtered[dataset_name][dataset_split] = 0
                    for key, val in tqdm(read_txn.cursor()):
                        val = pkl.loads(val)
                        if (
                            self._calc_input_len(*val) > self.model_max_length
                            or len(val[3]) > max_num_mol_per_sample
                        ):
                            self.dataset_filtered[dataset_name][dataset_split] += 1
                            continue
                        key = key.decode()
                        self.index_to_key_map.append((dataset_name, dataset_split, key))
                        self.dataset_count[dataset_name][dataset_split] += 1
                        self.in_memory_data[dataset_name][dataset_split][key] = val
                        self.len += 1
                    read_env.close()
                if self.dataset_ratios is not None:
                    self.weight_dict[(start_index, self.len)] = self.dataset_ratios[i]

        logger.info(f"{self.len} sentences loaded.")
        assert self.len > 0, "No data loaded."

        for dataset_name in self.dataset_count:
            for dataset_split in self.dataset_count[dataset_name]:
                logger.info(
                    f"Dataset {dataset_name} split {dataset_split}: {self.dataset_count[dataset_name][dataset_split]} loaded, {self.dataset_filtered[dataset_name][dataset_split]} filtered."
                )

        if len(self.weight_dict) == 0:
            self.weight_dict = None

        if self.weight_dict is not None:
            equal_ratio = True
            for begin, end in self.weight_dict:
                if int(self.weight_dict[(begin, end)]) != 1:
                    equal_ratio = False
            if equal_ratio:
                self.weight_dict = None

    def _load_multiprocessing(
        self, dataset_name, dataset_split, threshold_maxmol, num_processes
    ):
        logger.warning(f"Loading dataset {dataset_name} split {dataset_split}")
        with Pool(num_processes) as pool:
            results = pool.starmap(
                _load,
                zip(
                    [self.data_path] * num_processes,
                    [self.model_max_length] * num_processes,
                    [self.molecule_max_size] * num_processes,
                    range(num_processes),
                    [num_processes] * num_processes,
                    [dataset_name] * num_processes,
                    [dataset_split] * num_processes,
                    [threshold_maxmol] * num_processes,
                ),
            )
            total_dataset_count = 0
            total_dataset_filtered = 0
            total_index_to_key_map = []
            for dataset_count, dataset_filtered, index_to_key_map in results:
                total_dataset_count += dataset_count
                total_dataset_filtered += dataset_filtered
                total_index_to_key_map.extend(index_to_key_map)
            self.len += total_dataset_count
            self.dataset_count[dataset_name][dataset_split] = total_dataset_count
            self.dataset_filtered[dataset_name][dataset_split] = total_dataset_filtered
            self.index_to_key_map.extend(total_index_to_key_map)

    def _calc_input_len(self, input_ids, source_len, smiless, mol_sizes, nums=None):
        try:
            mol_sizes = torch.tensor(mol_sizes)
            if torch.any(torch.isnan(mol_sizes)) or torch.any(
                mol_sizes > self.molecule_max_size
            ):
                return self.model_max_length + 1
            mol_idxs = -input_ids[input_ids < 0] - 1
            return int(len(input_ids) + torch.sum(mol_sizes[mol_idxs]) - len(mol_idxs))
        except Exception as e:
            print(f"Failed to convert smiles to graph: {e} {input_ids} {smiless}")
            return self.model_max_length + 1

    def __del__(self):
        if not self.in_memory:
            for dataset_name in self.read_envs:
                for dataset_split in self.read_envs[dataset_name]:
                    self.read_envs[dataset_name][dataset_split].close()
        if hasattr(self, "molrep_dict_env") and self.molrep_dict_env is not None:
            self.molrep_dict_env.close()

    def __len__(self):
        return self.len

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        dataset_name, dataset_split, key = self.index_to_key_map[i]
        if not self.in_memory:
            data = pkl.loads(
                self.read_txns[dataset_name][dataset_split].get(key.encode())
            )
        else:
            data = self.in_memory_data[dataset_name][dataset_split][key]

        nums = None
        if len(data) == 4:
            input_ids, input_ids_len, smiless, mol_sizes = data
        elif len(data) == 5:
            input_ids, input_ids_len, smiless, mol_sizes, nums = data

        if dataset_name in self.skip_num_datasets:
            pass

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.int64)

        input_ids_len = int(input_ids_len)
        original_input_ids_len = input_ids_len
        if self.mol_embed_type == "atoms":
            mol_pos = torch.nonzero(input_ids < 0).squeeze(-1)
            mol_pos = torch.cat(
                [torch.tensor([0]), mol_pos, torch.tensor([len(input_ids)])]
            )
            new_input_ids = []
            for j in range(len(mol_pos) - 1):
                if self.pool_mode == "full":
                    new_input_ids.extend(input_ids[mol_pos[j] : mol_pos[j + 1]])
                    if j < len(mol_pos) - 2:
                        mol_idx = input_ids[mol_pos[j + 1]]
                        new_input_ids.extend(
                            torch.ones([mol_sizes[-mol_idx - 1] - 1]) * mol_idx
                        )
                        if mol_pos[j + 1] < original_input_ids_len:
                            input_ids_len += mol_sizes[-mol_idx - 1] - 1
                elif self.pool_mode == "qformer":
                    new_input_ids.extend(input_ids[mol_pos[j] : mol_pos[j + 1]])
                    if j < len(mol_pos) - 2:
                        mol_idx = input_ids[mol_pos[j + 1]]
                        new_input_ids.extend(
                            torch.ones([self.embedding_length - 1]) * mol_idx
                        )
                        if mol_pos[j + 1] < original_input_ids_len:
                            input_ids_len += self.embedding_length - 1
                else:
                    raise Exception(f"Pool mode {self.pool_mode} not supported yet")

            input_ids = torch.tensor(new_input_ids)

        input_ids = input_ids.to(dtype=torch.int64)

        labels = input_ids.clone()
        labels[:input_ids_len] = IGNORE_INDEX
        labels[labels < 0] = IGNORE_INDEX

        num_labels = torch.zeros_like(input_ids, dtype=torch.float)
        num_labels[:] = IGNORE_INDEX

        # if nums is not None and len(nums) > 0:
        #     nums = torch.tensor(nums, dtype=torch.float)

        #     if torch.sum(label_poss) == nums.shape[0]:
        #         num_labels[label_poss] = nums

        # # labels = torch.stack([labels, num_labels], dim=-1)

        return dict(
            input_ids=input_ids, labels=labels, smiless=smiless, num_labels=num_labels
        )

    def collater(self, samples):
        return self.data_collater(samples)

    def collate(self, samples):
        return self.collater(samples)


def preprocess_item(item, use_pbc=False):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    if item.pos is None:
        item.pos = torch.zeros([N, 3], dtype=torch.float)
        item.mask3d = torch.tensor([1.0]).bool()
    else:
        item.mask3d = torch.tensor([0.0]).bool()
        item.pos = item.pos - torch.mean(item.pos, dim=0, keepdim=True)

    if use_pbc and (not hasattr(item, "pbc") or item.pbc is None):
        item.pbc = torch.zeros([3], dtype=torch.bool)

    if use_pbc and (not hasattr(item, "cell") or item.cell is None):
        item.cell = torch.zeros([3, 3], dtype=torch.float)

    return item


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    assert torch.all(x[x > 0] >= 3)
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def collator(
    items, multi_hop_max_dist=20, spatial_pos_max=20, max_node_num=None, use_pbc=False
):
    items = [
        (
            item.attn_bias,
            item.spatial_pos,
            item.in_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.pos,
            item.mask3d,
            item.pbc if use_pbc else None,
            item.cell if use_pbc else None,
        )
        for item in items
    ]
    (
        attn_biases,
        spatial_poses,
        in_degrees,
        xs,
        edge_inputs,
        poses,
        mask3ds,
        pbcs,
        cells,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")

    if max_node_num is None:
        max_node_num = max(i.size(0) for i in xs)
    num_atoms = torch.tensor([int(i.size(0)) for i in xs]).long()
    max_dist = max(i.size(-2) for i in edge_inputs)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    mask3d = torch.cat([i for i in mask3ds])
    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    pbc = torch.cat([i.unsqueeze(0) for i in pbcs], dim=0) if use_pbc else None
    cell = torch.cat([i.unsqueeze(0) for i in cells], dim=0) if use_pbc else None

    node_type_edges = []
    for idx in range(len(items)):
        node_atom_type = items[idx][3][:, 0]
        n_nodes = items[idx][3].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb_last(node_atom_edge)
        node_type_edges.append(node_atom_edge.long())

    node_type_edge = torch.cat(node_type_edges)

    return dict(
        num_atoms=num_atoms,
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        x=x,
        edge_input=edge_input,
        pos=pos,
        mask3d=mask3d,
        node_type_edge=node_type_edge,
        attn_edge_type=None,
        pbc=pbc,
        cell=cell,
    )


def smiles2graph_removeh(smiles_string, pos=None):
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

    if pos is None:
        graph["pos"] = None
    else:
        graph["pos"] = torch.tensor(pos)

    return graph


def batch_collater_for_graphormer(
    smiless: List[str],
    poses: List[Any],
    use_pbc: bool = False,
    max_node_num: Optional[int] = None,
    multi_hop_max_dist: Optional[int] = 20,
):
    if type(smiless[0]) == Data:
        graphs = [preprocess_item(smiles, use_pbc=use_pbc) for smiles in smiless]
    else:
        graphs = [
            preprocess_item(Data(**smiles2graph_removeh(smiles, pos)), use_pbc=use_pbc)
            for smiles, pos in zip(smiless, poses)
        ]

    return collator(
        graphs,
        use_pbc=use_pbc,
        max_node_num=max_node_num,
        multi_hop_max_dist=multi_hop_max_dist,
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int
    use_pp: bool
    add_mfm: bool
    use_pbc: bool
    use_global_padding: bool
    model_max_length: int
    molecule_max_size: int
    max_num_mol_per_sample: int
    multi_hop_max_dist: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, smiless, num_labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "smiless", "num_labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        num_labels = torch.nn.utils.rnn.pad_sequence(
            num_labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        if self.use_global_padding:
            batch_size, max_seq_len = input_ids.size()
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full(
                        [batch_size, self.model_max_length - max_seq_len],
                        self.pad_token_id,
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    ),
                ],
                dim=-1,
            )
            labels = torch.cat(
                [
                    labels,
                    torch.full(
                        [batch_size, self.model_max_length - max_seq_len],
                        IGNORE_INDEX,
                        device=labels.device,
                        dtype=labels.dtype,
                    ),
                ],
                dim=-1,
            )
            num_labels = torch.cat(
                [
                    num_labels,
                    torch.full(
                        [batch_size, self.model_max_length - max_seq_len],
                        IGNORE_INDEX,
                        device=num_labels.device,
                        dtype=num_labels.dtype,
                    ),
                ],
                dim=-1,
            )

        batched_molecules = {
            "x": None,
            "attn_bias": None,
            "spatial_pos": None,
            "in_degree": None,
            "out_degree": None,
            "edge_input": None,
            "num_atoms": None,
        }
        smiles = []
        pos = []
        for smis in smiless:
            if len(smis) > 0 and type(smis[0]) == str:
                smiles.extend(smis)
                pos.extend([None for _ in range(len(smis))])
            elif len(smis) > 0 and type(smis[0]) == list:
                smiles.extend(smis[0])
                pos.extend(smis[1])
            elif len(smis) > 0 and type(smis[0]) == Data:
                smiles.extend(smis)
                pos.extend([None for _ in range(len(smis))])

        if self.use_global_padding:
            smiles_for_padding = smiles[0]
            pos_for_padding = pos[0]
            num_mols = len(smiles)
            for i in range(num_mols, self.max_num_mol_per_sample * batch_size):
                smiles.append(smiles_for_padding)
                pos.append(pos_for_padding)
            batched_molecules = batch_collater_for_graphormer(
                smiles,
                pos,
                self.use_pbc,
                self.molecule_max_size,
                self.multi_hop_max_dist,
            )
        else:
            batched_molecules = batch_collater_for_graphormer(smiles, pos, self.use_pbc)
        batched_molecules["out_degree"] = batched_molecules["in_degree"]

        # pad edge_input
        if self.use_global_padding:
            edge_input = batched_molecules["edge_input"]
            if edge_input.size()[3] < self.multi_hop_max_dist:
                (
                    batch_size,
                    max_num_atom,
                    _,
                    max_dist,
                    num_edge_features,
                ) = edge_input.size()
                edge_input = torch.cat(
                    [
                        edge_input,
                        torch.zeros(
                            [
                                batch_size,
                                max_num_atom,
                                max_num_atom,
                                self.multi_hop_max_dist - max_dist,
                                num_edge_features,
                            ],
                            device=edge_input.device,
                            dtype=edge_input.dtype,
                        ),
                    ],
                    dim=3,
                )
                batched_molecules["edge_input"] = edge_input

        if self.use_pp:
            return_input = [
                input_ids,
                input_ids.ne(self.pad_token_id),
                labels,
                batched_molecules["x"],
                batched_molecules["in_degree"],
                batched_molecules["out_degree"],
                batched_molecules["attn_bias"],
                batched_molecules["spatial_pos"],
                batched_molecules["edge_input"],
                batched_molecules["num_atoms"],
                batched_molecules["pos"],
                batched_molecules["mask3d"],
                batched_molecules["node_type_edge"],
            ]
            if self.use_pbc:
                return_input.extend(
                    [batched_molecules["pbc"], batched_molecules["cell"]]
                )
            return (
                return_input,
                (
                    labels,
                    input_ids.ne(self.pad_token_id),
                    num_labels,
                ),
            )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                num_labels=num_labels,
                attention_mask=input_ids.ne(self.pad_token_id),
                **batched_molecules,
            )


if __name__ == "__main__":
    pass
