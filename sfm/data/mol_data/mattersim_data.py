# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

import copy
import os
import pickle as pkl
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from . import algos


class MatterSimDataset:
    def __init__(self, data_path, split=None):
        self.data_name_to_lmdb = {}
        self.data_name_to_txn = {}
        self.index_to_dataset_name = []
        self.data_path = data_path
        for path_name in os.listdir(self.data_path):
            if os.path.isdir(f"{self.data_path}/{path_name}"):
                if split is None:
                    lmdb_path = f"{self.data_path}/{path_name}"
                else:
                    lmdb_path = f"{self.data_path}/{path_name}/{split}"
                self.data_name_to_lmdb[path_name] = lmdb.open(lmdb_path)
                self.data_name_to_txn[path_name] = self.data_name_to_lmdb[
                    path_name
                ].begin(write=False)
                for key, _ in tqdm(self.data_name_to_txn[path_name].cursor()):
                    self.index_to_dataset_name.append([path_name, key.decode()])

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data_name, key = self.index_to_dataset_name[idx]
        return preprocess_item(
            pkl.loads(self.data_name_to_txn[data_name].get(key.encode())), idx
        )

    def __len__(self):
        return len(self.index_to_dataset_name)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item, idx):
    numbers = item.pop("numbers") if "numbers" in item else item["node_feat"][:, 0]
    item["x"] = torch.tensor(numbers, dtype=torch.long)
    if "pbc" in item:
        item["x"] = torch.cat(
            [item["x"], torch.full([8], 128)], dim=-1
        )  # use 128 for unit cell corners
    item["x"] = item["x"].unsqueeze(-1)
    positions = item.pop("positions") if "positions" in item else item.pop("pos")
    item["pos"] = torch.tensor(positions, dtype=torch.float64)

    item["cell"] = (
        torch.tensor(item["cell"], dtype=torch.float64)
        if "cell" in item
        else torch.zeros([3, 3], dtype=torch.float64)
    )

    if "pbc" in item:
        cell_corner_pos_matrix = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        cell_corner_pos = torch.matmul(cell_corner_pos_matrix, item["cell"])
        item["pos"] = torch.cat(
            [item["pos"], cell_corner_pos], dim=0
        )  # expand pos with cell corners
        item["num_atoms"] = int(item["x"].size()[0] - 8)
        item["forces"] = torch.cat(
            [
                torch.tensor(item["forces"], dtype=torch.float64),
                torch.zeros([8, 3], dtype=torch.float64),
            ],
            dim=0,
        )  # expand forces for cell corners
        item["y"] = torch.tensor([item["info"]["energy"] / item["x"].size()[0]])
        item["stress"] = torch.tensor(item["info"]["stress"], dtype=torch.float64)
    else:
        item["num_atoms"] = int(item["x"].size()[0])
        item["y"] = torch.tensor([item["alpha_gap"]])
        item["stress"] = torch.zeros([3, 3], dtype=torch.float64)
        item["forces"] = torch.zeros_like(item["pos"])

    item["node_attr"] = (
        torch.tensor(item.pop("node_feat"), dtype=torch.long)
        if "node_feat" in item
        else torch.cat(
            [
                convert_to_single_emb(item["x"]),
                torch.zeros([item["x"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
    )
    item["edge_attr"] = (
        torch.tensor(item.pop("edge_feat"), dtype=torch.long)
        if "edge_feat" in item
        else torch.zeros([0, 3], dtype=torch.long)
    )
    item["edge_index"] = (
        torch.zeros([2, 0], dtype=torch.long)
        if "edge_index" not in item
        else torch.tensor(item["edge_index"], dtype=torch.long)
    )
    item["idx"] = idx
    item["protein_masked_pos"] = torch.zeros([item["x"].size()[0], 3], dtype=torch.bool)
    item["protein_masked_aa"] = torch.zeros([item["x"].size()[0], 1], dtype=torch.bool)

    item = Data(**item)

    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    N = x.size(0)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    if "pbc" in item:
        shortest_path_result = (
            torch.full(adj.size(), 511, dtype=torch.long, device=x.device).cpu().numpy()
        )
        edge_input = (
            torch.zeros([N, N, 0, 3], dtype=torch.long, device=x.device).cpu().numpy()
        )
    else:
        adj[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    item["pbc"] = (
        torch.tensor(item["pbc"], dtype=torch.bool)
        if "pbc" in item
        else torch.zeros([3], dtype=torch.bool)
    )

    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

    item.x = convert_to_single_emb(x)
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree
    item.edge_input = torch.from_numpy(edge_input).long()

    return item
