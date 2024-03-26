# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})

import copy

import numpy as np
import torch
from matbench import MatbenchBenchmark
from torch_geometric.data import Data

from . import algos


class MatbenchDataset:
    def __init__(self, task_name, fold_id, split):
        self.task_name = task_name
        self.task = MatbenchBenchmark(autoload=False).tasks_map[self.task_name]
        self.fold_id = fold_id
        self.task.load()
        if split == "train_val":
            train_val_inputs, train_val_outputs = self.task.get_train_and_val_data(
                self.fold_id
            )
            self.data = self._preprocess(train_val_inputs, train_val_outputs)
        elif split == "test":
            test_inputs, test_outputs = self.task.get_test_data(
                self.fold_id, include_target=True
            )
            self.data = self._preprocess(test_inputs, test_outputs)

    def _preprocess(self, mat_inputs, mat_outputs=None):
        data_list = []
        if mat_outputs is None:
            mat_outputs = [None for _ in mat_inputs]
        for i, (mat_data, mat_y) in enumerate(zip(mat_inputs, mat_outputs)):
            pos_list = []
            x_list = list(mat_data.atomic_numbers)
            num_atoms = 0
            data = Data()
            for atom in mat_data.as_dict()["sites"]:
                pos_list.append(atom["xyz"])
            pbc_list = list(mat_data.as_dict()["lattice"]["pbc"])
            cell = torch.as_tensor(mat_data.as_dict()["lattice"]["matrix"])
            num_atoms = len(mat_data.as_dict()["sites"])

            data.num_atoms = num_atoms
            data.cell = cell
            data.x = torch.as_tensor(x_list).unsqueeze(-1)
            data.pos = torch.as_tensor(pos_list)
            data.pbc = torch.as_tensor(pbc_list)
            data.y = torch.tensor(
                [float(mat_y) if mat_y is not None else np.nan], dtype=torch.float
            )
            data.idx = i

            edge_len = 0
            edge_feat_len = 3
            data["edge_attr"] = torch.zeros([edge_len, edge_feat_len], dtype=torch.long)
            data["edge_index"] = torch.zeros([2, edge_len], dtype=torch.long)
            assert list(data["edge_attr"].size()) == [edge_len, edge_feat_len]
            assert list(data["edge_index"].size()) == [2, edge_len]
            assert list(data["x"].size()) == [num_atoms, 1]
            assert list(data["pos"].size()) == [num_atoms, 3]

            data_list.append(data)

        return data_list

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.data[idx]
        return preprocess_item(copy.deepcopy(item))

    def __len__(self):
        return len(self.data)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index.to(torch.int64), item.x

    N = x.size(0)
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
    spatial_pos = torch.from_numpy(shortest_path_result)

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())

    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

    item.x = convert_to_single_emb(x)
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree
    item.edge_input = torch.from_numpy(edge_input).long()

    return item
