# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import os
import os.path as osp
import pickle
import pickle as pkl
import random
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import lmdb
import numpy as np
import torch
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from sfm.data.dataset import FoundationModelDataset
from sfm.data.mol_data import algos
from sfm.data.prot_data.dataset import LMDBDataset
from sfm.data.prot_data.sequence_masking import masking_registry
from sfm.data.prot_data.spatial_noise import noise_registry
from sfm.data.prot_data.util import bstr2obj
from sfm.data.prot_data.vocalubary import Alphabet


class PM6FullLMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        lmdb_path: Optional[str],
    ):
        self.lmdb_path = lmdb_path
        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None

    def _init_db(self):
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self._sizes, self._keys = metadata["size"], metadata["keys"]

    @property
    def env(self):
        if self._env is None:
            self._init_db()
        return self._env

    @property
    def txn(self):
        if self._txn is None:
            self._init_db()
        return self._txn

    @property
    def sizes(self):
        if self._sizes is None:
            self._init_db()
        return self._sizes

    @property
    def keys(self):
        if self._keys is None:
            self._init_db()
        return self._keys

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(666).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(self.lmdb_path)
        dataset_train._keys = [self._keys[idx] for idx in training_indices]
        dataset_train._sizes = [self._sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.lmdb_path)
        dataset_val._keys = [self._keys[idx] for idx in validation_indices]
        dataset_val._sizes = [self._sizes[idx] for idx in validation_indices]

        return dataset_train, dataset_val

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)

        x = data["node_feat"].to(torch.int64)[:, 0]
        coords = data["coords"].to(torch.float32)

        x = convert_to_single_emb(x)

        data["sample_type"] = 0
        data["token_type"] = x
        data["idx"] = idx

        coords = coords - coords.mean(dim=0, keepdim=True)
        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)

        data = self.generate_2dgraphfeat(data)

        return data

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data["edge_feat"], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.tensor(data["node_feat"], dtype=torch.long)

        data["edge_input"] = torch.tensor(edge_input, dtype=torch.long)
        data["attn_edge_type"] = attn_edge_type
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = torch.tensor(indgree, dtype=torch.long)
        data["spatial_pos"] = torch.tensor(spatial_pos, dtype=torch.long)

        return data

    def __len__(self) -> int:
        return len(self.keys)


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
                for key, _ in self.data_name_to_txn[path_name].cursor():
                    self.index_to_dataset_name.append([path_name, key.decode()])

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data_name, key = self.index_to_dataset_name[idx]
        data = pkl.loads(self.data_name_to_txn[data_name].get(key.encode()))
        numbers = data.pop("numbers")
        x = torch.tensor(numbers, dtype=torch.long).unsqueeze(-1)
        positions = data.pop("positions")

        data["sample_type"] = 1
        data["coords"] = torch.tensor(positions, dtype=torch.float64)
        data["token_type"] = convert_to_single_emb(x)
        data["idx"] = idx

        data["cell"] = torch.tensor(data["cell"], dtype=torch.float64)
        data["pbc"] = torch.tensor(data["pbc"], dtype=torch.bool)

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

        cell_corner_pos = torch.matmul(cell_corner_pos_matrix, data["cell"])
        data["coords"] = torch.cat(
            [data["coords"], cell_corner_pos], dim=0
        )  # expand pos with cell corners
        data["num_atoms"] = int(x.size()[0] - 8)
        data["forces"] = torch.cat(
            [
                torch.tensor(data["forces"], dtype=torch.float64),
                torch.zeros([8, 3], dtype=torch.float64),
            ],
            dim=0,
        )  # expand forces for cell corners
        data["energy"] = torch.tensor([data["info"]["energy"] / x.size()[0]])
        data["stress"] = torch.tensor(data["info"]["stress"], dtype=torch.float64)

        data = self.generate_2dgraphfeat(data)

        return data

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result = (
            torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
        )
        edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)

        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"],
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["edge_input"] = edge_input
        data["attn_edge_type"] = attn_edge_type
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = torch.tensor(indgree, dtype=torch.long)
        data["spatial_pos"] = torch.tensor(spatial_pos, dtype=torch.long)

        return data

    def __len__(self):
        return len(self.index_to_dataset_name)


class AFDBLMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        lmdb_path: Optional[str],
    ):
        self.lmdb_path = lmdb_path
        self.vocab = {
            "L": 129,
            "A": 130,
            "G": 131,
            "V": 132,
            "S": 133,
            "E": 134,
            "R": 135,
            "T": 136,
            "I": 137,
            "D": 138,
            "P": 139,
            "K": 140,
            "Q": 141,
            "N": 142,
            "F": 143,
            "Y": 144,
            "M": 145,
            "H": 146,
            "W": 147,
            "C": 148,
            "X": 149,
            "B": 150,
            "U": 151,
            "Z": 152,
            "O": 153,
            "-": 154,
            ".": 155,
            "<mask>": 156,
            "<cls>": 157,
            "<eos>": 158,
            "<unk>": 159,
        }

        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None

    def _init_db(self):
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self._sizes, self._keys = metadata["sizes"], metadata["keys"]

    @property
    def env(self):
        if self._env is None:
            self._init_db()
        return self._env

    @property
    def txn(self):
        if self._txn is None:
            self._init_db()
        return self._txn

    @property
    def sizes(self):
        if self._sizes is None:
            self._init_db()
        return self._sizes

    @property
    def keys(self):
        if self._keys is None:
            self._init_db()
        return self._keys

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)

        x = torch.tensor(
            [self.vocab.tok_to_idx[tok] for tok in data["aa"]], dtype=torch.int64
        )
        # CA atom positions, assume all values are valid.
        coords = data["coords"][:, 1, :].to(torch.float32)

        data["sample_type"] = 2
        data["token_type"] = x
        data["idx"] = idx

        coords = coords - coords.mean(dim=0, keepdim=True)
        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)

        data = self.generate_2dgraphfeat(data)

        return data

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(666).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(self.lmdb_path)
        dataset_train._keys = [self._keys[idx] for idx in training_indices]
        dataset_train._sizes = [self._sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.lmdb_path)
        dataset_val._keys = [self._keys[idx] for idx in validation_indices]
        dataset_val._sizes = [self._sizes[idx] for idx in validation_indices]

        return dataset_train, dataset_val

    # protein does not have 2dgraph, create one for mixing data
    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result = (
            torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
        )
        edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long).cpu().numpy()
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"],
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["edge_input"] = torch.tensor(edge_input, dtype=torch.long)
        data["attn_edge_type"] = attn_edge_type
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = torch.tensor(indgree, dtype=torch.long)
        data["spatial_pos"] = torch.tensor(spatial_pos, dtype=torch.long)

        return data

    def __len__(self) -> int:
        return len(self.keys)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x
