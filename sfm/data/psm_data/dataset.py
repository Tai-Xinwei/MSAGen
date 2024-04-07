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
from sfm.data.prot_data.dataset import LMDBDataset
from sfm.data.prot_data.sequence_masking import masking_registry
from sfm.data.prot_data.spatial_noise import noise_registry
from sfm.data.prot_data.util import bstr2obj


class PM6FullLMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        max_node: Optional[int] = 512,
    ):
        self.root = root
        self.max_node = max_node

        self.env = lmdb.open(
            str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.sizes, self.keys = metadata["sizes"], metadata["keys"]

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)
        item = {"id": idx, **data}
        # item keys
        # {'name', 'size', "atom_coords", "token_type", "id"}

        x = item["node_feat"].to(torch.int64)
        coords = item["coords"].to(torch.float32)
        data.idx = idx

        x = convert_to_single_emb(x)

        data.x = x
        data.coords = coords

        if data.coords is not None:
            data.coords = data.coords - data.coords.mean(dim=0, keepdim=True)

        return data

    def __len__(self) -> int:
        if self._indices is None:
            return self.total_len
        else:
            return len(self._indices)


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
        item = pkl.loads(self.data_name_to_txn[data_name].get(key.encode()))
        numbers = item.pop("numbers")
        item["x"] = torch.tensor(numbers, dtype=torch.long).unsqueeze(-1)
        positions = item.pop("positions")
        item["pos"] = torch.tensor(positions, dtype=torch.float64)
        item["edge_attr"] = torch.zeros([0, 3], dtype=torch.long)
        item["edge_index"] = torch.zeros([2, 0], dtype=torch.long)
        item["cell"] = torch.tensor(item["cell"], dtype=torch.float64)
        item["pbc"] = torch.tensor(item["pbc"], dtype=torch.bool)
        item["idx"] = idx
        item["y"] = torch.tensor([item["info"]["energy"] / item["x"].size()[0]])
        item["stress"] = torch.tensor(item["info"]["stress"], dtype=torch.float64)
        item["forces"] = torch.tensor(item["forces"], dtype=torch.float64)

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
        shortest_path_result = (
            torch.full(adj.size(), 511, dtype=torch.long, device=x.device).cpu().numpy()
        )
        edge_input = (
            torch.zeros([N, N, 0, 3], dtype=torch.long, device=x.device).cpu().numpy()
        )
        spatial_pos = torch.from_numpy((shortest_path_result)).long()

        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        item.x = convert_to_single_emb(x)
        item.attn_bias = attn_bias
        item.attn_edge_type = attn_edge_type
        item.spatial_pos = spatial_pos
        item.in_degree = adj.long().sum(dim=1).view(-1)
        item.out_degree = item.in_degree
        item.edge_input = torch.from_numpy(edge_input).long()

        return item

    def __len__(self):
        return len(self.index_to_dataset_name)


class ProteinLMDBDataset(LMDBDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any) -> None:
        # here calls self.set_default_args(args)
        super().__init__(args)
        self.seed = self.args.seed
        self.seq_masking_method = self.args.seq_masking_method
        self.noise_method = self.args.noise_method
        self.pos_noise = self.args.pos_noise
        self.ang_noise = self.args.ang_noise

    def set_default_args(self, args):
        args.data_path = getattr(args, "data_path", None)
        args.seed = getattr(args, "seed", "2023")
        args.max_length = getattr(args, "max_length", 1024)
        args.seq_masking_method = getattr(args, "seq_masking_method", "transformerM")

        args.mask_prob = getattr(args, "mask_prob", 0.15)
        args.leave_unmasked_prob = getattr(args, "leave_unmasked_prob", 0.1)
        args.random_token_prob = getattr(args, "random_token_prob", 0.1)
        args.mask_multiple_length = getattr(args, "mask_multiple_length", 1)
        args.mask_stdev = getattr(args, "mask_stdev", 0.0)

        args.noise_method = getattr(args, "noise_method", "normal")
        args.pos_noise = getattr(args, "pos_noise", True)
        args.ang_noise = getattr(args, "ang_noise", True)

        args.coord_noise_mean = getattr(args, "coord_noise_mean", 0.0)
        args.coord_noise_stdev = getattr(args, "coord_noise_stdev", 0.1)
        args.angle_noise_mean = getattr(args, "angle_noise_mean", 0.0)
        args.angle_noise_stdev = getattr(args, "angle_noise_stdev", 0.003)

        return args

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
        dataset_train = self.__class__(self.args)
        dataset_train.keys = [self.keys[idx] for idx in training_indices]
        dataset_train.sizes = [self.sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.args)
        dataset_val.keys = [self.keys[idx] for idx in validation_indices]
        dataset_val.sizes = [self.sizes[idx] for idx in validation_indices]

        if sort:
            dataset_train.__sort__()
            dataset_val.__sort__()

        return dataset_train, dataset_val

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)
        item = {"id": index, **data}
        # item keys
        # {'name', 'size', "pos": np.ndarray(N, 37, 3, dtype=float32), "pos_mask": np.ndarray(N, 37, 3, dtype=int32), "ang": np.ndarray(N, 9, dtype=float32), "ang_mask": np.ndarray(N, 9, dtype=int32), "aa", "id": int}

        """
        - convert string sequence to int index
        """
        tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        item["aa"] = np.array(tokens, dtype=np.int64)
        item["seq_length"] = len(tokens)

        """
        - mask the sequence in different ways
        """
        int(hash((self.seed, index)) % 1e6)
        assert (
            "mask_idx" not in item
        ), "Item already contains mask_idx key, this is not expected!"

        item["ang"] = item["ang"] / 180.0 * torch.pi  # + torch.pi
        item["coord"] = item["pos"]

        return item

    def __len__(self) -> int:
        return len(self.keys)

    def size(self, index: int) -> int:
        sz = self.sizes[index]
        if self.vocab.prepend_bos:
            sz += 1
        if self.vocab.append_eos:
            sz += 1
        raise sz

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x
