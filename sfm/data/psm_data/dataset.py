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

        x = data["x"].to(torch.int64)
        coords = data["coords"].to(torch.float32)

        x = convert_to_single_emb(x)

        data["sample_type"] = 0
        data["token_type"] = x
        data["idx"] = idx

        coords = coords - coords.mean(dim=0, keepdim=True)
        data["coords"] = coords

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64)
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)

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
                for key, _ in tqdm(self.data_name_to_txn[path_name].cursor()):
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
        data["stress"] = torch.tensor(data["info"]["stress"], dtype=torch.float64)
        data["forces"] = torch.tensor(data["forces"], dtype=torch.float64)
        data["energy"] = torch.tensor([data["info"]["energy"] / x.size()[0]])

        return data

    def __len__(self):
        return len(self.index_to_dataset_name)


class AFDBLMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        lmdb_path: Optional[str],
    ):
        self.lmdb_path = lmdb_path
        self.vocab = Alphabet()

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
        x = convert_to_single_emb(x)

        data["sample_type"] = 0
        data["token_type"] = x
        data["idx"] = idx

        coords = coords - coords.mean(dim=0, keepdim=True)
        data["coords"] = coords

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64)
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)

        return data

    def __len__(self) -> int:
        return len(self.keys)


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x
