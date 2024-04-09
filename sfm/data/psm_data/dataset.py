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
        root: Optional[str] = None,
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


class ProteinLMDBDataset(FoundationModelDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, data_path: str) -> None:
        super().__init__()
        assert (
            data_path and Path(data_path).is_dir()
        ), f"Processed file not found: {data_path}"
        self.lmdb_path = data_path
        self.vocab = Alphabet()

        self.env = lmdb.open(
            str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self.sizes, self.keys = metadata["sizes"], metadata["keys"]
        # self.comment = metadata["comment"]
        # self.filter_indices_by_size(
        #     indices=np.array(range(len(self.keys))), max_sizes=self.args.max_length
        # )

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
