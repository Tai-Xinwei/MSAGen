# -*- coding: utf-8 -*-
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List

import lmdb
from torch.utils.data import Dataset


@dataclass
class Data:
    pass


@dataclass
class Batch(Data):
    batch_size: int


class FoundationModelDataset(Dataset[Data]):
    def __init__(self) -> None:
        super().__init__()

    def collate(self, batch: List[Data]) -> Data:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class InMemoryFoundationModelDataset(FoundationModelDataset):
    def __init__(self, data: list) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index) -> Data:
        return self.data[index]


class LMDBFoundationModelDataset(FoundationModelDataset):
    def __init__(self, lmdb_path: Path) -> None:
        super().__init__()
        self.read_env = lmdb.open(lmdb_path)
        self.read_txn = self.read_env.begin(write=False)
        self.key_list = []
        for key, _ in self.read_txn.cursor():
            self.key_list.append(key.decode())

    def __getitem__(self, index) -> Data:
        return pkl.loads(self.read_txn.get(self.key_list[index].encode()))

    def __del__(self):
        if hasattr(self, "read_env"):
            self.read_env.close()
