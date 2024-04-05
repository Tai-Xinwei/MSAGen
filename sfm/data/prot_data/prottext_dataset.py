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
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm import tqdm

from sfm.data.prot_data.util import bstr2obj
from sfm.logging.loggers import logger

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class ProteinTextDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        in_memory: bool,
        model_max_length: int,
        protein_max_size: int,
        pad_token_id: int,
        max_pro_per_sample: int = 1,
        pool_mode: Optional[str] = "full",
        embedding_length: int = 20,
        num_token_id: int = 32003,
        use_pp: bool = True,
        use_pbc: bool = False,
        local_rank: int = 0,
        use_global_padding: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.in_memory = in_memory
        self.model_max_length = model_max_length
        self.protein_max_size = protein_max_size
        self.pool_mode = pool_mode
        self.embedding_length = embedding_length
        self.num_token_id = num_token_id
        self.local_rank = local_rank

        self.max_pro_per_sample = max_pro_per_sample

        self.len = 0
        self.index_to_key_map = []
        self.in_memory_data = {}
        self.read_txns = {}
        self.read_envs = {}
        self.weight_dict = {}
        self.dataset_count = {}
        self.dataset_filtered = {}

        self.env = lmdb.open(
            str(self.data_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self.len, self.keys = metadata["sizes"], metadata["keys"]

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        key = self.keys[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")

        data = pkl.loads(value)
        return data

    def collater(self, samples):
        return self.data_collater(samples)

    def collate(self, samples):
        return self.collater(samples)


if __name__ == "__main__":
    pass
