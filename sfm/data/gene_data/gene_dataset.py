# -*- coding: utf-8 -*-
import bisect
import logging
import os

import torch

logging.getLogger().setLevel(logging.ERROR)

import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import lmdb
import numpy as np
from torch.utils.data import Dataset

from sfm.data.prot_data.util import bstr2obj
from sfm.logging.loggers import logger


class GeneDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        pad_token_id: int,
        num_token_id: int = 0,
        max_len: int = 16384,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_token_id = num_token_id
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.len = 0
        self.index_to_key_map = []
        self.read_txns = {}
        self.read_envs = {}
        self.dataset_count = {}
        self.dataset_filtered = {}
        env_list = []
        txn_list = []
        file_list = []
        keys_list = []
        data_size_list = []
        data_size_list.append(0)

        processed_seq_len = None
        if self.data_path.endswith(".lmdb"):
            file_list.append(self.data_path)
        else:
            for file_name in os.listdir(self.data_path):
                if file_name.endswith(".lmdb"):
                    file_list.append(os.path.join(self.data_path, file_name))
        for file_path in file_list:
            env = lmdb.open(
                file_path, subdir=True, readonly=True, lock=False, readahead=False
            )
            logger.info(f"Load {file_path} into the dataset.")
            env_list.append(env)
            txn = env.begin(write=False)
            txn_list.append(txn)
            metadata = bstr2obj(txn.get("metadata".encode()))
            cur_processed_seq_len = metadata["processed_seq_len"]
            cur_keys = metadata["keys"]
            keys_list.append(cur_keys)

            if processed_seq_len is not None:
                if cur_processed_seq_len != processed_seq_len:
                    raise ValueError(
                        f"{file_path} ({cur_processed_seq_len}) is inconsistent with processed_seq_len in other files ({processed_seq_len})"
                    )
            else:
                processed_seq_len = cur_processed_seq_len
        self.replicate = processed_seq_len // max_len
        for keys in keys_list:
            self.len += len(keys) * self.replicate
            data_size_list.append(self.len)

        self.env_list = env_list
        self.txn_list = txn_list
        self.data_size_list = data_size_list
        self.keys_list = keys_list

        # self.env = lmdb.open(
        #     str(self.data_path), subdir=True, readonly=True, lock=False, readahead=False
        # )
        # self.txn = self.env.begin(write=False)
        # metadata = bstr2obj(self.txn.get("metadata".encode()))
        # self.len, self.keys = metadata["size"], metadata["keys"]
        print(len(self.keys_list[0]))
        print(self.replicate)
        logger.info(f"Dataset size: {self.len}")

    def __len__(self):
        return self.len

    def get_real_index(self, index):
        list_index = bisect.bisect_right(self.data_size_list, index)
        return list_index - 1, index - self.data_size_list[list_index - 1]

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        list_index, data_index = self.get_real_index(index)
        data_index, offset = divmod(data_index, self.replicate)
        key = self.keys_list[list_index][data_index]
        # print(f"list_index{list_index}  key{key}")
        value = self.txn_list[list_index].get(str(key).encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        value = np.frombuffer(value, dtype=np.uint16)[
            offset * self.max_len : (offset + 1) * self.max_len
        ]

        # value = self.txn_list[list_index].get(str(key).encode())

        # value = np.frombuffer(value, dtype=np.uint16)
        input_ids = torch.from_numpy(value.astype(np.int64))

        labels = input_ids.clone()
        return input_ids, labels

    def collate(self, samples):
        input_ids_list, labels_list = zip(*samples)
        # print(type(input_ids_list))
        # print(input_ids_list)
        # print(input_ids_list[0].shape)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        padding_mask = input_ids.ne(self.pad_token_id)

        input = tuple([input_ids, padding_mask])
        return (input, labels)


if __name__ == "__main__":
    pass
