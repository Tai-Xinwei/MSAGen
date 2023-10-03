# -*- coding: utf-8 -*-
import os
import random
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from sfm.data.dataset import Batch, Data, InMemoryFoundationModelDataset
from sfm.data.sci_data import SFMDecTokenizer
from sfm.logging import logger


# allow pad_num to be int or float
def pad_1d_unsqueeze(
    x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
):
    # (N) -> (1, padlen)
    xlen = x.size(0)
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen], pad_num, dtype=x.dtype)
    new_x[start : start + xlen] = x
    x = new_x
    return x.unsqueeze(0)


def collate_fn(samples: List[dict], vocab: SFMDecTokenizer):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    # max_tokens = Nres+2 (<cls> and <eos>)
    max_tokens = max(len(s["tokens"]) for s in samples)

    batch = dict()

    # naa = [Nres+2, ...] for each sample
    batch["ntokens"] = torch.tensor(
        [len(s["tokens"]) for s in samples], dtype=torch.long
    )

    # (Nres+2,) -> (B, Nres+2)
    batch["x"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    return batch


def collate_fn_pp(samples: List[dict], vocab: SFMDecTokenizer):
    """
    Overload BaseWrapperDataset.collater
    May be future changes need config

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """
    max_tokens = max(len(s["tokens"]) for s in samples)

    input_ids = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]), max_tokens, 0, vocab.padding_idx
            )
            for s in samples
        ]
    )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=vocab.padding_idx
    )
    input = tuple([input_ids, input_ids.ne(vocab.padding_idx)])
    labels = input
    return (input, labels)


class BatchedDataDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        args=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.args = args
        self.vocab = dataset.vocab

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        if self.args is None or self.args.pipeline_model_parallel_size == 0:
            return collate_fn(samples, self.vocab)
        else:
            return collate_fn_pp(samples, self.vocab)

    def num_tokens(self, index: int) -> int:
        return self.dataset.sizes[index]


class ProcessedSciDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, padding_idx, max_len: int):
        super().__init__()
        self.data = np.load(path, mmap_mode="r")
        processed_seq_len = self.data.shape[1]
        if processed_seq_len % max_len != 0:
            raise ValueError(
                f"processed_seq_len {processed_seq_len} is not divisible by max_len {max_len}"
            )
        self.replicate = processed_seq_len // max_len
        self.max_len = max_len
        self.padding_idx = padding_idx

        logger.info(
            f"Loaded {path} with shape {self.data.shape}, max_len {max_len}, replicate {self.replicate}"
        )

    def __getitem__(self, index):
        index, offset = divmod(index, self.replicate)
        data = self.data[index][offset * self.max_len : (offset + 1) * self.max_len]
        return torch.from_numpy(data.astype(np.int64))

    def __len__(self):
        return self.data.shape[0] * self.replicate

    def collate(self, samples):
        input_ids = torch.stack(samples, dim=0)
        padding_mask = input_ids.ne(self.padding_idx)
        input = tuple([input_ids, padding_mask])
        labels = input
        return (input, labels)


class SciDataset(InMemoryFoundationModelDataset):
    def __init__(self, dict_path, data_path, args):
        self.vocab = SFMDecTokenizer.from_file(dict_path)
        self.args = args
        self.max_position_embeddings = args.max_position_embeddings

        with open(data_path, "r") as f:
            lines = f.read().splitlines()
            self.data = list(
                filter(lambda x: len(x) <= self.max_position_embeddings, lines)
            )
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = dict()
        item["tokens"] = self.vocab.encode(self.data[index])
        return item
