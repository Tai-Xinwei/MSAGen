# -*- coding: utf-8 -*-
import os
import random
from dataclasses import dataclass
from typing import Any, List, Union

import torch

from sfm.data.dataset import Batch, Data, InMemoryFoundationModelDataset
from sfm.logging import logger

from .tokenizer import SciTokenizer


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


def collate_fn(samples: List[dict], vocab: SciTokenizer):
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


def collate_fn_pp(samples: List[dict], vocab: SciTokenizer):
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


class SciDataset(InMemoryFoundationModelDataset):
    def __init__(self, dict_path, data_path, args):
        self.vocab = SciTokenizer.from_file(dict_path)
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


if __name__ == "__main__":

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Namespace()
    args.dict_path = "/mnt/protein/scigpt/sample/dict.txt"
    args.train_data_path = "/mnt/protein/scigpt/sample/scigpt.train.txt"

    print(args)
    print("=================")
    print("Test sci dataset")
    dataset = SciDataset(args.dict_path, args.train_data_path)
    print(len(dataset))
    print(dataset[12])
    print()
    batch_dataset = BatchedDataDataset(dataset)
    print(
        batch_dataset.collate([dataset[0], dataset[1111], dataset[2222], dataset[3333]])
    )
    # print()
    # print(batch_dataset.collate([dataset[6123], dataset[6001], dataset[6299], dataset[6599]]))
