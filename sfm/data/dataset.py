# -*- coding: utf-8 -*-
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List

import lmdb
import numpy as np
from torch.utils.data import Dataset

from .data_utils import batch_by_size


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

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
        """

        fixed_shapes = self.get_batch_shapes()
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if bsz is None:
                    assert max_tokens is not None, "Must specify --max-tokens"
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif (
                    bsz >= required_batch_size_multiple
                    and bsz % required_batch_size_multiple != 0
                ):
                    bsz -= bsz % required_batch_size_multiple
                return bsz

            fixed_shapes = np.array(
                [
                    [adjust_bsz(bsz, num_tokens), num_tokens]
                    for (bsz, num_tokens) in fixed_shapes
                ]
            )

        try:
            num_tokens_vec = self.num_tokens_vec(indices).astype("int64")
        except NotImplementedError:
            num_tokens_vec = None

        return batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            num_tokens_vec=num_tokens_vec,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=None,
        )

    def num_tokens(self, index: int) -> int:
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        raise NotImplementedError

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example:

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None


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
