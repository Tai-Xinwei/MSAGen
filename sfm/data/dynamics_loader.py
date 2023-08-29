# -*- coding: utf-8 -*-

import math

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Sampler

from sfm.logging import logger


class DynamicDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_by_size_fn, num_tokens_fn, collate_fn, *args, **kwargs
    ):
        self.dataset = dataset
        self.batch_by_size_fn = batch_by_size_fn
        self.num_tokens_fn = num_tokens_fn
        self.collate_fn = collate_fn
        self.max_samples = kwargs.pop("max_sample", None)
        self.max_tokens = kwargs.pop("max_tokens", None)
        self.max_length = kwargs.pop("max_length", 1024)
        self.num_replicas = kwargs.pop("num_replicas", 1)
        self.rank = kwargs.pop("rank", 0)
        self.shuffle = kwargs.pop("shuffle", False)
        self.seed = kwargs.pop("seed", 0)
        self.epoch = kwargs.pop("epoch", 0)
        super().__init__(dataset, collate_fn=collate_fn, *args, **kwargs)

    def __set_dist_indices(self):
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        local_indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(local_indices) == self.num_samples
        return local_indices

    def __iter__(self):
        local_indices = self.__set_dist_indices()
        # logger.debug(f"local_indices: {len(local_indices)}")
        batches = self.batch_by_size_fn(
            indices=local_indices,
            max_length=self.max_length,
            num_tokens_fn=self.num_tokens_fn,
            max_tokens=self.max_tokens,
            max_samples=self.max_samples,
            required_batch_size_multiple=1,
        )

        for batch_indices in batches:
            # logger.debug(f"batch_indices: {len(batch_indices)}")
            batch_data = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch_data)


class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        sampler,
        num_tokens_fn,
        num_buckets=128,
        min_size=0,
        max_size=1000,
        max_tokens=None,
        max_sentences=None,
        drop_last=False,
    ):
        super(DynamicBatchSampler, self).__init__(sampler)
        self.sampler = sampler
        self.num_tokens_fn = num_tokens_fn
        self.num_buckets = num_buckets

        self.min_size = min_size
        self.max_size = max_size

        assert max_size <= max_tokens, "max_size should be smaller than max tokens"
        assert (
            max_tokens is not None or max_sentences is not None
        ), "max_tokens and max_sentences should not be null at the same time, please specify one parameter at least"
        self.max_tokens = max_tokens if max_tokens is not None else float("Inf")
        self.max_sentences = (
            max_sentences if max_sentences is not None else float("Inf")
        )
        self.drop_last = drop_last

    def is_batch_full(self, num_tokens, batch):
        if len(batch) == 0:
            return False
        if len(batch) == self.max_sentences:
            return True
        if num_tokens > self.max_tokens:
            return True
        return False

    def __iter__(self):
        buckets = [[] for _ in range(self.num_buckets)]
        sample_len = [0] * self.num_buckets

        for idx in self.sampler:
            idx_length = self.num_tokens_fn(idx)
            if not (self.min_size <= idx_length <= self.max_size):
                # logger.warning(
                #     "sentence at index {} of size {} exceeds max_tokens, the sentence is ignored".format(
                #         idx, idx_length
                #     )
                # )
                continue

            index_buckets = math.floor(
                (idx_length - self.min_size)
                / (self.max_size - self.min_size + 1)
                * self.num_buckets
            )
            sample_len[index_buckets] = max(sample_len[index_buckets], idx_length)

            num_tokens = (len(buckets[index_buckets]) + 1) * sample_len[index_buckets]
            if self.is_batch_full(num_tokens, buckets[index_buckets]):
                yield buckets[index_buckets]
                buckets[index_buckets] = []
                sample_len[index_buckets] = 0

            buckets[index_buckets].append(idx)

        leftover_batch = []
        leftover_sample_len = 0
        leftover = [idx for bucket in buckets for idx in bucket]
        for idx in leftover:
            idx_length = self.num_tokens_fn(idx)
            leftover_sample_len = max(leftover_sample_len, idx_length)
            num_tokens = (len(leftover_batch) + 1) * leftover_sample_len
            if self.is_batch_full(num_tokens, leftover_batch):
                yield leftover_batch
                leftover_batch = []
                leftover_sample_len = 0
            leftover_batch.append(idx)

        if len(leftover_batch) > 0 and not self.drop_last:
            yield leftover_batch

    def __len__(self):
        raise NotImplementedError
        # we do not know the exactly batch size, so do not call len(dataloader)
