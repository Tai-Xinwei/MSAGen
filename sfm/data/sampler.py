# -*- coding: utf-8 -*-
import math
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from deepspeed import (
    __git_branch__,
    __git_hash__,
    __version__,
    __version_major__,
    __version_minor__,
    __version_patch__,
    dist,
)
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler, T_co


class WeightedDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        weight_dict: Dict[Tuple[int, int], float],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.weight_dict = weight_dict

        dataset_indices_len = 0
        for begin, end in weight_dict:
            ratio = weight_dict[(begin, end)]
            sampled_len = math.ceil((end - begin) * ratio)
            dataset_indices_len += sampled_len

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and dataset_indices_len % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (dataset_indices_len - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(dataset_indices_len / self.num_replicas)  # type: ignore[arg-type]
        self.dataset_indices_len = dataset_indices_len
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.shuffle = True
        self.num_skip_batches = None
        self.micro_batch_size = None

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            generator = np.random.default_rng(self.epoch + self.seed)
            torch_generator = torch.Generator()
            torch_generator.manual_seed(self.seed + self.epoch)
            indices = []
            for begin, end in np.sort(list(self.weight_dict.keys())):
                ratio = self.weight_dict[(begin, end)]
                sampled_len = math.ceil((end - begin) * ratio)
                if sampled_len > end - begin:
                    choice_replace = True
                else:
                    choice_replace = False
                indices.extend(
                    list(
                        generator.choice(
                            end - begin, sampled_len, replace=choice_replace
                        )
                        + begin
                    )
                )
            sorted_indices = torch.randperm(
                len(indices), generator=torch_generator
            ).tolist()
            indices = np.array(indices)[np.array(sorted_indices)].tolist()
        else:
            raise ValueError("WeightedDistributedSampler must be shuffled.")

        assert self.dataset_indices_len == len(indices)

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
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if self.num_skip_batches is not None:
            indices = indices[self.num_skip_batches * self.micro_batch_size :]
            self.num_samples = len(indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_skip_batches(self, num_skip_batches, micro_batch_size):
        self.num_skip_batches = num_skip_batches
        self.micro_batch_size = micro_batch_size

    def set_epoch(self, epoch) -> None:
        self.num_skip_batches = None
        self.micro_batch_size = None
        return super().set_epoch(epoch)
