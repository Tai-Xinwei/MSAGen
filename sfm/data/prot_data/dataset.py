# -*- coding: utf-8 -*-
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, List

import lmdb
import numpy as np
import torch

from sfm.data.data_utils import _filter_by_size_dynamic
from sfm.data.dataset import FoundationModelDataset
from sfm.logging import logger

from .collater import collate_fn
from .process import bstr2obj
from .sequence_masking import masking_registry
from .spatial_noise import noise_registry
from .vocalubary import Alphabet


class ProteinLMDBDataset(FoundationModelDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any) -> None:
        super().__init__()

        self.args = self.set_default_args(args)

        # logger.info(self.args)

        self.lmdb_path = Path(self.args.data_path)
        assert self.lmdb_path.is_dir(), f"Processed file not found: {self.lmdb_path}"

        self.vocab = Alphabet()

        self.seed = self.args.seed
        self.seq_masking_method = self.args.seq_masking_method

        self.noise_method = self.args.noise_method
        self.pos_noise = self.args.pos_noise
        self.ang_noise = self.args.ang_noise

        self.env = lmdb.open(
            str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.sizes, self.names = metadata["sizes"], metadata["names"]
        self.comment = metadata["comment"]

        self.filter_indices_by_size(
            indices=np.array(range(len(self.names))), max_sizes=self.args.max_num_aa
        )

    def __sort__(self):
        sorted_names_sizes = sorted(zip(self.names, self.sizes), key=lambda x: x[1])
        self.names = [name for name, size in sorted_names_sizes]
        self.sizes = [size for name, size in sorted_names_sizes]

    def set_default_args(self, args):
        args.data_path = getattr(args, "data_path", None)

        args.seed = getattr(args, "seed", "2023")
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
        args.coord_noise_stdev = getattr(args, "coord_noise_stdev", 1.0)
        args.angle_noise_mean = getattr(args, "angle_noise_mean", 0.0)
        args.angle_noise_stdev = getattr(args, "angle_noise_stdev", 1.0)

        return args

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.names)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(666).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(self.args)
        dataset_train.names = [self.names[idx] for idx in training_indices]
        dataset_train.sizes = [self.sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.args)
        dataset_val.names = [self.names[idx] for idx in validation_indices]
        dataset_val.sizes = [self.sizes[idx] for idx in validation_indices]

        if sort:
            dataset_train.__sort__()
            dataset_val.__sort__()

        return dataset_train, dataset_val

    def __getitem__(self, index: int) -> dict:
        key = self.names[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)
        item = {"id": index, **data}
        # if len(item["aa"]) > self.args.max_num_aa:
        #     return self.__getitem__(index + 1)

        """
        - add physichemical properties
        """
        # tokens = [self.vocab.tok_to_idx[tok] for tok in item["raw_aa"]]

        """
        - convert string sequence to int index
        """
        tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        if self.vocab.prepend_bos:
            tokens.insert(0, self.vocab.cls_idx)
        if self.vocab.append_eos:
            tokens.append(self.vocab.eos_idx)
        item["aa"] = np.array(tokens, dtype=np.int64)

        """
        - add residue properties
        """
        properties = self.vocab.feat_idx(item["aa"])
        for prop_name in ["chem_polar", "net_charge"]:
            item[prop_name] = np.array(properties[prop_name], dtype=np.int64)
        for prop_name in ["hydropathy", "mol_mass"]:
            item[prop_name] = np.array(properties[prop_name], dtype=np.float32)

        """
        - mask the sequence in different ways
        """
        seed = int(hash((self.seed, index)) % 1e6)
        item["aa"]
        # {"id": index, 'aa': aa, 'pos': pos, 'ang': ang, 'conf': conf_score, "name": name}
        assert (
            "mask_idx" not in item
        ), "Item already contains mask_idx key, this is not expected!"

        # masked_seq, mask, replace_mask = masking_registry[self.seq_masking_method](
        #     item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        # )

        masked_seq, mask_type, mask_pos = masking_registry[self.seq_masking_method](
            item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        )

        item["masked_aa"] = mask_type
        # item["mask"] = mask
        # item["replace_mask"] = replace_mask
        item["mask_pos"] = mask_pos

        """
        Add noise to the coordinate or angles, manupilate the item['pos']/item['ang']:
        - add noise to the coordinate
        - add noise to the angles
        """
        # keys in {"id", 'aa', 'pos', 'ang', 'conf', "name", "masked_aa", "mask", "replace_mask"}
        assert (
            "pos_noise" or "ang_noise" not in item
        ), "Item already contains mask_idx key, this is not expected!"
        pos_noise, ang_noise = noise_registry[self.noise_method](
            item, self.args, seed, self.pos_noise, self.ang_noise
        )
        item["pos_noise"] = pos_noise
        item["ang_noise"] = ang_noise

        return item

    def __len__(self) -> int:
        return len(self.names)

    def size(self, index: int) -> int:
        sz = self.sizes[index]
        if self.vocab.prepend_bos:
            sz += 1
        if self.vocab.append_eos:
            sz += 1
        raise sz

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        raise NotImplementedError

    def collate(self, samples: List[dict]) -> dict:
        return collate_fn(samples, self.vocab)

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif hasattr(self, "sizes") and isinstance(self.sizes, list):
                sizes = np.array(self.sizes)
                ignored = indices[np.array(sizes[indices]) > max_sizes].tolist()
                indices = indices[np.array(sizes[indices]) <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(indices, self.size, max_sizes)

        self.sizes = [self.sizes[idx] for idx in indices]
        self.names = [self.names[idx] for idx in indices]


class BatchedDataDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        vocab,
        args=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.args = args
        self.vocab = vocab

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        return collate_fn(samples, self.vocab)

    def num_tokens(self, index: int) -> int:
        return self.dataset.sizes[index]


if __name__ == "__main__":

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    args = Namespace()
    args.lmdb_path = "/mnt/protein/48organism.lmdb/"

    print(args)
    print("=================")
    print("Test ProteinLMDBDataset")
    dataset = ProteinLMDBDataset(args)
    print(len(dataset))
    data = dataset[12]
    for k, v in data.items():
        print(k, v.shape if isinstance(v, np.ndarray) else v)
    print(data)
