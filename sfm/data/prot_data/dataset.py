# -*- coding: utf-8 -*-
import itertools
import math
import os
import pickle as pkl
import random
from pathlib import Path
from typing import Any, List, Tuple, Union

import lmdb
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler

from sfm.data.data_utils import _filter_by_size_dynamic

try:
    from sfm.data.prot_data.token_block_utils_fast import (
        _get_block_to_dataset_index_fast,
        _get_slice_indices_fast,
    )
except ImportError:
    raise ImportError(
        "Please build Cython components with: `pip install --editable .` "
        "or `python setup.py build_ext --inplace`"
    )

from sfm.data.dataset import FoundationModelDataset
from sfm.data.prot_data.collater import (
    collate_downstream_fn,
    collate_fn,
    collate_multiseq_downstream_fn,
    collate_secondary_structure_fn,
    collate_stack_fn,
    collate_ur50_fn,
)
from sfm.data.prot_data.sequence_masking import masking_registry
from sfm.data.prot_data.spatial_noise import noise_registry
from sfm.data.prot_data.util import bstr2obj
from sfm.data.prot_data.vocalubary import Alphabet
from sfm.logging import logger


class LMDBDataset(FoundationModelDataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = self.set_default_args(args)
        assert (
            self.args.data_path and Path(self.args.data_path).is_dir()
        ), f"Processed file not found: {self.args.data_path}"
        self.lmdb_path = Path(self.args.data_path)
        self.vocab = Alphabet()

        self.env = lmdb.open(
            str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self.sizes, self.keys = metadata["sizes"], metadata["keys"]
        self.comment = metadata["comment"]
        self.filter_indices_by_size(
            indices=np.array(range(len(self.keys))),
            max_sizes=self.args.max_length - 2,
        )

    def __sort__(self):
        sorted_names_sizes = sorted(zip(self.keys, self.sizes), key=lambda x: x[1])
        self.keys = [name for name, size in sorted_names_sizes]
        self.sizes = [size for name, size in sorted_names_sizes]

    def set_default_args(self, args):
        raise NotImplementedError()

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def size(self, index: int) -> int:
        raise NotImplementedError()

    def num_tokens(self, index: int) -> int:
        raise NotImplementedError()

    def num_tokens_vec(self, indices):
        raise NotImplementedError()

    def collate(self, samples: List[dict]) -> dict:
        raise NotImplementedError()

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
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

        logger.warning(
            f"Removed {len(ignored)} examples from the dataset because they are longer than {max_sizes}."
        )
        self.sizes = [self.sizes[idx] for idx in indices]
        self.keys = [self.keys[idx] for idx in indices]


class DownstreamLMDBDataset(LMDBDataset):
    """
    DownstreamLMDBDataset is a base class for downstream tasks. Different from the other ProteinLMDBDataset,
    this class does not have the noise and masking method. It contains the labeled data for downstream tasks.
    It should be used for finetuning the model.

    """

    TASKINFO = {
        # single sequence --> single label
        "beta_lactamase": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [0.7383112351980983, 0.31642946622284757],
            "classes": None,
        },
        "fluorescence": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [3.180586883940159, 0.8339910288860691],
            "classes": None,
        },
        "solubility": {
            "type": "binary",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },  # 0-1
        "stability": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [0.1790524860555312, 0.5662245232305079],
            "classes": None,
        },
        "subcellular_localization": {
            "type": "classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(10)),
        },
        "subcellular_localization_2": {
            "type": "binary",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },
        # below three are same files, but different splits
        "remote_homology_fold": {
            "type": "classification",
            "splits": ["train", "valid", "test_fold_holdout"],
            "mean_std": [None, None],
            "classes": list(range(1195)),
        },
        "remote_homology_superfamily": {
            "type": "classification",
            "splits": ["train", "valid", "test_superfamily_holdout"],
            "mean_std": [None, None],
            "classes": list(range(1822)),
        },
        "remote_homology_family": {
            "type": "classification",
            "splits": ["train", "valid", "test_family_holdout"],
            "mean_std": [None, None],
            "classes": list(range(3439)),
        },
        # single sequence --> multiple labels
        "EnzymeCommission": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(538)),
        },
        "GeneOntology_mf": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(489)),
        },
        "GeneOntology_bp": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(1943)),
        },
        "GeneOntology_cc": {
            "type": "multi_classification",
            "splits": ["train", "valid", "test"],
            "mean_std": [None, None],
            "classes": list(range(320)),
        },
        # single sequence --> residue labels + residue masks
        "secondary_structure": {
            "type": "residue_classification",
            "splits": ["train", "valid", "casp12", "cb513", "ts115"],
            "mean_std": [None, None],
            "classes": list(range(3)),
        },
        # multiple sequences --> single label
        "human_ppi": {
            "type": "binary",
            "splits": ["train", "valid", "test", "cross_species_test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },
        "yeast_ppi": {
            "type": "binary",
            "splits": ["train", "valid", "test", "cross_species_test"],
            "mean_std": [None, None],
            "classes": [0, 1],
        },
        "ppi_affinity": {
            "type": "regression",
            "splits": ["train", "valid", "test"],
            "mean_std": [-11.654589870742205, 3.0832061340498975],
            "classes": None,
        },
        # TODO: single sequence --> contact map
    }

    def __init__(self, args: Any, direct=True) -> None:
        if direct:
            raise ValueError(
                "DownstreamLMDBDataset should not be initialized directly, please use DownstreamLMDBDataset.load_dataset(args) instead."
            )
        super().__init__(args)
        self.lmdb_basepath = Path(self.args.data_basepath)
        self.lmdb_path = Path(self.args.data_path)
        self.max_length = self.args.max_length
        self.task_name = self.args.task_name
        self.label_field = self.args.label_field
        self.split = self.args.split
        self.normalize_label = self.args.normalize_label
        assert (
            self.split in DownstreamLMDBDataset.TASKINFO[self.task_name]["splits"]
        ), f"split must be one of {self.TASKINFO[self.task_name]['splits']} for task {self.task_name}, but got {self.split}"

    def set_default_args(self, args):
        args.data_basepath = getattr(args, "data_basepath", None)
        args.task_name = getattr(args, "task_name", None)
        args.label_field = getattr(args, "label_field", "target")
        args.split = getattr(args, "split", None)
        args.max_length = getattr(args, "max_length", 1024)
        args.normalize_label = getattr(args, "normalize_label", False)
        # this should be set by self.load_dataset
        args.data_path = getattr(args, "data_path", None)
        # currently, we do not use seed in DownstreamLMDBDataset since all datasets are splitted already, but we keep it for future use
        # args.seed = getattr(args, "seed", "2023")
        return args

    def __getitem__(self, index: int) -> dict:
        key = self.keys[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)
        item = {"id": index, **data}

        # multi-instance
        if isinstance(item["aa"][0], list):
            for idx, seq in enumerate(item["aa"]):
                tokens = [self.vocab.tok_to_idx[tok] for tok in seq]
                if self.vocab.prepend_bos:
                    tokens.insert(0, self.vocab.cls_idx)
                if self.vocab.append_eos:
                    tokens.append(self.vocab.eos_idx)
                item[f"aa_{idx}"] = np.array(tokens, dtype=np.int64)
                properties = self.vocab.feat_idx(item[f"aa_{idx}"])
                for prop_name in ["chem_polar", "net_charge"]:
                    item[f"{prop_name}_{idx}"] = np.array(
                        properties[prop_name], dtype=np.int64
                    )
                for prop_name in ["hydropathy", "mol_mass"]:
                    item[f"{prop_name}_{idx}"] = np.array(
                        properties[prop_name], dtype=np.float32
                    )
            del item["aa"]
        else:
            tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
            if self.vocab.prepend_bos:
                tokens.insert(0, self.vocab.cls_idx)
            if self.vocab.append_eos:
                tokens.append(self.vocab.eos_idx)
            item["aa"] = np.array(tokens, dtype=np.int64)

            properties = self.vocab.feat_idx(item["aa"])
            for prop_name in ["chem_polar", "net_charge"]:
                item[prop_name] = np.array(properties[prop_name], dtype=np.int64)
            for prop_name in ["hydropathy", "mol_mass"]:
                item[prop_name] = np.array(properties[prop_name], dtype=np.float32)

        # make the label's type right
        if self.TASKINFO[self.task_name]["type"] == "regression":
            item[self.label_field] = np.array(item[self.label_field], dtype=np.float32)
        elif self.TASKINFO[self.task_name]["type"] in {
            "classification",
            "binary",
            "multi_classification",
        }:
            # patch for remote_homology
            if self.task_name == "remote_homology_fold":
                item[self.label_field] = np.array(
                    [item[self.label_field][1]], dtype=np.int64
                )
            elif self.task_name == "remote_homology_superfamily":
                item[self.label_field] = np.array(
                    [item[self.label_field][2]], dtype=np.int64
                )
            elif self.task_name == "remote_homology_family":
                item[self.label_field] = np.array(
                    [item[self.label_field][3]], dtype=np.int64
                )
            else:
                item[self.label_field] = np.array(
                    item[self.label_field], dtype=np.int64
                )
            # patch end
        elif self.TASKINFO[self.task_name]["type"] == "residue_classification":
            item[self.label_field], item[f"{self.label_field}_mask"] = item[
                self.label_field
            ]
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

    def collate(self, samples: List[dict]) -> dict:
        if self.task_name in {"human_ppi", "yeast_ppi", "ppi_affinity"}:
            return collate_multiseq_downstream_fn(samples, self.vocab)
        elif self.task_name == "secondary_structure":
            return collate_secondary_structure_fn(samples, self.vocab)
        else:
            return collate_downstream_fn(samples, self.vocab)

    @classmethod
    def load_dataset(cls, args):
        if not hasattr(args, "task_name"):
            raise ValueError("args must have task_name to load DownstreamLMDBDataset.")
        if args.task_name not in DownstreamLMDBDataset.TASKINFO:
            raise ValueError(
                f"args.task_name = {args.task_name} not support yet, must be one of {DownstreamLMDBDataset.TASKINFO.keys()}"
            )
        dset_dict = {}
        for split in DownstreamLMDBDataset.TASKINFO[args.task_name]["splits"]:
            args.split = split
            args.data_path = str(
                Path(args.data_basepath)
                / args.task_name
                / f"{args.task_name}_{split}.lmdb"
            )
            dset_dict[split] = cls(args, direct=False)
        return dset_dict


# not used for now, but keep it for future use
class ProteinLMDBDataset(LMDBDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any) -> None:
        # here calls self.set_default_args(args)
        super().__init__(args)
        self.seed = self.args.seed
        self.seq_masking_method = self.args.seq_masking_method
        self.noise_method = self.args.noise_method
        self.pos_noise = self.args.pos_noise
        self.ang_noise = self.args.ang_noise

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
        # if len(tokens) > self.args.max_length - 2:
        #     start = random.randint(0, len(tokens) - self.args.max_length + 2)
        #     tokens = tokens[start : start + self.args.max_length - 2]
        # assert len(tokens) <= self.args.max_length - 2, f"len(tokens) = {len(tokens)} > {self.args.max_length - 2} = max_length - 2"

        if self.vocab.prepend_bos:
            tokens.insert(0, self.vocab.cls_idx)
        if self.vocab.append_eos:
            tokens.append(self.vocab.eos_idx)

        item["aa"] = np.array(tokens, dtype=np.int64)
        item["seq_length"] = len(tokens)

        """
        - mask the sequence in different ways
        """
        seed = int(hash((self.seed, index)) % 1e6)
        assert (
            "mask_idx" not in item
        ), "Item already contains mask_idx key, this is not expected!"

        masked_seq, mask_type, mask_pos = masking_registry[self.seq_masking_method](
            item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        )

        """
        Add noise to the coordinate or angles, manupilate the item['pos']/item['ang']:
        - add noise to the coordinate
        - add noise to the angles
        """
        assert (
            "pos_noise" or "ang_noise" not in item
        ), "Item already contains mask_idx key, this is not expected!"
        pos_noise, ang_noise = noise_registry[self.noise_method](
            item, self.args, seed, self.pos_noise, self.ang_noise
        )
        item["masked_aa"] = mask_type
        item["mask_pos"] = mask_pos

        item["pos_noise"] = pos_noise
        item["ang_noise"] = ang_noise

        item["ang"] = item["ang"] / 180.0 * torch.pi
        # insert inf to the first place and the end
        item["ang"] = np.concatenate(
            [np.zeros((1, 9)), item["ang"], np.zeros((1, 9))], axis=0
        )
        item["pos"] = np.concatenate(
            [np.zeros((1, 37, 3)), item["pos"], np.zeros((1, 37, 3))], axis=0
        )
        item["ang_mask"] = np.concatenate(
            [np.zeros((1, 9)), item["ang_mask"], np.zeros((1, 9))], axis=0
        )
        item["pos_mask"] = np.concatenate(
            [np.zeros((1, 37)), item["pos_mask"], np.zeros((1, 37))], axis=0
        )

        # ang_time has same shape as aa, random number with scale of (0, 1]
        ang_t = 1.0 - np.random.rand(1).astype(np.float32)
        item["ang_time"] = ang_t * np.ones(len(item["aa"])).astype(np.float32)
        # set cls and eos's time to 0
        item["ang_time"][0] = 0.0
        item["ang_time"][-1] = 0.0

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

    def collate(self, samples: List[dict]) -> dict:
        return collate_stack_fn(samples, self.vocab, offset=0)


class UR50LMDBDataset(FoundationModelDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any, data_path: str = None) -> None:
        super().__init__()

        self.args = self.set_default_args(args)

        # logger.info(self.args)
        if data_path is None:
            self.lmdb_path = Path(self.args.data_path)
        else:
            self.lmdb_path = Path(data_path)

        assert self.lmdb_path.is_dir(), f"Processed file not found: {self.lmdb_path}"

        self.vocab = Alphabet()

        self.seed = self.args.seed
        # self.seq_masking_method = self.args.seq_masking_method
        self.seq_masking_method = "bert"

        self.noise_method = self.args.noise_method
        self.pos_noise = self.args.pos_noise
        self.ang_noise = self.args.ang_noise

        self.env = lmdb.open(
            str(self.lmdb_path), subdir=True, readonly=True, lock=False, readahead=False
        )
        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.sizes, self.names = metadata["lengths"], metadata["prot_accessions"]

        self.filter_indices_by_size(
            indices=np.array(range(len(self.names))), max_sizes=self.args.max_length
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
        args.mask_multiple_length = getattr(args, "mask_multiple_length", 3)
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

        # item = {"id": index, **data}
        aatypes = list(data)
        item = {
            # "id": index,
            "aa": aatypes,
        }
        # if len(item["aa"]) > self.args.max_length:
        #     return self.__getitem__(index + 1)

        """
        - add physichemical properties
        """
        # tokens = [self.vocab.tok_to_idx[tok] for tok in item["raw_aa"]]

        """
        - convert string sequence to int index
        """
        # if self.vocab.prepend_bos:
        #     tokens.insert(0, self.vocab.cls_idx)
        # if self.vocab.append_eos:
        #     tokens.append(self.vocab.eos_idx)
        tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        if self.args.ifstack:
            tokens.insert(0, self.vocab.cls_idx)
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
        # item["aa"]
        # {"id": index, 'aa': aa, 'pos': pos, 'ang': ang, 'conf': conf_score, "name": name}
        assert (
            "mask_idx" not in item
        ), "Item already contains mask_idx key, this is not expected!"

        new_seq, mask_type, rand_mask = masking_registry[self.seq_masking_method](
            item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks_idx
        )
        mask_pos = mask_type

        # masked_seq, mask_type, mask_pos = masking_registry[self.seq_masking_method](
        #     item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        # )

        item["masked_aa"] = mask_type
        item["mask_pos"] = mask_pos
        item["new_seq"] = new_seq

        return item

    def __len__(self) -> int:
        return len(self.names)

    def size(self, index: int) -> int:
        sz = self.sizes[index]
        # if self.vocab.prepend_bos:
        #     sz += 1
        # if self.vocab.append_eos:
        #     sz += 1
        if self.args.ifstack:
            sz += 2
        return sz

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]

    def num_tokens_vec(self, indices):
        raise NotImplementedError

    def collate(self, samples: List[dict]) -> dict:
        return collate_ur50_fn(samples, self.vocab)

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
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


class PackedUR50LMDBDataset(FoundationModelDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any, data_path: str = None, init_lmdb=True) -> None:
        super().__init__()

        self.args = self.set_default_args(args)

        # logger.info(self.args)
        if data_path is None:
            self.lmdb_path = Path(self.args.data_path)
        else:
            self.lmdb_path = Path(data_path)

        self.vocab = Alphabet()

        self.seed = self.args.seed
        self.seq_masking_method = self.args.seq_masking_method

        self.noise_method = self.args.noise_method
        self.pos_noise = self.args.pos_noise
        self.ang_noise = self.args.ang_noise

        if init_lmdb:
            assert (
                self.lmdb_path.is_dir()
            ), f"Processed file not found: {self.lmdb_path}"

            self.env = lmdb.open(
                str(self.lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
            )

            self.txn = self.env.begin(write=False)

            metadata = bstr2obj(self.txn.get("metadata".encode()))
            self.sizes, self.names = metadata["lengths"], metadata["prot_accessions"]

            logger.info(f"Loaded {len(self.names)} proteins from {self.lmdb_path}")

    def __sort__(self):
        sorted_names_sizes = sorted(zip(self.names, self.sizes), key=lambda x: x[1])
        self.names = [name for name, size in sorted_names_sizes]
        self.sizes = [size for name, size in sorted_names_sizes]

    def set_default_args(self, args):
        args.data_path = getattr(args, "data_path", None)

        args.seed = getattr(args, "seed", "2023")
        args.seq_masking_method = getattr(args, "seq_masking_method", "bert")

        args.mask_prob = getattr(args, "mask_prob", 0.15)
        args.leave_unmasked_prob = getattr(args, "leave_unmasked_prob", 0.1)
        args.random_token_prob = getattr(args, "random_token_prob", 0.1)
        args.mask_multiple_length = getattr(args, "mask_multiple_length", 3)
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
        value = self.txn.get(f"{key}".encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = pkl.loads(value)

        # item = {"id": index, **data}
        tokens = list(data)

        item = {
            # "id": index,
            "aa": tokens,
        }
        # if len(item["aa"]) > self.args.max_length:
        #     return self.__getitem__(index + 1)

        """
        - add physichemical properties
        """
        # tokens = [self.vocab.tok_to_idx[tok] for tok in item["raw_aa"]]

        """
        - convert string sequence to int index
        """

        # tokens = [self.vocab.tok_to_idx[tok] for tok in item["aa"]]
        # if self.vocab.prepend_bos:
        #     tokens.insert(0, self.vocab.cls_idx)
        # if self.vocab.append_eos:
        #     tokens.append(self.vocab.eos_idx)
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
        # item["aa"]
        # {"id": index, 'aa': aa, 'pos': pos, 'ang': ang, 'conf': conf_score, "name": name}
        assert (
            "mask_idx" not in item
        ), "Item already contains mask_idx key, this is not expected!"

        # new_seq, mask_type, mask_pos = masking_registry[self.seq_masking_method](
        #     item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        # )

        # bert like mask
        new_seq, mask_type, rand_mask = masking_registry[self.seq_masking_method](
            item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks_idx
        )

        item["masked_aa"] = mask_type
        item["mask_pos"] = mask_type
        item["new_seq"] = new_seq

        return item

    def __len__(self) -> int:
        return len(self.names)

    def collate(self, samples: List[dict]) -> dict:
        return collate_ur50_fn(samples, self.vocab)


class PackedUR50LMDBMultiSrcDataset(PackedUR50LMDBDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any, data_path: str = None) -> None:
        super(PackedUR50LMDBMultiSrcDataset, self).__init__(
            args, data_path, init_lmdb=False
        )

        temp_path = str(self.lmdb_path).split(".")[0]
        if self.args.rank % 8 == 0:
            self.lmdb_path = temp_path + "_1.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs1")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        elif self.args.rank % 8 == 1:
            self.lmdb_path = temp_path + "_2.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs2")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        elif self.args.rank % 8 == 2:
            self.lmdb_path = temp_path + "_3.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs3")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        elif self.args.rank % 8 == 3:
            self.lmdb_path = temp_path + "_4.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs4")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        elif self.args.rank % 8 == 4:
            self.lmdb_path = temp_path + "_5.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs5")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        elif self.args.rank % 8 == 5:
            self.lmdb_path = temp_path + "_6.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs6")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        elif self.args.rank % 8 == 6:
            self.lmdb_path = temp_path + "_7.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs7")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        elif self.args.rank % 8 == 7:
            self.lmdb_path = temp_path + "_8.lmdb"
            self.lmdb_path = self.lmdb_path.replace("nfs", "nfs8")
            self.env = lmdb.open(
                self.lmdb_path, subdir=True, readonly=True, lock=False, readahead=False
            )
        else:
            raise ValueError(f"Rank must be in [0, 7], but got {self.args.rank}")

        self.txn = self.env.begin(write=False)

        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self.sizes, self.names = metadata["lengths"], metadata["prot_accessions"]

        logger.info(f"Loaded {len(self.names)} proteins from {self.lmdb_path}")


class PackedBpeUR50LMDBDataset(FoundationModelDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any, data_path: str = None, init_lmdb=True) -> None:
        super().__init__()

        self.args = self.set_default_args(args)

        # logger.info(self.args)
        if data_path is None:
            self.lmdb_path = Path(self.args.data_path)
        else:
            self.lmdb_path = Path(data_path)

        self.vocab = Alphabet()

        self.seed = self.args.seed
        self.seq_masking_method = self.args.seq_masking_method

        self.noise_method = self.args.noise_method
        self.pos_noise = self.args.pos_noise
        self.ang_noise = self.args.ang_noise

        if init_lmdb:
            assert (
                self.lmdb_path.is_dir()
            ), f"Processed file not found: {self.lmdb_path}"

            self.env = lmdb.open(
                str(self.lmdb_path),
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
            )

            self.txn = self.env.begin(write=False)

            metadata = bstr2obj(self.txn.get("metadata".encode()))
            self.sizes, self.names = metadata["lengths"], metadata["prot_accessions"]

            logger.info(f"Loaded {len(self.names)} proteins from {self.lmdb_path}")

    def __sort__(self):
        sorted_names_sizes = sorted(zip(self.names, self.sizes), key=lambda x: x[1])
        self.names = [name for name, size in sorted_names_sizes]
        self.sizes = [size for name, size in sorted_names_sizes]

    def set_default_args(self, args):
        args.data_path = getattr(args, "data_path", None)

        args.seed = getattr(args, "seed", "2023")
        args.seq_masking_method = getattr(args, "seq_masking_method", "bert")

        args.mask_prob = getattr(args, "mask_prob", 0.15)
        args.leave_unmasked_prob = getattr(args, "leave_unmasked_prob", 0.1)
        args.random_token_prob = getattr(args, "random_token_prob", 0.1)
        args.mask_multiple_length = getattr(args, "mask_multiple_length", 5)
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
        value = self.txn.get(f"{key}".encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = pkl.loads(value)

        tokens = list(data["aa_seq"])
        bpe_token = list(data["bpe_seq"])
        item = {
            # "id": index,
            "aa": tokens,
        }

        item["aa"] = np.array(tokens, dtype=np.int64)
        item["bpe"] = np.array(bpe_token, dtype=np.int64)

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
        # item["aa"]
        # {"id": index, 'aa': aa, 'pos': pos, 'ang': ang, 'conf': conf_score, "name": name}
        assert (
            "mask_idx" not in item
        ), "Item already contains mask_idx key, this is not expected!"

        # new_seq, mask_type, mask_pos = masking_registry[self.seq_masking_method](
        #     item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks
        # )

        # bert like mask
        new_seq, mask_type, rand_mask = masking_registry[self.seq_masking_method](
            item, self.args, seed, self.vocab.mask_idx, self.vocab.standard_toks_idx
        )

        item["masked_aa"] = mask_type
        item["mask_pos"] = mask_type
        item["new_seq"] = new_seq

        return item

    def __len__(self) -> int:
        return len(self.names)

    def collate(self, samples: List[dict]) -> dict:
        return collate_ur50_fn(samples, self.vocab)


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
        self.collate_fn = dataset.collate

        self.sequence_length = args.max_length

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        return self.collate_fn(samples)

    def num_tokens(self, index: int) -> int:
        return self.dataset.sizes[index]


class StackedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, args):
        self.dataset = list(dataset)  # Convert the iterable dataset to a list
        self.sequence_length = args.max_length
        self.args = args
        self.collate_fn = dataset.collate
        self.stacked_data = self.stack_data()

    def stack_data(self):
        # Preprocess the iterable dataset into fixed-length sequences
        # This is just a placeholder function, you'll need to implement this based on your data
        processed_data = []
        buffer = {}
        buffer_len = 0

        for item in self.dataset:
            for key, value in item.items():
                if key not in buffer:
                    buffer[key] = []
                buffer[key].append(value)

            buffer_len += len(item["aa"])
            if buffer_len >= self.sequence_length:
                # Add a new sequence to processed_data
                sequence = {
                    key: torch.cat(buf[: self.sequence_length], dim=0)
                    for key, buf in buffer.items()
                }
                processed_data.append(sequence)
                # Update buffer
                if buffer_len > self.sequence_length:
                    for key in buffer.keys():
                        buffer[key] = [buffer[key][-1]]
                    buffer_len = len(buffer["aa"][0])
                else:
                    buffer = {}
                    buffer_len = 0

        # Handle any remaining items in the buffer
        if buffer_len > 0:
            # You may need to pad the sequences to reach the desired length
            sequence = {key: torch.cat(buf, dim=0) for key, buf in buffer.items()}
            processed_data.append(sequence)

        return processed_data

    def collate(self, samples):
        return self.collate_fn(samples)

    def __len__(self):
        # Return the total number of sequences in the dataset
        return len(self.stacked_data)

    def __getitem__(self, idx):
        # Retrieve a sequence by index
        return self.stacked_data[idx]


class StackedSequenceIterableDataset(IterableDataset):
    def __init__(self, dataset, args, shuffle=True):
        self.dataset = dataset  # An iterable source
        self.collate_fn = collate_stack_fn
        self.sequence_length = args.max_length
        self.args = args
        self.buffer_len = 0
        self.buffers = {}
        self.shuffle = shuffle
        self.epoch = 0
        self.break_mode = "complete_doc"
        self.document_sep_len = 1
        self.sizes = dataset.sizes
        self.num_blocks = 0

        # DDP-related attributes
        self.rank = args.rank
        self.world_size = args.world_size
        self.sampler = DistributedSampler(
            dataset,
            shuffle=self.shuffle,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        self._create_subset_iterator()

    def __iter__(self):
        # Reset the buffers when creating a new iterator
        if self.shuffle:
            self.sampler.set_epoch(self.epoch)
            self.epoch += 1

        # # Reset the buffers when creating a new iterator
        self.buffer_len = 0
        self.buffers = {}

        self.subset_iterator = self._create_subset_iterator()

        return self

    def _create_subset_iterator(self):
        # # Get the list of indices from the sampler and iterate through them
        indices = list(iter(self.sampler))
        sizes = [self.sizes[i] for i in indices]
        slice_indices = _get_slice_indices_fast(
            np.array(sizes),
            self.break_mode,
            self.sequence_length,
            self.document_sep_len,
        )
        blocks = _get_block_to_dataset_index_fast(np.array(sizes), slice_indices)

        # Split blocks among workers for DDP
        self.num_blocks = len(blocks)
        logger.success(
            f"number of stacked block in epoch {self.epoch-1} of rank {self.rank} is {self.num_blocks}"
        )

        for block in blocks:
            start, start_offset, end = block
            for idx in range(start, end + 1):
                yield self.dataset[idx]

    def __next__(self):
        # Continue to read from the subset iterator and fill the buffers until we have enough data
        while self.buffer_len < self.sequence_length:
            try:
                item = next(self.subset_iterator)

                for key, value in item.items():
                    if key not in self.buffers:
                        self.buffers[key] = []
                    self.buffers[key].append(value)

                self.buffer_len += len(item["aa"])
            except StopIteration:
                # If there's no more data and the buffer is partially filled, return what's left
                if self.buffers:
                    for key in self.buffers.keys():
                        self.buffers[key].clear()
                self.buffer_len = 0
                raise

        # Extract a sequence of exactly `sequence_length` from the buffers for each key
        result = {}
        for key, buf in self.buffers.items():
            if key not in [
                "aa",
                "masked_aa",
                "mask_pos",
                "ang",
                "pos",
                "ang_mask",
                "pos_mask",
                "ang_time",
            ]:
                continue

            result[key] = np.concatenate(buf)[: self.sequence_length]
            if self.buffer_len == self.sequence_length:
                self.buffers[key] = []
            elif len(self.buffers[key][-1]) > self.sequence_length:
                self.buffers[key] = []
            else:
                self.buffers[key] = buf[-1:]

        if self.buffer_len == self.sequence_length:
            self.buffer_len = 0
            for key in self.buffers.keys():
                self.buffers[key].clear()
        elif len(self.buffers["aa"]) == 0:
            self.buffer_len = 0
        else:
            self.buffer_len = len(self.buffers["aa"][0])

        return result

    def collate(self, samples):
        return self.collate_fn(samples, self.dataset.vocab)

    def __len__(self):
        return self.num_blocks


if __name__ == "__main__":
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from torch.utils.data import DataLoader, RandomSampler
    from tqdm import tqdm

    class Namespace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def reverse2str(vocab, tokens):
        idx_to_tok = {v: k for k, v in vocab.tok_to_idx.items()}
        aaseq = []
        for i in tokens:
            if i in [
                vocab.unk_idx,
                vocab.padding_idx,
                vocab.cls_idx,
                vocab.mask_idx,
                vocab.eos_idx,
            ]:
                continue
            aaseq.append(idx_to_tok[i])
        return "".join(aaseq)

    args = Namespace()
    args.data_basepath = "/mnta/yaosen/data/bfm_benchmark"
    args.task_name = "solubility"
    args.max_length = 2048

    dsets = DownstreamLMDBDataset.load_dataset(args)
    for name, dset in dsets.items():
        seqrecords = []
        for data in tqdm(dset):
            name = data["name"]
            aa = reverse2str(dset.vocab, data["aa"])
            seqrecords.append(
                SeqRecord(
                    Seq(aa),
                    id=name,
                    description="",
                )
            )
        SeqIO.write(
            seqrecords,
            Path(args.data_basepath) / f"{args.task_name}_{name}.fasta",
            "fasta",
        )
