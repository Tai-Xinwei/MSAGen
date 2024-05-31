# -*- coding: utf-8 -*-
import copy
import random
import sys
from abc import ABC
from typing import Any, Callable, Iterable, List, Tuple, Union

import lmdb
import numpy as np
import torch

from sfm.data.dataset import FoundationModelDataset
from sfm.data.prot_data.collater import pad_nd_seq_unsqueeze
from sfm.data.prot_data.crab import (
    BondAngleCalculator,
    BondLengthCalculator,
    DihedralAngleCalculator,
    FourthAtomCalculator,
)
from sfm.data.prot_data.util import bstr2obj
from sfm.data.prot_data.vocalubary import Alphabet
from sfm.logging import logger

VOCAB = Alphabet()


class BaseTransform(ABC):
    """An abstract base class for data transformations."""

    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    def forward(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(BaseTransform):
    r"""Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def forward(self, data: Union[Iterable[dict], dict]) -> Union[Iterable[dict], dict]:
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data

    def __repr__(self) -> str:
        args = [f"  {transform}" for transform in self.transforms]
        return "{}([\n{}\n])".format(self.__class__.__name__, ",\n".join(args))


class ComposeFilters:
    r"""Composes several filters together.

    Args:
        filters (List[Callable]): List of filters to compose.
    """

    def __init__(self, filters: List[Callable]):
        self.filters = filters

    def __call__(
        self,
        data: Union[Iterable[dict], dict],
    ) -> bool:
        for filter_fn in self.filters:
            if isinstance(data, (list, tuple)):
                if not all([filter_fn(d) for d in data]):
                    return False
            elif not filter_fn(data):
                return False
        return True

    def __repr__(self) -> str:
        args = [f"  {filter_fn}" for filter_fn in self.filters]
        return "{}([\n{}\n])".format(self.__class__.__name__, ",\n".join(args))


class FromNumpy(BaseTransform):
    def __init__(self):
        pass

    def forward(self, item: Union[Iterable[dict], dict]) -> Union[Iterable[dict], dict]:
        if isinstance(item, dict):
            return {k: self.forward(v) for k, v in item.items()}
        elif isinstance(item, np.ndarray):
            return torch.from_numpy(item)
        elif isinstance(item, Iterable) and isinstance(item[0], dict):
            return [self.forward(i) for i in item]
        else:
            return item


class SetNormalNoise(BaseTransform):
    def __init__(
        self, seed: int, key: str, noise_attr: str, mean: float = 0, std: float = 1
    ):
        self.seed = seed
        self.generator = torch.manual_seed(seed)
        self.mean = mean
        self.std = std
        self.key = key
        self.noise_attr = noise_attr

    def forward(self, item: Union[Iterable[dict], dict]) -> Union[Iterable[dict], dict]:
        if isinstance(item, Iterable):
            return [self.forward(i) for i in item]
        elif isinstance(item, dict):
            tsr = item[self.key]
            noise = (
                torch.rand(tsr.size(), generator=self.generator) * self.std + self.mean
            )
            item[self.noise_attr] = noise
            return item
        else:
            raise ValueError(
                f"Expected item to be a dict or an iterable of dicts, but got {type(item)}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BERTMasking(BaseTransform):
    def __init__(
        self,
        set_key="mask",
        mask_prob: float = 0.5,
        mask_multiple_length: int = 1,
        mask_stdev: float = 0.0,
        seed: int = 666,
    ):
        self.set_key = set_key
        self.mask_prob = mask_prob
        self.mask_multiple_length = mask_multiple_length
        self.mask_stdev = mask_stdev
        self.seed = seed
        assert 0.0 < mask_prob < 1.0
        assert mask_multiple_length >= 1
        assert mask_stdev >= 0.0
        self.rng = np.random.default_rng(self.seed)

    def forward(self, item: dict) -> dict:
        size = len(item["input"]["aa"]) - 2  # cls and eos token
        mask_seq = np.full(size, False)
        mask_str = np.full(size, False)
        # at least mask one element or one span
        num_mask = int(self.mask_prob * size / float(self.mask_multiple_length) + 1)
        # GLM like masking, mask size - 1 because some internal coordinates have N_res - 1 elements.
        mask_seq_idx = self.rng.choice(size - 1, num_mask, replace=False)
        mask_str_idx = self.rng.choice(size - 1, num_mask, replace=False)

        mask_seq[mask_seq_idx] = True
        mask_str[mask_str_idx] = True

        if self.mask_stdev > 0.0:
            lengths_seq = self.rng.normal(
                self.mask_multiple_length, self.mask_stdev, size=num_mask
            )
            lengths_seq = [max(0, int(round(x))) for x in lengths_seq]
            mask_seq_idx = np.asarray(
                [
                    mask_seq_idx[j] + offset
                    for j in range(len(mask_seq_idx))
                    for offset in range(lengths_seq[j])
                ],
                dtype=np.int64,
            )

            lengths_str = self.rng.normal(
                self.mask_multiple_length, self.mask_stdev, size=num_mask
            )
            lengths_str = [max(0, int(round(x))) for x in lengths_str]
            mask_str_idx = np.asarray(
                [
                    mask_str_idx[j] + offset
                    for j in range(len(mask_str_idx))
                    for offset in range(lengths_str[j])
                ],
                dtype=np.int64,
            )
        else:
            mask_seq_idx = np.concatenate(
                [mask_seq_idx + i for i in range(self.mask_multiple_length)]
            )
            mask_str_idx = np.concatenate(
                [mask_str_idx + i for i in range(self.mask_multiple_length)]
            )

        mask_seq_idx = mask_seq_idx[mask_seq_idx < len(mask_seq)]
        mask_str_idx = mask_str_idx[mask_str_idx < len(mask_str)]

        try:
            mask_seq[mask_seq_idx] = True
            mask_str[mask_str_idx] = True
        except:  # something wrong
            logger.error("Assigning mask indexes to mask failed!")
            raise

        item[self.set_key] = {
            "mask_seq": torch.from_numpy(mask_seq),
            "mask_str": torch.from_numpy(mask_str),
        }
        return item

    def collate(self, samples: List[dict], batch: dict, padlen: int) -> dict:
        batch[self.set_key] = dict()
        for key in ["mask_seq", "mask_str"]:
            padval = False
            batch[self.set_key][key] = torch.cat(
                [
                    pad_nd_seq_unsqueeze(sample[self.set_key][key], padlen, 0, padval)
                    for sample in samples
                ],
                dim=0,
            )
        return batch


class ItemToCRABBackBone(BaseTransform):
    def __init__(self, set_key: str = "crab"):
        super().__init__()
        self.set_key = set_key

    def forward(self, item: dict) -> dict:
        cls_mask = item["input"]["aa"] == VOCAB.cls_idx
        cls_idx = (cls_mask).nonzero()[0]
        eos_mask = item["input"]["aa"] == VOCAB.eos_idx
        eos_idx = (eos_mask).nonzero()[0]
        lengths = eos_idx - cls_idx + 1

        C = torch.cat(
            [
                torch.ones((length), dtype=torch.int64) * i
                for i, length in enumerate(lengths, VOCAB.padding_idx)
            ]
        )[(~cls_mask) & (~eos_mask)]
        R = item["input"]["aa"][(~cls_mask) & (~eos_mask)]
        # from (N_res, 37, 3) to (N_res, 4, 3) only the backbone atoms for now, e.g., ["N", "CA", "C", "O"]
        # TODO: currently here are N Ca C, maybe we can support side chain, at least CB
        A = item["input"]["pos"][:, [0, 1, 2], :][(~cls_mask) & (~eos_mask)]
        item[self.set_key] = {"C": C, "R": R, "A": A}
        return item

    @staticmethod
    def collate(samples: List[dict], batch: dict, padlen: int):
        batch["crab"] = dict()
        for key in ["C", "R", "A"]:
            padval = (
                VOCAB.padding_idx
                if samples[0]["crab"][key].dtype == torch.int64
                else 0.0  # torch.inf
            )
            batch["crab"][key] = torch.cat(
                [
                    pad_nd_seq_unsqueeze(sample["crab"][key], padlen, 0, padval)
                    for sample in samples
                ],
                dim=0,
            )
        batch["crab"]["padding_mask"] = batch["crab"]["R"].eq(VOCAB.padding_idx)
        return batch


class CRABToInternal(BaseTransform):
    def __init__(self, set_key="internal"):
        self.blc = BondLengthCalculator()
        self.bac = BondAngleCalculator()
        self.dac = DihedralAngleCalculator()
        self.set_key = set_key

    def forward(self, item: dict) -> dict:
        """Convert the Structure instance to internal coordinates.
        TODO: Someone recheck this!!!

                O     :             O  :             O
                |     :             |  :             |
        N -- CA -- C -:- N -- CA -- C -:- N -- CA -- C -
                      :                :
        \........../     \........../     \.........../
            i-1 th           i th             i+1 th

        bond_length: N[i]-CA[i], CA[i]-C[i], C[i]-N[i+1]
        bond_angle: C[i-1]-N[i]-CA[i], N[i]-CA[i]-C[i], CA[i]-C[i]-N[i+1]
            NOTE: C[i-1]-N[i]-CA[i] is invalid for i=0, CA[i]-C[i]-N[i+1] is invalid for i=last
        dihedral_angle: CA[i-1]-C[i-1]-N[i]-CA[i] (omega ω), C[i-1]-N[i]-CA[i]-C[i] (phi ϕ), N[i]-CA[i]-C[i]-N[i+1] (psi ψ)
            NOTE: CA[i-1]-C[i-1]-N[i]-CA[i] (omega ω) and C[i-1]-N[i]-CA[i]-C[i] (phi ϕ) are invalid for i=0, N[i]-CA[i]-C[i]-N[i+1] (psi ψ) is invalid for i=last

        """
        A = item["crab"]["A"]
        # assert (
        #     isinstance(C, torch.Tensor)
        #     and isinstance(R, torch.Tensor)
        #     and isinstance(A, torch.Tensor)
        # )
        atom_N, atom_CA, atom_C = A[:, 0, :], A[:, 1, :], A[:, 2, :]
        # bond_length: N[i]-CA[i], CA[i]-C[i], C[i]-N[i+1]
        bl_N_CA = self.blc(atom_N, atom_CA)  # N_res
        bl_CA_C = self.blc(atom_CA, atom_C)  # N_res
        bl_C_N = self.blc(atom_C[:-1], atom_N[1:])  # N_res - 1
        # bond_angle: C[i-1]-N[i]-CA[i], N[i]-CA[i]-C[i], CA[i]-C[i]-N[i+1]
        ba_C_N_CA = self.bac(atom_C[:-1], atom_N[1:], atom_CA[1:])  # N_res - 1
        ba_N_CA_C = self.bac(atom_N, atom_CA, atom_C)  # N_res
        ba_CA_C_N = self.bac(atom_CA[:-1], atom_C[:-1], atom_N[1:])  # N_res - 1
        # dihedral_angle: CA[i-1]-C[i-1]-N[i]-CA[i] (omega ω), C[i-1]-N[i]-CA[i]-C[i] (phi ϕ), N[i]-CA[i]-C[i]-N[i+1] (psi ψ)
        da_CA_C_N_CA = self.dac(
            atom_CA[:-1], atom_C[:-1], atom_N[1:], atom_CA[1:]
        )  # N_res - 1
        da_C_N_CA_C = self.dac(
            atom_C[:-1], atom_N[1:], atom_CA[1:], atom_C[1:]
        )  # N_res - 1
        da_N_CA_C_N = self.dac(
            atom_N[:-1], atom_CA[:-1], atom_C[:-1], atom_N[1:]
        )  # N_res - 1

        item[self.set_key] = {
            "bl_N_CA": bl_N_CA,  # N_res
            "bl_CA_C": bl_CA_C,  # N_res
            "bl_C_N": bl_C_N,  # N_res - 1
            "ba_C_N_CA": ba_C_N_CA,  # N_res - 1
            "ba_N_CA_C": ba_N_CA_C,  # N_res
            "ba_CA_C_N": ba_CA_C_N,  # N_res - 1
            "da_CA_C_N_CA": da_CA_C_N_CA,  # N_res - 1
            "da_C_N_CA_C": da_C_N_CA_C,  # N_res - 1
            "da_N_CA_C_N": da_N_CA_C_N,  # N_res - 1
        }
        return item

    def collate(self, samples: List[dict], batch: dict, padlen: int) -> dict:
        batch["internal"] = dict()
        for key in [
            "bl_N_CA",
            "bl_CA_C",
            "bl_C_N",
            "ba_C_N_CA",
            "ba_N_CA_C",
            "ba_CA_C_N",
            "da_CA_C_N_CA",
            "da_C_N_CA_C",
            "da_N_CA_C_N",
        ]:
            batch["internal"][key] = torch.cat(
                [
                    pad_nd_seq_unsqueeze(
                        sample["internal"][key], padlen, 0, 1e-6
                    )  # torch.inf)
                    for sample in samples
                ]
            )
        return batch


class InternalToCRAB(BaseTransform):
    def __init__(
        self,
        set_key: str = "crab_rebuilt",
        eps: float = 1e-6,
    ):
        self.fac = FourthAtomCalculator(eps)
        self.set_key = set_key
        self.eps = eps

    def forward(
        self,
        item: dict,
        # bl_N_CA_mark: bool,
        # bl_CA_C_mark: bool,
        # bl_C_N_mark: bool,
        # ba_C_N_CA_mark: bool,
        # ba_N_CA_C_mark: bool,
        # ba_CA_C_N_mark: bool,
    ) -> dict:
        """Convert the Structure instance to internal coordinates."""
        # input:
        #   aa int64 [B, L] pos, pos_mask, ang, ang_mask, name, size
        # crab: do not have cls and eos token
        #   C int64 [B, L], chain identifier
        #   R int64 [B, L], amino acid token from Alphabet()
        #   A float32 [B, L, 4, 3], coordinates of protein backbone, in order of [N, CA, C, O]
        #   padding_mask: [B, L]
        # internal:
        #   bl_N_CA float32 [B, L], bond length between N and CA, first N_res are valid
        #   bl_CA_C float32 [B, L], bond length between CA and C, first N_res are valid
        #   bl_C_N float32 [B, L], bond length between C and N, first N_res-1 are valid
        #   ba_C_N_CA float32 [B, L], bond angle between C, N, CA, first N_res-1 are valid
        #   ba_N_CA_C float32 [B, L], bond angle between N, CA, C, first N_res are valid
        #   ba_CA_C_N float32 [B, L], bond angle between CA, C, N, first N_res-1 are valid
        #   da_CA_C_N_CA float32 [B, L], dihedral angle between CA, C, N, CA, first N_res-1 are valid
        #   da_C_N_CA_C float32 [B, L], dihedral angle between C, N, CA, C, first N_res-1 are valid
        #   da_N_CA_C_N float32 [B, L], dihedral angle between N, CA, C, N, first N_res-1 are valid

        # use internal coordinates to generate crab coordinates
        # bl_N_CA = item["internal"]["bl_N_CA"]
        # bl_CA_C = item["internal"]["bl_CA_C"]
        # bl_C_N = item["internal"]["bl_C_N"]
        # ba_C_N_CA = item["internal"]["ba_C_N_CA"]
        # ba_N_CA_C = item["internal"]["ba_N_CA_C"]
        # ba_CA_C_N = item["internal"]["ba_CA_C_N"]
        # da_CA_C_N_CA = item["internal"]["da_CA_C_N_CA"]
        # da_C_N_CA_C = item["internal"]["da_C_N_CA_C"]
        # da_N_CA_C_N = item["internal"]["da_N_CA_C_N"]

        bl_N_CA = item["rebuilt"]["bl_N_CA"]
        bl_CA_C = item["rebuilt"]["bl_CA_C"]
        bl_C_N = item["rebuilt"]["bl_C_N"]
        ba_C_N_CA = item["rebuilt"]["ba_C_N_CA"]
        ba_N_CA_C = item["rebuilt"]["ba_N_CA_C"]
        ba_CA_C_N = item["rebuilt"]["ba_CA_C_N"]
        da_CA_C_N_CA = item["rebuilt"]["da_CA_C_N_CA"]
        da_C_N_CA_C = item["rebuilt"]["da_C_N_CA_C"]
        da_N_CA_C_N = item["rebuilt"]["da_N_CA_C_N"]

        C, R, A = (
            item["crab"]["C"],
            item["crab"]["R"],
            torch.full(item["crab"]["A"].shape, 0.0, device=item["crab"]["A"].device),
        )

        # from sfm.data.prot_data import crab

        # batch_lengths = []
        # for batch in R:
        #     length = torch.sum(batch != 1)
        #     batch_lengths.append(length.item())

        # bl_N_CA_statistic = torch.tensor(
        #     [[crab.amino_acid_dict[VOCAB.all_toks[v]][0] for v in row] for row in R]
        # )
        # bl_CA_C_statistic = torch.tensor(
        #     [[crab.amino_acid_dict[VOCAB.all_toks[v]][1] for v in row] for row in R]
        # )
        # bl_C_N_statistic = torch.tensor(
        #     [[crab.amino_acid_dict[VOCAB.all_toks[v]][2] for v in row] for row in R]
        # )
        # ba_C_N_CA_statistic = torch.deg2rad(
        #     torch.tensor(
        #         [
        #             [crab.amino_acid_dict_angle[VOCAB.all_toks[v]][2] for v in row]
        #             for row in R[:, 1:]
        #         ]
        #     )
        # )
        # ba_N_CA_C_statistic = torch.deg2rad(
        #     torch.tensor(
        #         [
        #             [crab.amino_acid_dict_angle[VOCAB.all_toks[v]][0] for v in row]
        #             for row in R
        #         ]
        #     )
        # )
        # ba_CA_C_N_statistic = torch.deg2rad(
        #     torch.tensor(
        #         [
        #             [crab.amino_acid_dict_angle[VOCAB.all_toks[v]][1] for v in row]
        #             for row in R
        #         ]
        #     )
        # )

        # for i, length in enumerate(batch_lengths):
        #     batch_index = torch.tensor(i)
        #     inf_index = length - 1
        #     bl_C_N_statistic[batch_index, inf_index] = torch.inf
        #     ba_CA_C_N_statistic[batch_index, inf_index] = torch.inf

        # inf_tensor = torch.full(
        #     (len(R), 1),
        #     float("inf"),
        #     dtype=ba_C_N_CA_statistic.dtype,
        #     device=ba_C_N_CA_statistic.device,
        # )
        # ba_C_N_CA_statistic = torch.cat((ba_C_N_CA_statistic, inf_tensor), dim=1)

        # if bl_N_CA_mark is True:
        #     bl_N_CA = bl_N_CA_statistic
        # if bl_C_N_mark is True:
        #     bl_C_N = bl_C_N_statistic
        # if bl_CA_C_mark is True:
        #     bl_CA_C = bl_CA_C_statistic
        # if ba_C_N_CA_mark is True:
        #     ba_C_N_CA = ba_C_N_CA_statistic
        # if ba_N_CA_C_mark is True:
        #     ba_N_CA_C = ba_N_CA_C_statistic
        # if ba_CA_C_N_mark is True:
        #     ba_CA_C_N = ba_CA_C_N_statistic

        # set the coordinates for the first residue,
        # N is at (-bl_N_CA, 0, 0), CA is at (0, 0, 0), C is at the xy plane.

        A[:, 0, 1, :] = torch.tensor([0.0, 0.0, 0.0])
        A[:, 0, 0, :] = A[:, 0, 1, :]
        A[:, 0, 0, 0] = A[:, 0, 0, 0] - bl_N_CA[:, 0]
        A[:, 0, 2, :] = A[:, 0, 1, :]
        A[:, 0, 2, 0] = A[:, 0, 2, 0] - bl_CA_C[:, 0] * torch.cos(ba_N_CA_C[:, 0])
        A[:, 0, 2, 1] = A[:, 0, 2, 1] + bl_CA_C[:, 0] * torch.sin(ba_N_CA_C[:, 0])

        # rebuild the coordinates for the rest of the residues
        for i in range(1, len(R[0])):
            A[:, i, 0, :] = self.fac(
                A[:, i - 1, 0, :],
                A[:, i - 1, 1, :],
                A[:, i - 1, 2, :],
                bl_C_N[:, i - 1],
                ba_CA_C_N[:, i - 1],
                da_N_CA_C_N[:, i - 1],
            )

            A[:, i, 1, :] = self.fac(
                A[:, i - 1, 1, :],
                A[:, i - 1, 2, :],
                A[:, i, 0, :],
                bl_N_CA[:, i],
                ba_C_N_CA[:, i - 1],
                da_CA_C_N_CA[:, i - 1],
            )

            A[:, i, 2, :] = self.fac(
                A[:, i - 1, 2, :],
                A[:, i, 0, :],
                A[:, i, 1, :],
                bl_CA_C[:, i],
                ba_N_CA_C[:, i],
                da_C_N_CA_C[:, i - 1],
            )

        item[self.set_key] = {
            "C": C,  # [Batch, Length]
            "R": R,  # [Batch, Length]
            "A": A,  # [Batch, Length, 3, 3], only contain N, CA, C
        }

        return item


class ToxInternalLMDBDataset(FoundationModelDataset):
    """
    This is a dataset for protein information, including amino acid, position, angles and confidence score.
    All the information are raw data. Please ues other dataset to process the data, eg, tokenize, encode...

    The process pipeline will be changed in the future, but the interface will not change.
    """

    def __init__(self, args: Any) -> None:
        self.args = self.check_args(args)
        self.data_path = args.data_path
        self.seed = args.seed
        self.max_length = args.max_length
        self.vocab = Alphabet()
        assert args.num_residues == len(
            self.vocab.tok_to_idx
        ), f"vocab size is not equal to args.num_residues. {len(self.vocab.tok_to_idx)} != {args.num_residues}"
        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None
        # eg: transform_str = "FromNumpy(), ItemToCRABBackBone(self.vocab.cls_idx, self.vocab.eos_idx), CRABToInternal(),"
        # all globals and locals are available
        self.data_transform = Compose(eval(args.transform_str, globals(), locals()))

    # Temp method to check if all required arguments are present
    def check_args(self, args):
        required_lst = [
            "data_path",
            "seed",
            "max_length",
            "min_length",
            "num_residues",
            "transform_str",
        ]
        for k in required_lst:
            assert hasattr(
                args, k
            ), f"args should have {k} attribute in {self.__class__.__name__} class."
        return args

    def _init_db(self):
        self._env = lmdb.open(
            str(self.data_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self._sizes, self._keys = metadata["sizes"], metadata["keys"]
        self.filter_indices_by_size(self.args.min_length, 1000000)

    @property
    def env(self):
        if self._env is None:
            self._init_db()
        return self._env

    @property
    def txn(self):
        if self._txn is None:
            self._init_db()
        return self._txn

    @property
    def sizes(self):
        if self._sizes is None:
            self._init_db()
        return self._sizes

    @property
    def keys(self):
        if self._keys is None:
            self._init_db()
        return self._keys

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(self.seed).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(self.args)
        dataset_train._keys = [self.keys[idx] for idx in training_indices]
        dataset_train._sizes = [self.sizes[idx] for idx in training_indices]
        dataset_train.filter_indices_by_size(self.args.min_length, 1000000)

        dataset_val = self.__class__(self.args)
        dataset_val._keys = [self.keys[idx] for idx in validation_indices]
        dataset_val._sizes = [self.sizes[idx] for idx in validation_indices]
        dataset_val.filter_indices_by_size(self.args.min_length, 1000000)

        if sort:
            dataset_train.__sort__()
            dataset_val.__sort__()

        return dataset_train, dataset_val

    def __getitem__(self, index: int) -> dict:
        def pad_concat(nparr, pre_pad, post_pad, value):
            return np.concatenate(
                [
                    np.full(
                        [pre_pad] + list(nparr.shape[1:]),
                        fill_value=value,
                        dtype=nparr.dtype,
                    ),
                    nparr,
                    np.full(
                        [post_pad] + list(nparr.shape[1:]),
                        fill_value=value,
                        dtype=nparr.dtype,
                    ),
                ],
                axis=0,
            )

        key = self.keys[index]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)

        raw_length = len(data["aa"])
        if raw_length > self.max_length - 2:
            start = random.randint(0, raw_length - self.max_length + 2)
            end = start + self.max_length - 2
        else:
            start, end = 0, raw_length

        aa = np.array(
            [self.vocab.cls_idx]
            + [self.vocab.tok_to_idx[tok] for tok in data["aa"][start:end]]
            + [self.vocab.eos_idx],
            dtype=np.int64,
        )
        pos = pad_concat(data["pos"][start:end], 1, 1, 0.0)  # np.inf)
        # pos_mask = pad_concat(data["pos_mask"][start:end], 1, 1, 0)
        # ang = pad_concat(data["ang"][start:end], 1, 1, np.inf)
        # ang_mask = pad_concat(data["ang_mask"][start:end], 1, 1, 0)

        item = {
            "input": {
                "aa": aa,
                "pos": pos,
                # "pos_mask": pos_mask,
                # "ang": ang,
                # "ang_mask": ang_mask,
                "name": data["name"],
                # "size": data["size"],
            }
        }
        item = self.data_transform(item)
        return item

    def __len__(self) -> int:
        return len(self.keys)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index] + 2

    def filter_indices_by_size(
        self, min_size: Union[int, float], max_size: Union[int, float]
    ):
        indices = [idx for idx, s in enumerate(self.sizes) if min_size <= s < max_size]
        logger.warning(
            f"Removed {len(self.sizes) - len(indices)} examples from the dataset because they are outside of interval [{min_size}+2, {max_size}+2) (CLS + EOS)."
        )
        self._sizes = [self.sizes[idx] for idx in indices]
        self._keys = [self.keys[idx] for idx in indices]

    def collate_input(self, samples: List[dict], batch: dict, padlen: int) -> dict:
        batch["input"] = dict()
        batch["input"]["aa"] = torch.cat(
            [
                pad_nd_seq_unsqueeze(
                    sample["input"]["aa"], padlen, 0, self.vocab.padding_idx
                )
                for sample in samples
            ],
            dim=0,
        )
        batch["input"]["pos"] = torch.cat(
            [
                pad_nd_seq_unsqueeze(sample["input"]["pos"], padlen, 0, torch.inf)
                for sample in samples
            ],
            dim=0,
        )
        # batch["input"]["pos_mask"] = torch.cat(
        #     [
        #         pad_nd_seq_unsqueeze(sample["input"]["pos_mask"], padlen, 0, 0)
        #         for sample in samples
        #     ],
        #     dim=0,
        # )
        # batch["input"]["ang"] = torch.cat(
        #     [
        #         pad_nd_seq_unsqueeze(sample["input"]["ang"], padlen, 0, torch.inf)
        #         for sample in samples
        #     ],
        #     dim=0,
        # )
        # batch["input"]["ang_mask"] = torch.cat(
        #     [
        #         pad_nd_seq_unsqueeze(sample["input"]["ang_mask"], padlen, 0, 0)
        #         for sample in samples
        #     ],
        #     dim=0,
        # )
        batch["input"]["name"] = [sample["input"]["name"] for sample in samples]
        return batch

    def collate(self, samples: List[dict]) -> dict:
        padlen = max([len(sample["input"]["aa"]) for sample in samples])
        batch = dict()

        # batching input
        batch = self.collate_input(samples, batch, padlen)
        # batching in transforms
        for transform in self.data_transform.transforms:
            if hasattr(transform, "collate"):
                batch = transform.collate(samples, batch, padlen)

        return batch


class BatchedDataDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        args=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.args = args

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        return item

    def __len__(self):
        return len(self.dataset)

    def collate(self, samples):
        return self.dataset.collate(samples)

    def num_tokens(self, index: int) -> int:
        return self.dataset.sizes[index]


def print_data(dic, prefix=""):
    for k, v in dic.items():
        if isinstance(v, torch.Tensor):
            print(f"{prefix}{k}: Tensor {v.dtype} {v.size()}")
        elif isinstance(v, np.ndarray):
            print(f"{prefix}{k}: np.ndarray {v.dtype} {v.shape}")
        elif isinstance(v, dict):
            print(f"{prefix}{k}: dict with keys: {v.keys()}")
            print_data(v, prefix + "  ")
        else:
            print(f"{prefix}{k}: {type(v)} {v}")


def validata_item(item):
    assert "input" in item
    assert "aa" in item["input"]
    assert len(item["input"]["aa"]) == len(item["input"]["pos"])
    assert "pos" in item["input"]
    assert "pos_mask" in item["input"]
    assert (item["input"]["pos_mask"][1:-1, [0, 1, 2, 4]] == 1).all()
    if not (item["input"]["aa"] == 6).any():
        print("GLY not in aa, CB test not work")
    assert (
        (item["input"]["pos_mask"][1:-1, 3] == 0) == (item["input"]["aa"][1:-1] == 6)
    ).all()
    # assert "ang" in item["input"]
    # assert "ang_mask" in item["input"]
    assert "name" in item["input"]
    assert "size" in item["input"]
    assert item["input"]["size"] == len(item["input"]["aa"]) - 2
    if "crab" in item:
        assert "C" in item["crab"]
        assert len(item["crab"]["C"]) == item["input"]["size"]
        assert "R" in item["crab"]
        assert len(item["crab"]["R"]) == item["input"]["size"]
        assert "A" in item["crab"]
        assert item["crab"]["A"].size()[0] == item["input"]["size"]
    else:
        print("No crab in item")

    if "internal" in item:
        assert "bl_N_CA" in item["internal"]
        assert item["internal"]["bl_N_CA"].size()[0] == item["input"]["size"]
        assert "bl_CA_C" in item["internal"]
        assert item["internal"]["bl_CA_C"].size()[0] == item["input"]["size"]
        assert "bl_C_N" in item["internal"]
        assert item["internal"]["bl_C_N"].size()[0] == item["input"]["size"] - 1
        assert "ba_C_N_CA" in item["internal"]
        assert item["internal"]["ba_C_N_CA"].size()[0] == item["input"]["size"] - 1
        assert "ba_N_CA_C" in item["internal"]
        assert item["internal"]["ba_N_CA_C"].size()[0] == item["input"]["size"]
        assert "ba_CA_C_N" in item["internal"]
        assert item["internal"]["ba_CA_C_N"].size()[0] == item["input"]["size"] - 1
        assert "da_CA_C_N_CA" in item["internal"]
        assert item["internal"]["da_CA_C_N_CA"].size()[0] == item["input"]["size"] - 1
        assert "da_C_N_CA_C" in item["internal"]
        assert item["internal"]["da_C_N_CA_C"].size()[0] == item["input"]["size"] - 1
        assert "da_N_CA_C_N" in item["internal"]
        assert item["internal"]["da_N_CA_C_N"].size()[0] == item["input"]["size"] - 1
    else:
        print("No internal in item")

    if "mask" in item:
        assert "mask_seq" in item["mask"]
        assert item["mask"]["mask_seq"].size()[0] == item["input"]["size"]
        assert item["mask"]["mask_seq"][item["input"]["size"] :].sum() == 0
        assert "mask_str" in item["mask"]
        assert item["mask"]["mask_str"].size()[0] == item["input"]["size"]
        assert item["mask"]["mask_str"][item["input"]["size"] :].sum() == 0


if __name__ == "__main__":
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from torch.utils.data import DataLoader as Dataloader
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

    args.data_path = "/mnta/yaosen/data/AFDB30-plddt70.lmdb"
    args.seed = 666
    args.max_length = 1024
    args.num_residues = 32
    args.transform_str = (
        "[FromNumpy(), ItemToCRABBackBone(), CRABToInternal(), BERTMasking()]"
    )

    dataset = ToxInternalLMDBDataset(args)
    trainset, valset = dataset.split_dataset(sort=False)
    trainset_batch = Dataloader(
        BatchedDataDataset(trainset, args),
        batch_size=7,
        collate_fn=trainset.collate,
        num_workers=8,
    )

    for i, t in enumerate(trainset.vocab.all_toks):
        print(i, t, sep="\t")

    print(f"lenght of trainset: {len(trainset)}, lenght of valset: {len(valset)}")
    print_data(trainset[0])
    validata_item(trainset[2])
    print()
    print_data(next(iter(trainset_batch)))

    print("#####")
    print_data(next(iter(trainset_batch)))
    i2c = InternalToCRAB()
    i2c.forward(next(iter(trainset_batch)), True, True, True, True, True, True)
