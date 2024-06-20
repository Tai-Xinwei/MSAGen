# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import bisect
import glob
import os
import pickle as pkl
import random
from pathlib import Path
from typing import Any, List, Optional, Union

import lmdb
import torch
from numpy import dot
from numpy.linalg import norm
from sympy.utilities.iterables import multiset_permutations
from torch.utils.data import Subset
from torch_geometric.data import Data
from tqdm import tqdm

from sfm.data.data_utils import _filter_by_size_dynamic
from sfm.data.dataset import FoundationModelDataset
from sfm.data.mol_data import algos
from sfm.data.prot_data.util import bstr2obj
from sfm.data.psm_data.collator import collate_fn
from sfm.data.psm_data.utils import (
    PM6_ATOM_REFERENCE_LIST,
    get_conv_variable_lin,
    get_data_defult_config,
    matrixtoblock_lin,
)
from sfm.logging import logger
from sfm.models.psm.psm_config import PSMConfig


class MoleculeLMDBDataset(FoundationModelDataset):
    energy_mean: float = 0.0
    energy_std: float = 1.0
    energy_per_atom_mean: float = 0.0
    energy_per_atom_std: float = 1.0
    force_mean: float = 0.0  # force mean should always be 0.0 to keep equivariance
    force_std: float = 1.0

    def __init__(self, args: PSMConfig, lmdb_path: str) -> None:
        assert lmdb_path, "LMDB path must be provided"
        self.lmdb_path = lmdb_path

        self.args = args
        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None
        self.PM6_ATOM_REFERENCE_tensor = torch.tensor(
            PM6_ATOM_REFERENCE_LIST, dtype=torch.float64
        )
        self.filter_indices_by_size(
            indices=np.array(range(len(self.keys))), max_sizes=self.args.max_length - 2
        )

    def _ensure_init_db(self):
        if self._env is not None:
            return
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self._env.begin(write=False)
        metadata = bstr2obj(self._txn.get("metadata".encode()))
        self._keys = [str(key) for key in metadata["keys"]]
        self._sizes = metadata["size"] if "size" in metadata else metadata["sizes"]

    def _close_db(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._txn = None

    @property
    def env(self):
        self._ensure_init_db()
        return self._env

    @property
    def txn(self):
        self._ensure_init_db()
        return self._txn

    @property
    def sizes(self):
        self._ensure_init_db()
        return self._sizes

    @property
    def keys(self):
        self._ensure_init_db()
        return self._keys

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(self.args, self.lmdb_path)
        dataset_train._keys = [self._keys[idx] for idx in training_indices]
        dataset_train._sizes = [self._sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.args, self.lmdb_path)
        dataset_val._keys = [self._keys[idx] for idx in validation_indices]
        dataset_val._sizes = [self._sizes[idx] for idx in validation_indices]

        return dataset_train, dataset_val

    def raw(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = pkl.loads(value)

        return data

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = pkl.loads(value)

        # node features conversion for embedding, [6, 1, 0, 2] -> [6, 1 + 512, 0 + 512 x 2, 2 + 512 x 3]
        # note that in node_feat from ogb smiles2graph, hydrogen is represented by number 0 in the first dimension of node features
        # see https://github.com/snap-stanford/ogb/blob/f631af76359c9687b2fe60905557bbb241916258/ogb/utils/features.py#L60
        # +1 for the atomic number here to be consistent with other datasets
        data["node_feat"][:, 0] += 1
        data["node_feat"] = convert_to_single_emb(
            torch.tensor(data["node_feat"], dtype=torch.long)
        )
        if "pos" in data:
            coords = torch.tensor(data["pos"], dtype=torch.float64)
        elif "coords" in data:
            coords = torch.tensor(data["coords"], dtype=torch.float64)
        else:
            coords = torch.zeros((data["node_feat"].size()[0], 3), dtype=torch.float64)

        x = data["node_feat"]

        data["sample_type"] = 0
        data["token_type"] = data["node_feat"][
            :, 0
        ]  # token type only records the atomic numbers
        data["idx"] = idx

        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64)
        data["forces"] = torch.zeros((x.size()[0], 3), dtype=torch.float64)

        if "energy" in data or "total_energy" in data:
            total_energy = data["energy"] if "energy" in data else data["total_energy"]
            data["energy"] = torch.tensor(
                [(total_energy - self.energy_mean) / self.energy_std]
            )
            # data["energy_per_atom"] = torch.tensor(
            #     [
            #         (
            #             total_energy / float(data["num_atoms"])
            #             - self.energy_per_atom_mean
            #         )
            #         / self.energy_per_atom_std
            #     ]
            # )

            reference_energy = (
                torch.gather(self.PM6_ATOM_REFERENCE_tensor, 0, data["token_type"] - 1)
                .sum()
                .unsqueeze(0)
            )
            data["energy_per_atom"] = (
                torch.tensor(total_energy) - reference_energy
            ) / data["num_atoms"]
        else:
            data["energy"] = torch.tensor([0.0], dtype=torch.float64)
            data["energy_per_atom"] = torch.tensor([0.0], dtype=torch.float64)

        data["has_energy"] = torch.tensor([1], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)
        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False

        return data

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data["edge_feat"], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(
            edge_attr
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)
        adj[edge_index[1, :], edge_index[0, :]] = True

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["node_feat"]

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            max_dist = np.amax(shortest_path_result)
            edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = torch.tensor(edge_input, dtype=torch.long)
            data["spatial_pos"] = spatial_pos

        return data

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
            if hasattr(self, "_sizes") and isinstance(self._sizes, np.ndarray):
                ignored = indices[self._sizes[indices] > max_sizes].tolist()
                indices = indices[self._sizes[indices] <= max_sizes]
            elif hasattr(self, "_sizes") and isinstance(self._sizes, list):
                sizes = np.array(self._sizes)
                ignored = indices[np.array(sizes[indices]) > max_sizes].tolist()
                indices = indices[np.array(sizes[indices]) <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self._sizes, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(indices, self._sizes, max_sizes)

        logger.warning(
            f"Removed {len(ignored)} examples from the {self.lmdb_path} because they are longer than {max_sizes}."
        )
        self._sizes = [self._sizes[idx] for idx in indices]
        self._keys = [self._keys[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.keys)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]


class PM6FullLMDBDataset(MoleculeLMDBDataset):
    latest_version = "20240527.1"
    energy_mean: float = -42774.16038176129
    energy_std: float = 25029.68158883449
    energy_per_atom_mean: float = -994.0920019593214
    energy_per_atom_std: float = 770.7496116135809

    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: str,
        version: Optional[str] = None,
    ):
        path = os.path.normpath(lmdb_path)
        if path.endswith("PubChemQC-B3LYP-PM6"):
            path = os.path.join(
                path, version or PM6FullLMDBDataset.latest_version, "full"
            )
        super().__init__(args, path)


class PlainPM6FullLMDBDataset(PM6FullLMDBDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
    ):
        super().__init__(args, lmdb_path)

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data["edge_feat"], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(
            edge_attr
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["node_feat"]

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        data["adj"] = adj
        data["attn_edge_type"] = attn_edge_type

        return data


class MatterSimDataset:
    def __init__(self, args: PSMConfig, data_path, split):
        self.data_lmdb = None
        self.data_txn = None
        self.index_to_key_name = []
        self.data_path = data_path
        lmdb_path = f"{self.data_path}/{split}"
        self.data_lmdb = lmdb.open(
            lmdb_path,
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.data_txn = self.data_lmdb.begin(write=False)
        self.index_to_key_name = bstr2obj(
            self.data_txn.get("index_to_key_name".encode())
        )
        self.args = args

    def switch_lattice_vectors(self, pbc, cell):
        # simple algorithm to switch lattice vectors so that they are more aligned with the initial lattice vectors
        initial_lattice_vectors = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        best_permutation = None
        best_lattice_flip_sign = None
        max_cosine_sum = 0.0
        for permutation in multiset_permutations(np.arange(3)):
            cosine = 0.0
            lattice_flip_sign = []
            for i in range(3):
                index = permutation[i]
                original_lattice_vector = cell[index]
                initial_lattice_vector = initial_lattice_vectors[i]
                cosine_similarity = dot(
                    original_lattice_vector, initial_lattice_vector
                ) / (norm(original_lattice_vector) * norm(initial_lattice_vector))
                cosine += np.abs(cosine_similarity)
                lattice_flip_sign.append(-1.0 if cosine_similarity < 0.0 else 1.0)
            if cosine > max_cosine_sum:
                best_permutation = permutation
                max_cosine_sum = cosine
                best_lattice_flip_sign = lattice_flip_sign
        pbc = pbc[best_permutation]
        cell = cell[best_permutation] * np.array(best_lattice_flip_sign)[:, None]
        return pbc, cell

    # energy and std calculated over training part of the dataset
    @property
    def energy_mean(self):
        return -66.0996156928496

    @property
    def energy_std(self):
        return 102.91694201560776

    @property
    def energy_per_atom_mean(self):
        return -4.707989414326259

    @property
    def energy_per_atom_std(self):
        return 3.7324579639110653

    @property
    def force_mean(self):  # force mean should always be 0.0 to keep equivariance
        return 0.0

    @property
    def force_std(self):
        return 2.155674863803223

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        key = self.index_to_key_name[idx]
        data = pkl.loads(self.data_txn.get(key.encode()))
        numbers = data.pop(
            "numbers"
        )  # atomic numbers, starting from 1 for hydrogen atoms
        x = torch.tensor(numbers, dtype=torch.long)
        x = torch.cat([x, torch.full([8], 128)], dim=-1)

        positions = data.pop("positions")

        data["sample_type"] = 1
        data["coords"] = torch.tensor(positions, dtype=torch.float64)
        data["token_type"] = x
        data["idx"] = idx

        data["pbc"], data["cell"] = self.switch_lattice_vectors(
            data["pbc"], data["cell"]
        )
        data["cell"] = torch.tensor(data["cell"], dtype=torch.float64)
        data["pbc"] = torch.tensor(data["pbc"], dtype=torch.bool)

        cell_corner_pos_matrix = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        cell_corner_pos = torch.matmul(cell_corner_pos_matrix, data["cell"])
        data["coords"] = torch.cat(
            [data["coords"], cell_corner_pos], dim=0
        )  # expand pos with cell corners
        data["num_atoms"] = int(x.size()[0] - 8)
        data["forces"] = torch.cat(
            [
                (torch.tensor(data["forces"], dtype=torch.float64) - self.force_mean)
                / self.force_std,
                torch.zeros([8, 3], dtype=torch.float64),
            ],
            dim=0,
        )  # expand forces for cell corners
        data["energy"] = torch.tensor(
            [(data["info"]["energy"] - self.energy_mean) / self.energy_std]
        )
        data["energy_per_atom"] = torch.tensor(
            [
                (
                    (data["info"]["energy"] / float(data["num_atoms"]))
                    - self.energy_per_atom_mean
                )
                / self.energy_per_atom_std
            ]
        )
        data["stress"] = torch.tensor(data["info"]["stress"], dtype=torch.float64)

        data["has_energy"] = torch.tensor([1], dtype=torch.bool)
        data["has_forces"] = torch.tensor([1], dtype=torch.bool)
        data = self.generate_2dgraphfeat(data)

        if self.data_path.find("force-filtered") != -1:
            data["is_stable_periodic"] = True
        else:
            data["is_stable_periodic"] = False

        return data

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"] + 8
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"].unsqueeze(-1),
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result = (
                torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
            )
            edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = edge_input
            data["spatial_pos"] = spatial_pos

        return data

    def __len__(self):
        return len(self.index_to_key_name)


class AFDBLMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
    ):
        self.lmdb_path = lmdb_path
        self.args = args

        self.vocab = {
            # "<pad>": 0,  # padding
            # "1"-"127": 1-127, # atom type
            # "<cell_corner>": 128, use for pbc material
            "L": 130,
            "A": 131,
            "G": 132,
            "V": 133,
            "S": 134,
            "E": 135,
            "R": 136,
            "T": 137,
            "I": 138,
            "D": 139,
            "P": 140,
            "K": 141,
            "Q": 142,
            "N": 143,
            "F": 144,
            "Y": 145,
            "M": 146,
            "H": 147,
            "W": 148,
            "C": 149,
            "X": 150,
            "B": 151,
            "U": 152,
            "Z": 153,
            "O": 154,
            "-": 155,
            ".": 156,
            "<mask>": 157,
            "<cls>": 158,
            "<eos>": 159,
            # "<unk>": 160,
        }

        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None

        self.filter_indices_by_size(
            indices=np.array(range(len(self.keys))), max_sizes=self.args.max_length - 2
        )

    def _init_db(self):
        self._env = lmdb.open(
            str(self.lmdb_path),
            subdir=True,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._txn = self.env.begin(write=False)
        metadata = bstr2obj(self.txn.get("__metadata__".encode()))
        self._sizes, self._keys = metadata["sizes"], metadata["keys"]

    def _close_db(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._txn = None

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

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)

        # minus 1 due to add padding index=0 in collator
        x = torch.tensor([self.vocab[tok] - 1 for tok in data["aa"]], dtype=torch.int64)
        # CA atom positions, assume all values are valid.
        coords = data["pos"][:, 1, :]

        data["sample_type"] = 2
        data["token_type"] = x
        data["idx"] = idx

        coords = torch.tensor(coords, dtype=torch.float64)
        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)
        data["energy_per_atom"] = torch.tensor(
            [0.0], dtype=torch.float64, device=x.device
        )

        data["has_energy"] = torch.tensor([0], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False

        return data

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = len(self.keys)
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        # Create training and validation datasets
        dataset_train = self.__class__(self.args, self.lmdb_path)
        dataset_train._keys = [self._keys[idx] for idx in training_indices]
        dataset_train._sizes = [self._sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.args, self.lmdb_path)
        dataset_val._keys = [self._keys[idx] for idx in validation_indices]
        dataset_val._sizes = [self._sizes[idx] for idx in validation_indices]

        return dataset_train, dataset_val

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
            if hasattr(self, "_sizes") and isinstance(self._sizes, np.ndarray):
                ignored = indices[self._sizes[indices] > max_sizes].tolist()
                indices = indices[self._sizes[indices] <= max_sizes]
            elif hasattr(self, "_sizes") and isinstance(self._sizes, list):
                sizes = np.array(self._sizes)
                ignored = indices[np.array(sizes[indices]) > max_sizes].tolist()
                indices = indices[np.array(sizes[indices]) <= max_sizes]
            else:
                indices, ignored = _filter_by_size_dynamic(
                    indices, self._sizes, max_sizes
                )
        else:
            indices, ignored = _filter_by_size_dynamic(indices, self._sizes, max_sizes)

        logger.warning(
            f"Removed {len(ignored)} examples from the AFDBLMDBDataset because they are longer than {max_sizes}."
        )
        self._sizes = [self._sizes[idx] for idx in indices]
        self._keys = [self._keys[idx] for idx in indices]

    # protein does not have 2dgraph, create one for mixing data
    def generate_2dgraphfeat(self, data):
        N = data["token_type"].shape[0]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.zeros([2, 0], dtype=torch.long)
        edge_attr = torch.zeros([0, 3], dtype=torch.long)
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = torch.cat(
            [
                data["token_type"].unsqueeze(-1),
                torch.zeros([data["token_type"].size()[0], 8], dtype=torch.long),
            ],
            dim=-1,
        )
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result = (
                torch.full(adj.size(), 511, dtype=torch.long).cpu().numpy()
            )
            edge_input = torch.zeros([N, N, 0, 3], dtype=torch.long)
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = edge_input
            data["spatial_pos"] = spatial_pos

        return data

    def __len__(self) -> int:
        return len(self.keys)

    def num_tokens(self, index: int) -> int:
        return self.sizes[index]


class SmallMolDataset(FoundationModelDataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.
    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.
    Args:
            @path: the path to store the data
            @task: the task for the lmdb dataset
            @split: splitting of the data
            @transform: some transformation of the data
    """

    energy = "energy"
    forces = "forces"

    def __init__(
        self,
        args: PSMConfig,
        path,
        data_name="pubchem5w",
        transforms=[],
        enable_hami=False,
        remove_init=False,
        remove_atomref_energy=True,
        Htoblock_otf=True,  ## on save H matrix, H to block is process in collate unifined for memory saving.
        basis="def2-tzvp",
        ## 1 kcal/mol = 0.0433634 eV, transform to eV by default here
    ):
        super(SmallMolDataset, self).__init__()
        if data_name.lower() == "pubchem5w":
            if basis != "def2-tzvp":
                raise ValueError(
                    "sorry, when using pubchem the basis should be def2-tzvp"
                )
        (
            self.atom_reference,
            self.system_ref,
            _,
            _,
            _,
            self.has_energy,
            self.has_forces,
            self.is_pbc,
            self.unit,
        ) = get_data_defult_config(data_name)
        db_paths = []
        if isinstance(path, str):
            if path.endswith("lmdb"):
                db_paths.append(path)
            else:
                db_paths.extend(glob.glob(path + "/*.lmdb"))

        elif isinstance(path, list):
            for p in path:
                if p.endswith("lmdb"):
                    db_paths.append(p)
                else:
                    db_paths.extend(glob.glob(p + "/*.lmdb"))
        # print(db_paths)
        assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
        self.enable_hami = enable_hami
        self._keys, self.envs = [], []
        self.db_paths = sorted(db_paths)
        self.open_db()
        self.transforms = transforms  # unused
        self.remove_init = remove_init
        self.remove_atomref_energy = remove_atomref_energy
        self.conv, self.orbitals_ref, self.mask, self.chemical_symbols = (
            None,
            None,
            None,
            None,
        )
        self.Htoblock_otf = Htoblock_otf
        if self.enable_hami:
            self.conv, _, self.mask, _ = get_conv_variable_lin(basis)
        self.args = args

    def open_db(self):
        for db_path in self.db_paths:
            self.envs.append(self.connect_db(db_path))
            length = self.envs[-1].begin().get("length".encode("ascii"))
            if length is not None:
                length = pkl.loads(length)
            else:
                length = self.envs[-1].stat()["entries"]

            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        datapoint_pickled = (
            self.envs[db_idx]
            .begin()
            .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        )
        data_object = pkl.loads(datapoint_pickled)
        data_object.id = el_idx  # f"{db_idx}_{el_idx}"

        energy = data_object.energy
        # out["pyscf_energy"] = copy.deepcopy(energy.astype(np.float32))  # this is pyscf energy ground truth
        if self.remove_atomref_energy:
            unique, counts = np.unique(
                data_object.atomic_numbers.int().numpy(), return_counts=True
            )
            energy = energy - np.sum(self.atom_reference[unique] * counts)
            energy = torch.Tensor([energy - self.system_ref])

        out = {
            "coords": data_object.pos,
            "forces": data_object.forces * self.unit,
            "num_atoms": data_object.pos.shape[0],
            "token_type": data_object.atomic_numbers.int().reshape(-1),
            "idx": idx,
            "edge_index": data_object.edge_index,
            "energy": energy.reshape(-1)
            * self.unit,  # this is used from model training, mean/ref is removed.
            "has_energy": torch.tensor([self.has_energy], dtype=torch.bool),
            "has_forces": torch.tensor([self.has_forces], dtype=torch.bool),
        }
        if self.is_pbc:
            cell_corner_pos_matrix = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )
            out["token_type"] = torch.cat(
                [out["token_type"], torch.full([8], 128)], dim=-1
            )  # convert_to_single_emb(x)
            cell_corner_pos = torch.matmul(
                cell_corner_pos_matrix, data_object.cell.squeeze(dim=0).float()
            )
            out["coords"] = torch.cat(
                [out["coords"], cell_corner_pos], dim=0
            )  # expand pos with cell corners
            out["forces"] = torch.cat(
                [
                    torch.tensor(out["forces"].clone().detach(), dtype=torch.float32),
                    torch.zeros([8, 3], dtype=torch.float32),
                ],
                dim=0,
            )  # expand forces for cell corners

            out["cell"] = data_object.cell.squeeze(dim=0)
            out["pbc"] = torch.ones(3, dtype=torch.float32).bool()
            out["stress"] = torch.zeros(
                (3, 3), dtype=torch.float32, device=energy.device
            )
            out["cell_offsets"] = data_object.cell_offsets.numpy()

            out["energy_per_atom"] = out["energy"] / out["num_atoms"]

        else:
            out["cell"] = torch.zeros((3, 3), dtype=torch.float32)
            out["pbc"] = torch.zeros(3, dtype=torch.float32).bool()
            out["stress"] = torch.zeros(
                (3, 3), dtype=torch.float32, device=energy.device
            )

            out["energy_per_atom"] = out["energy"] / out["num_atoms"]

        if self.enable_hami:
            # out.update({"init_fock":data_object.init_fock.astype(np.float32)})
            if self.remove_init:
                data_object.fock = data_object.fock - data_object.init_fock
            if self.Htoblock_otf is True:
                out.update(
                    {
                        "buildblock_mask": self.mask,
                        "max_block_size": self.conv.max_block_size,
                        "fock": data_object.fock * self.unit,
                    }
                )
            else:
                diag, non_diag, diag_mask, non_diag_mask = None, None, None, None
                diag, non_diag, diag_mask, non_diag_mask = matrixtoblock_lin(
                    data_object.fock,
                    data_object.atomic_numbers,
                    self.mask,
                    self.conv.max_block_size,
                )
                out.update(
                    {
                        "diag_hamiltonian": diag * self.unit,
                        "non_diag_hamiltonian": non_diag * self.unit,
                        "diag_mask": diag_mask,
                        "non_diag_mask": non_diag_mask,
                    }
                )
            out.update({"init_fock": data_object.init_fock * self.unit})
            out.update({"s1e": data_object.s1e * self.unit})

        for key in out.keys():
            if key not in [
                "num_atoms",
                "token_type",
                "idx",
                "edge_index",
                "has_energy",
                "has_forces",
            ]:
                out[key] = torch.tensor(out[key], dtype=torch.float32)

        out = self.generate_2dgraphfeat(out)

        return out

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"].clone().detach(), dtype=torch.long)
        edge_attr = torch.ones((data["edge_index"].shape[1], 1), dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = edge_attr + 1
        adj[edge_index[0, :], edge_index[1, :]] = True
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["token_type"].reshape(-1, 1)

        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree

        if self.args.preprocess_2d_bond_features_with_cuda:
            data["adj"] = adj
            data["attn_edge_type"] = attn_edge_type
        else:
            shortest_path_result, path = algos.floyd_warshall(adj.numpy())
            max_dist = np.amax(shortest_path_result)
            edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            data["edge_input"] = torch.tensor(edge_input, dtype=torch.long)
            data["spatial_pos"] = spatial_pos

        return data

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=32,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
            self.envs = []
        else:
            self.env.close()
            self.env = None

    def split_dataset(self, validation_ratio=0.03, sort=False):
        num_samples = self.num_samples
        # Shuffle the indices and split them into training and validation sets
        indices = list(range(num_samples))
        random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * validation_ratio)
        num_training_samples = num_samples - num_validation_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[num_training_samples:]

        dataset_train = Subset(self, training_indices)
        dataset_val = Subset(self, validation_indices)
        return dataset_train, dataset_val

    def split_train_valid_test(self, ratio_list: list, sort=False, shuffle=True):
        num_samples = self.num_samples

        indices = list(range(num_samples))
        # Shuffle the indices and split them into training and validation sets
        if shuffle:
            random.Random(12345).shuffle(indices)

        num_validation_samples = int(num_samples * ratio_list[1])
        num_test_samples = int(num_samples * ratio_list[2])
        num_training_samples = num_samples - num_validation_samples - num_test_samples

        training_indices = indices[:num_training_samples]
        validation_indices = indices[
            num_training_samples : num_training_samples + num_validation_samples
        ]
        test_indices = indices[num_training_samples + num_validation_samples :]

        dataset_train = Subset(self, training_indices)
        dataset_val = Subset(self, validation_indices)
        dataset_test = Subset(self, test_indices)
        return dataset_train, dataset_val, dataset_test


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class DaliPM6DataSource:
    def __init__(self, args: Any, dataset: Any, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        self.generator = np.random.default_rng(args.seed)

        self.rank = args.rank
        self.world_size = args.world_size

        self._set_sample_indices()

    def _set_sample_indices(self):
        indices = self.generator.permutation(self.size)
        indices = indices[self.rank : self.size : self.world_size]

        self.indices = indices
        self.rank_size = len(indices)

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch

        if sample_info.iteration >= self.rank_size // self.batch_size:
            raise StopIteration

        idx = self.indices[sample_idx]
        data = self.dataset.raw(idx % self.size)

        N = data["node_feat"].shape[0]
        F = data["edge_feat"].shape[-1]

        node_feat = torch.tensor(data["node_feat"], dtype=torch.long)
        if "pos" in data:
            coords = torch.tensor(data["pos"], dtype=torch.float)
        elif "coords" in data:
            coords = torch.tensor(data["coords"], dtype=torch.float)
        else:
            coords = torch.zeros((data["node_feat"].size()[0], 3), dtype=torch.float)

        cell = torch.zeros([3, 3], dtype=torch.float)
        pbc = torch.zeros([3], dtype=torch.long)
        forces = torch.zeros([N, 3], dtype=torch.float)
        energy = torch.tensor(data["energy"], dtype=torch.float)
        adj = torch.zeros([N, N], dtype=torch.long)
        edge_index = torch.tensor(
            np.ascontiguousarray(data["edge_index"]), dtype=torch.long
        )
        edge_attr = torch.tensor(
            np.ascontiguousarray(data["edge_feat"]), dtype=torch.long
        )
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
        attn_edge_type = torch.zeros([N, N, F], dtype=torch.long)
        is_stable_periodic = torch.zeros(1, dtype=torch.bool)

        return (
            node_feat,
            coords,
            cell,
            pbc,
            forces,
            energy,
            adj,
            edge_index,
            edge_attr,
            attn_bias,
            attn_edge_type,
            is_stable_periodic,
        )

    def __len__(self):
        return self.size


class DaliUnifiedDataSource:
    def __init__(self, args: Any, dataset: Any, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

        self.generator = np.random.default_rng(args.seed)

        self.shard_id = args.rank
        self.world_size = args.world_size

        self.shard_size = len(dataset) // self.world_size
        self.total_size = self.shard_size * self.world_size
        self.iterations = self.shard_size // self.batch_size

        self._set_sample_indices()

    def _set_sample_indices(self):
        indices = self.generator.permutation(self.total_size)
        indices = indices[self.shard_id :: self.world_size]

        self.indices = indices

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch

        if sample_info.iteration % self.iterations == 0:
            self._set_sample_indices()

        data_idx = self.indices[sample_idx % self.shard_size]
        data = self.dataset[data_idx % self.total_size]

        attn_bias = data["attn_bias"]
        in_degree = data["in_degree"]
        node_attr = data["node_attr"]
        energy = data["energy"].to(dtype=torch.float)
        energy_per_atom = data["energy_per_atom"].to(dtype=torch.float)
        forces = data["forces"].to(dtype=torch.float)
        pos = data["coords"].to(dtype=torch.float)
        token_type = data["token_type"].contiguous().to(dtype=torch.long)
        pbc = data["pbc"].to(dtype=torch.long)
        cell = data["cell"].to(dtype=torch.float)
        num_atoms = torch.tensor(data["num_atoms"], dtype=torch.long)
        adj = data["adj"]
        attn_edge_type = data["attn_edge_type"]
        is_stable_periodic = torch.tensor(data["is_stable_periodic"], dtype=torch.bool)
        has_energy = data["has_energy"]
        has_forces = data["has_forces"]

        return (
            attn_bias,
            in_degree,
            node_attr,
            energy,
            energy_per_atom,
            forces,
            pos,
            token_type,
            pbc,
            cell,
            num_atoms,
            adj,
            attn_edge_type,
            is_stable_periodic,
            has_energy,
            has_forces,
        )

    def __len__(self):
        return self.size
