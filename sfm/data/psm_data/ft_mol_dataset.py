# -*- coding: utf-8 -*-
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import random
from typing import Optional, Union

import lmdb
import torch
from torch_geometric.data import Data

from sfm.data.data_utils import _filter_by_size_dynamic
from sfm.data.dataset import FoundationModelDataset
from sfm.data.mol_data import algos
from sfm.data.prot_data.util import bstr2obj
from sfm.data.psm_data.collator import collate_fn, pad_pos_unsqueeze
from sfm.data.psm_data.dataset import convert_to_single_emb
from sfm.logging import logger


class GenericMoleculeLMDBDataset(FoundationModelDataset):
    def __init__(
        self,
        args,
        lmdb_path: Optional[str],
    ):
        self.lmdb_path = lmdb_path
        self.args = args
        # for dataloader with num_workers > 1
        self._env, self._txn = None, None
        self._sizes, self._keys = None, None

        self.filter_indices_by_size(
            indices=np.array(range(len(self.keys))), max_sizes=self.args.max_length
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
        metadata = bstr2obj(self.txn.get("metadata".encode()))
        self._sizes, self._keys = metadata["size"], metadata["keys"]

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

    def get_dataset(self, sort=False):
        num_samples = len(self.keys)
        indices = list(range(num_samples))

        dataset = self.__class__(self.args, self.lmdb_path)
        dataset._keys = [self._keys[idx] for idx in indices]
        dataset._sizes = [self._sizes[idx] for idx in indices]

        return dataset

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
        dataset_train = self.__class__(self.args, self.lmdb_path)
        dataset_train._keys = [self._keys[idx] for idx in training_indices]
        dataset_train._sizes = [self._sizes[idx] for idx in training_indices]

        dataset_val = self.__class__(self.args, self.lmdb_path)
        dataset_val._keys = [self._keys[idx] for idx in validation_indices]
        dataset_val._sizes = [self._sizes[idx] for idx in validation_indices]

        return dataset_train, dataset_val

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx]
        value = self.txn.get(key.encode())
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        data = bstr2obj(value)

        data["node_feat"] = torch.tensor(data["node_feat"], dtype=torch.long)
        x = data["node_feat"][:, 0]
        if "pos" in data:
            coords = torch.tensor(data["pos"], dtype=torch.float64)
        elif "coords" in data:
            coords = torch.tensor(data["coords"], dtype=torch.float64)
        else:
            coords = torch.zeros((data["num_nodes"], 3), dtype=torch.float64)

        # x = convert_to_single_emb(x)

        data["sample_type"] = 0
        data["token_type"] = x
        data["idx"] = idx

        data["coords"] = coords
        data["num_atoms"] = x.size()[0]

        data["cell"] = torch.zeros((3, 3), dtype=torch.float64)
        data["pbc"] = torch.zeros(3, dtype=torch.float64).bool()
        data["stress"] = torch.zeros((3, 3), dtype=torch.float64, device=x.device)
        data["forces"] = torch.zeros(
            (x.size()[0], 3), dtype=torch.float64, device=x.device
        )
        if "energy" in data:
            data["energy"] = torch.tensor(
                data["energy"], dtype=torch.float64, device=x.device
            )
        else:
            data["energy"] = torch.tensor([0.0], dtype=torch.float64, device=x.device)

        if "homo_lumo_gap" in data:
            data["homo_lumo_gap"] = torch.tensor(
                data["homo_lumo_gap"], dtype=torch.float64
            )

        data = self.generate_2dgraphfeat(data)

        return data

    def generate_2dgraphfeat(self, data):
        N = data["num_atoms"]
        adj = torch.zeros([N, N], dtype=torch.bool)

        edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(data["edge_feat"], dtype=torch.long)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
            convert_to_single_emb(edge_attr) + 1
        )
        adj[edge_index[0, :], edge_index[1, :]] = True
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        indgree = adj.long().sum(dim=1).view(-1)

        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_attr"] = data["node_feat"]

        data["edge_input"] = torch.tensor(edge_input, dtype=torch.long)
        data["attn_edge_type"] = attn_edge_type
        data["attn_bias"] = torch.zeros([N + 1, N + 1], dtype=torch.float)
        data["in_degree"] = indgree
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
            f"Removed {len(ignored)} examples from the PCQM4Mv2LMDBDataset because they are longer than {max_sizes}."
        )
        self._sizes = [self._sizes[idx] for idx in indices]
        self._keys = [self._keys[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.keys)

    def collate(self, samples):
        batch = collate_fn(samples)

        if all(["homo_lumo_gap" in s for s in samples]):
            batch["homo_lumo_gap"] = torch.cat([s["homo_lumo_gap"] for s in samples])

        if all(["mol" in s for s in samples]):
            batch["mol"] = [s["mol"] for s in samples]

        max_edge_num = max(s["edge_index"].shape[1] for s in samples)
        batch["edge_index"] = torch.cat(
            [pad_pos_unsqueeze(torch.t(s["edge_index"]), max_edge_num) for s in samples]
        )
        batch["edge_attr"] = torch.cat(
            [pad_pos_unsqueeze(s["edge_attr"], max_edge_num) for s in samples]
        )

        return batch
