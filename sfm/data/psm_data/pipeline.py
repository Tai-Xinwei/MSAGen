# -*- coding: utf-8 -*-

import math
from typing import Any

import numpy as np
import nvidia.dali as dali
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.utils.data import IterableDataset

from sfm.models.psm.psm_config import PSMConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.utils import env_init


class UnifiedBatchedIterableDataset(IterableDataset):
    def __init__(
        self,
        args: PSMConfig,
        dataset_list,
        dataset_len,
        ft=False,
        infer=False,
    ):
        super().__init__()

        self.args = args
        self.generator = np.random.default_rng(args.seed)

        num_datasets = len(dataset_list)
        dataset_split_ratios = list(map(float, args.dataset_split_raito.split(",")))
        dataset_batch_sizes = list(map(int, args.dataset_micro_batch_size.split(",")))
        dataset_names = args.dataset_name_list.split(",")

        self.dataset_len = dataset_len

        dataset_params_mismatch = "Dataset parameters mismatched, please check data_path_list, dataset_name_list, dataset_split_raito, and dataset_micro_batch_size"
        assert num_datasets == len(dataset_split_ratios), dataset_params_mismatch
        assert num_datasets == len(dataset_batch_sizes), dataset_params_mismatch
        assert num_datasets == len(dataset_names), dataset_params_mismatch
        assert (
            sum(dataset_split_ratios) == 1.0
        ), f"sum of split ratio {self.dataset_split_ratios} is not 1.0"

        dataset_sizes = list(map(len, dataset_list))
        dataset_ranges = np.cumsum([0] + dataset_sizes)

        total_size = dataset_ranges[-1]
        dataset_indices = []

        for i in range(num_datasets):
            world_batch_size = dataset_batch_sizes[i] * args.world_size
            sample_size = math.ceil(total_size * dataset_split_ratios[i])

            iterations = (sample_size + world_batch_size - 1) // world_batch_size
            dataset_indices.append(np.full((iterations,), i, dtype=int))

        self.dataset_indices = np.concatenate(dataset_indices)

        self._permute_indices()

        dataset_parallel = [
            True if name not in ["mattersim"] else False for name in dataset_names
        ]

        self.dataset_list = [
            get_dali_pipeline(
                args, dataset=dataset, batch_size=batch_size, parallel=parallel
            )
            for dataset, batch_size, parallel in zip(
                dataset_list, dataset_batch_sizes, dataset_parallel
            )
        ]
        # TODO: get_dali_pm6_pipeline needs to be updated with dataset in main branch
        # self.dataset_list[0] = get_dali_pm6_pipeline(
        #     args, dataset_list[0], dataset_batch_sizes[0], True
        # )

    def _permute_indices(self):
        self.dataset_indices = self.generator.permutation(self.dataset_indices)

    def __iter__(self):
        self.dataset_idx_iter = np.nditer(self.dataset_indices)

        return self

    def __next__(self):
        dataset_idx = next(self.dataset_idx_iter)

        return next(self.dataset_list[dataset_idx])

    def __len__(self):
        return self.dataset_len

    def collate(self, samples):
        return samples[0]

    def num_tokens(self, idx: int) -> int:
        return super().num_tokens(idx)


def build_atom_edge_matrix(node_atom_i, node_atom_j):
    l = node_atom_i.shape[0]

    node_atom_i = node_atom_i.repeat(1, l)
    node_atom_j = node_atom_j.repeat(l, 1)

    return node_atom_i, node_atom_j


def convert_2d_to_single_emb(x):
    return x + torch.arange(0, x.shape[1] * 512, 512, dtype=torch.long, device=x.device)


def set_edge_attr(x, edge_index, v):
    x[edge_index[0, :], edge_index[1, :]] = v

    return x


@dali.pipeline_def
def dali_pm6_pipeline(source, source_batch_size, source_num_outputs, parallel):
    energy_mean: float = -42774.16038176129
    energy_std: float = 25029.68158883449
    energy_per_atom_mean: float = -994.0920019593214
    energy_per_atom_std: float = 770.7496116135809

    (
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
        has_energy,
        has_forces,
    ) = dali.fn.external_source(
        source=source,
        num_outputs=source_num_outputs,
        batch=False,
        parallel=parallel,
    )

    node_feat = node_feat.gpu()
    token_type = node_feat[:, 0] + 1
    node_feat = node_feat[:, 1:]
    node_feat = dali.fn.cat(token_type, node_feat, axis=1)
    node_feat = dali.plugin.pytorch.fn.torch_python_function(
        node_feat,
        function=convert_2d_to_single_emb,
        num_outputs=1,
    )
    num_atoms = dali.fn.shapes(node_feat)[0]
    edge_attr_emb = dali.plugin.pytorch.fn.torch_python_function(
        edge_attr,
        function=convert_2d_to_single_emb,
        num_outputs=1,
    )
    attn_edge_type = dali.plugin.pytorch.fn.torch_python_function(
        attn_edge_type,
        edge_index,
        edge_attr_emb,
        function=set_edge_attr,
        num_outputs=1,
    )
    adj = dali.plugin.pytorch.fn.torch_python_function(
        adj,
        edge_index,
        torch.ones(1, dtype=torch.long),
        function=set_edge_attr,
        num_outputs=1,
    )
    in_degree = dali.fn.reductions.sum(adj, axes=1)
    in_degree = dali.fn.reshape(in_degree, shape=[-1])

    attn_bias = dali.fn.pad(attn_bias.gpu(), axes=(0, 1))
    in_degree = dali.fn.pad(in_degree + 1, axes=(0,))
    node_attr = dali.fn.pad(node_feat + 1, axes=(0,))
    token_id = node_attr[:, 0]
    energy = (energy.gpu() - energy_mean) / energy_std
    energy_per_atom = (energy / num_atoms - energy_per_atom_mean) / energy_per_atom_std
    energy = dali.fn.squeeze(energy, axes=0)
    energy_per_atom = dali.fn.squeeze(energy_per_atom, axes=0)
    forces = dali.fn.pad(forces.gpu(), axes=(0,))
    pos = dali.fn.pad(coords.gpu(), axes=(0,))
    node_atom_i = dali.fn.reshape(token_type, shape=[-1, 1])
    node_atom_j = dali.fn.reshape(token_type, shape=[1, -1])
    node_atom_i, node_atom_j = dali.plugin.pytorch.fn.torch_python_function(
        node_atom_i,
        node_atom_j,
        function=build_atom_edge_matrix,
        num_outputs=2,
    )
    node_atom_i = dali.fn.pad(node_atom_i, axes=(0, 1))
    node_atom_j = dali.fn.pad(node_atom_j, axes=(0, 1))
    node_type_edge = dali.fn.cat(node_atom_i, node_atom_j, axis=-1)
    node_type_edge = dali.plugin.pytorch.fn.torch_python_function(
        node_type_edge,
        function=convert_2d_to_single_emb,
        num_outputs=1,
    )
    adj = dali.fn.pad(adj.gpu(), axes=(0, 1))
    attn_edge_type = dali.fn.pad(attn_edge_type.gpu() + 1, axes=(0, 1))
    is_stable_periodic = dali.fn.squeeze(is_stable_periodic, axes=0)

    return (
        attn_bias,
        in_degree,
        in_degree,
        token_id,
        node_attr,
        energy,
        energy_per_atom,
        forces,
        pos,
        node_type_edge,
        pbc,
        cell,
        num_atoms,
        adj,
        attn_edge_type,
        is_stable_periodic,
        has_energy,
        has_forces,
    )


@dali.pipeline_def
def dali_pipeline(source, source_batch_size, source_num_outputs, parallel):
    (
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
    ) = dali.fn.external_source(
        source=source,
        num_outputs=source_num_outputs,
        batch=False,
        parallel=parallel,
    )

    attn_bias = dali.fn.pad(attn_bias.gpu(), axes=(0, 1))
    in_degree = dali.fn.pad(in_degree.gpu() + 1, axes=(0,))
    node_attr = dali.fn.pad(node_attr.gpu() + 1, axes=(0,))
    token_id = node_attr[:, 0]
    energy = dali.fn.squeeze(energy.gpu(), axes=0)
    energy_per_atom = dali.fn.squeeze(energy_per_atom.gpu(), axes=0)
    forces = dali.fn.pad(forces.gpu(), axes=(0,))
    pos = dali.fn.pad(pos.gpu(), axes=(0,))
    node_atom_type = token_type.gpu()
    node_atom_i = dali.fn.reshape(node_atom_type, shape=[-1, 1])
    node_atom_j = dali.fn.reshape(node_atom_type, shape=[1, -1])
    node_atom_i, node_atom_j = dali.plugin.pytorch.fn.torch_python_function(
        node_atom_i,
        node_atom_j,
        function=build_atom_edge_matrix,
        num_outputs=2,
    )
    node_atom_i = dali.fn.pad(node_atom_i, axes=(0, 1))
    node_atom_j = dali.fn.pad(node_atom_j, axes=(0, 1))
    node_type_edge = dali.fn.cat(node_atom_i, node_atom_j, axis=-1)
    node_type_edge = dali.plugin.pytorch.fn.torch_python_function(
        node_type_edge,
        function=convert_2d_to_single_emb,
        num_outputs=1,
    )
    adj = dali.fn.pad(adj.gpu(), axes=(0, 1))
    attn_edge_type = dali.fn.pad(attn_edge_type.gpu() + 1, axes=(0, 1))
    has_energy = dali.fn.squeeze(has_energy.gpu(), axes=0)
    has_forces = dali.fn.squeeze(has_forces.gpu(), axes=0)

    return (
        attn_bias,
        in_degree,
        in_degree,
        token_id,
        node_attr,
        energy,
        energy_per_atom,
        forces,
        pos,
        node_type_edge,
        pbc,
        cell,
        num_atoms,
        adj,
        attn_edge_type,
        is_stable_periodic,
        has_energy,
        has_forces,
    )


def get_dali_pm6_pipeline(args: Any, dataset: Any, batch_size: int, parallel: bool):
    source = DaliPM6DataSource(args, dataset=dataset, batch_size=batch_size)

    if parallel is True:
        dataset._close_db()

    pipe = dali_pm6_pipeline(
        batch_size=batch_size,
        num_threads=4,
        device_id=args.local_rank,
        prefetch_queue_depth=4,
        py_num_workers=4,
        py_start_method="spawn",
        source=source,
        source_batch_size=batch_size,
        source_num_outputs=12,
        parallel=parallel,
    )
    pipe.build()

    return DALIGenericIterator(
        pipe,
        [
            "attn_bias",
            "in_degree",
            "out_degree",
            "token_id",
            "node_attr",
            "energy",
            "energy_per_atom",
            "forces",
            "pos",
            "node_type_edge",
            "pbc",
            "cell",
            "num_atoms",
            "adj",
            "attn_edge_type",
            "is_stable_periodic",
        ],
    )


def get_dali_pipeline(args: Any, dataset: Any, batch_size: int, parallel: bool):
    source = DaliUnifiedDataSource(args, dataset=dataset, batch_size=batch_size)

    if parallel is True:
        dataset._close_db()

    pipe = dali_pipeline(
        batch_size=batch_size,
        num_threads=2,
        device_id=args.local_rank,
        prefetch_queue_depth=2,
        py_num_workers=2,
        py_start_method="spawn",
        source=source,
        source_batch_size=batch_size,
        source_num_outputs=16,
        parallel=parallel,
    )
    pipe.build()

    return DALIGenericIterator(
        pipe,
        [
            "attn_bias",
            "in_degree",
            "out_degree",
            "token_id",
            "node_attr",
            "energy",
            "energy_per_atom",
            "forces",
            "pos",
            "node_type_edge",
            "pbc",
            "cell",
            "num_atoms",
            "adj",
            "attn_edge_type",
            "is_stable_periodic",
            "has_energy",
            "has_forces",
        ],
    )


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
