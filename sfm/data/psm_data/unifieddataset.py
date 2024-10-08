# -*- coding: utf-8 -*-

import os
import random
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler, T_co

from sfm.data.dataset import FoundationModelDataset
from sfm.data.psm_data.collator import collate_fn
from sfm.data.psm_data.dataset import (
    AFDBLMDBDataset,
    ESMDataset,
    MatterSimDataset,
    PDBComplexDataset,
    PDBDataset,
    PlainPM6FullLMDBDataset,
    PM6FullLMDBDataset,
    PubChemQCB3lypPM6Dataset,
    SmallMolDataset,
    UR50LMDBDataset,
)
from sfm.data.psm_data.ft_mat_dataset import MatBenchDataset
from sfm.data.psm_data.ft_mol_dataset import (
    PCQM4Mv2LMDBDataset,
    PubChemQCB3LYPLMDBDataset,
)
from sfm.data.psm_data.ft_prot_dataset import ComplexDataset
from sfm.data.sampler import WeightedDistributedSampler
from sfm.logging import logger
from sfm.models.psm.psm_config import PSMConfig

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

import math

import torch.distributed as dist


class UnifiedPSMDataset(FoundationModelDataset):
    def __init__(
        self,
        data_dir: str,
        data_path_list: str,
        dataset_name_list: str,
        args=None,
        split="train",
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_path_list = data_path_list
        self.dataset_name_list = dataset_name_list

        data_path_list = data_path_list.split(",")
        dataset_name_list = dataset_name_list.split(",")

        self.sizes = []
        self.dataset_lens = {}

        file_list = []

        for data_path in data_path_list:
            file_list.append(os.path.join(data_dir, data_path))

        self.train_dataset_list = []
        self.valid_dataset_list = []
        self.train_len = 0
        self.valid_len = 0

        self.molecule_energy_mean = 0.0
        self.molecule_energy_std = 1.0
        self.periodic_energy_mean = 0.0
        self.periodic_energy_std = 1.0
        self.molecule_energy_per_atom_mean = 0.0
        self.molecule_energy_per_atom_std = 1.0
        self.periodic_energy_per_atom_mean = 0.0
        self.periodic_energy_per_atom_std = 1.0
        self.molecule_force_mean = 0.0
        self.molecule_force_std = 1.0
        self.periodic_force_mean = 0.0
        self.periodic_force_std = 1.0

        for data_path, dataset_name in zip(file_list, dataset_name_list):
            if dataset_name in ["pm6", "pm6-b3lyp", "pm6-wb97xd3"]:
                if dataset_name == "pm6-wb97xd3":
                    dataset = PubChemQCB3lypPM6Dataset(args, data_path, **kwargs)
                elif dataset_name == "pm6-b3lyp":
                    data_path = os.path.join(data_path, "b3lyp/1.0.0")
                    dataset = PubChemQCB3lypPM6Dataset(args, data_path, **kwargs)
                else:
                    if args.backbone.find("vanilla") != -1:
                        dataset = PlainPM6FullLMDBDataset(args, data_path, **kwargs)
                    else:
                        dataset = PM6FullLMDBDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
                self.sizes.append(train_dataset.sizes)
                self.molecule_energy_mean = dataset.energy_mean
                self.molecule_energy_std = dataset.energy_std
                self.molecule_energy_per_atom_mean = dataset.energy_per_atom_mean
                self.molecule_energy_per_atom_std = dataset.energy_per_atom_std
                self.molecule_force_mean = dataset.force_mean
                self.molecule_force_std = dataset.force_std
            elif dataset_name == "afdb":
                dataset = AFDBLMDBDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
                self.sizes.append(train_dataset.sizes)
            elif dataset_name == "esm":
                dataset = ESMDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
                self.sizes.append(train_dataset.sizes)
            elif dataset_name == "mattersim":
                train_dataset = MatterSimDataset(
                    args, data_path, split="train", **kwargs
                )
                valid_dataset = MatterSimDataset(
                    args, data_path, split="valid", **kwargs
                )
                self.dataset_lens[dataset_name] = len(train_dataset)
                len_total = len(train_dataset) + len(valid_dataset)
                self.periodic_energy_mean = train_dataset.energy_mean
                self.periodic_energy_std = train_dataset.energy_std
                self.periodic_energy_per_atom_mean = train_dataset.energy_per_atom_mean
                self.periodic_energy_per_atom_std = train_dataset.energy_per_atom_std
                self.periodic_force_mean = train_dataset.force_mean
                self.periodic_force_std = train_dataset.force_std
            elif dataset_name == "matbench":
                train_dataset = MatBenchDataset(args, split="train_val", **kwargs)
                valid_dataset = MatBenchDataset(args, split="test", **kwargs)
                self.dataset_lens[dataset_name] = len(train_dataset)
                len_total = len(train_dataset) + len(valid_dataset)
                self.periodic_energy_mean = 0.0
                self.periodic_energy_std = 1.0
                self.periodic_energy_per_atom_mean = 0.0
                self.periodic_energy_per_atom_std = 1.0
                self.periodic_force_mean = 0.0
                self.periodic_force_std = 1.0
            elif dataset_name in [
                "SPICE",
                "pubchem5w",
                "Ac_Ala3_NHMe",
                "AT_AT",
                "AT_AT_CG_CG",
                "DHA",
                "stachyose",
                "buckyball_catcher",  # buckyball_catcher/radius3_broadcast_kmeans
                "double_walled_nanotube",  # double_walled_nanotube/radius3_broadcast_kmeans
                "oc20",
                "deshaw",
                "deshaw_120",
                "deshaw_400",
                "deshaw_650",
                "GEMS",
            ]:
                dataset = SmallMolDataset(
                    args, data_path, data_name=dataset_name, **kwargs
                )
                # train_dataset, valid_dataset = dataset.split_dataset_cutoff(
                #     dataset_len_ratio=dataset_len_ratio
                # )
                train_dataset, valid_dataset = dataset.split_dataset(
                    validation_ratio=0.8
                )
                len_total = len(dataset)
            elif dataset_name == "pcqm4mv2":
                train_dataset = PCQM4Mv2LMDBDataset(
                    args, data_path, split="train", **kwargs
                )
                valid_dataset = PCQM4Mv2LMDBDataset(
                    args, data_path, split="valid", **kwargs
                )
                len_total = len(train_dataset) + len(valid_dataset)
            elif dataset_name == "ur50":
                dataset = UR50LMDBDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
                self.sizes.append(train_dataset.sizes)
            elif dataset_name == "pdb":
                dataset = PDBDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
                self.sizes.append(train_dataset.sizes)
            elif dataset_name == "proteintest":
                dataset = PDBDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset, dataset
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
                self.sizes.append(train_dataset.sizes)
            elif dataset_name == "complex":
                dataset = ComplexDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
                # self.sizes.append(train_dataset.sizes)
            elif dataset_name == "pdbcomplexmultimer":
                dataset = PDBComplexDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
            elif dataset_name == "complextest":
                dataset = PDBComplexDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset, dataset
                len_total = len(dataset)
                self.dataset_lens[dataset_name] = len(train_dataset)
            elif dataset_name == "pubchemqc-b3lyp":
                dataset = PubChemQCB3LYPLMDBDataset(args, data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset(
                    validation_ratio=0.01
                )
                len_total = len(dataset)
            else:
                raise ValueError(f"Invalid dataset name:{dataset_name}")

            self.train_dataset_list.append(train_dataset)
            self.valid_dataset_list.append(valid_dataset)

            self.train_len += len(train_dataset)
            self.valid_len += len(valid_dataset)
            logger.info(
                f"Loaded dataset {dataset_name} with total {len_total/1000/1000:0.2f} samples, {len(train_dataset)/1000/1000:0.2f} for training, {len(valid_dataset)/1000/1000:0.2f} for validation"
            )

        self.num_datasets = len(self.train_dataset_list)

    def split_dataset(self):
        return self.train_dataset_list, self.valid_dataset_list


class BatchedDataDataset(FoundationModelDataset):
    def __init__(
        self,
        args: PSMConfig,
        dataset_list,
        len_data,
        multi_hop_max_dist=5,
        spatial_pos_max=1024,
        ft=False,
        infer=False,
        extra_collate_fn=None,
    ):
        super().__init__()
        self.dataset_list = dataset_list
        self.num_datasets = len(dataset_list)
        self.len = len_data
        self.dataset_split_raito = [
            float(i) for i in args.dataset_split_raito.split(",")
        ]
        assert (
            len(self.dataset_split_raito) == self.num_datasets
        ), "split ratio mismatch with number of datasets"
        if sum(self.dataset_split_raito) != 1.0:
            logger.info(
                f"sum of split ratio {self.dataset_split_raito} == {sum(self.dataset_split_raito)} is not 1.0, use default ratio"
            )
            self.dataset_split_raito[-1] = 1.0 - sum(self.dataset_split_raito[:-1])

        logger.info(f"Total data Length is {len_data:0.2f}")

        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.args = args
        self.ft = ft
        self.infer = infer
        self.extra_collate_fn = extra_collate_fn

    def __getitem__(self, idx):
        # select dataset_idx based on split ratio
        dataset_idx = random.choices(
            range(self.num_datasets), weights=self.dataset_split_raito
        )[0]
        pick_idx = idx % len(self.dataset_list[dataset_idx])
        return self.dataset_list[dataset_idx][pick_idx]

    def __len__(self):
        return self.len

    def collate(self, samples):
        batched_data = collate_fn(
            samples,
            multi_hop_max_dist=self.multi_hop_max_dist,
            preprocess_2d_bond_features_with_cuda=self.args.preprocess_2d_bond_features_with_cuda,
            sample_in_validation=self.args.sample_in_validation,
        )
        if self.extra_collate_fn is not None:
            batched_data = self.extra_collate_fn(samples, batched_data)
        return batched_data

    def num_tokens(self, idx: int) -> int:
        return super().num_tokens(idx)


class BatchedDataDatasetForUnifiedSampler(BatchedDataDataset):
    def __init__(
        self,
        args,
        dataset_list,
        len_data,
        **kwargs,
    ):
        super().__init__(args, dataset_list, len_data, **kwargs)
        self.dataset_lens = [len(dataset) for dataset in self.dataset_list]
        self.dataset_ranges = np.cumsum([0] + self.dataset_lens)

    def __getitem__(self, idx):
        for i in range(self.num_datasets):
            if idx >= self.dataset_ranges[i] and idx < self.dataset_ranges[i + 1]:
                return self.dataset_list[i][idx - self.dataset_ranges[i]]
        raise ValueError(f"Data with index {idx} not found in any subset.")


class StackedIterableDataset(IterableDataset):
    def __init__(self, dataset_list, args: PSMConfig, sizes, shuffle=True):
        self.dataset_list = dataset_list  # An iterable source
        self.collate_fn = collate_fn
        self.sequence_length = args.max_length
        self.args = args
        self.buffer_len = 0
        self.buffers = {}
        self.shuffle = shuffle
        self.epoch = 0
        self.break_mode = "complete_doc"
        self.document_sep_len = 1
        self.sizes_list = sizes
        self.num_blocks = 0

        # logger.info(
        #     "Dataset split ratio is disabled in stacked dataset, all data will be used equally"
        # )
        self.dataset_split_raito = [
            float(i) for i in args.dataset_split_raito.split(",")
        ]
        assert (
            self.dataset_split_raito[1] == 0.0
        ), "stacked dataset does not support pbc crystal right now"
        assert len(self.dataset_split_raito) == len(
            self.dataset_list
        ), f"split ratio mismatch len is {len(self.dataset_split_raito)} with number of datasets is {len(self.dataset_list)}"
        if sum(self.dataset_split_raito) != 1.0:
            logger.info(
                f"sum of split ratio {self.dataset_split_raito} is not 1.0, use default ratio"
            )
            self.dataset_split_raito = [0.9, 0.0, 0.1]

        self.dataset_split_raito = [
            self.dataset_split_raito[0],
            self.dataset_split_raito[1],
        ]

        # DDP-related attributes
        self.rank = args.rank
        self.world_size = args.world_size
        self.sampler0 = DistributedSampler(
            dataset_list[0],
            shuffle=self.shuffle,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        self.sampler1 = DistributedSampler(
            dataset_list[1],
            shuffle=self.shuffle,
            num_replicas=self.world_size,
            rank=self.rank,
        )
        self._create_subset_iterator()

    def __iter__(self):
        # Reset the buffers when creating a new iterator
        if self.shuffle:
            self.sampler0.set_epoch(self.epoch)
            self.sampler1.set_epoch(self.epoch)
            self.epoch += 1

        # # Reset the buffers when creating a new iterator
        self.buffer_len = 0
        self.buffers = {}

        self.subset_iterator = self._create_subset_iterator()

        return self

    def _get_blocks(self, sizes):
        slice_indices = _get_slice_indices_fast(
            np.array(sizes),
            self.break_mode,
            self.sequence_length,
            self.document_sep_len,
        )
        blocks = _get_block_to_dataset_index_fast(np.array(sizes), slice_indices)
        return blocks

    def _create_subset_iterator(self):
        # # Get the list of indices from the sampler and iterate through them
        indices0 = list(iter(self.sampler0))
        indices1 = list(iter(self.sampler1))
        sizes0 = [self.sizes_list[0][i] for i in indices0]
        sizes1 = [self.sizes_list[1][i] for i in indices1]
        blocks0 = self._get_blocks(sizes0)
        blocks1 = self._get_blocks(sizes1)
        blocks = [blocks0, blocks1]
        block_len = [len(blocks0), len(blocks1)]

        # Split blocks among workers for DDP
        self.num_blocks = len(blocks0) + len(blocks1)
        logger.success(
            f"number of stacked block in epoch {self.epoch-1} of rank {self.rank} is {self.num_blocks}"
        )

        # pick_idx = idx % len(self.dataset_list[dataset_idx])
        # return self.dataset_list[dataset_idx][pick_idx]
        for _ in range(self.num_blocks):
            dataset_idx = random.choices(range(2), weights=self.dataset_split_raito)[0]
            block_idx = random.choices(range(block_len[dataset_idx]))[0]
            start, start_offset, end = blocks[dataset_idx][block_idx]
            for idx in range(start, end + 1):
                yield self.dataset_list[dataset_idx][idx]

    def __next__(self):
        # Continue to read from the subset iterator and fill the buffers until we have enough data
        while self.buffer_len < self.sequence_length:
            try:
                item = next(self.subset_iterator)

                for key, value in item.items():
                    if key not in self.buffers:
                        self.buffers[key] = []
                    self.buffers[key].append(value)

                self.buffer_len += len(item["token_type"])
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
                # "sample_type", # do not need in collate_fn
                "coords",
                "token_type",
                # "pbc", # only use for pbc, do not support
                # "cell", # only use for pbc, do not support
                # "num_atoms", # only use for pbc, do not support
                "forces",
                "energy",
                # "stress", # only use for pbc, do not support
                "edge_index",
                "edge_attr",
                "node_attr",
                "edge_input",
                "attn_bias",
                "in_degree",
                "spatial_pos",
            ]:
                continue
            try:
                result[key] = torch.cat(buf)[: self.sequence_length]
            except Exception as e:
                logger.error(f"Error in stacking: {e}")
                logger.error(f"key: {key}, buf: {buf}")
                logger.error(f"len buf: {[len(b) for b in buf]}")
                raise
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
        elif len(self.buffers["token_type"]) == 0:
            self.buffer_len = 0
        else:
            self.buffer_len = len(self.buffers["token_type"][0])

        return result

    def collate(self, samples):
        return self.collate_fn(
            samples,
            use_pbc=False,
            preprocess_2d_bond_features_with_cuda=self.args.preprocess_2d_bond_features_with_cuda,
        )

    def __len__(self):
        return self.num_blocks


class UnifiedDataSampler(WeightedDistributedSampler):
    # samples data from different modalities
    def __init__(
        self,
        dataset: UnifiedPSMDataset,
        dataset_split_ratios: str,
        dataset_batch_sizes: str,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.dataset_split_ratios = [
            float(ratio) for ratio in dataset_split_ratios.split(",")
        ]
        self.dataset_batch_sizes = [
            int(batch_size) for batch_size in dataset_batch_sizes.split(",")
        ]
        assert len(dataset.dataset_lens) == len(self.dataset_split_ratios) and len(
            dataset.dataset_lens
        ) == len(
            self.dataset_batch_sizes
        ), "Dataset parameters mismatched, please check data_path_list, dataset_name_list, dataset_split_raito, and dataset_micro_batch_size"
        self.dataset_ranges = np.cumsum([0] + dataset.dataset_lens)
        total_len = self.dataset_ranges[-1]
        dataset_sampled_lens = [
            total_len * ratio for ratio in self.dataset_split_ratios
        ]
        weight_dict = {}
        for i in range(len(self.dataset_ranges) - 1):
            start = self.dataset_ranges[i]
            end = self.dataset_ranges[i + 1]
            weight_dict[(start, end)] = dataset_sampled_lens[i] * 1.0 / (end - start)

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
        self.weight_dict = weight_dict
        self.dataset_sampled_len = {}

        dataset_indices_len = 0
        num_samples = 0
        for i in range(len(self.dataset_ranges) - 1):
            start = self.dataset_ranges[i]
            end = self.dataset_ranges[i + 1]
            ratio = weight_dict[(start, end)]
            sampled_len = math.ceil((end - start) * ratio)
            micro_batch_size = self.dataset_batch_sizes[i]
            sampled_len = (
                (sampled_len + micro_batch_size * num_replicas - 1)
                // (micro_batch_size * num_replicas)
                * micro_batch_size
                * num_replicas
            )
            self.dataset_sampled_len[(start, end)] = sampled_len
            dataset_indices_len += sampled_len
            num_samples += sampled_len // num_replicas

        self.dataset_indices_len = dataset_indices_len
        self.num_total_samples = num_samples
        self.num_samples = num_samples
        self.total_size = self.num_samples * self.num_replicas
        assert self.total_size == self.dataset_indices_len
        self.seed = seed
        self.shuffle = True
        self.drop_last = False
        self.num_skip_batches = None
        self.micro_batch_size = None

    def __iter__(self) -> Iterator[T_co]:
        # deterministically shuffle based on epoch and seed
        generator = np.random.default_rng(self.epoch + self.seed)
        torch_generator = torch.Generator()
        torch_generator.manual_seed(self.seed + self.epoch)
        indices = []
        for begin, end in np.sort(list(self.dataset_sampled_len.keys())):
            sampled_len = self.dataset_sampled_len[(begin, end)]
            indices_for_dataset = []
            while sampled_len > end - begin:
                indices_for_dataset.extend(
                    torch.randperm(end - begin, generator=torch_generator).numpy()
                    + begin
                )
                sampled_len -= end - begin
            indices_for_dataset.extend(
                list(generator.choice(end - begin, sampled_len, replace=False) + begin)
            )
            indices_for_dataset = list(
                torch.tensor(indices_for_dataset, dtype=torch.long)[
                    torch.randperm(len(indices_for_dataset), generator=torch_generator)
                ].numpy()
            )
            indices.extend(
                indices_for_dataset[self.rank : self.total_size : self.num_replicas]
            )
        sorted_indices = torch.randperm(
            len(indices), generator=torch_generator
        ).tolist()
        indices = np.array(indices)[np.array(sorted_indices)].tolist()

        assert len(indices) == self.num_total_samples
        self.num_samples = self.num_total_samples

        num_datasets = len(self.dataset_ranges) - 1
        split_indices = [[] for _ in range(num_datasets)]
        for index in indices:
            for tag in range(num_datasets):
                if (
                    index >= self.dataset_ranges[tag]
                    and index < self.dataset_ranges[tag + 1]
                ):
                    split_indices[tag].append(index)

        batch_seqs = [
            [
                split_indices[j][
                    i * self.dataset_batch_sizes[j] : i * self.dataset_batch_sizes[j]
                    + self.dataset_batch_sizes[j]
                ]
                for i in range(len(split_indices[j]) // self.dataset_batch_sizes[j])
            ]
            for j in range(num_datasets)
        ]

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        total_num_batches = np.sum([len(batch_seq) for batch_seq in batch_seqs])
        all_batches = []
        for batch_seq in batch_seqs:
            all_batches += batch_seq
        batch_indices = torch.randperm(total_num_batches, generator=g).tolist()

        if self.num_skip_batches is not None:
            batch_indices = batch_indices[self.num_skip_batches :]
        all_batches = [all_batches[i] for i in batch_indices]
        if self.num_skip_batches is not None:
            self.num_samples = np.sum(len(batch) for batch in all_batches)
        return iter(all_batches)

    def set_epoch(self, epoch) -> None:
        return super().set_epoch(epoch)


if __name__ == "__main__":
    data_dir = "/data/peiran/"
    data_path_list = "pm6_10M_refined4.lmdb,AFDB50-plddt70.lmdb,matter-sim-3M"
    dataset_name_list = "pm6,afdb,mattersim"
    train_data = UnifiedPSMDataset(data_dir, data_path_list, dataset_name_list)
