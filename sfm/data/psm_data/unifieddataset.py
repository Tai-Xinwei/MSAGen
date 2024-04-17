# -*- coding: utf-8 -*-

import os
import random

from sfm.data.dataset import FoundationModelDataset
from sfm.data.psm_data.collator import collate_fn
from sfm.data.psm_data.dataset import (
    AFDBLMDBDataset,
    MatterSimDataset,
    PM6FullLMDBDataset,
)
from sfm.logging import logger


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

        file_list = []

        for data_path in data_path_list:
            file_list.append(os.path.join(data_dir, data_path))

        self.train_dataset_list = []
        self.valid_dataset_list = []
        self.train_len = 0
        self.valid_len = 0

        for data_path, dataset_name in zip(file_list, dataset_name_list):
            if dataset_name == "pm6":
                dataset = PM6FullLMDBDataset(data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
            elif dataset_name == "afdb":
                dataset = AFDBLMDBDataset(data_path, **kwargs)
                train_dataset, valid_dataset = dataset.split_dataset()
                len_total = len(dataset)
            elif dataset_name == "mattersim":
                train_dataset = MatterSimDataset(data_path, split="train", **kwargs)
                valid_dataset = MatterSimDataset(data_path, split="valid", **kwargs)
                len_total = len(train_dataset) + len(valid_dataset)
            else:
                raise ValueError(f"Invalid dataset name: {dataset_name}")

            self.train_dataset_list.append(train_dataset)
            self.valid_dataset_list.append(valid_dataset)

            self.train_len += len(train_dataset)
            self.valid_len += len(valid_dataset)
            logger.info(
                f"Loaded dataset {dataset_name} with total {len_total/1024/1024:0.2f}M samples, {len(train_dataset)/1024/1024:0.2f}M for training, {len(valid_dataset)/1024/1024:0.2f}M for validation"
            )

        self.num_datasets = len(self.train_dataset_list)

    def split_dataset(self):
        return self.train_dataset_list, self.valid_dataset_list


class BatchedDataDataset(FoundationModelDataset):
    def __init__(
        self,
        dataset_list,
        len_data,
        min_node=32,
        max_node=128,
        multi_hop_max_dist=5,
        spatial_pos_max=1024,
        args=None,
        ft=False,
        infer=False,
    ):
        super().__init__()
        self.dataset_list = dataset_list
        self.num_datasets = len(dataset_list)
        self.len = len_data
        self.min_node = min_node
        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max
        self.args = args
        self.ft = ft
        self.infer = infer

    def __getitem__(self, idx):
        # randomly select a dataset
        dataset_idx = random.randint(0, self.num_datasets - 1)
        pick_idx = idx % len(self.dataset_list[dataset_idx])
        return self.dataset_list[dataset_idx][pick_idx]

    def __len__(self):
        return self.len

    def collate(self, samples):
        return collate_fn(samples)


if __name__ == "__main__":
    data_dir = "/data/peiran/"
    data_path_list = "pm6_10M_refined4.lmdb,AFDB50-plddt70.lmdb,matter-sim-3M"
    dataset_name_list = "pm6,afdb,mattersim"
    train_data = UnifiedPSMDataset(data_dir, data_path_list, dataset_name_list)
