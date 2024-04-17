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
        **kwargs,
    ):
        super().__init__()
        data_path_list = data_path_list.split(",")
        dataset_name_list = dataset_name_list.split(",")

        self.data_path_list = []

        for data_path in data_path_list:
            self.data_path_list.append(os.path.join(data_dir, data_path))

        self.dataset_list = []
        self.len = 0

        for data_path, dataset_name in zip(self.data_path_list, dataset_name_list):
            if dataset_name == "pm6":
                dataset = PM6FullLMDBDataset(data_path, **kwargs)
            elif dataset_name == "afdb":
                dataset = AFDBLMDBDataset(data_path, **kwargs)
            elif dataset_name == "mattersim":
                dataset = MatterSimDataset(data_path, split="train", **kwargs)
            else:
                raise ValueError(f"Invalid dataset name: {dataset_name}")

            self.dataset_list.append(dataset)
            self.len += len(dataset)
            logger.info(f"Loaded dataset {dataset_name} with {len(dataset)} samples")

        self.num_datasets = len(self.dataset_list)

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
