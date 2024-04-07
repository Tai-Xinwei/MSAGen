# -*- coding: utf-8 -*-
import copy
import itertools
import random
from functools import lru_cache
from multiprocessing import Pool
from typing import List

from sfm.data.dataset import FoundationModelDataset
from sfm.data.psm_data.dataset import (
    MatterSimDataset,
    PM6FullLMDBDataset,
    ProteinLMDBDataset,
)


class UnifiedPSMDataset(FoundationModelDataset):
    def __init__(self, data_path_list: List, dataset_name_list: List, **kwargs):
        super().__init__()
        self.data_path_list = data_path_list
        self.dataset_name_list = dataset_name_list

        self.dataset_list = []
        self.len = 0

        for data_path, dataset_name in zip(data_path_list, dataset_name_list):
            if dataset_name == "mattersim":
                self.dataset_list.append(MatterSimDataset(data_path, **kwargs))
            elif dataset_name == "pm6":
                self.dataset_list.append(PM6FullLMDBDataset(data_path, **kwargs))
            elif dataset_name == "afdb":
                self.dataset_list.append(ProteinLMDBDataset(data_path, **kwargs))
            else:
                raise ValueError(f"Invalid dataset name: {dataset_name}")

            self.len += len(self.dataset_list[-1])

        self.num_datasets = len(self.dataset_list)

    def __getitem__(self, idx):
        # randomly select a dataset
        dataset_idx = random.randint(0, self.num_datasets - 1)
        pick_idx = idx % len(self.dataset_list[dataset_idx])
        return self.dataset_list[dataset_idx][pick_idx]

    def __len__(self):
        return self.len

    def collate(self, samples):
        pass


if __name__ == "__main__":
    pass
