# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import safetensors
import torch


# this is a basic dataset class that loads the SPICE dataset
class SpiceDataset:
    def __init__(self, data_path, dataset_label):
        self.data_path = data_path
        self.dataset_label = dataset_label
        self.total_sample_count = 2008628
        self.train_sample_count = 1606902
        self.val_sample_count = 200863
        self.test_sample_count = 200863

        self.sample_count = self.total_sample_count
        if dataset_label.startswith("train"):
            self.sample_count = self.train_sample_count
        elif dataset_label.startswith("val"):
            self.sample_count = self.val_sample_count
        elif dataset_label.startswith("test"):
            self.sample_count = self.test_sample_count

    def load_data(self):
        self.tensors = {}
        with safetensors.safe_open(
            self.data_path, framework="pt", device="cuda:0"
        ) as f:
            for k in f.keys():
                self.tensors[k] = f.get_tensor(k)
                print(
                    f"loaded tensors[{k}].shape ({self.dataset_label}) = {self.tensors[k].shape}"
                )

    def __getitem__(self, idx):
        return (
            self.tensors["atom_counts"][idx],
            self.tensors["atomic_numbers"][idx],
            self.tensors["conformations"][idx],
            self.tensors["dft_total_gradients"][idx],
            self.tensors["dft_total_energies"][idx],
        )

    def __len__(self):
        return self.sample_count
