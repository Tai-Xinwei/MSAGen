# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import safetensors
import torch


# this is a basic dataset class that loads the SPICE dataset
class SpiceDataset:
    def __init__(self, data_path):
        self.total_sample_count = 2008628
        self.tensors = {}
        with safetensors.safe_open(data_path, framework="pt") as f:
            for k in f.keys():
                self.tensors[k] = f.get_tensor(k)

    def __getitem__(self, idx):
        return (
            self.tensors["atom_counts"][idx],
            self.tensors["atomic_numbers"][idx],
            self.tensors["conformations"][idx],
            self.tensors["dft_total_gradients"][idx],
            self.tensors["dft_total_energies"][idx],
        )

    def __len__(self):
        return self.total_sample_count
