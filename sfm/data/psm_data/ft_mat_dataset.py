# -*- coding: utf-8 -*-
from functools import lru_cache

import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
from copy import deepcopy

import torch
from torch_geometric.data import Data

from sfm.data.psm_data.dataset import MatterSimDataset
from sfm.logging import logger
from sfm.models.psm.psm_config import PSMConfig

try:
    from matbench import MatbenchBenchmark
except:
    logger.error("matbench is not installed successfully")


class MatBenchDataset(MatterSimDataset):
    def __init__(self, args: PSMConfig, split):
        self.args = args
        self.task_name = args.psm_matbench_task_name
        self.task = MatbenchBenchmark(autoload=False).tasks_map[self.task_name]
        self.fold_id = args.psm_matbench_fold_id
        self.task.load()
        if split == "train_val":
            train_val_inputs, train_val_outputs = self.task.get_train_and_val_data(
                self.fold_id
            )
            self.data = self._preprocess(train_val_inputs, train_val_outputs)
        elif split == "test":
            test_inputs, test_outputs = self.task.get_test_data(
                self.fold_id, include_target=True
            )
            self.data = self._preprocess(test_inputs, test_outputs)

    def _preprocess(self, mat_inputs, mat_outputs=None):
        data_list = []
        if mat_outputs is None:
            mat_outputs = [None for _ in mat_inputs]
        for i, (mat_data, mat_y) in enumerate(zip(mat_inputs, mat_outputs)):
            pos_list = []
            x_list = list(mat_data.atomic_numbers)
            num_atoms = 0
            data = Data()
            for atom in mat_data.as_dict()["sites"]:
                pos_list.append(atom["xyz"])
            pbc_list = list(mat_data.as_dict()["lattice"]["pbc"])
            cell = np.array(mat_data.as_dict()["lattice"]["matrix"])
            num_atoms = len(mat_data.as_dict()["sites"])

            data.num_atoms = num_atoms
            data.cell = cell
            data.x = torch.as_tensor(x_list)
            data.pos = torch.as_tensor(pos_list)
            data.pbc = np.array(pbc_list)
            data.y = torch.tensor(
                [float(mat_y) if mat_y is not None else np.nan], dtype=torch.float
            )
            data.idx = i

            edge_len = 0
            edge_feat_len = 3
            data["edge_attr"] = torch.zeros([edge_len, edge_feat_len], dtype=torch.long)
            data["edge_index"] = torch.zeros([2, edge_len], dtype=torch.long)
            assert list(data["edge_attr"].size()) == [edge_len, edge_feat_len]
            assert list(data["edge_index"].size()) == [2, edge_len]
            assert list(data["x"].size()) == [num_atoms]
            assert list(data["pos"].size()) == [num_atoms, 3]

            data_list.append(data)

        return data_list

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = deepcopy(self.data[idx])
        data["sample_type"] = 1
        data["coords"] = data.pos.clone()
        data.pos = None
        x = data.x.clone()
        data.x = None
        x = torch.cat([x, torch.full([8], 128)], dim=-1)
        data["token_type"] = x
        data["idx"] = idx
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
        data["forces"] = torch.zeros_like(data["coords"])
        data["energy"] = data.y.clone()
        data["energy_per_atom"] = data.y.clone()
        data.y = None
        data["stress"] = torch.zeros_like(data["cell"])
        data["has_energy"] = torch.tensor([1], dtype=torch.bool)
        data["has_forces"] = torch.tensor([0], dtype=torch.bool)

        data = self.generate_2dgraphfeat(data)

        data["is_stable_periodic"] = False

        return data

    def __len__(self):
        return len(self.data)
