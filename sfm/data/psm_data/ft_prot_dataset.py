# -*- coding: utf-8 -*-
import numpy as np
import pyximport

pyximport.install(setup_args={"include_dirs": np.get_include()})
import random
from typing import Optional, Union

import lmdb
import torch
from torch_geometric.data import Data

from sfm.data.prot_data.util import bstr2obj
from sfm.data.psm_data.collator import collate_fn
from sfm.data.psm_data.dataset import AFDBLMDBDataset
from sfm.logging import logger
from sfm.models.psm.psm_config import PSMConfig


class ProteinSamplingDataset(AFDBLMDBDataset):
    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: Optional[str],
    ):
        super().__init__(args, lmdb_path)

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        key = self.keys[idx].encode()
        value = self.txn.get(key)
        if value is None:
            raise IndexError(f"Name {key} has no data in the dataset")
        toks = bstr2obj(value)["seq"]

        data = {}

        x = torch.tensor([self.vocab[tok] - 1 for tok in toks], dtype=torch.int64)
        coords = torch.zeros([x.size()[0], 3], dtype=torch.float32)

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

        data = self.generate_2dgraphfeat(data)

        return data

    def collate(self, samples):
        return collate_fn(
            samples,
            multi_hop_max_dist=5,
            preprocess_2d_bond_features_with_cuda=True,
            sample_in_validation=True,
        )
