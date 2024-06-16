# -*- coding: utf-8 -*-
import os
from typing import Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data

from sfm.data.psm_data.dataset import MoleculeLMDBDataset
from sfm.models.psm.psm_config import PSMConfig


class PCQM4Mv2LMDBDataset(MoleculeLMDBDataset):
    latest_version: str = "20240604.1"

    def __init__(
        self,
        args: PSMConfig,
        lmdb_path: str,
        split: str = "train",
        version: Optional[str] = None,
    ):
        assert split in ["train", "valid", "test-dev", "test-challenge"]
        path = os.path.normpath(lmdb_path)
        if path.endswith("PCQM4Mv2"):
            path = os.path.join(
                path, version or PCQM4Mv2LMDBDataset.latest_version, split
            )
        super().__init__(args, path)

    def __getitem__(self, idx: Union[int, np.integer]) -> Data:
        data = super().__getitem__(idx)
        data["homo_lumo_gap"] = torch.tensor(
            [data["homo_lumo_gap"]], dtype=torch.float64
        )

        # don't use 3d structure
        # data["coords"] = torch.zeros_like(data["coords"])

        return data


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Config:
        max_length = 1e10
        preprocess_2d_bond_features_with_cuda = True

    ds = PCQM4Mv2LMDBDataset(Config(), "/data/psm/PCQM4Mv2")
    train_ds, valid_ds = ds.split_dataset()
    test_ds = PCQM4Mv2LMDBDataset(Config(), "/data/psm/PCQM4Mv2", "valid")
    print(f"train/valid/test: {len(train_ds)}/{len(valid_ds)}/{len(test_ds)}")
    print(train_ds[0])
