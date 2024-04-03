# -*- coding: utf-8 -*-
import copy
import itertools
from functools import lru_cache
from multiprocessing import Pool

from sfm.data.dataset import FoundationModelDataset
from sfm.data.psm_data.dataset import (
    MatterSimDataset,
    PM6FullLMDBDataset,
    ProteinLMDBDataset,
)


class UnifiedPSMDataset(FoundationModelDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return super().__len__()


if __name__ == "__main__":
    pass
