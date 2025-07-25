# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Union

from sfm.data.dataset import Batch, Data, InMemoryFoundationModelDataset
from sfm.logging import logger


@dataclass
class TextToMolData(Data):
    smiles: str
    text: str


@dataclass
class BatchTextToMolData(Batch):
    smiles: List[str]
    text: List[str]


class TextToMolDataset(InMemoryFoundationModelDataset):
    def __init__(self, data: List[TextToMolData]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def collate(self, samples):
        return BatchTextToMolData(
            smiles=[s.smiles for s in samples],
            text=[s.text for s in samples],
            batch_size=len(samples),
        )

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def from_files(cls, mol_path, text_path):
        with open(mol_path, "r") as f:
            lines1 = f.read().splitlines()
        with open(text_path, "r") as f:
            lines2 = f.read().splitlines()

        assert len(lines1) == len(lines2)
        data_size = len(lines1)

        data = []
        logger.info(
            "Loading data from {} and {}. Data size {}", mol_path, text_path, data_size
        )
        for l1, l2 in zip(lines1, lines2):
            data.append(
                TextToMolData(smiles=l1.replace("<m>", "").replace(" ", ""), text=l2)
            )

            if len(data) % 10000 == 0:
                logger.info("Loaded {}/{} data", len(data), data_size)

        logger.info("Loaded {}/{} data", len(data), data_size)
        logger.info("First example:\n {}", data[0])

        return cls(data)
