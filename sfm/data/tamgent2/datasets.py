# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Union

from torch.utils.data.dataloader import default_collate

from sfm.data.dataset import Data, InMemoryFoundationModelDataset


@dataclass
class TextToMolData(Data):
    smiles: Union[str, list[str]]
    text: Union[str, list[str]]


class TextToMolDataset(InMemoryFoundationModelDataset):
    def __init__(self, data: list[TextToMolData]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def collate(self, samples):
        return TextToMolData(
            smiles=[s.smiles for s in samples],
            text=[s.text for s in samples],
        )

    def __getitem__(self, index):
        return self.data[index]

    @classmethod
    def from_files(cls, mol_path, text_path):
        with open(mol_path, "r") as f:
            lines1 = f.read().splitlines()
        with open(text_path, "r") as f:
            lines2 = f.read().splitlines()

        data = []
        for l1, l2 in zip(lines1, lines2):
            data.append(
                TextToMolData(smiles=l1.replace("<m>", "").replace(" ", ""), text=l2)
            )

        return cls(data)
