# -*- coding: utf-8 -*-
from dataclasses import dataclass

from torch.utils.data.dataloader import default_collate

from sfm.data.dataset import Data, InMemoryFoundationModelDataset


@dataclass
class TextToMolData(Data):
    smiles: str
    text: str


class TextToMolDataset(InMemoryFoundationModelDataset):
    def __init__(self, mol_path, text_path):
        self.data = []
        with open(mol_path, "r") as f:
            lines1 = f.read().splitlines()
        with open(text_path, "r") as f:
            lines2 = f.read().splitlines()
        for l1, l2 in zip(lines1, lines2):
            self.data.append(
                TextToMolData(smiles=l1.replace("<m>", "").replace(" ", ""), text=l2)
            )

    def __len__(self):
        return len(self.data)

    def collate(self, samples):
        return default_collate(samples)

    def __getitem__(self, index):
        return self.data[index]
