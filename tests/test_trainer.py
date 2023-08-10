# -*- coding: utf-8 -*-
import tempfile
import unittest
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.data.dataset import Batch, Data, FoundationModelDataset
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer, TrainerConfig


@dataclass
class DummyData(Data):
    x: torch.Tensor


@dataclass
class DummyBatch(Batch):
    x: torch.Tensor


class DummyNN(Model):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x: DummyBatch):
        return self.fc(x.x)

    def compute_loss(self, pred, batch) -> ModelOutput:
        loss = F.l1_loss(pred, batch.x)
        return ModelOutput(
            loss=loss, log_output={"l2": F.mse_loss(pred, batch.x).item()}
        )

    def config_optimizer(self) -> Tuple[Optimizer, LRScheduler]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)

        return optimizer, lr_scheduler


class DummyDataset(FoundationModelDataset):
    def __init__(self):
        super().__init__()
        self.data = [DummyData(torch.randn(10)) for _ in range(100)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        return DummyBatch(
            x=torch.stack([b.x for b in batch]),
            batch_size=len(batch),
        )


class Test_Trainer(unittest.TestCase):
    def test_trainer(self):
        with tempfile.TemporaryDirectory() as save_dir:
            config = TrainerConfig(
                epochs=3,
                save_dir=save_dir,
                gradient_accumulation_steps=2,
                log_interval=10,
            )

            model = DummyNN()
            train_data = DummyDataset()
            valid_data = DummyDataset()

            trainer = Trainer(
                config,
                model=model,
                train_data=train_data,
                valid_data=valid_data,
            )
            trainer.train()


if __name__ == "__main__":
    unittest.main()
