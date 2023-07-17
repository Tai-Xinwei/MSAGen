# -*- coding: utf-8 -*-
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data.dataset import Dataset

from sfm.pipeline.accelerator.trainer import Model, ModelOutput, Trainer, TrainerConfig


class DummyNN(Model):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

    def compute_loss(self, pred, batch) -> ModelOutput:
        loss = F.l1_loss(pred, batch)
        return ModelOutput(loss=loss, log_output={"l2": F.mse_loss(pred, batch).item()})

    def config_optimizer(self) -> tuple[Optimizer, LRScheduler]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9)

        return optimizer, lr_scheduler


class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(1024, 10)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collater(batch):
    return torch.stack(batch)


def test_trainer():
    with tempfile.TemporaryDirectory() as save_dir:
        config = TrainerConfig(
            epochs=5, save_dir=save_dir, fp16=True, update_freq=2, log_interval=10
        )
        print(config)

        model = DummyNN()
        train_data = DummyDataset()
        valid_data = DummyDataset()
        test_data = DummyDataset()

        trainer = Trainer(
            config,
            model=model,
            collater=collater,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
        )
        trainer.train()


if __name__ == "__main__":
    test_trainer()
