# -*- coding: utf-8 -*-
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sfm.data.dataset import Batch, Data, FoundationModelDataset
from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import ModelOutput
from sfm.pipeline.accelerator.trainer import Model, Trainer, TrainerConfig
from sfm.utils.mypp_module import partition_by_layers


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


class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        with tempfile.TemporaryDirectory() as save_dir:
            config = TrainerConfig(
                total_num_epochs=3,
                save_dir=save_dir,
                gradient_accumulation_steps=1,
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


class TestPPLayerPartition(unittest.TestCase):
    def test_pp_layer_partition(self):
        for num_layers in range(100):
            binary_weights = [0] + [1] * num_layers + [0] * 4
            for num_stages in range(1, num_layers):
                partition_by_layers(binary_weights, num_stages, len(binary_weights))


class TestLoadCheckpoint(unittest.TestCase):
    def test_load_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as save_dir, tempfile.TemporaryDirectory() as finetune_from_dir:
            config = TrainerConfig(
                save_dir=save_dir,
                total_num_epochs=3,
                finetune_from_checkpoint_dir=str(finetune_from_dir),
            )
            model = DummyNN()
            train_data = DummyDataset()
            valid_data = DummyDataset()

            trainer = Trainer(
                config, model=model, train_data=train_data, valid_data=valid_data
            )

            with StringIO() as out:
                out_id = logger.add(out)
                trainer.resume()
                logger.complete()
                output = out.getvalue().strip()
                assert (
                    output.find(
                        f"Non-empty checkpoint_list.txt or latest file is not present in {save_dir}, "
                        "or finetune_from_checkpoint_id is not provided. No checkpoint is loaded."
                    )
                    != -1
                )
                logger.remove(out_id)

            with StringIO() as out:
                out_id = logger.add(out)
                with open(
                    Path(save_dir) / "checkpoint_list.txt", "w"
                ) as checkpoint_list:
                    checkpoint_list.write("global_step10")
                trainer.resume()
                logger.complete()
                output = out.getvalue().strip()
                assert (
                    output.find(
                        f"Checkpoint path {save_dir}/global_step10 does not exist."
                    )
                    != -1
                )
                logger.remove(out_id)

            with StringIO() as out:
                out_id = logger.add(out)
                trainer.train()
                trainer.resume()
                logger.complete()
                output = out.getvalue().strip()
                assert (
                    output.find(f"Resume from checkpoint: {save_dir}/checkpoint_E2.pt")
                    != -1
                )
                logger.remove(out_id)

            with StringIO() as out:
                out_id = logger.add(out)
                shutil.copy(f"{save_dir}/checkpoint_E2.pt", f"{finetune_from_dir}/")
                config = TrainerConfig(
                    save_dir=save_dir,
                    total_num_epochs=3,
                    finetune_from_checkpoint_dir=str(finetune_from_dir),
                    finetune_from_checkpoint_id="checkpoint_E2.pt",
                )
                finetune_trainer = Trainer(
                    config, model=model, train_data=train_data, valid_data=valid_data
                )
                finetune_trainer.train()
                logger.complete()
                output = out.getvalue().strip()
                assert (
                    output.find(
                        f"Finetune from checkpoint: {finetune_from_dir}/checkpoint_E2.pt"
                    )
                    != -1
                )
                logger.remove(out_id)


if __name__ == "__main__":
    unittest.main()
