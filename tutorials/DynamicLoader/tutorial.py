# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from config import TutConfig
from dynamic_dataset import TextDataset
from model import TutModel

from sfm.logging import logger
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli


@cli(DistributedTrainConfig, TutConfig)
def main(args) -> None:
    texts = ["Hello, world!", "Goodbye, world!", "PyTorch is fun!"]

    train_data = TextDataset(texts, args=args)

    model = TutModel(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.max_lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    trainer = Trainer(
        args,
        model,
        train_data=train_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
