# -*- coding: utf-8 -*-
from dataclasses import dataclass  # isort:skip

from apex.optimizers import FusedAdam as Adam  # isort:skip

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from sfm.data.mol_data.dataset import BatchedDataDataset
from sfm.data.mol_data.mattersim_data import MatterSimDataset
from sfm.logging import logger
from sfm.models.graphormer.graphormer_mattersim import GraphormerMatterSim
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.optim.set_lr import groupWarmupDecayLR


@dataclass
class MatbenchFinetuneConfig:
    data_path: str
    use_simple_head: bool = False
    force_loss_factor: float = 1.0


@cli(DistributedTrainConfig, PSMConfig, MatbenchFinetuneConfig)
def main(args):
    train_data = MatterSimDataset(args.data_path, split="train")
    valid_data = MatterSimDataset(args.data_path, split="valid")
    test_data = MatterSimDataset(args.data_path, split="test")

    logger.info(f"length of train_data {len(train_data)}.")

    max_node = 512
    train_data = BatchedDataDataset(
        train_data,
        dataset_version="3D",
        max_node=max_node,
        args=args,
        ft=True,
    )

    valid_data = BatchedDataDataset(
        valid_data,
        dataset_version="3D",
        max_node=max_node,
        args=args,
        ft=True,
    )

    test_data = BatchedDataDataset(
        test_data,
        dataset_version="3D",
        max_node=max_node,
        args=args,
        ft=True,
    )

    model = PSMModel(
        args,
        # energy_mean=-4.708115539360253,
        # energy_std=3.7354437106542777,
        # force_mean=0.0,
        # force_std=2.6599926818734785,
        # force_loss_factor=args.force_loss_factor,
    )
    optimizer = Adam(model.parameters(), lr=args.max_lr)

    lr_scheduler = groupWarmupDecayLR(
        optimizer,
        total_num_steps=args.total_num_steps,
        warmup_max_lr=args.max_lr,
        warmup_num_steps=args.warmup_num_steps,
    )

    logger.info(
        f"finetune: {args.ft}, add_3d: {args.add_3d}, infer: {args.infer}, no_2d: {args.no_2d}"
    )

    trainer = Trainer(
        args,
        model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()
    # trainer.validate()


if __name__ == "__main__":
    main()
