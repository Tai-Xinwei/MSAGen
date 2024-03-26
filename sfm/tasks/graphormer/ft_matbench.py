# -*- coding: utf-8 -*-
import os

# import ogb
import sys

# import pytorch_forecasting
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import dataclass

from sfm.criterions.l1ft import L1Criterions
from sfm.data.mol_data.dataset import BatchedDataDataset
from sfm.data.mol_data.matbench_data import MatbenchDataset
from sfm.logging import logger
from sfm.models.graphormer.graphormer import GraphormerModel
from sfm.models.graphormer.graphormer_config import GraphormerConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli

# from sfm.pipeline.graphormer.graphormer_fter_bk import Finetuner
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import groupWarmupDecayLR


@dataclass
class MatbenchFinetuneConfig:
    matbench_task_name: str
    matbench_task_fold_id: int


@cli(DistributedTrainConfig, GraphormerConfig, MatbenchFinetuneConfig)
def main(args):
    train_data = MatbenchDataset(
        args.matbench_task_name, args.matbench_task_fold_id, "train_val"
    )
    test_data = MatbenchDataset(
        args.matbench_task_name, args.matbench_task_fold_id, "test"
    )

    logger.info(
        f"length of train_data {len(train_data)}, length of test_data {len(test_data)}."
    )

    max_node = 512
    train_data = BatchedDataDataset(
        train_data,
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

    model = GraphormerModel(args, loss_fn=L1Criterions)

    optimizer, _ = myAdam(
        model,
        lr=args.max_lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

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
        valid_data=test_data,
        test_data=test_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
