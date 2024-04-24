# -*- coding: utf-8 -*-
import os
import sys

import torch

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from sfm.criterions.mae3ddiff import ProteinMAE3dCriterions
from sfm.data.prot_data.dataset import (
    BatchedDataDataset,
    ProteinLMDBDataset,
    StackedSequenceIterableDataset,
)
from sfm.logging import logger
from sfm.models.tox.modules.mae3ddiff import ProteinMAEDistCriterions
from sfm.models.tox.tox_config import TOXConfig
from sfm.models.tox.tox_optimizer import groupWarmupDecayLR, myAdam
from sfm.models.tox.toxmodel import TOXModel, TOXPDEModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli


@cli(DistributedTrainConfig, TOXConfig)
def main(args) -> None:
    assert (
        args.data_path is not None and len(args.data_path) > 0
    ), f"lmdb_path is {args.data_path} it should not be None or empty"
    if not args.fp16 and not args.bf16:
        torch.set_float32_matmul_precision("high")

    dataset = ProteinLMDBDataset(args)

    trainset, valset = dataset.split_dataset(sort=False)
    logger.info(f"lenght of trainset: {len(trainset)}, lenght of valset: {len(valset)}")

    if args.ifstack:
        train_data = StackedSequenceIterableDataset(trainset, args)
    else:
        train_data = BatchedDataDataset(
            trainset,
            args=args,
            vocab=dataset.vocab,
        )

    val_data = BatchedDataDataset(
        valset,
        args=args,
        vocab=dataset.vocab,
    )

    model = TOXModel(args, loss_fn=ProteinMAEDistCriterions)

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
        valid_data=val_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt!")
    finally:
        wandb.finish()  # support to finish wandb logging
        logger.info("wandb finish logging!")
