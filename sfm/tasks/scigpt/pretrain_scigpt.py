# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.data.sci_data.dataset import BatchedDataDataset, SciDataset
from sfm.logging import logger
from sfm.models.scigpt.scigpt import ScigptModel
from sfm.models.scigpt.scigpt_config import ScigptConfig
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import groupWarmupDecayLR


@cli(DistributedTrainConfig, ScigptConfig)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    assert (
        args.valid_data_path is not None and len(args.valid_data_path) > 0
    ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    train_dataset = SciDataset(args.dict_path, args.train_data_path, args)
    valid_dataset = SciDataset(args.dict_path, args.valid_data_path, args)

    train_data = BatchedDataDataset(train_dataset, args)
    valid_data = BatchedDataDataset(valid_dataset, args)

    model = ScigptModel(args, loss_fn=AutoregressiveCriterion)

    optimizer, _ = myAdam(
        model,
        lr=args.max_lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    groupWarmupDecayLR(
        optimizer,
        total_num_steps=args.total_num_steps,
        warmup_max_lr=args.max_lr,
        warmup_num_steps=args.warmup_num_steps,
    )

    logger.info(f"finetune: {args.ft}, infer: {args.infer}")

    trainer = Trainer(
        args,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        # optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
