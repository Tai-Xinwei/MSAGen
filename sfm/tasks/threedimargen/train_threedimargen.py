# -*- coding: utf-8 -*-
import math
import os
import sys

from sfm.criterions.autoregressive import AutoregressiveCriterion
from sfm.data.threedimargen_data.dataset import BatchedDataDataset, ThreeDimGenDataset
from sfm.data.threedimargen_data.tokenizer import ThreeDimTokenizer
from sfm.logging import logger
from sfm.models.threedimargen.threedimargen import ThreeDimARGenModel
from sfm.models.threedimargen.threedimargen_config import ThreeDimARGenConfig
from sfm.pipeline.accelerator.dataclasses import (
    DistributedTrainConfig,
    ModelOutput,
    TrainStrategy,
)
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.optim.optimizer import myAdam
from sfm.utils.optim.set_lr import groupWarmupDecayLR


@cli(DistributedTrainConfig, ThreeDimARGenConfig)
def main(args) -> None:
    assert (
        args.train_data_path is not None and len(args.train_data_path) > 0
    ), f"train_dataset is {args.train_data_path} it should not be None or empty"

    # assert (
    #    args.valid_data_path is not None and len(args.valid_data_path) > 0
    # ), f"valid_dataset is {args.valid_data_path} it should not be None or empty"

    tokenizer = ThreeDimTokenizer.from_file(args.dict_path, args)
    args.vocab_size = len(tokenizer)

    train_dataset = ThreeDimGenDataset(tokenizer, args.train_data_path, args)
    logger.info(f"loadded {len(train_dataset)} samples from train_dataset")
    train_data = BatchedDataDataset(train_dataset, args)
    if args.valid_data_path is not None and len(args.valid_data_path) > 0:
        valid_dataset = ThreeDimGenDataset(tokenizer, args.valid_data_path, args)
        logger.info(f"loadded {len(valid_dataset)} samples from valid_dataset")
        valid_data = BatchedDataDataset(valid_dataset, args)
    else:
        valid_data = None

    if os.path.exists(args.save_dir):
        args.ifresume = True
    model = ThreeDimARGenModel(args)

    optimizer, _ = myAdam(
        model,
        lr=args.max_lr,
        betas=[0.9, 0.999],
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    if args.total_num_epochs is not None and args.total_num_epochs > 0:
        if args.strategy == TrainStrategy.Single or args.strategy == TrainStrategy.DDP:
            world_size = (
                int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
            )
            total_batch_size = (
                args.train_batch_size * args.gradient_accumulation_steps * world_size
            )
        else:
            total_batch_size = args.train_batch_size
        args.total_num_steps = (
            math.ceil((len(train_data) // total_batch_size)) * args.total_num_epochs
        )

    lr_scheduler = groupWarmupDecayLR(
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
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
