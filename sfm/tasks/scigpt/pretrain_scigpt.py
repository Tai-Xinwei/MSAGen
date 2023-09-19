# -*- coding: utf-8 -*-

from sfm.data.sci_data.dataset import BatchedDataDataset, ProcessedSciDataset
from sfm.data.sci_data.SFMDecTokenizer import SFMDecTokenizer
from sfm.logging import logger
from sfm.models.scigpt.config import ScigptConfig
from sfm.models.scigpt.scigpt import ScigptModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import arg_utils
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

    tokenizer = SFMDecTokenizer.from_pretrained(args.dict_path)
    args.vocab_size = len(tokenizer)  # now we have new tokens
    args.pad_token_id = tokenizer.pad_token_id

    train_dataset = ProcessedSciDataset(args.train_data_path, tokenizer.pad_token_id)
    valid_dataset = ProcessedSciDataset(args.valid_data_path, tokenizer.pad_token_id)

    config = arg_utils.from_args(args, ScigptConfig)
    model = ScigptModel(config)

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

    trainer = Trainer(
        args,
        model=model,
        train_data=train_dataset,
        valid_data=valid_dataset,
        # optimizer=optimizer,
        # lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
