# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from sfm.data.prot_data.dataset import (
    BatchedDataDataset,
    PackedBpeUR50LMDBDataset,
    PackedUR50LMDBDataset,
    PackedUR50LMDBMultiSrcDataset,
    StackedSequenceIterableDataset,
    UR50LMDBDataset,
)
from sfm.logging import logger
from sfm.models.pfm.bfm_loss import (
    ProteinMAE3dCriterions,
    ProteinMLM,
    ProteinPMLM,
    ProteinPMLMMSA,
)
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.pfm.pfmmodel import PFMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli


@cli(DistributedTrainConfig, PFMConfig)
def main(args) -> None:
    trainset = PackedBpeUR50LMDBDataset(args, args.train_data_path)
    valset = PackedBpeUR50LMDBDataset(args, args.valid_data_path)

    if args.stack_seq:
        train_data = StackedSequenceIterableDataset(
            trainset,
            args=args,
        )
    else:
        train_data = BatchedDataDataset(
            trainset,
            args=args,
            vocab=trainset.vocab,
        )

    val_data = BatchedDataDataset(
        valset,
        args=args,
        vocab=trainset.vocab,
    )

    model = PFMModel(args, loss_fn=ProteinPMLMMSA)

    optimizer, _ = myAdam(
        model,
        lr=args.max_lr,
        betas=[0.9, 0.98],
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    lr_scheduler = groupWarmupDecayLR(
        optimizer,
        total_num_steps=args.total_num_steps,
        warmup_max_lr=args.max_lr,
        warmup_num_steps=args.warmup_num_steps,
        d_tilde=2,
        decay_type=DECAY_COSINE_RATE,
    )

    logger.info(
        f"finetune: {args.ft}, add_3d: {args.add_3d}, infer: {args.infer}, no_2d: {args.no_2d}"
    )

    trainer = Trainer(
        args,
        model,  # =torch.compile(model),
        train_data=train_data,
        valid_data=val_data,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    main()
