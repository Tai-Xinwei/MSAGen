# -*- coding: utf-8 -*-
import os
import sys

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from torch import nn

from sfm.data.prot_data.internal_dataset import (
    BatchedDataDataset,
    ToxInternalLMDBDataset,
    print_data,
)
from sfm.logging import logger
from sfm.models.tox.modules.tox_internal_loss import (
    InternalSeqMAELoss,
    InternalStruMAELoss,
    RebuiltCaDistogramLoss,
)
from sfm.models.tox.tox_internal import ToxInternalModel
from sfm.models.tox.tox_internal_config import ToxInternalConfig
from sfm.models.tox.tox_optimizer import groupWarmupDecayLR, myAdam
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli

VOCAB = None


class InitialLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seqmae = InternalSeqMAELoss(args)
        self.strumae = InternalStruMAELoss(args)
        self.distogram_loss = RebuiltCaDistogramLoss(args)
        self.args = self.check_args(args)

    def check_args(self, args):
        required_lst = [
            "seq_type_loss_weight",
            "disto_loss_weight",
            "bl_loss_weight",
            "ba_loss_weight",
            "ba_norm_loss_weight",
            "da_loss_weight",
            "da_norm_loss_weight",
        ]
        for k in required_lst:
            assert hasattr(
                args, k
            ), f"args should have {k} attribute in {self.__class__.__name__} class."
        return args

    def forward(self, batched_data: dict):
        seq_loss = self.seqmae(batched_data)
        stru_loss = self.strumae(batched_data)
        distogram_loss = self.distogram_loss(batched_data)
        loss_dict = {**seq_loss, **stru_loss, **distogram_loss}

        loss_dict["seq_type_loss"] = (
            self.args.seq_type_loss_weight * loss_dict["seq_type_loss"]
        )
        loss_dict["disto_loss"] = self.args.disto_loss_weight * loss_dict["disto_loss"]
        loss_dict["bl_loss"] = self.args.bl_loss_weight * loss_dict["bl_loss"]
        loss_dict["ba_loss"] = self.args.ba_loss_weight * loss_dict["ba_loss"]
        loss_dict["ba_norm_loss"] = (
            self.args.ba_norm_loss_weight * loss_dict["ba_norm_loss"]
        )
        loss_dict["da_loss"] = self.args.da_loss_weight * loss_dict["da_loss"]
        loss_dict["da_norm_loss"] = (
            self.args.da_norm_loss_weight * loss_dict["da_norm_loss"]
        )
        # logger.info(f"My logs, loss_dict: {loss_dict}")
        loss_dict["tot_loss"] = (
            loss_dict["seq_type_loss"]
            + loss_dict["disto_loss"]
            + loss_dict["bl_loss"]
            + loss_dict["ba_loss"]
            + loss_dict["ba_norm_loss"]
            + loss_dict["da_loss"]
            + loss_dict["da_norm_loss"]
        )
        return loss_dict


@cli(DistributedTrainConfig, ToxInternalConfig)
def main(args) -> None:
    global VOCAB
    dataset = ToxInternalLMDBDataset(args)
    VOCAB = dataset.vocab

    trainset, valset = dataset.split_dataset(sort=False, validation_ratio=0.05)
    logger.info(f"lenght of trainset: {len(trainset)}, lenght of valset: {len(valset)}")

    train_data = BatchedDataDataset(
        trainset,
        args=args,
    )
    val_data = BatchedDataDataset(
        valset,
        args=args,
    )

    model = ToxInternalModel(args, loss_fn=InitialLoss)

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
