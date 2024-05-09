# -*- coding: utf-8 -*-
import os
import sys

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

from sfm.data.psm_data.unifieddataset import (
    BatchedDataDataset,
    StackedIterableDataset,
    UnifiedPSMDataset,
)
from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils import dist_utils
from sfm.utils.cli_utils import wandb_init


@dataclass
class Config(DistributedTrainConfig, PSMConfig):
    backbone_config: Dict[str, Any] = MISSING
    backbone: str = "graphormer"
    ode_mode: bool = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


# @cli(DistributedTrainConfig, PSMConfig)
@hydra.main(
    version_base="1.3", config_path="../../../config_file", config_name="config_psm"
)
def main(args: DictConfig) -> None:
    schema = OmegaConf.structured(Config)
    args = OmegaConf.merge(schema, args)
    args = Config(**OmegaConf.to_container(args, resolve=True))

    wandb_init(args)

    ### define psm dataset here
    dataset = UnifiedPSMDataset(
        args.data_path, args.data_path_list, args.dataset_name_list, args
    )
    train_data, valid_data = dataset.split_dataset()

    if args.ifstack:
        raise NotImplementedError("ifstack is not finished yet!")
        # train_data = StackedIterableDataset(train_data, args, dataset.sizes)
    else:
        train_data = BatchedDataDataset(args, train_data, dataset.train_len)

    valid_data = BatchedDataDataset(args, valid_data, dataset.valid_len)

    ### define psm models here, define the diff loss in DiffMAE3dCriterions
    model = PSMModel(args, loss_fn=DiffMAE3dCriterions)

    # define optimizer here
    optimizer = myAdam(
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
        decay_type=DECAY_COSINE_RATE,
    )

    trainer = Trainer(
        args,
        model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=train_data,
        valid_data=valid_data,
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
