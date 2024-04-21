# -*- coding: utf-8 -*-
import os
import sys

import wandb  # isort:skip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from sfm.data.psm_data.unifieddataset import BatchedDataDataset, UnifiedPSMDataset
from sfm.logging import logger
from sfm.models.psm.loss.mae3ddiff import DiffMAE3dCriterions
from sfm.models.psm.psm_config import PSMConfig
from sfm.models.psm.psmmodel import PSMModel
from sfm.pipeline.accelerator.dataclasses import DistributedTrainConfig
from sfm.pipeline.accelerator.trainer import Trainer
from sfm.utils.cli_utils import cli


@cli(DistributedTrainConfig, PSMConfig)
def main(args) -> None:
    ### define psm dataset here
    dataset = UnifiedPSMDataset(
        args.data_path, args.data_path_list, args.dataset_name_list, args
    )
    train_data, valid_data = dataset.split_dataset()

    train_data = BatchedDataDataset(args, train_data, dataset.train_len)
    valid_data = BatchedDataDataset(args, valid_data, dataset.valid_len)

    ### define psm models here, define the diff loss in DiffMAE3dCriterions
    model = PSMModel(args, loss_fn=DiffMAE3dCriterions)

    trainer = Trainer(
        args,
        model,
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
