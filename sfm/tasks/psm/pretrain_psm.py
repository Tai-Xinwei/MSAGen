# -*- coding: utf-8 -*-
import os
import sys

import deepspeed
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])
from argparse import ArgumentParser

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
    ### train_data = ...
    ### valid_data = ...

    ### define psm models here, define the diff loss in DiffMAE3dCriterions
    model = PSMModel(args, loss_fn=DiffMAE3dCriterions)

    logger.info(
        f"finetune: {args.ft}, add_3d: {args.add_3d}, infer: {args.infer}, no_2d: {args.no_2d}"
    )

    trainer = Trainer(
        args,
        model,
        # train_data=train_data,
        # valid_data=valid_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()
