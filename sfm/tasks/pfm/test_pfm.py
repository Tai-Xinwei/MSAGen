# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import asdict, dataclass

import numpy as np
from finetune_pfm import DownstreamConfig, init_model, load_batched_dataset
from scipy import stats

from sfm.criterions.mae3d import ProteinPMLM
from sfm.data.prot_data.dataset import BatchedDataDataset, DownstreamLMDBDataset
from sfm.logging import logger, metric_logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.pfm.pfmmodel import PFMModel
from sfm.pipeline.accelerator.dataclasses import (
    DistributedTrainConfig,
    ModelOutput,
    ValidLogOutput,
)
from sfm.pipeline.accelerator.trainer import Model, Trainer
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def mse(pred, true):
    return np.mean(np.square(pred - true))


def rmse(pred, true):
    return np.sqrt(np.mean(np.square(pred - true)))


def pearsonr(pred, true):
    return stats.pearsonr(pred, true)[0]


def test(args, trainer, test_fns):
    """
    Validate the model on the validation data loader.
    """
    if trainer.valid_data_loader is None:
        logger.warning("No validation data, skip validation")
        return

    logger.info(
        "Start validation for epoch: {}, global step: {}",
        trainer.state.epoch,
        trainer.state.global_step,
    )

    pred, true = [], []

    for idx, batch_data in enumerate(trainer.valid_data_loader):
        trainer.model.eval()
        trainer.model.to(trainer.accelerator.device)
        batch_data = move_to_device(batch_data, trainer.accelerator.device)
        with torch.no_grad():
            output = trainer.model(batch_data)
            pred.append(output.to(torch.float32).squeeze().detach().cpu().numpy())
        true.append(batch_data["target"].detach().cpu().numpy())

    if DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "regression":
        mean, std = DownstreamLMDBDataset.TASKINFO[args.task_name]["mean_std"]
        pred = np.concatenate(pred, axis=0) * std + mean
    else:
        pred = np.concatenate(pred, axis=0)
    true = np.concatenate(true, axis=0)
    assert pred.shape == true.shape
    results = dict()
    for fn in test_fns:
        results[fn.__name__] = fn(pred, true)
    metric_logger.log(results, "test")


@cli(DistributedTrainConfig, PFMConfig, DownstreamConfig)
def test_checkpoint(args) -> None:
    train_data, val_data, testset_dict = load_batched_dataset(args)

    model = init_model(args)
    checkpoints_state = torch.load(args.loadcheck_path, map_location="cpu")
    if "model" in checkpoints_state:
        checkpoints_state = checkpoints_state["model"]
    elif "module" in checkpoints_state:
        checkpoints_state = checkpoints_state["module"]

    IncompatibleKeys = model.load_state_dict(checkpoints_state, strict=False)
    IncompatibleKeys = IncompatibleKeys._asdict()
    logger.warning(f"Following keys are incompatible: {IncompatibleKeys.keys()}")

    # TODO: now no way to calculate the accuracy, where we should get all the output from testset and calculate
    # only batched accuracy or batched something else are available
    for name, testset in testset_dict.items():
        trainer = Trainer(
            args,
            model,
            train_data=train_data,
            valid_data=testset,
        )
        test(args, trainer, [pearsonr, mae, mse])


if __name__ == "__main__":
    test_checkpoint()
