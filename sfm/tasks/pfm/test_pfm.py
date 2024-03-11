# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.extend([".", ".."])

from dataclasses import asdict, dataclass

from torch_scatter import scatter_add, scatter_max, scatter_mean

from sfm.criterions.mae3d import ProteinPMLM
from sfm.data.prot_data.dataset import BatchedDataDataset, DownstreamLMDBDataset
from sfm.logging import logger, metric_logger
from sfm.models.pfm.pfm_config import PFMConfig
from sfm.models.pfm.pfm_mlm_config import PfmMlmConfig
from sfm.models.pfm.pfm_mlm_model import PfmMlmBpeModel
from sfm.models.pfm.pfm_optimizer import DECAY_COSINE_RATE, groupWarmupDecayLR, myAdam
from sfm.models.pfm.pfmmodel import PFMModel
from sfm.pipeline.accelerator.dataclasses import (
    DistributedTrainConfig,
    ModelOutput,
    ValidLogOutput,
)
from sfm.pipeline.accelerator.trainer import Model, Trainer
from sfm.tasks.pfm.finetune_pfm import (
    DownstreamConfig,
    init_model,
    load_batched_dataset,
)
from sfm.utils.cli_utils import cli
from sfm.utils.move_to_device import move_to_device


def mae(pred, true):
    return torch.mean(torch.abs(pred - true))


def mse(pred, true):
    return torch.mean(torch.square(pred - true))


def rmse(pred, true):
    return torch.sqrt(torch.mean(torch.square(pred - true)))


def f1_max(pred, target):
    """
    F1 score with the optimal threshold.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


def accuracy(pred, target):
    """
    Classification accuracy.

    Suppose there are :math:`N` sets and :math:`C` categories.

    Parameters:
        pred (Tensor): prediction of shape :math:`(N, C)`
        target (Tensor): target of shape :math:`(N,)`
    """
    return (pred.argmax(dim=-1) == target).float().mean()


def binary_accuracy(pred, target):
    """
    Binary classification accuracy.

    Parameters:
        pred (Tensor): prediction of shape :math:`(N,)`
        target (Tensor): target of shape :math:`(N,)`
    """
    return ((pred > 0) == target).float().mean()


def pearsonr(pred, target):
    """
    Pearson correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    pred_mean = pred.float().mean()
    target_mean = target.float().mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_normalized = pred_centered / pred_centered.norm(2)
    target_normalized = target_centered / target_centered.norm(2)
    pearsonr = pred_normalized @ target_normalized
    return pearsonr


def spearmanr(pred, target):
    """
    Spearman correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(input):
        input_set, input_inverse = input.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = torch.zeros(len(input_inverse), device=input.device)
        ranking[order] = torch.arange(
            1, len(input) + 1, dtype=torch.float, device=input.device
        )

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(
            ranking, input_inverse, dim=0, dim_size=len(input_set)
        )
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr = covariance / (pred_std * target_std + 1e-10)
    return spearmanr


def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).

    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    pred, target = pred.flatten(), target.flatten()
    order = pred.argsort(descending=True)
    target = target[order]
    precision = target.cumsum(0) / torch.arange(
        1, len(target) + 1, device=target.device
    )
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc


def test(args, trainer):
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
            pred.append(output.to(torch.float32).squeeze().detach().cpu())
        if (
            DownstreamLMDBDataset.TASKINFO[args.task_name]["type"]
            == "multi_classification"
        ):
            target = batch_data["target"].unsqueeze(1)
            target_offset = batch_data["target_offset"]
            # logits: (B, N_classes)
            # multi_hot_target: (B, N_classes)
            multi_hot_target = torch.zeros(
                (output.shape[0], trainer.model.n_classes),
                dtype=torch.float32,
                device=target.device,
            )
            curr = 0
            for idx, n in enumerate(target_offset):
                label_idx = target[curr : curr + n]  # (n, 1)
                multi_hot_target[idx, label_idx] = 1
                curr += n
            target = multi_hot_target
            true.append(target.cpu())
        else:
            true.append(batch_data["target"].detach().cpu())

    if DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "regression":
        mean, std = DownstreamLMDBDataset.TASKINFO[args.task_name]["mean_std"]
        pred = torch.cat(pred, axis=0) * std + mean
        true = torch.cat(true, axis=0)
        test_fns = [pearsonr, spearmanr, mae, mse, rmse]
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "binary":
        pred = torch.cat(pred, axis=0)
        true = torch.cat(true, axis=0)
        test_fns = [binary_accuracy]
    elif DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "classification":
        pred = torch.cat(pred, axis=0)
        true = torch.cat(true, axis=0)
        test_fns = [accuracy]
    elif (
        DownstreamLMDBDataset.TASKINFO[args.task_name]["type"] == "multi_classification"
    ):
        pred = torch.cat(pred, axis=0)  # (B, N_classes)
        true = torch.cat(true, axis=0)
        test_fns = [f1_max, area_under_prc]
    else:
        raise NotImplementedError()
    assert pred.shape == true.shape
    results = dict()
    for fn in test_fns:
        results[fn.__name__] = fn(pred, true)
    logger.info(f"Checkpoint: {args.loadcheck_path}; Test results: {results}")
    metric_logger.log(results, "test")


@cli(DistributedTrainConfig, PFMConfig, PfmMlmConfig, DownstreamConfig)
def test_checkpoint(args) -> None:
    train_data, val_data, testset_dict = load_batched_dataset(args)

    model = init_model(args, load_ckpt=False)
    checkpoints_state = torch.load(args.loadcheck_path, map_location="cpu")
    if "model" in checkpoints_state:
        checkpoints_state = checkpoints_state["model"]
    elif "module" in checkpoints_state:
        checkpoints_state = checkpoints_state["module"]

    IncompatibleKeys = model.load_state_dict(checkpoints_state, strict=False)
    IncompatibleKeys = IncompatibleKeys._asdict()
    logger.info(f"checkpoint: {args.loadcheck_path} is loaded")
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
        test(args, trainer)


if __name__ == "__main__":
    test_checkpoint()
