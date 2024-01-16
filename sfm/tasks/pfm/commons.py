# -*- coding: utf-8 -*-
import torch
from torch_scatter import scatter_add, scatter_max, scatter_mean


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
    return ((torch.sigmoid(pred) > 0.5) == target).float().mean()


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


def variadic_mean(input, size):
    """
    Compute mean over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        input (Tensor): input of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    index2sample = torch.repeat_interleave(size)
    index2sample = index2sample.view([-1] + [1] * (input.ndim - 1))
    index2sample = index2sample.expand_as(input)

    value = scatter_mean(input, index2sample, dim=0)
    return value
