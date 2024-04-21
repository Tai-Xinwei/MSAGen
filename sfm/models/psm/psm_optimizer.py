# -*- coding: utf-8 -*-
# Copyright 2022 Microsoft Corporation.
import math
from typing import Callable, List, Optional, Tuple

from deepspeed.runtime.lr_schedules import WarmupLR
from torch.optim import Optimizer

from sfm.logging.loggers import logger

from torch.optim import Adam  # isort:skip


def process_param(
    net,
    freeze_list: List = [],
    unfreeze_list: List = [],
    lr: float = 1e-5,
    mfm_lora: bool = False,
    **kwargs,
):
    param_groups = [{}, {}]
    param_groups[0]["lr"] = 0.0
    param_groups[0]["params"] = []
    param_groups[1]["lr"] = 0.0
    param_groups[1]["params"] = []

    for name, param in net.named_parameters():
        if (
            name.find("fc_pmlm_q") != -1
            or name.find("fc_pmlm_k") != -1
            or name.find("embed_out") != -1
            or name.find("head") != -1
            or name.find("structure_module") != -1
        ):
            param_groups[1]["params"].append(param)
        else:
            param_groups[0]["params"].append(param)

    for param_group in param_groups:
        if "lr" not in param_group:
            param_group["lr"] = kwargs["lr"]
        if "weight_decay" not in param_group:
            param_group["weight_decay"] = kwargs.get("weight_decay", 0.0)

    return param_groups


def myAdam(
    net,
    impl=Adam,
    freeze_list: List = [],
    unfreeze_list: List = [],
    mfm_lora=False,
    **kwargs,
):
    assert (
        len(freeze_list) == 0 or len(unfreeze_list) == 0
    ), "freeze_list and unfreeze_list cannot be set at the same time"

    new_param_groups = []
    param_groups = process_param(
        net,
        freeze_list=freeze_list,
        unfreeze_list=unfreeze_list,
        mfm_lora=mfm_lora,
        **kwargs,
    )
    for param_group in param_groups:
        new_param_groups.extend([param_group])
    return impl(new_param_groups, **kwargs), param_groups


WARMUP_LOG_RATE = "log"
WARMUP_LINEAR_RATE = "linear"
DECAY_LINEAR_RATE = "linear"
DECAY_COSINE_RATE = "cosine"


class groupWarmupDecayLR(WarmupLR):
    """Increase the learning rate of each parameter group from min lr to max lr
    over warmup_num_steps steps, and then decay at linear rate over the remaining training steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_num_steps (int): total number of training steps
        warmup_min_lr (float or list): minimum learning rate. Default: 0
        warmup_max_lr (float or list): maximum learning rate. Default: 0.001
        warmup_num_steps (int): number of steps to warm up from min_lr to max_lr. Default: 1000
        warmup_type {'log', 'linear'}: increasing function from min_lr to max_lr during warmup. Default: log
        last_batch_iteration (int): The index of the last batch. Default: -1.
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = WarmupDecayLR(optimizer, 1000000)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_num_steps: int,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 0.001,
        warmup_num_steps: int = 1000,
        warmup_type: str = WARMUP_LINEAR_RATE,
        last_batch_iteration: int = -1,
        d_tilde: float = 1.0,
        decay_type: str = DECAY_LINEAR_RATE,
    ):
        self.total_num_steps = total_num_steps
        super(groupWarmupDecayLR, self).__init__(
            optimizer,
            warmup_min_lr,
            warmup_max_lr,
            warmup_num_steps,
            warmup_type,
            last_batch_iteration,
        )
        self.d_tilde = d_tilde
        self.decay_type = decay_type

        if self.total_num_steps < self.warmup_num_steps:
            logger.warning(
                "total_num_steps {} is less than warmup_num_steps {}".format(
                    total_num_steps, warmup_num_steps
                )
            )
        for group in self.optimizer.param_groups:
            group["lr"] = 0.0

    def step(self, last_batch_iteration=None):
        """Update the learning rate of each parameter group."""
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self.optimizer.param_groups[1]["lr"] /= self.d_tilde

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _get_gamma(self):
        if self.last_batch_iteration < self.warmup_num_steps:
            if self.warmup_type == WARMUP_LOG_RATE:
                return self.inverse_log_warm_up * math.log(
                    self.last_batch_iteration + 1
                )
            elif self.warmup_type == WARMUP_LINEAR_RATE:
                return self.last_batch_iteration / self.warmup_num_steps
        else:
            if self.decay_type == DECAY_LINEAR_RATE:
                return max(
                    0.0,
                    float(self.total_num_steps - self.last_batch_iteration)
                    / float(max(1.0, self.total_num_steps - self.warmup_num_steps)),
                )
            else:
                return 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * float(self.last_batch_iteration - self.warmup_num_steps)
                        / float(max(1.0, self.total_num_steps - self.warmup_num_steps))
                    )
                )
