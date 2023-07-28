# -*- coding: utf-8 -*-
# Copyright 2022 Microsoft Corporation.
import math
from typing import Callable, List, Optional, Tuple

from deepspeed.runtime.lr_schedules import WarmupLR
from sfmlogging.loggers import sfm_logger
from torch.optim import Adam, Optimizer


def process_param(
    net,
    freeze_list: List = [],
    unfreeze_list: List = [],
    lr: float = 1e-5,
    mfm_lora: bool = False,
    **kwargs,
):
    param_groups = [{}]
    param_groups[0]["lr"] = lr
    param_groups[0]["params"] = []
    if len(unfreeze_list) > 0:
        for name, param in net.named_parameters():
            for unfreeze_name in unfreeze_list:
                if name.find(unfreeze_name) != -1:
                    param_groups[0]["params"].append(param)
    elif len(freeze_list) > 0:
        for name, param in net.named_parameters():
            for freeze_name in freeze_list:
                flag = False
                if name.find(freeze_name) != -1:
                    flag = True
                    break
            if not flag:
                param_groups[0]["params"].append(param)
    else:
        for name, param in net.named_parameters():
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
    return impl(new_param_groups, **kwargs)
