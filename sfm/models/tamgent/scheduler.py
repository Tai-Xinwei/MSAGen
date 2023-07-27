# -*- coding: utf-8 -*-
import math

from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_epoch,
        iters_per_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs,
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        print(self.warmup_steps)
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        total_cur_step = self._step_count
        print(self._step_count)
        if total_cur_step < self.warmup_steps:
            return warmup_lr_schedule(
                step=total_cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            return cosine_lr_schedule(
                epoch=total_cur_step,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch * self.iters_per_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    return [lr for param_group in optimizer.param_groups]


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    # print(step)
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    return [lr for param_group in optimizer.param_groups]


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
