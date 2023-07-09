import copy
import os
from typing import Optional, Tuple

import deepspeed
import torch
import torch.nn as nn
from deepspeed.runtime.utils import see_memory_usage
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        args,
        train_data: Dataset,
        val_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
        strategy: str = "zero1",
        data_mean: float = 0.0,
        data_std: float = 1.0,
    ):
        super().__init__()

        assert strategy in [
            "zero1",
            "zero2",
            "zero3",
            "ddp",
        ], "Strategy should be one of ['zero1', 'zero2', 'zero3', 'ddp']"
        # define model
        # net = GraphormerModel(args)
        # count_paranum(net)

        # define criterion
        # self.L1loss = L1_criterions(args, reduction='mean', data_mean=data_mean, data_std=data_std)

        # define optimizer
        # parameters = filter(lambda p: p.requires_grad, net.parameters())
        # self.optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

        # define scheduler
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0.0)

        # define model engine
        # self.model_engine, _, self.train_loader, _ = deepspeed.initialize(args=args,
        #                                                                   model=net,
        #                                                                   model_parameters=parameters,
        #                                                                   training_data=train_data,
        #                                                                   collate_fn=train_data.collater2,
        #                                                                   optimizer=self.optimizer,
        #                                                                   lr_scheduler=self.scheduler,
        #                                                                   loss_scale=args.loss_scale,
        #                                                                   loss_fn=self.L1loss,
        #                                                                   )

        # define dataloader
        # len_val, self.val_dataloader = self.set_dataloader(data=val_data)
        # len_test, self.test_dataloader = self.set_dataloader(data=test_data)

        # load checkpoints
        # self.load_checkpoint()

    def load_checkpoint(self, checkpoint_path=None):
        pass

    def set_dataloader(self, data=None):
        len_data = 0
        if data is not None:
            len_data = len(data)
            validsampler = torch.utils.data.distributed.DistributedSampler(
                data, num_replicas=self.model_engine.dp_world_size, shuffle=True
            )
            dataloader = torch.utils.data.DataLoader(
                data,
                sampler=validsampler,
                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                collate_fn=data.collater,
            )

        return len_data, dataloader

    def run(self, dataloader, iftrain=True):
        pass

    def __call__(self, iftrain=True):
        print("start training")
        for ep in range(self.args.epochs):
            pass
