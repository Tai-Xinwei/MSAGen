# -*- coding: utf-8 -*-
import math
import os

import deepspeed
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from criterions.mae3d import MAE3dCriterions
from criterions.mae3ddiff import DiffMAE3dCriterions
from deepspeed.runtime.utils import see_memory_usage
from models.graphormer.graphormer import GraphormerModel
from models.graphormer.graphormerdiff import GraphormerDiffModel
from sfmlogging.loggers import sfm_logger
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils.get_paranum import count_paranum
from utils.move_to_device import move_to_device


class Trainer:
    def __init__(self, args, train_data, train_loader=None):
        super().__init__()

        net = GraphormerModel(args)
        count_paranum(net)
        self.args = args

        see_memory_usage("Model built", force=True)

        parameters = filter(lambda p: p.requires_grad, net.parameters())

        self.criterion_3d = MAE3dCriterions(args)

        self.model_engine, _, self.train_loader, _ = deepspeed.initialize(
            args=args,
            model=net,
            model_parameters=parameters,
            training_data=train_data,
            collate_fn=train_data.collater2,
        )

    def __call__(self):
        sfm_logger.info("start training")
        self.model_engine.module.train()
        global_step = 0
        for epoch in range(self.args.epochs):
            for i, batch_data in enumerate(self.train_loader):
                batch_data = move_to_device(
                    batch_data, device=self.args.local_rank, non_blocking=True
                )

                model_output = self.model_engine(batch_data)
                logits, node_output = model_output[0], model_output[1]

                loss = self.criterion_3d(batch_data, logits, node_output)

                self.model_engine.backward(loss)
                self.model_engine.step()

                if (i + 1) % 1000 == 0:
                    if self.args.local_rank == 0:
                        virt_mem = psutil.virtual_memory()
                        sfm_logger.info(
                            "epoch={}, micro_step={}, vm %: {}, global_rank: {}".format(
                                epoch, i, virt_mem.percent, self.args.rank
                            )
                        )

                global_step += 1

                if global_step % 10000 == 0:
                    self.model_engine.save_checkpoint(
                        save_dir=self.args.output_path,
                        client_state={"checkpoint_step": global_step},
                    )

                del loss
                torch.cuda.empty_cache()


class DiffTrainer:
    def __init__(self, args, train_data, train_loader=None):
        super().__init__()

        net = GraphormerDiffModel(args)
        count_paranum(net)
        self.args = args

        see_memory_usage("Model built", force=True)

        parameters = filter(lambda p: p.requires_grad, net.parameters())

        self.criterion_3d = DiffMAE3dCriterions(args)

        self.model_engine, _, self.train_loader, _ = deepspeed.initialize(
            args=args,
            model=net,
            model_parameters=parameters,
            training_data=train_data,
            collate_fn=train_data.collater2,
        )

    def __call__(self):
        sfm_logger.info("start training")
        self.model_engine.module.train()
        global_step = 0
        for epoch in range(self.args.epochs):
            for i, batch_data in enumerate(self.train_loader):
                batch_data = move_to_device(
                    batch_data, device=self.args.local_rank, non_blocking=True
                )

                model_output = self.model_engine(batch_data)
                logits, node_output, y_pred = (
                    model_output[0],
                    model_output[1],
                    model_output[2],
                )

                loss = self.criterion_3d(batch_data, logits, node_output, y_pred)

                self.model_engine.backward(loss)
                self.model_engine.step()

                if (i + 1) % 1000 == 0:
                    if self.args.local_rank == 0:
                        virt_mem = psutil.virtual_memory()
                        sfm_logger.info(
                            "epoch={}, micro_step={}, vm %: {}, global_rank: {}".format(
                                epoch, i, virt_mem.percent, self.args.rank
                            )
                        )

                global_step += 1

                if global_step % 10000 == 0:
                    self.model_engine.save_checkpoint(
                        save_dir=self.args.output_path,
                        client_state={"checkpoint_step": global_step},
                    )

                del loss
                torch.cuda.empty_cache()
