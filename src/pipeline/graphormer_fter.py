import copy
import math
import os

import deepspeed
# from tqdm import tqdm
# import logging
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.runtime.utils import see_memory_usage
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from criterions.L1ft import Binary_criterions, L1_criterions
from criterions.mae3d import MAE3d_criterions
from models.graphormer import GraphormerModel
from utils.get_paranum import count_paranum
# from graphormer.data.dataset import PCQPreprocessedData, BatchedDataDataset
# from graphormer.data.wrapper import MyPygPCQM4MDataset
# from torch.nn.parallel import DistributedDataParallel as DDP
from utils.move_to_device import move_to_device


class Finetuner:
    def __init__(
        self,
        args,
        train_data,
        val_data=None,
        test_data=None,
        data_mean=0.0,
        data_std=1.0,
    ):
        super().__init__()

        net = GraphormerModel(args)
        count_paranum(net)
        self.args = args
        if args.rank == 0:
            self.writer = SummaryWriter("../output")

        see_memory_usage("Model built", force=True)

        parameters = filter(lambda p: p.requires_grad, net.parameters())

        self.L1loss = L1_criterions(
            args, reduction="mean", data_mean=data_mean, data_std=data_std
        )

        self.model_engine, _, self.train_loader, _ = deepspeed.initialize(
            args=args,
            model=net,
            model_parameters=parameters,
            training_data=train_data,
            collate_fn=train_data.collater2,
        )
        self.len_train = len(train_data)

        self.load_checkpoint()
        self.set_dataloader(val_data=val_data, test_data=test_data)

    def load_checkpoint(self, checkpoint_path=None):
        pass

    def set_dataloader(self, val_data=None, test_data=None):
        self.len_val = 0
        self.len_test = 0
        if val_data is not None:
            self.len_val = len(val_data)
            validsampler = torch.utils.data.distributed.DistributedSampler(
                val_data, num_replicas=self.model_engine.dp_world_size, shuffle=True
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                val_data,
                sampler=validsampler,
                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                collate_fn=val_data.collaterft,
            )

        if test_data is not None:
            self.len_test = len(test_data)
            testsampler = torch.utils.data.distributed.DistributedSampler(
                test_data, num_replicas=self.model_engine.dp_world_size, shuffle=True
            )
            self.test_dataloader = torch.utils.data.DataLoader(
                test_data,
                sampler=testsampler,
                batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                collate_fn=val_data.collaterft,
            )

    def run(self, dataloader, iftrain=True):
        if iftrain:
            self.model_engine.module.train()
        else:
            self.model_engine.module.eval()

        running_loss = 0.0
        for i, batch_data in enumerate(dataloader):
            batch_data = move_to_device(
                batch_data, device=self.args.local_rank, non_blocking=True
            )

            if iftrain:
                model_output = self.model_engine(batch_data)
            else:
                with torch.no_grad():
                    model_output = self.model_engine(batch_data)

            logits, node_output = model_output[0], model_output[1]

            loss = self.L1loss(batch_data, logits, node_output)
            running_loss += loss.detach().item()

            if iftrain:
                self.model_engine.backward(loss)
                self.model_engine.step()

            del loss
            torch.cuda.empty_cache()

        return running_loss / len(dataloader)

    def __call__(self, iftrain=True):
        print("start training")
        best_val_loss = 100000
        best_test_loss = 100000
        for ep in range(self.args.epochs):
            # train
            running_loss = self.run(self.train_loader, iftrain=True)

            if self.args.rank == 0:
                print("Epoch: {}, Avg Train loss: {}".format(ep, running_loss))

            # valid
            if self.len_val != 0:
                vallaoder = copy.deepcopy(self.valid_dataloader)
                running_loss = self.run(vallaoder, iftrain=False)
                del vallaoder

                if running_loss < best_val_loss:
                    best_val_loss = running_loss

                if self.args.rank == 0 and self.len_val != 0:
                    print(
                        "Epoch: {}, Avg eval loss: {}, best eval loss: {}".format(
                            ep, running_loss, best_val_loss
                        )
                    )

            # test
            if self.len_test != 0:
                testloader = copy.deepcopy(self.test_dataloader)
                running_loss = self.run(testloader, iftrain=False)
                del testloader

                if running_loss < best_test_loss:
                    best_test_loss = running_loss

                if self.args.rank == 0 and self.len_test != 0:
                    print(
                        "Epoch: {}, Avg test loss: {}, best test loss: {}".format(
                            ep, running_loss, best_test_loss
                        )
                    )

            torch.cuda.empty_cache()

        self.writer.flush()
        self.writer.close()
